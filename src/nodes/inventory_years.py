"""
Add and remove the historical inventory years of an instance.

The inventory years are the historical span without its skipped years
(`YearsSpec.historical`). Adding a year declares it and seeds blank input cells
for it; blank cells make the year present to the dataset rules, so an added
year is expected to be completed before the instance publishes. A year outside
the span widens the span, and the years it jumps over are skipped.

Removing a year is the inverse: it deletes the year's local cells and
un-declares it. An inner year becomes skipped; an edge year shrinks the span to
the nearest inventory year that remains.

Submissions are not touched. A submission is its own interaction, and a year
that has one cannot be removed.

An instance whose inventory years were never declared (`skipped is None`) is
taken at its first add or remove to have declared its whole span. That is what
its first and last historical years already asserted; it claims nothing about
whether those years hold observed data.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from django.core.exceptions import ValidationError
from django.db.models import Q

from kausal_common.datasets.models import DataPoint, DataPointComment, Dataset, DatasetSourceReference

from datasets.change_snapshots import data_point_snapshot
from datasets.materialization import refresh_dataset_materialization
from datasets.year_slots import ensure_empty_year
from frameworks.models import DataEvidenceKind, Submission
from frameworks.models.evidence import DataPointEvidence
from nodes.change_ops import record_change

if TYPE_CHECKING:
    from django.db.models import QuerySet

    from kausal_common.datasets.models import DatasetQuerySet

    from nodes.models import InstanceConfig


class InventoryYearError(ValidationError):
    pass


@dataclass(frozen=True)
class InventoryYears:
    """A declared set of inventory years, with the span and skipped years it implies."""

    years: frozenset[int]

    @property
    def min_historical(self) -> int:
        return min(self.years)

    @property
    def max_historical(self) -> int:
        return max(self.years)

    @property
    def skipped(self) -> list[int]:
        return [year for year in range(self.min_historical, self.max_historical + 1) if year not in self.years]

    def as_log(self) -> dict[str, object]:
        return {'min_historical': self.min_historical, 'max_historical': self.max_historical, 'skipped': self.skipped}


@dataclass(frozen=True)
class AddedYear:
    year: int
    inventory_years: InventoryYears
    created_cells: int


@dataclass(frozen=True)
class RemovedYear:
    year: int
    inventory_years: InventoryYears
    deleted_cells: int


@dataclass(frozen=True)
class YearContents:
    """What has been entered in a year's local cells, beyond the blank cells themselves."""

    year: int
    values: int
    evidence: int
    comments: int
    source_references: int

    @property
    def is_empty(self) -> bool:
        return not (self.values or self.evidence or self.comments or self.source_references)


def declared_inventory_years(ic: InstanceConfig) -> InventoryYears:
    """Return the instance's inventory years, counting the whole span of an undeclared instance."""
    years = ic.ensure_spec().years
    max_historical = years.max_historical or years.reference
    if max_historical is None:
        raise InventoryYearError('The instance has no historical years.')
    min_historical = years.min_historical or min(max_historical, years.reference or max_historical)
    skipped = set(years.skipped or ())
    return InventoryYears(frozenset(y for y in range(min_historical, max_historical + 1) if y not in skipped))


def _local_datasets(ic: InstanceConfig) -> DatasetQuerySet:
    return Dataset.objects.for_instance_config(ic).select_related('schema')


def _save(ic: InstanceConfig, *, action: str, year: int, before: InventoryYears, after: InventoryYears, **log: object) -> None:
    ic.update_years(min_historical=after.min_historical, max_historical=after.max_historical, skipped=after.skipped)
    record_change(ic, action=action, before=before.as_log(), after={'year': year, **after.as_log(), **log})


def add_inventory_year(ic: InstanceConfig, year: int) -> AddedYear:
    """Declare `year` an inventory year and seed blank local input cells for it."""
    ic.refresh_from_db()
    before = declared_inventory_years(ic)
    model_end = ic.ensure_spec().years.model_end
    if model_end is not None and year > model_end:
        raise InventoryYearError(f'{year} exceeds the model end year {model_end}.')
    if year in before.years:
        raise InventoryYearError(f'{year} is already an inventory year.')

    sources: dict[str, Dataset] = {}
    if ic.has_framework_config() and (template := ic.framework_config.framework.template_instance) is not None:
        sources = {
            dataset.identifier: dataset
            for dataset in Dataset.objects.for_instance_config(template)
            if dataset.identifier is not None
        }
    created_cells = 0
    for dataset in _local_datasets(ic):
        prototype = sources.get(dataset.identifier) if dataset.identifier is not None else None
        created_cells += ensure_empty_year(dataset, year, prototype=prototype)

    after = InventoryYears(before.years | {year})
    _save(ic, action='inventory.year.add', year=year, before=before, after=after, created_cells=created_cells)
    return AddedYear(year=year, inventory_years=after, created_cells=created_cells)


def _year_points(ic: InstanceConfig, year: int) -> QuerySet[DataPoint]:
    """
    Select the year's local cells, apart from provider defaults.

    A provider default (such as a seeded weather-correction factor) is the provider's
    data, not work entered in the year, and it often covers forecast years too; a
    year's removal neither counts nor deletes it.
    """
    return DataPoint.objects.filter(dataset__in=_local_datasets(ic), date__year=year).exclude(
        evidence__kind=DataEvidenceKind.PROVIDER_DEFAULT
    )


def year_contents(ic: InstanceConfig, year: int) -> YearContents:
    points = _year_points(ic, year)
    return YearContents(
        year=year,
        values=points.filter(value__isnull=False).count(),
        evidence=DataPointEvidence.objects.filter(data_point__in=points).count(),
        comments=DataPointComment.objects.filter(data_point__in=points).count(),
        source_references=DatasetSourceReference.objects.filter(data_point__in=points).count(),
    )


def remove_inventory_year(ic: InstanceConfig, year: int, *, force: bool) -> RemovedYear | YearContents:
    """
    Un-declare `year` and delete its local cells.

    Without `force`, a year holding anything beyond blank cells is left alone and its
    contents are returned, so the caller can ask before deleting them. Every refusal
    comes first, so a caller that confirms and retries with `force` is not refused then.
    """
    ic.refresh_from_db()
    before = declared_inventory_years(ic)
    if year == ic.ensure_spec().years.reference:
        raise InventoryYearError(f'{year} is the reference year and cannot be removed.')
    if year not in before.years:
        raise InventoryYearError(f'{year} is not an inventory year of this instance.')
    if len(before.years) == 1:
        raise InventoryYearError(f'{year} is the only inventory year.')
    submission = Submission.objects.filter(instance_config=ic, period_start=year).first()
    if submission is not None:
        raise InventoryYearError(
            f'{year} has a submission ({submission.get_status_display().lower()}); a year with a submission cannot be removed.'
        )

    datasets = list(Dataset.objects.filter(pk__in=_year_points(ic, year).values('dataset_id')).select_related('schema'))
    locked = [dataset for dataset in datasets if dataset.schema is None or not dataset.schema.is_editable]
    if locked:
        names = ', '.join(sorted(dataset.identifier or str(dataset.uuid) for dataset in locked))
        raise InventoryYearError(f'{year} holds imported data that cannot be removed here: {names}.')

    contents = year_contents(ic, year)
    if not contents.is_empty and not force:
        return contents

    deleted_cells = 0
    for candidate in datasets:
        dataset = Dataset.objects.select_for_update().get(pk=candidate.pk)
        points = _year_points(ic, year).filter(dataset=dataset)
        worked = points.filter(
            Q(value__isnull=False) | Q(evidence__isnull=False) | Q(comments__isnull=False) | Q(source_references__isnull=False)
        ).distinct()
        # Blank cells were seeded, not entered; only the ones someone worked on are worth a change record.
        for point in worked.prefetch_related('dimension_categories'):
            record_change(point, action='dataset.datapoint.delete', before=data_point_snapshot(point), after=None)
        deleted_cells += points.delete()[1].get(DataPoint._meta.label, 0)
        refresh_dataset_materialization(dataset, touch=False)

    after = InventoryYears(before.years - {year})
    _save(
        ic,
        action='inventory.year.remove',
        year=year,
        before=before,
        after=after,
        deleted_cells=deleted_cells,
        forced=not contents.is_empty,
    )
    return RemovedYear(year=year, inventory_years=after, deleted_cells=deleted_cells)
