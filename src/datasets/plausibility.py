"""Advisory checks against curated, versioned dataset reference ranges."""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from django.db.models import Q
from pydantic import BaseModel, Field

import polars as pl

from datasets.coordinates import DatasetCoordinate, DatasetCoordinateIndex
from datasets.models import DatasetMetricPlausibilityRange
from datasets.validation import RuleViolation, _category_domain_coordinates, evaluate_dataset_rules
from frameworks.models import OrganizationPopulation
from nodes.constants import YEAR_COLUMN
from nodes.datasets import DBDataset

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kausal_common.datasets.models import Dataset

    from nodes.models import InstanceConfig

Range = DatasetMetricPlausibilityRange


class PlausibilityFinding(BaseModel):
    """
    A value, or a sum of values, outside one advisory reference range.

    For a sum, ``categories`` and ``coordinates`` hold the dimensions the
    selection fixes to one category, and ``selection`` every selected pair.
    ``normalized`` is the quantity compared with the bounds: the value per
    denominator unit, or its ratio to ``reference_value`` from ``reference_year``.
    """

    rule_uuid: UUID
    metric_uuid: UUID
    metric: str
    dataset_uuid: UUID
    dataset_identifier: str | None
    aggregation: str
    reference: str
    years: list[int]
    categories: dict[str, str] = Field(default_factory=dict)
    coordinates: list[DatasetCoordinate] = Field(default_factory=list)
    selection: list[DatasetCoordinate] = Field(default_factory=list)
    combination_ids: list[UUID] = Field(default_factory=list)
    observed: float
    normalized: float
    denominator_value: int | None
    reference_year: int | None = None
    reference_value: float | None = None
    complete: bool = True
    """False for a sum with empty cells; such a sum is only checked against the upper bound."""
    component_count: int = 1
    lower: float
    upper: float
    unit: str
    source_identifier: str
    source_url: str
    source_revision: str
    rule_revision: int
    is_example: bool
    message: str


def _invalid_cell(violations: Iterable[RuleViolation], metric_uuid: UUID, year: int, categories: dict[str, str]) -> bool:
    for violation in violations:
        if violation.metric_uuid != metric_uuid:
            continue
        if violation.years and year not in violation.years:
            continue
        if all(categories.get(dim) == category for dim, category in violation.categories.items()):
            return True
    return False


def applicable_plausibility_ranges(dataset: Dataset) -> list[DatasetMetricPlausibilityRange]:
    """Return reference ranges visible to the owning instance and its framework, if any."""
    if dataset.schema is None or dataset.is_external_placeholder:
        return []
    instance = dataset.scope_instance
    framework_id = instance.framework_config.framework_id if instance.has_framework_config() else None
    scope = Q(instance_config=instance)
    if framework_id is not None:
        scope |= Q(framework_id=framework_id)
    return list(
        DatasetMetricPlausibilityRange.objects
        .filter(scope, metric__schema=dataset.schema, enabled=True)
        .select_related('metric', 'source')
        .order_by('metric_id', 'identifier')
    )


def _observed_population(dataset: Dataset, rules: Iterable[DatasetMetricPlausibilityRange]) -> dict[int, int]:
    if not any(rule.denominator == Range.Denominator.POPULATION for rule in rules):
        return {}
    instance = dataset.scope_instance
    framework_id = instance.framework_config.framework_id if instance.has_framework_config() else None
    if framework_id is None:
        return {}
    return dict(
        OrganizationPopulation.objects.filter(framework_id=framework_id, organization_id=instance.organization_id).values_list(
            'year', 'value'
        )
    )


def _in_years(rule: DatasetMetricPlausibilityRange, year: int) -> bool:
    return (rule.first_year is None or year >= rule.first_year) and (rule.last_year is None or year <= rule.last_year)


def _denominator_multiplier(rule: DatasetMetricPlausibilityRange, population: int | None) -> float | None:
    """How many bound units one metric unit is per denominator; None when unobserved."""
    if rule.denominator == Range.Denominator.NONE:
        return 1.0
    if rule.denominator == Range.Denominator.POPULATION:
        return float(population) if population and population > 0 else None
    raise ValueError(f'Unknown plausibility denominator: {rule.denominator}')


def resolved_bounds(
    rule: DatasetMetricPlausibilityRange, population: int | None, earlier_value: float | None = None
) -> tuple[float | None, float | None]:
    """
    Express a cell range in the linked metric's input unit.

    Absolute bounds are already in the metric unit per denominator unit, so the
    denominator is the only conversion. Previous-year bounds need the earlier value.
    """
    if rule.aggregation != Range.Aggregation.CELL:
        return None, None
    if rule.reference == Range.Reference.PREVIOUS_YEAR:
        if earlier_value is None or earlier_value <= 0:
            return None, None
        return rule.lower * earlier_value, rule.upper * earlier_value
    multiplier = _denominator_multiplier(rule, population)
    if multiplier is None:
        return None, None
    return rule.lower * multiplier, rule.upper * multiplier


@dataclass
class _Cells:
    """The metric column of one dataset, with the context every rule needs."""

    dataset: Dataset
    frame: pl.DataFrame
    dim_cols: list[str]
    coordinate_index: DatasetCoordinateIndex
    violations: list[RuleViolation]
    combinations: dict[UUID, dict[str, str]]
    population: dict[int, int]
    _selected: dict[UUID, pl.DataFrame] = field(default_factory=dict)

    @classmethod
    def load(cls, dataset: Dataset, rules: list[DatasetMetricPlausibilityRange]) -> _Cells:
        ppdf = DBDataset.deserialize_df(dataset)
        frame = pl.DataFrame({col: ppdf.get_column(col) for col in ppdf.columns})
        dim_cols = [col for col in ppdf.primary_keys if col != YEAR_COLUMN]
        if dim_cols:
            frame = frame.with_columns([pl.col(col).cast(pl.Utf8) for col in dim_cols])
        return cls(
            dataset=dataset,
            frame=frame,
            dim_cols=dim_cols,
            coordinate_index=DatasetCoordinateIndex(dataset),
            violations=evaluate_dataset_rules(dataset),
            combinations=_category_domain_coordinates(dataset),
            population=_observed_population(dataset, rules),
        )

    def selected(self, rule: DatasetMetricPlausibilityRange) -> pl.DataFrame | None:
        """Rows of the rule's metric inside its selection: year, dimension columns, ``value``."""
        column = _metric_column(rule)
        if column not in self.frame.columns:
            return None
        if rule.uuid not in self._selected:
            allowed: dict[str, set[str]] = {}
            for coordinate in self.coordinate_index.resolve_selection(rule.selected_categories()):
                allowed.setdefault(coordinate.dimension, set()).add(coordinate.category)
            frame = self.frame.select(YEAR_COLUMN, *self.dim_cols, pl.col(column).cast(pl.Float64).alias('value'))
            for dimension, categories in allowed.items():
                if dimension not in self.dim_cols:
                    frame = frame.clear()
                    break
                frame = frame.filter(pl.col(dimension).is_in(sorted(categories)))
            self._selected[rule.uuid] = frame
        return self._selected[rule.uuid]

    def categories(self, row: dict[str, object]) -> dict[str, str]:
        return {col: str(row[col]) for col in self.dim_cols if row[col] is not None}

    def invalid(self, rule: DatasetMetricPlausibilityRange, year: int, categories: dict[str, str]) -> bool:
        return _invalid_cell(self.violations, rule.metric.uuid, year, categories)

    def combination_ids(self, cells: list[dict[str, str]]) -> list[UUID]:
        return [cid for cid, coordinates in self.combinations.items() if coordinates in cells]


def _metric_column(rule: DatasetMetricPlausibilityRange) -> str:
    return rule.metric.name or rule.metric.label or str(rule.metric.uuid)


@dataclass(frozen=True)
class _Comparison:
    """An observed value and the quantity compared with a rule's bounds."""

    observed: float
    normalized: float
    denominator_value: int | None = None
    reference_year: int | None = None
    reference_value: float | None = None


def _earlier(rule: DatasetMetricPlausibilityRange, year: int, values: dict[int, float]) -> tuple[int, float] | None:
    """Find the latest usable earlier value within the rule's maximum gap."""
    assert rule.max_gap_years is not None
    for earlier in range(year - 1, year - rule.max_gap_years - 1, -1):
        value = values.get(earlier)
        if value is not None and value > 0:
            return earlier, value
    return None


def _compare(
    rule: DatasetMetricPlausibilityRange, cells: _Cells, year: int, observed: float, history: dict[int, float]
) -> _Comparison | None:
    """Normalize one observation for its rule; return None when it cannot be assessed."""
    if rule.reference == Range.Reference.PREVIOUS_YEAR:
        earlier = _earlier(rule, year, history)
        if earlier is None:
            return None
        reference_year, reference_value = earlier
        return _Comparison(observed, observed / reference_value, None, reference_year, reference_value)
    population = cells.population.get(year)
    multiplier = _denominator_multiplier(rule, population)
    if multiplier is None:
        return None
    denominator = population if rule.denominator == Range.Denominator.POPULATION else None
    return _Comparison(observed, observed / multiplier, denominator)


def _finding(
    rule: DatasetMetricPlausibilityRange,
    cells: _Cells,
    *,
    year: int,
    categories: dict[str, str],
    components: list[dict[str, str]],
    comparison: _Comparison,
    complete: bool,
) -> PlausibilityFinding:
    unit = str(rule.bound_unit)
    value = comparison.normalized
    if rule.reference == Range.Reference.PREVIOUS_YEAR:
        described = f'{value:.3g} times the {comparison.reference_year} value is outside the range {rule.lower:g}-{rule.upper:g}.'
    else:
        described = f'{value:g} {unit} is outside the reference range {rule.lower:g}-{rule.upper:g}.'
    is_sum = rule.aggregation == Range.Aggregation.SUM
    if is_sum:
        described = f'The sum of {len(components)} cells{"" if complete else ", some still empty,"}: {described}'
    return PlausibilityFinding(
        rule_uuid=rule.uuid,
        metric_uuid=rule.metric.uuid,
        metric=_metric_column(rule),
        dataset_uuid=cells.dataset.uuid,
        dataset_identifier=cells.dataset.identifier,
        aggregation=rule.aggregation,
        reference=rule.reference,
        years=[year],
        categories=categories,
        coordinates=cells.coordinate_index.resolve(categories),
        selection=cells.coordinate_index.resolve_selection(rule.selected_categories()) if is_sum else [],
        combination_ids=cells.combination_ids(components),
        observed=comparison.observed,
        normalized=value,
        denominator_value=comparison.denominator_value,
        reference_year=comparison.reference_year,
        reference_value=comparison.reference_value,
        complete=complete,
        component_count=len(components),
        lower=rule.lower,
        upper=rule.upper,
        unit=unit,
        source_identifier=rule.source.identifier,
        source_url=rule.source.url,
        source_revision=rule.source.revision,
        rule_revision=rule.revision,
        is_example=rule.source.is_example,
        message=described,
    )


def _cell_findings(rule: DatasetMetricPlausibilityRange, cells: _Cells) -> list[PlausibilityFinding]:
    frame = cells.selected(rule)
    if frame is None:
        return []
    history: dict[tuple[str, ...], dict[int, float]] = {}
    entered: list[tuple[int, dict[str, str], tuple[str, ...], float]] = []
    for row in frame.iter_rows(named=True):
        categories = cells.categories(row)
        year = int(row[YEAR_COLUMN])
        value = row['value']
        if value is None or not math.isfinite(value) or cells.invalid(rule, year, categories):
            continue
        key = tuple(categories.get(col, '') for col in cells.dim_cols)
        history.setdefault(key, {})[year] = float(value)
        entered.append((year, categories, key, float(value)))
    findings = []
    for year, categories, key, observed in entered:
        if not _in_years(rule, year):
            continue
        comparison = _compare(rule, cells, year, observed, history[key])
        if comparison is None or rule.lower <= comparison.normalized <= rule.upper:
            continue
        findings.append(
            _finding(rule, cells, year=year, categories=categories, components=[categories], comparison=comparison, complete=True)
        )
    return findings


@dataclass(frozen=True)
class _YearSum:
    total: float
    complete: bool
    components: list[dict[str, str]]


def _year_sums(rule: DatasetMetricPlausibilityRange, cells: _Cells) -> dict[int, _YearSum]:
    """
    Sum the selected cells per year.

    An empty (null) cell makes the sum incomplete. A year where any selected cell
    violates a validation rule has no sum, as validation is evaluated first.
    """
    frame = cells.selected(rule)
    if frame is None:
        return {}
    sums: dict[int, _YearSum] = {}
    for (year,), group in frame.group_by(YEAR_COLUMN):
        year_int = int(str(year))
        components = [cells.categories(row) for row in group.iter_rows(named=True)]
        if any(cells.invalid(rule, year_int, categories) for categories in components):
            continue
        values = group['value']
        entered = [float(value) for value in values.to_list() if value is not None]
        if not entered or not all(math.isfinite(value) for value in entered):
            continue
        sums[year_int] = _YearSum(total=math.fsum(entered), complete=values.null_count() == 0, components=components)
    return sums


def _fixed_categories(rule: DatasetMetricPlausibilityRange, cells: _Cells) -> dict[str, str]:
    """Return the dimensions a sum's selection fixes to a single category."""
    single = {dimension: categories for dimension, categories in rule.selected_categories().items() if len(categories) == 1}
    return {coordinate.dimension: coordinate.category for coordinate in cells.coordinate_index.resolve_selection(single)}


def _sum_findings(rule: DatasetMetricPlausibilityRange, cells: _Cells) -> list[PlausibilityFinding]:
    sums = _year_sums(rule, cells)
    # A ratio between partial sums says nothing about either year.
    history = {year: year_sum.total for year, year_sum in sums.items() if year_sum.complete}
    fixed = _fixed_categories(rule, cells)
    findings = []
    for year, year_sum in sorted(sums.items()):
        if not _in_years(rule, year):
            continue
        if rule.reference == Range.Reference.PREVIOUS_YEAR and not year_sum.complete:
            continue
        comparison = _compare(rule, cells, year, year_sum.total, history)
        if comparison is None:
            continue
        too_high = comparison.normalized > rule.upper
        # An incomplete sum only grows as cells are filled in, so only its upper bound means anything.
        too_low = comparison.normalized < rule.lower and year_sum.complete
        if not (too_high or too_low):
            continue
        findings.append(
            _finding(
                rule,
                cells,
                year=year,
                categories=fixed,
                components=year_sum.components,
                comparison=comparison,
                complete=year_sum.complete,
            )
        )
    return findings


def evaluate_dataset_plausibility(dataset: Dataset) -> list[PlausibilityFinding]:
    """Evaluate applicable bands against entered cells, using observed population only."""
    rules = applicable_plausibility_ranges(dataset)
    if not rules:
        return []
    cells = _Cells.load(dataset, rules)
    findings: list[PlausibilityFinding] = []
    for rule in rules:
        if rule.aggregation == Range.Aggregation.SUM:
            findings.extend(_sum_findings(rule, cells))
        else:
            findings.extend(_cell_findings(rule, cells))
    return findings


class DatasetPlausibilityLookup:
    """Request-local resolution of cell ranges for the data points of one dataset."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.rules = applicable_plausibility_ranges(dataset)
        self.coordinate_index = DatasetCoordinateIndex(dataset) if self.rules else None
        self.rules_by_metric: dict[int, list[DatasetMetricPlausibilityRange]] = {}
        for rule in self.rules:
            if rule.aggregation == Range.Aggregation.CELL:
                self.rules_by_metric.setdefault(rule.metric_id, []).append(rule)
        self.population = _observed_population(dataset, self.rules)
        self._values_by_metric: dict[int, dict[tuple[int, frozenset[str]], float]] = {}

    def _values(self, rule: DatasetMetricPlausibilityRange) -> dict[tuple[int, frozenset[str]], float]:
        """Entered values of one metric by (year, category UUIDs), loaded once for previous-year ranges."""
        if rule.metric_id not in self._values_by_metric:
            from kausal_common.datasets.models import DataPoint

            values: dict[tuple[int, frozenset[str]], float] = {}
            points = DataPoint.objects.filter(
                dataset=self.dataset, metric_id=rule.metric_id, value__isnull=False
            ).prefetch_related('dimension_categories')
            for point in points:
                assert point.value is not None
                key = (point.date.year, frozenset(str(category.uuid) for category in point.dimension_categories.all()))
                values[key] = float(point.value)
            self._values_by_metric[rule.metric_id] = values
        return self._values_by_metric[rule.metric_id]

    def for_cell(
        self, metric_id: int, year: int, category_uuids: dict[str, str]
    ) -> list[tuple[DatasetMetricPlausibilityRange, float | None, float | None]]:
        matches = []
        for rule in self.rules_by_metric.get(metric_id, []):
            if not _in_years(rule, year):
                continue
            if any(category_uuids.get(dim) not in categories for dim, categories in rule.selection.items()):
                continue
            earlier_value = None
            if rule.reference == Range.Reference.PREVIOUS_YEAR:
                values = self._values(rule)
                cell = frozenset(category_uuids.values())
                earlier = _earlier(rule, year, {y: v for (y, c), v in values.items() if c == cell})
                earlier_value = earlier[1] if earlier else None
            lower, upper = resolved_bounds(rule, self.population.get(year), earlier_value)
            matches.append((rule, lower, upper))
        return matches


def collect_instance_dataset_plausibility_findings(instance_config: InstanceConfig) -> list[PlausibilityFinding]:
    """Use the same current bound-dataset scope as instance validation findings."""
    from kausal_common.datasets.models import Dataset

    from nodes.instance_serialization import build_instance_snapshot

    snapshot = build_instance_snapshot(instance_config)
    datasets = Dataset.objects.filter(uuid__in=[dataset.id for dataset in snapshot.datasets]).exclude(
        uuid__in=[pin.dataset_uuid for pin in snapshot.dataset_revisions],
    )
    return [finding for dataset in datasets for finding in evaluate_dataset_plausibility(dataset)]
