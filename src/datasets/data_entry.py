"""Request-local, permission-checked data behind a structural data-entry layout."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING
from uuid import UUID, uuid3

import polars as pl

from common import qualifiers
from datasets.editions import (
    DatasetEdition,
    DatasetEditions,
    EditionFrame,
    evaluate_edition_plausibility,
    evaluate_edition_rules,
)
from frameworks.qualifiers import qualifier_catalog_for_instance
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN
from nodes.data_entry import EntryResolver, EntrySelection, Rectangle, intersect, rectangle
from nodes.defs.binding_def import DatasetBindingDef
from nodes.value_validation import contract_requirements, requirement_failures

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from kausal_common.datasets.models import Dataset

    from datasets.plausibility import PlausibilityFinding
    from datasets.snapshot import DataPointEvidenceSnapshot, DatasetSnapshot
    from nodes.defs.graph import DatasetMeta
    from nodes.instance_graph import InstanceGraph
    from nodes.value_validation import ValueContract, ValueRequirement
    from users.models import User


@dataclass(frozen=True)
class EntryPoint:
    id: UUID | None
    dataset_id: UUID
    metric_id: UUID
    year: int
    categories: tuple[tuple[UUID, UUID], ...]
    value: float | None
    evidence_kind: str | None = None
    quality_level_id: UUID | None = None
    quality_level_identifier: str | None = None


@dataclass(frozen=True)
class EntryFinding:
    id: UUID
    code: str
    message: str
    years: tuple[int, ...]
    section_ids: tuple[UUID, ...]
    dataset_id: UUID | None = None
    metric_id: UUID | None = None
    coordinates: tuple[tuple[UUID, UUID], ...] = ()


@dataclass
class EntryDatasetData:
    edition: DatasetEdition
    table: EditionFrame
    point_ids: dict[tuple[UUID, int, tuple[UUID, ...]], UUID] = field(default_factory=dict)
    evidence: dict[tuple[UUID, int, tuple[UUID, ...]], DataPointEvidenceSnapshot] = field(default_factory=dict)

    @property
    def meta(self) -> DatasetMeta:
        return self.edition.meta

    @property
    def snapshot(self) -> DatasetSnapshot:
        return self.edition.snapshot

    @property
    def frame(self) -> pl.DataFrame:
        return self.table.frame

    @property
    def dimensions(self) -> dict[str, UUID]:
        return self.table.dimensions

    @property
    def categories(self) -> dict[tuple[str, str], UUID]:
        return self.table.categories

    def coordinates(self, row: Mapping[str, object]) -> tuple[tuple[UUID, UUID], ...]:
        return self.table.coordinates(row)


def matches(selection: EntrySelection, metric: UUID, coordinates: tuple[tuple[UUID, UUID], ...]) -> bool:
    if selection.metric_id != metric:
        return False
    values = dict(coordinates)
    return any(all(values.get(dim) in cats for dim, cats in item) for item in selection.rectangles)


class DataEntryQuery:
    def __init__(self, graph: InstanceGraph, user: User, *, published: bool = False, editions: DatasetEditions | None = None):
        self.graph = graph
        self.published = published
        self.user = user
        self.editions = editions if editions is not None else DatasetEditions(graph, user, published=published)
        self._data: dict[UUID, EntryDatasetData] = {}
        self._findings: dict[UUID, tuple[EntryFinding, ...]] = {}
        self.plausibility: dict[UUID, list[PlausibilityFinding]] = {}

    @property
    def years(self) -> list[int]:
        return self.graph.spec.years.historical or []

    def selected_years(self, years: list[int] | None) -> list[int]:
        if years is None:
            return self.years
        if set(years) - set(self.years):
            raise ValueError('Requested years must belong to the declared inventory calendar')
        return sorted(set(years))

    @cached_property
    def qualifier_catalog(self) -> qualifiers.QualifierCatalog:
        return qualifier_catalog_for_instance(self.editions.config)

    @property
    def datasets(self) -> dict[UUID, Dataset]:
        return self.editions.viewable

    def is_editable(self, id: UUID) -> bool:
        if self.published or id in self.graph.pinned_revisions:
            return False
        dataset = self.datasets[id]
        return bool(
            dataset.schema
            and (dataset.schema.is_editable or dataset._schema_is_shared)
            and dataset.permission_policy().user_has_perm(self.user, 'change', dataset)
        )

    def load(self, dataset_ids: Iterable[UUID]) -> None:
        wanted = set(dataset_ids) & self.datasets.keys() - self._data.keys()
        if not wanted:
            return
        for edition in self.editions.load(wanted):
            data = EntryDatasetData(edition, edition.frame(self.qualifier_catalog))
            declared = set(edition.meta.declared_dimension_ids)
            for point in edition.snapshot.points:
                key = (point.metric, point.date.year, tuple(sorted(c for d, c in point.categories.items() if d in declared)))
                data.point_ids[key] = point.id
                if point.evidence is not None:
                    data.evidence[key] = point.evidence
            self._data[edition.id] = data

    def points(self, selections: tuple[EntrySelection, ...], years: list[int] | None) -> list[EntryPoint]:
        selected = self.selected_years(years)
        if not selected:
            return []
        dataset_ids = {item.dataset_id for item in selections}
        self.load(dataset_ids)
        result: dict[tuple[UUID, UUID, int, tuple[tuple[UUID, UUID], ...]], EntryPoint] = {}
        for id in dataset_ids & self.datasets.keys():
            data = self._data[id]
            for row in data.frame.filter(pl.col(YEAR_COLUMN).is_in(selected)).iter_rows(named=True):
                coords = data.coordinates(row)
                year = int(row[YEAR_COLUMN])
                for metric in data.meta.metrics:
                    if not any(matches(item, metric.id, coords) for item in selections if item.dataset_id == id):
                        continue
                    if metric.identifier not in row:
                        continue
                    key = (metric.id, year, tuple(sorted(cat for _, cat in coords)))
                    proof = data.evidence.get(key)
                    quality = proof.quality_level if proof else None
                    value = row[metric.identifier]
                    point = EntryPoint(
                        data.point_ids.get(key),
                        id,
                        metric.id,
                        year,
                        coords,
                        float(value) if value is not None else None,
                        proof.kind if proof else None,
                        UUID(quality.uuid) if quality else None,
                        quality.level if quality else None,
                    )
                    result[id, metric.id, year, coords] = point
        return list(result.values())

    def affected_sections(self, dataset: UUID, metric: UUID, selections: tuple[Rectangle, ...]) -> tuple[UUID, ...]:
        return tuple(
            section.id
            for section in self.graph.data_entry.sections
            if any(
                claim.dataset_id == dataset
                and claim.metric_id == metric
                and any(intersect(owned, selected) is not None for owned in claim.rectangles for selected in selections)
                for claim in section.selections
            )
        )

    def contract_findings(self, data: EntryDatasetData) -> list[EntryFinding]:
        """Project declared input completeness onto source coordinates, without filling data."""
        results: list[EntryFinding] = []
        resolver = EntryResolver(self.graph)
        for binding in self.graph.bindings:
            if not isinstance(binding, DatasetBindingDef) or binding.dataset_uuid != data.meta.id:
                continue
            contract = binding.target_port.validation
            if contract is None or binding.metric_uuid not in data.meta.metric_by_id:
                continue
            metric = data.meta.metric_by_id[binding.metric_uuid]
            column = metric.identifier
            if column is None or column not in data.frame.columns:
                continue
            if contract.required_if_positive or contract.combinations_from_positive:
                # Model-level validation evaluates these dependencies. A reminder
                # about the evaluator is not a finding about the entered data.
                continue
            shape = self.graph.shapes.get(contract.shape) if contract.shape is not None else None
            for requirement in contract_requirements(contract, shape):
                results.extend(self._requirement_findings(resolver, binding, data, metric.id, column, contract, requirement))
        return results

    def _requirement_findings(
        self,
        resolver: EntryResolver,
        binding: DatasetBindingDef,
        data: EntryDatasetData,
        metric_id: UUID,
        column: str,
        contract: ValueContract,
        requirement: ValueRequirement,
    ) -> list[EntryFinding]:
        """
        Locate one port requirement in the dataset that feeds the port.

        A requirement with an alternative this binding cannot deliver is skipped: another
        binding of the port may satisfy it.
        """
        alternatives: list[tuple[pl.DataFrame, tuple[tuple[UUID, UUID], ...], Rectangle]] = []
        for categories in requirement.alternatives:
            selection: dict[UUID, frozenset[UUID]] = {}
            for dimension, category in categories.items():
                dim = self.graph.dimension_by_identifier[dimension]
                category_id = next(c.id for c in dim.categories if c.identifier == category)
                selection[dim.id] = frozenset((category_id,))
            source, reasons = resolver.translate(binding, rectangle(selection))
            if source is None:
                return []
            if reasons:
                return [
                    EntryFinding(
                        uuid3(binding.id, f'unresolved-contract:{selection}'),
                        'unresolved_contract',
                        'Input requirements cannot be translated to dataset coordinates',
                        (),
                        self.affected_sections(data.meta.id, metric_id, ((),)),
                        data.meta.id,
                        metric_id,
                    )
                ]
            source = rectangle({dim: cats for dim, cats in source if dim in data.meta.declared_dimension_ids})
            rows = data.frame
            for dim, cats in source:
                dimension_column = next(name for name, id in data.dimensions.items() if id == dim)
                labels = [label for (name, label), id in data.categories.items() if name == dimension_column and id in cats]
                rows = rows.filter(pl.col(dimension_column).is_in(labels))
            coordinates = tuple((dim, next(iter(cats))) for dim, cats in source if len(cats) == 1)
            alternatives.append((rows, coordinates, source))
        sections = self.affected_sections(data.meta.id, metric_id, tuple(source for _, _, source in alternatives))
        coordinates = alternatives[0][1] if len(alternatives) == 1 else ()
        key = coordinates if len(alternatives) == 1 else requirement.identifier
        renamed = {column: VALUE_COLUMN}
        qualifier_column = qualifiers.qualifier_column(column)
        if qualifier_column in data.frame.columns:
            renamed[qualifier_column] = qualifiers.qualifier_column(VALUE_COLUMN)
        results: list[EntryFinding] = []
        for year in self.years:
            present = [
                rows.filter((pl.col(YEAR_COLUMN) == year) & pl.col(column).is_not_null() & pl.col(column).is_finite())
                for rows, _, _ in alternatives
            ]
            if contract.years == 'active' and all(rows.is_empty() for rows in present):
                continue
            assessments = [rows.select(list(renamed)).rename(renamed) for rows in present]
            results.extend(
                EntryFinding(
                    uuid3(binding.id, f'{code}:{message}:{key}:{year}'),
                    code,
                    message,
                    (year,),
                    sections,
                    data.meta.id,
                    metric_id,
                    coordinates,
                )
                for code, message in requirement_failures(assessments, requirement.qualifiers)
            )
        return results

    def dataset_findings(self, id: UUID) -> tuple[EntryFinding, ...]:
        if id in self._findings:
            return self._findings[id]
        self.load([id])
        data = self._data[id]
        frame = data.frame
        # The rules must also see wholly empty declared years.
        # Null rows add years without inventing observed category combinations.
        absent = set(self.years) - set(frame[YEAR_COLUMN].to_list())
        if absent:
            frame = pl.concat([frame, pl.DataFrame({YEAR_COLUMN: sorted(absent)})], how='diagonal_relaxed')
        violations = evaluate_edition_rules(data.edition, data.table.with_frame(frame))
        results: list[EntryFinding] = []
        for violation in violations:
            coordinates = data.coordinates(violation.categories)
            sections = self.affected_sections(
                id, violation.metric_uuid, (rectangle({dim: frozenset((cat,)) for dim, cat in coordinates}),)
            )
            results.append(
                EntryFinding(
                    uuid3(id, f'{violation.rule_uuid}:{violation.kind}:{coordinates}'),
                    violation.kind,
                    violation.message,
                    tuple(violation.years),
                    sections,
                    id,
                    violation.metric_uuid,
                    coordinates,
                )
            )
        self.plausibility[id] = evaluate_edition_plausibility(data.edition, data.table, violations)
        for finding in self.plausibility[id]:
            coords = tuple(sorted((coord.dimension_uuid, coord.category_uuid) for coord in finding.coordinates))
            selection: dict[UUID, frozenset[UUID]] = {}
            for coord in finding.selection:
                selection[coord.dimension_uuid] = selection.get(coord.dimension_uuid, frozenset()) | {coord.category_uuid}
            for dimension, category in coords:
                selection[dimension] = frozenset((category,))
            sections = self.affected_sections(id, finding.metric_uuid, (rectangle(selection),))
            key = f'plausibility:{finding.rule_uuid}:{finding.metric_uuid}:{coords}:{finding.years}'
            results.append(
                EntryFinding(
                    uuid3(id, key),
                    'plausibility',
                    f'{finding.metric}: {finding.normalized:g} outside {finding.lower:g}-{finding.upper:g}',
                    tuple(finding.years),
                    sections,
                    id,
                    finding.metric_uuid,
                    coords,
                )
            )
        results.extend(self.contract_findings(data))
        self._findings[id] = tuple(results)
        return self._findings[id]

    def findings(self, section_id: UUID | None, years: list[int] | None) -> list[EntryFinding]:
        selected = set(self.selected_years(years))
        sections = [section for section in self.graph.data_entry.sections if section_id is None or section.id == section_id]
        datasets = {item.dataset_id for section in sections for item in section.selections} & self.datasets.keys()
        self.load(datasets)
        result: dict[UUID, EntryFinding] = {}
        for id in datasets:
            for finding in self.dataset_findings(id):
                if section_id is not None and section_id not in finding.section_ids:
                    continue
                if finding.years and not selected.intersection(finding.years):
                    continue
                result[finding.id] = finding
        return list(result.values())
