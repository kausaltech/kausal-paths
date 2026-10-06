"""Embedded data-entry views; observation UUIDs remain the mutation identities."""

from typing import TYPE_CHECKING
from uuid import UUID, uuid3

import strawberry as sb

from datasets.data_entry import DataEntryQuery
from datasets.graphql.types import (
    DataQualitySchemeType,
    DatasetCategoryCombinationType,
    DatasetCategoryCoordinateType,
    DatasetCategoryDomainType,
    DatasetDimensionCategoryType,
    DatasetDimensionType,
    DatasetMetricType,
    MetricValidationRuleType,
)
from datasets.validation_rules import rule_to_gql
from frameworks.evidence import quality_schemes_for_dataset
from nodes.data_entry import EntrySelection
from nodes.graphql.types.problems import DatasetPlausibilityFindingType

if TYPE_CHECKING:
    from datasets.data_entry import EntryFinding, EntryPoint
    from nodes.data_entry import EntrySection


@sb.type(name='DataEntryCoordinate')
class CoordinateType:
    dimension_id: UUID
    category_id: UUID


@sb.type(name='DataEntryDimensionSelection')
class DimensionSelectionType:
    dimension_id: UUID
    category_ids: list[UUID]


@sb.type(name='DataEntryRectangle')
class RectangleType:
    dimensions: list[DimensionSelectionType]


@sb.type(name='DataEntrySelectionFragment')
class FragmentType:
    rectangles: list[RectangleType]
    placement: str
    precision: str
    path: list[UUID]
    reasons: list[str]
    node_id: UUID | None
    port_id: UUID | None

    @classmethod
    def from_selection(cls, item: EntrySelection) -> FragmentType:
        return cls(
            rectangles=[
                RectangleType(
                    dimensions=[DimensionSelectionType(dimension_id=dim, category_ids=sorted(cats)) for dim, cats in rectangle]
                )
                for rectangle in item.rectangles
            ],
            placement=item.placement,
            precision='approximate' if item.approximate else 'exact',
            path=list(item.path),
            reasons=list(item.reasons),
            node_id=item.node_id,
            port_id=item.port_id,
        )


@sb.type(name='DataEntryMetricSelection')
class MetricSelectionType:
    metric_id: UUID
    fragments: list[FragmentType]


@sb.type(name='DataEntryPoint')
class PointType:
    id: UUID | None
    metric_id: UUID
    year: int
    coordinates: list[CoordinateType]
    value: float | None
    evidence_kind: str | None
    quality_level_id: UUID | None
    quality_level_identifier: str | None

    @classmethod
    def from_point(cls, point: EntryPoint) -> PointType:
        return cls(
            id=point.id,
            metric_id=point.metric_id,
            year=point.year,
            coordinates=[CoordinateType(dimension_id=d, category_id=c) for d, c in point.categories],
            value=point.value,
            evidence_kind=point.evidence_kind,
            quality_level_id=point.quality_level_id,
            quality_level_identifier=point.quality_level_identifier,
        )


@sb.type(name='DataEntryProblem')
class ProblemType:
    id: UUID
    code: str
    message: str
    years: list[int]
    shared: bool
    affected_section_ids: list[UUID]
    dataset_id: UUID | None
    metric_id: UUID | None
    coordinates: list[CoordinateType]

    @classmethod
    def from_finding(cls, finding: EntryFinding, years: list[int]) -> ProblemType:
        return cls(
            id=finding.id,
            code=finding.code,
            message=finding.message,
            years=sorted(set(finding.years) & set(years)),
            shared=len(finding.section_ids) > 1,
            affected_section_ids=list(finding.section_ids),
            dataset_id=finding.dataset_id,
            metric_id=finding.metric_id,
            coordinates=[CoordinateType(dimension_id=d, category_id=c) for d, c in finding.coordinates],
        )


@sb.type(name='DataEntryYearCount')
class YearCountType:
    year: int
    count: int


@sb.type(name='DataEntryProblemCounts')
class CountsType:
    total: int
    annual: int
    yearless: int
    by_year: list[YearCountType]

    @classmethod
    def from_findings(cls, findings: list[EntryFinding], years: list[int]) -> CountsType:
        annual = sum(bool(finding.years) for finding in findings)
        return cls(
            total=len(findings),
            annual=annual,
            yearless=len(findings) - annual,
            by_year=[YearCountType(year=year, count=sum(year in finding.years for finding in findings)) for year in years],
        )


@sb.type(name='DataEntryUnresolvedInput')
class UnresolvedInputType:
    node_id: UUID | None
    port_id: UUID | None
    dataset_id: UUID | None
    reason: str
    message: str


@sb.type(name='DataEntryDataset')
class EntryDatasetType:
    """Dataset metadata from the selected graph edition, embedded in its table."""

    _query: sb.Private[DataEntryQuery]
    id: UUID

    @sb.field
    def is_editable(self) -> bool:
        return self._query.is_editable(self.id)

    @sb.field
    def name(self) -> str:
        self._query.load([self.id])
        snapshot = self._query._data[self.id].snapshot
        return str(snapshot.name or snapshot.identifier or self.id)

    @sb.field
    def dimensions(self) -> list[DatasetDimensionType]:
        return [
            DatasetDimensionType(
                id=sb.ID(str(dim.id)),
                name=str(dim.label or dim.identifier),
                categories=[
                    DatasetDimensionCategoryType(
                        uuid=cat.id, identifier=cat.identifier, label=str(cat.label or cat.identifier or cat.id)
                    )
                    for cat in dim.categories
                ],
            )
            for id in self._query.graph.dataset_by_id[self.id].declared_dimension_ids
            for dim in [self._query.graph.dimension_by_id[id]]
        ]

    @sb.field
    def category_domain(self) -> DatasetCategoryDomainType:
        domain = self._query.graph.dataset_by_id[self.id].category_domain
        return DatasetCategoryDomainType(
            mode=domain.mode,
            combinations=[
                DatasetCategoryCombinationType(
                    id=item.id,
                    identifier=item.identifier,
                    coordinates=[
                        DatasetCategoryCoordinateType(dimension_id=dim, category_id=cat) for dim, cat in item.categories.items()
                    ],
                )
                for item in domain.combinations
            ],
        )

    @sb.field
    def metrics(self) -> list[DatasetMetricType]:
        self._query.load([self.id])
        snapshot = self._query._data[self.id].snapshot
        rules = {metric.identifier: metric.validation_rules for metric in snapshot.metrics}
        return [
            DatasetMetricType(
                id=sb.ID(str(metric.id)),
                name=metric.identifier,
                label=str(metric.label or metric.identifier or metric.id),
                unit=metric.unit,
                previous_sibling=None,
                next_sibling=None,
                quality_of=sb.ID(str(metric.quality_of)) if metric.quality_of else None,
                validation_rules=[
                    MetricValidationRuleType(id=sb.ID(str(rule.uuid)), rule=rule_to_gql(rule.rule))
                    for rule in rules.get(metric.identifier or '', [])
                ],
            )
            for metric in self._query.graph.dataset_by_id[self.id].metrics
        ]

    @sb.field
    def quality_schemes(self) -> list[DataQualitySchemeType]:
        return [DataQualitySchemeType.from_model(scheme) for scheme in quality_schemes_for_dataset(self._query.datasets[self.id])]


@sb.type(name='DataEntryTable')
class TableType:
    id: UUID
    entry_id: UUID | None
    _query: sb.Private[DataEntryQuery]
    _selections: sb.Private[tuple['EntrySelection', ...]]
    dataset: EntryDatasetType
    metric_selections: list[MetricSelectionType]

    @sb.field
    def is_editable(self) -> bool:
        return self._query.is_editable(self._selections[0].dataset_id)

    @sb.field
    def plausibility_findings(self, years: list[int] | None = None) -> list[DatasetPlausibilityFindingType]:
        id = self._selections[0].dataset_id
        selected = set(self._query.selected_years(years))
        if not selected:
            return []
        self._query.dataset_findings(id)
        return [
            DatasetPlausibilityFindingType.from_finding(finding)
            for finding in self._query.plausibility[id]
            if selected.intersection(finding.years)
        ]

    @sb.field
    def data_points(self, years: list[int] | None = None) -> list[PointType]:
        return [PointType.from_point(point) for point in self._query.points(self._selections, years)]


@sb.type(name='DataEntrySection')
class SectionType:
    _query: sb.Private[DataEntryQuery]
    _section: sb.Private['EntrySection']
    id: UUID
    identifier: str | None
    name: str
    description: str | None
    kind: str

    @sb.field
    def tables(self, dataset_id: UUID | None = None) -> list[TableType]:
        by_dataset: dict[tuple[UUID | None, UUID], list[EntrySelection]] = {}
        for item in self._section.selections:
            if item.dataset_id in self._query.datasets and (dataset_id is None or item.dataset_id == dataset_id):
                by_dataset.setdefault((item.table_id, item.dataset_id), []).append(item)
        result = []
        for (entry_id, id), selections in by_dataset.items():
            metrics: dict[UUID, list[FragmentType]] = {}
            for selection in selections:
                metrics.setdefault(selection.metric_id, []).append(FragmentType.from_selection(selection))
            result.append(
                TableType(
                    id=uuid3(entry_id or self.id, f'dataset:{id}'),
                    entry_id=entry_id,
                    _query=self._query,
                    _selections=tuple(selections),
                    dataset=EntryDatasetType(_query=self._query, id=id),
                    metric_selections=[
                        MetricSelectionType(metric_id=metric, fragments=fragments) for metric, fragments in metrics.items()
                    ],
                )
            )
        return result

    @sb.field
    def unresolved_inputs(self) -> list[UnresolvedInputType]:
        return [
            UnresolvedInputType(node_id=p.node_id, port_id=p.port_id, dataset_id=p.dataset_id, reason=p.code, message=p.message)
            for p in self._section.problems
            if p.code != 'competing_claim' and (p.dataset_id is None or p.dataset_id in self._query.datasets)
        ]

    @sb.field
    def problems(self, years: list[int] | None = None) -> list[ProblemType]:
        return [
            ProblemType.from_finding(finding, self._query.selected_years(years))
            for finding in self._query.findings(self.id, years)
        ]

    @sb.field
    def problem_counts(self, years: list[int] | None = None) -> CountsType:
        return CountsType.from_findings(self._query.findings(self.id, years), self._query.selected_years(years))


@sb.type(name='DataEntry')
class DataEntryType:
    _query: sb.Private[DataEntryQuery]
    can_manage_years: bool

    @sb.field
    def years(self) -> list[int]:
        return self._query.years

    @sb.field
    def default_year(self) -> int | None:
        return max(self._query.years, default=None)

    @sb.field
    def sections(self, section_id: UUID | None = None, identifier: str | None = None) -> list[SectionType]:
        return [
            SectionType(
                _query=self._query,
                _section=section,
                id=section.id,
                identifier=section.identifier,
                name=str(section.name),
                description=str(section.description) if section.description else None,
                kind=section.kind,
            )
            for section in self._query.graph.data_entry.sections
            if (section_id is None or section.id == section_id) and (identifier is None or section.identifier == identifier)
        ]

    @sb.field
    def problem_counts(self, years: list[int] | None = None) -> CountsType:
        return CountsType.from_findings(self._query.findings(None, years), self._query.selected_years(years))

    @sb.field
    def resolution_problems(self) -> list[UnresolvedInputType]:
        return [
            UnresolvedInputType(node_id=p.node_id, port_id=p.port_id, dataset_id=p.dataset_id, reason=p.code, message=p.message)
            for p in self._query.graph.data_entry.problems
            if p.dataset_id is None or p.dataset_id in self._query.datasets
        ]
