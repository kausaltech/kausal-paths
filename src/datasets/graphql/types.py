"""Strawberry GraphQL types for DB-backed datasets."""

from collections import defaultdict
from datetime import date, datetime
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any, cast
from uuid import UUID

import strawberry as sb
import strawberry_django
from django.contrib.contenttypes.models import ContentType
from django.db.models import Q, prefetch_related_objects
from django.utils import timezone
from strawberry import auto

from pint.errors import UndefinedUnitError

from kausal_common.datasets.models import (
    DataPointComment as DataPointCommentModel,
    Dataset as DatasetModel,
    DatasetSchema as DatasetSchemaModel,
    DatasetSourceReference as DatasetSourceReferenceModel,
    DataSource as DataSourceModel,
)
from kausal_common.strawberry.ordering import with_sibling_ids
from kausal_common.strawberry.permissions import UserPermissionsMixin
from kausal_common.strawberry.registry import register_strawberry_type

from paths import gql
from paths.graphql_types import UnitType

from datasets.coordinates import DatasetCoordinateIndex
from datasets.editions import (
    DatasetEdition,
    DatasetEditions,
    edition_plausibility_ranges,
    evaluate_edition_plausibility,
    evaluate_edition_rules,
)
from datasets.graphql.enums import PlausibilityAggregation, PlausibilityDenominator, PlausibilityReference
from datasets.materialization import ensure_dataset_materializations
from datasets.plausibility import DatasetPlausibilityLookup, _log_unreadable
from datasets.snapshot import SourceReferenceSnapshot
from datasets.validation import load_violations
from datasets.validation_rules import (
    DimensionSumRule,
    NoGapsRule,
    RequiredCombinationsRule,
    ValidationRuleGQLInterface,
    ValueRangeRule,
    rule_to_gql,
    validation_rule_adapter,
)
from frameworks.evidence import QUALITY_OF_SPEC_KEY, get_evidence, quality_schemes_for_dataset
from frameworks.models import (
    DataEvidenceKind,
    DataPointEvidence as DataPointEvidenceModel,
    DataQualityLevel as DataQualityLevelModel,
    DataQualityScheme as DataQualitySchemeModel,
    Framework,
)
from nodes.defs.binding_def import DatasetBindingDef
from nodes.graphql.types.graph import DatasetExternalRefType
from nodes.graphql.types.problems import (
    DatasetDimensionCoordinateType,
    DatasetPlausibilityFindingType,
    DatasetValidationViolationType,
)
from nodes.graphql.types.shape import ShapeType
from nodes.metric_gen import metric_from_dataframe_standalone
from nodes.models import InstanceConfig, NodeInputPortBinding, PreferredInstanceSource
from nodes.quantities import get_registry
from nodes.units import Unit, unit_registry
from users.models import User
from users.schema import UserType

# The concrete rule types are only reachable through the ValidationRule
# interface, so they must be registered as extra schema types explicitly.
register_strawberry_type(ValueRangeRule.ObjectType)
register_strawberry_type(DimensionSumRule.ObjectType)
register_strawberry_type(NoGapsRule.ObjectType)
register_strawberry_type(RequiredCombinationsRule.ObjectType)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kausal_common.datasets.models import (
        DataPoint as DataPointModel,
        DatasetMetric as DatasetMetricModel,
        DatasetMetricValidationRule as DatasetMetricValidationRuleModel,
        DatasetSchemaDimension,
        Dimension as DimensionModel,
        DimensionCategory as DimensionCategoryModel,
    )

    from datasets.models import (
        DatasetMetricPlausibilityRange,
        PlausibilitySource,
    )
    from datasets.snapshot import DataPointSnapshot
    from nodes.defs.graph import DatasetMeta, DatasetMetricMeta, DimensionCategoryMeta, DimensionMeta
    from nodes.graphql.types.graph import DatasetPortType
    from nodes.graphql.types.metric import DimensionalMetricType
    from nodes.graphql.types.node import QuantityKindType  # used in lazy strawberry annotations
    from nodes.metric import DimensionalMetric


@sb.type(name='DatasetDimensionCategory')
class DatasetDimensionCategoryType:
    """A category within a dataset dimension (e.g. 'North', 'South')."""

    uuid: UUID
    identifier: str | None
    label: str
    short_label: str | None

    @classmethod
    def from_model(cls, cat: DimensionCategoryModel) -> DatasetDimensionCategoryType:
        return cls(
            uuid=cat.uuid,
            identifier=cat.identifier,
            label=cat.label_i18n or str(cat.uuid),
            short_label=cat.short_label_i18n or None,
        )

    @classmethod
    def from_meta(cls, cat: DimensionCategoryMeta) -> DatasetDimensionCategoryType:
        return cls(
            uuid=cat.id,
            identifier=cat.identifier,
            label=str(cat.label or cat.identifier or cat.id),
            short_label=str(cat.short_label) if cat.short_label is not None else None,
        )


@sb.type(name='DatasetDimension')
class DatasetDimensionType:
    """A dimension attached to a dataset schema (e.g. 'Region', 'Sector')."""

    id: sb.ID
    name: str
    categories: list[DatasetDimensionCategoryType]

    @classmethod
    def from_schema_dimension(cls, sd: DatasetSchemaDimension) -> DatasetDimensionType:
        dim: DimensionModel = sd.dimension
        cats = [DatasetDimensionCategoryType.from_model(cat) for cat in dim.categories.all()]
        return cls(
            id=sb.ID(str(dim.uuid)),
            name=dim.name_i18n or str(dim.uuid),
            categories=cats,
        )

    @classmethod
    def from_meta(cls, dim: DimensionMeta) -> DatasetDimensionType:
        return cls(
            id=sb.ID(str(dim.id)),
            name=str(dim.label or dim.identifier),
            categories=[DatasetDimensionCategoryType.from_meta(cat) for cat in dim.categories],
        )


@sb.type(name='DatasetCategoryCoordinate')
class DatasetCategoryCoordinateType:
    dimension_id: UUID
    category_id: UUID


@sb.type(name='DatasetCategoryCombination')
class DatasetCategoryCombinationType:
    id: UUID
    identifier: str
    coordinates: list[DatasetCategoryCoordinateType]


@sb.type(name='DatasetCategoryDomain')
class DatasetCategoryDomainType:
    mode: str
    combinations: list[DatasetCategoryCombinationType]


@sb.type(name='DatasetMetricValidationRule')
class MetricValidationRuleType:
    """A declarative validation rule bound to a dataset metric."""

    id: sb.ID
    rule: ValidationRuleGQLInterface = sb.field(
        description='The rule; the concrete type carries the kind-specific parameters.',
    )

    @classmethod
    def from_model(cls, obj: DatasetMetricValidationRuleModel) -> MetricValidationRuleType:
        return cls(id=sb.ID(str(obj.uuid)), rule=rule_to_gql(validation_rule_adapter.validate_python(obj.rule)))


@sb.type(name='PlausibilitySource', description='Where a set of reference ranges comes from.')
class PlausibilitySourceType:
    id: sb.ID
    identifier: str
    name: str
    url: str
    revision: str
    method: str
    is_example: bool = sb.field(description='Local demonstration data, not an empirical benchmark.')

    @classmethod
    def from_model(cls, obj: PlausibilitySource) -> PlausibilitySourceType:
        return cls(
            id=sb.ID(str(obj.uuid)),
            identifier=obj.identifier,
            name=obj.name,
            url=obj.url,
            revision=obj.revision,
            method=obj.method,
            is_example=obj.is_example,
        )


@sb.type(name='DatasetMetricPlausibilityRange')
class DatasetMetricPlausibilityRangeType:
    id: sb.ID
    identifier: str
    metric_uuid: UUID
    selection: list[Annotated['DatasetDimensionCoordinateType', sb.lazy('nodes.graphql.types.problems')]] = sb.field(
        description='Every selected dimension and category; a dimension may appear several times.',
    )
    aggregation: PlausibilityAggregation
    denominator: PlausibilityDenominator
    reference: PlausibilityReference
    max_gap_years: int | None = sb.field(description='For a previous-year reference: how far back the earlier value may be.')
    lower: float
    upper: float
    unit: Unit = sb.field(
        graphql_type=UnitType,
        description="Unit of the bounds: the metric's unit per denominator, or dimensionless for a ratio.",
    )
    first_year: int | None
    last_year: int | None
    sample_size: int | None
    revision: int
    source: PlausibilitySourceType

    @classmethod
    def from_model(
        cls, obj: DatasetMetricPlausibilityRange, coordinate_index: DatasetCoordinateIndex
    ) -> DatasetMetricPlausibilityRangeType:
        return cls(
            id=sb.ID(str(obj.uuid)),
            identifier=obj.identifier,
            metric_uuid=obj.metric.uuid,
            selection=[
                DatasetDimensionCoordinateType.from_coordinate(coordinate)
                for coordinate in coordinate_index.resolve_selection(obj.selected_categories())
            ],
            aggregation=PlausibilityAggregation(obj.aggregation),
            denominator=PlausibilityDenominator(obj.denominator),
            reference=PlausibilityReference(obj.reference),
            max_gap_years=obj.max_gap_years,
            lower=obj.lower,
            upper=obj.upper,
            unit=obj.bound_unit,
            first_year=obj.first_year,
            last_year=obj.last_year,
            sample_size=obj.sample_size,
            revision=obj.revision,
            source=PlausibilitySourceType.from_model(obj.source),
        )


@sb.type(name='ResolvedPlausibilityRange')
class ResolvedPlausibilityRangeType:
    """One reference interval expressed in the data point's metric unit."""

    lower: float | None = sb.field(description='Lower input-unit bound; null when population is unavailable.')
    upper: float | None = sb.field(description='Upper input-unit bound; null when population is unavailable.')
    reference: DatasetMetricPlausibilityRangeType


@register_strawberry_type
@sb.type(name='DatasetMetric')
class DatasetMetricType:
    """A metric (value column) defined in a dataset schema."""

    id: sb.ID
    name: str | None = sb.field(description='Column name used in DataFrames.')
    label: str = sb.field(description='Human-readable label.')
    unit: str = sb.field(deprecation_reason='Use unitInfo instead.')
    previous_sibling: sb.ID | None
    next_sibling: sb.ID | None
    validation_rules: list[MetricValidationRuleType] = sb.field(
        description='Validation rules evaluated against this metric, in order.',
    )
    quality_of: sb.ID | None = sb.field(
        default=None,
        description=(
            'When set, this metric is the numeric projection of the grades of the metric with this id: its values are '
            'derived from data-point evidence and cannot be written directly.'
        ),
    )
    _quantity_id: sb.Private[str | None] = None

    @sb.field(
        graphql_type=Annotated['UnitType', sb.lazy('paths.graphql_types')] | None,  # type: ignore[operator]
        description=(
            'Parsed unit of the metric, e.g. for checking compatibility with an input port. '
            'Null when the metric has no unit or its unit string does not parse.'
        ),
    )
    @staticmethod
    def unit_info(root: 'DatasetMetricType') -> Any:

        if not root.unit:
            return None
        try:
            return unit_registry.parse_units(root.unit)
        except Exception:
            return None

    @sb.field(
        graphql_type=Annotated['QuantityKindType', sb.lazy('nodes.graphql.types.node')] | None,  # type: ignore[operator]
        description=(
            'What the metric measures, e.g. for checking compatibility with an input port. '
            'Null means the metric is compatible with any quantity.'
        ),
    )
    @staticmethod
    def quantity(root: 'DatasetMetricType') -> Any:
        from nodes.graphql.types.node import QuantityKindType

        if not root._quantity_id:
            return None
        kind = get_registry().get(root._quantity_id)
        if kind is None:
            return None
        return QuantityKindType.from_kind(kind)

    @classmethod
    def from_model(
        cls,
        metric: DatasetMetricModel,
        previous_sibling: sb.ID | None = None,
        next_sibling: sb.ID | None = None,
    ) -> DatasetMetricType:
        return cls(
            id=sb.ID(str(metric.uuid)),
            name=metric.name,
            label=metric.label_i18n or metric.name or '',
            unit=metric.unit or '',
            previous_sibling=previous_sibling,
            next_sibling=next_sibling,
            validation_rules=[MetricValidationRuleType.from_model(rule) for rule in metric.validation_rules.all()],
            quality_of=sb.ID(quality_of) if (quality_of := (metric.spec or {}).get(QUALITY_OF_SPEC_KEY)) else None,
            _quantity_id=(metric.spec or {}).get('quantity'),
        )

    @classmethod
    def from_meta(
        cls,
        metric: DatasetMetricMeta,
        previous_sibling: sb.ID | None = None,
        next_sibling: sb.ID | None = None,
    ) -> DatasetMetricType:
        return cls(
            id=sb.ID(str(metric.id)),
            name=metric.identifier,
            label=str(metric.label or metric.identifier or metric.id),
            unit=metric.unit,
            previous_sibling=previous_sibling,
            next_sibling=next_sibling,
            validation_rules=[
                MetricValidationRuleType(id=sb.ID(str(rule.id)), rule=rule_to_gql(rule.rule))
                for rule in metric.validation_rules
                if rule.id is not None
            ],
            quality_of=sb.ID(str(metric.quality_of)) if metric.quality_of else None,
            _quantity_id=metric.quantity,
        )


@register_strawberry_type
@strawberry_django.type(DataPointCommentModel, name='DataPointComment')
class DataPointCommentType:
    """A user comment attached to a single data point."""

    text: auto
    is_sticky: auto
    is_review: auto
    review_state: auto
    resolved_at: auto
    resolved_by: Annotated['UserType', sb.lazy('users.schema')] | None
    created_at: auto
    created_by: Annotated['UserType', sb.lazy('users.schema')] | None
    last_modified_at: auto
    last_modified_by: Annotated['UserType', sb.lazy('users.schema')] | None

    @strawberry_django.field
    @staticmethod
    def id(root: sb.Parent[DataPointCommentModel]) -> sb.ID:
        return sb.ID(str(root.uuid))


@sb.enum
class DatasetSourceReferenceTarget(Enum):
    """
    Filter for `DatasetSourceReference` queries scoped to a dataset.

    `DATASET` returns refs bound to the dataset itself; `DATA_POINT` returns
    refs bound to one of its data points; `ALL` returns both.
    """

    DATASET = 'dataset'
    DATA_POINT = 'data_point'
    ALL = 'all'


@register_strawberry_type
@strawberry_django.type(DataSourceModel, name='DataSource')
class DataSourceType:
    """A published data source (study, dataset, report, …) usable as a reference."""

    name: auto
    edition: auto
    authority: auto
    description: auto
    url: auto
    created_at: auto
    created_by: Annotated['UserType', sb.lazy('users.schema')] | None
    last_modified_at: auto
    last_modified_by: Annotated['UserType', sb.lazy('users.schema')] | None

    @strawberry_django.field
    @staticmethod
    def id(root: sb.Parent[DataSourceModel]) -> sb.ID:
        return sb.ID(str(root.uuid))

    @strawberry_django.field(description='Single-line human-readable label (name, authority, edition).')
    @staticmethod
    def label(root: sb.Parent[DataSourceModel]) -> str:
        return root.get_label()


def _source_references_queryset_for_data_point(data_point: DataPointModel) -> Any:
    return (
        DatasetSourceReferenceModel.objects
        .filter(data_point=data_point)
        .select_related('data_source', 'created_by', 'last_modified_by')
        .order_by('-created_at')
    )


def _source_references_queryset_for_dataset(
    dataset: DatasetModel,
    target: DatasetSourceReferenceTarget,
) -> Any:
    qs = DatasetSourceReferenceModel.objects.select_related(
        'data_source', 'data_point', 'dataset', 'created_by', 'last_modified_by'
    ).order_by('-created_at')
    if target == DatasetSourceReferenceTarget.DATASET:
        return qs.filter(dataset=dataset)
    if target == DatasetSourceReferenceTarget.DATA_POINT:
        return qs.filter(data_point__dataset=dataset)
    return qs.filter(Q(dataset=dataset) | Q(data_point__dataset=dataset))


def _data_sources_queryset_for_dataset(dataset: DatasetModel) -> Any:
    """DataSources referenced from inside this dataset (via refs on it or its data points)."""
    return (
        DataSourceModel.objects
        .filter(Q(references__dataset=dataset) | Q(references__data_point__dataset=dataset))
        .distinct()
        .order_by('name')
    )


def _comments_queryset_for_data_point(data_point: DataPointModel) -> Any:
    return (
        DataPointCommentModel.objects
        .filter(data_point=data_point)
        .select_related('created_by', 'last_modified_by', 'resolved_by')
        .order_by('-created_at')
    )


def _comments_queryset_for_dataset(dataset: DatasetModel) -> Any:
    return (
        DataPointCommentModel.objects
        .filter(data_point__dataset=dataset)
        .select_related('data_point', 'created_by', 'last_modified_by', 'resolved_by')
        .order_by('-created_at')
    )


@sb.type(name='DataQualityLevel')
class DataQualityLevelType:
    """One grade of a framework quality scheme."""

    id: sb.ID
    identifier: str
    name: str
    description: str
    order: int
    score: float = sb.field(description='Numeric weight of the grade, 0 (worst) to 1 (best).')

    @classmethod
    def from_model(cls, level: DataQualityLevelModel) -> DataQualityLevelType:
        return cls(
            id=sb.ID(str(level.uuid)),
            identifier=level.identifier,
            name=level.name,
            description=level.description,
            order=level.order,
            score=float(level.score),
        )


@sb.type(name='DataQualityScheme')
class DataQualitySchemeType:
    """A versioned quality scale owned by a framework."""

    id: sb.ID
    identifier: str
    version: str
    name: str
    description: str
    levels: list[DataQualityLevelType] = sb.field(description='Grades, best first.')

    @classmethod
    def from_model(cls, scheme: DataQualitySchemeModel) -> DataQualitySchemeType:
        return cls(
            id=sb.ID(str(scheme.uuid)),
            identifier=scheme.identifier,
            version=scheme.version,
            name=scheme.name,
            description=scheme.description,
            levels=[DataQualityLevelType.from_model(level) for level in scheme.levels.all()],
        )


@sb.type(name='DataPointEvidence')
class DataPointEvidenceType:
    """What is asserted about a data point's value. Absent when nothing has been asserted."""

    kind: DataEvidenceKind | None = sb.field(description='How the value was obtained; null when unknown.')
    quality_level: DataQualityLevelType | None = sb.field(description='Assessed grade; null when ungraded.')
    last_modified_at: datetime = sb.field(
        description=(
            'When the assertion was last changed. A frozen edition does not record it, '
            'and gives the time the edition was saved instead.'
        ),
    )
    last_modified_by: User | None = sb.field(graphql_type=UserType | None)

    @classmethod
    def from_model(cls, evidence: DataPointEvidenceModel) -> DataPointEvidenceType:
        level = evidence.quality_level
        return cls(
            kind=DataEvidenceKind(evidence.kind) if evidence.kind else None,
            quality_level=DataQualityLevelType.from_model(level) if level is not None else None,
            last_modified_at=evidence.last_modified_at,
            last_modified_by=evidence.last_modified_by,
        )


class EditionPoints:
    """
    What the data points of one dataset edition share, loaded once for all of them.

    A point's value, categories, evidence and citations are the edition's. Its comments are
    not: a comment thread is a conversation about a cell rather than part of its value, so
    every edition shows the live thread, which follows the cell by its uuid.
    """

    def __init__(self, edition: DatasetEdition):
        self.edition = edition

    @cached_property
    def metrics(self) -> dict[UUID, DatasetMetricType]:
        return {metric.id: DatasetMetricType.from_meta(metric) for metric in self.edition.meta.metrics}

    @cached_property
    def categories(self) -> dict[UUID, DatasetDimensionCategoryType]:
        return {
            category.id: DatasetDimensionCategoryType.from_meta(category)
            for dimension in self.edition.dimensions.values()
            for category in dimension.categories
        }

    @cached_property
    def comments(self) -> dict[UUID, list[DataPointCommentModel]]:
        row = self.edition.row
        if row is None:
            return {}
        result: dict[UUID, list[DataPointCommentModel]] = defaultdict(list)
        ids = [point.id for point in self.edition.snapshot.points]
        for comment in _comments_queryset_for_dataset(row).filter(data_point__uuid__in=ids):
            assert comment.data_point is not None
            result[comment.data_point.uuid].append(comment)
        return result

    @cached_property
    def source_references(self) -> dict[UUID, DatasetSourceReferenceModel]:
        """The rows of the references this edition cites; one deleted since is left out."""
        snapshot = self.edition.snapshot
        ids = {ref.id for ref in snapshot.source_references} | {ref.id for point in snapshot.points for ref in point.sources}
        rows = DatasetSourceReferenceModel.objects.filter(uuid__in=ids).select_related(
            'data_source', 'data_point', 'dataset', 'created_by', 'last_modified_by'
        )
        return {ref.uuid: ref for ref in rows}

    def references(self, refs: Iterable[SourceReferenceSnapshot]) -> list[DatasetSourceReferenceModel]:
        rows = [self.source_references[ref.id] for ref in refs if ref.id in self.source_references]
        return sorted(rows, key=lambda ref: ref.created_at, reverse=True)

    @cached_property
    def quality_levels(self) -> dict[str, DataQualityLevelModel]:
        ids = {
            point.evidence.quality_level.uuid
            for point in self.edition.snapshot.points
            if point.evidence is not None and point.evidence.quality_level is not None
        }
        return {str(level.uuid): level for level in DataQualityLevelModel.objects.filter(uuid__in=ids)}

    @cached_property
    def users(self) -> dict[str, User]:
        ids = {
            point.evidence.last_modified_by
            for point in self.edition.snapshot.points
            if point.evidence is not None and point.evidence.last_modified_by is not None
        }
        return {str(user.uuid): user for user in User.objects.filter(uuid__in=ids)}

    @cached_property
    def live_evidence(self) -> dict[UUID, DataPointEvidenceModel]:
        """The evidence rows of a live edition, which alone record when each was changed."""
        if not self.edition.is_live or self.edition.row is None:
            return {}
        rows = DataPointEvidenceModel.objects.filter(data_point__dataset=self.edition.row).select_related(
            'data_point', 'last_modified_by'
        )
        return {row.data_point.uuid: row for row in rows}

    def evidence(self, point: DataPointSnapshot) -> DataPointEvidenceType | None:
        evidence = point.evidence
        if evidence is None:
            return None
        level = self.quality_levels.get(evidence.quality_level.uuid) if evidence.quality_level is not None else None
        live = self.live_evidence.get(point.id)
        if live is not None:
            modified_at, modified_by = live.last_modified_at, live.last_modified_by
        else:
            revision = self.edition.revision
            modified_at = revision.created_at if revision is not None else timezone.now()
            modified_by = self.users.get(evidence.last_modified_by) if evidence.last_modified_by is not None else None
        return DataPointEvidenceType(
            kind=DataEvidenceKind(evidence.kind) if evidence.kind else None,
            quality_level=DataQualityLevelType.from_model(level) if level is not None else None,
            last_modified_at=modified_at,
            last_modified_by=modified_by,
        )

    @cached_property
    def plausibility(self) -> DatasetPlausibilityLookup | None:
        row = self.edition.row
        if row is None or self.edition.meta.is_external_placeholder:
            return None
        values: dict[UUID, dict[tuple[int, frozenset[str]], float]] = defaultdict(dict)
        for point in self.edition.snapshot.points:
            if point.value is not None:
                cell = frozenset(str(category) for category in point.categories.values())
                values[point.metric][point.date.year, cell] = point.value
        try:
            data = self.edition.frame()
        except UndefinedUnitError as exc:
            _log_unreadable(row, exc)
            return None
        return DatasetPlausibilityLookup(row, history=(data.frame, data.dim_cols), values=values)

    def point(self, point: DataPointSnapshot) -> DataPointType:
        obj = DataPointType(
            id=sb.ID(str(point.id)),
            date=point.date,
            value=point.value,
            metric=self.metrics[point.metric],
            dimension_categories=[
                self.categories[category] for category in point.categories.values() if category in self.categories
            ],
        )
        obj._point = point
        obj._points = self
        return obj


@register_strawberry_type
@sb.type(name='DataPoint')
class DataPointType:
    """A dataset data point, as the edition of its dataset records it."""

    id: sb.ID
    date: date
    value: float | None
    metric: DatasetMetricType
    dimension_categories: list[DatasetDimensionCategoryType]

    _model: sb.Private['DataPointModel | None'] = None
    _dataset: sb.Private[DatasetModel | None] = None
    _comments: sb.Private[list[DataPointCommentModel] | None] = None
    _point: sb.Private['DataPointSnapshot | None'] = None
    _points: sb.Private[EditionPoints | None] = None

    @sb.field(
        graphql_type=list[ResolvedPlausibilityRangeType],
        description="Matching reference intervals in this metric's input unit, one entry per source.",
    )
    @staticmethod
    def plausibility_ranges(root: 'DataPointType', info: gql.Info) -> list[ResolvedPlausibilityRangeType]:
        if root._point is not None and root._points is not None:
            lookup = root._points.plausibility
            metric_uuid, year = root._point.metric, root._point.date.year
            category_uuids = {str(dim): str(cat) for dim, cat in root._point.categories.items()}
        elif root._model is not None:
            point = root._model
            dataset = root._dataset or point.dataset
            cache = info.context.dataset_plausibility_lookups
            if dataset.pk not in cache:
                cache[dataset.pk] = DatasetPlausibilityLookup(dataset)
            lookup = cache[dataset.pk]
            metric_uuid, year = point.metric.uuid, point.date.year
            category_uuids = {str(category.dimension.uuid): str(category.uuid) for category in point.dimension_categories.all()}
        else:
            return []
        if lookup is None or lookup.coordinate_index is None:
            return []
        coordinate_index = lookup.coordinate_index
        return [
            ResolvedPlausibilityRangeType(
                lower=lower,
                upper=upper,
                reference=DatasetMetricPlausibilityRangeType.from_model(rule, coordinate_index),
            )
            for rule, lower, upper in lookup.for_cell(metric_uuid, year, category_uuids)
        ]

    @sb.field(graphql_type=list[DataPointCommentType], description='Comments attached to this data point, newest first.')
    @staticmethod
    def comments(root: 'DataPointType') -> list[DataPointCommentModel]:
        if root._comments is not None:
            return root._comments
        if root._point is not None and root._points is not None:
            return root._points.comments.get(root._point.id, [])
        if root._model is None:
            return []
        return list(_comments_queryset_for_data_point(root._model))

    @sb.field(
        graphql_type=list[Annotated['DatasetSourceReferenceType', sb.lazy('datasets.graphql.types')]],
        description='Source references attached directly to this data point, newest first.',
    )
    @staticmethod
    def source_references(root: 'DataPointType') -> list[DatasetSourceReferenceModel]:
        if root._point is not None and root._points is not None:
            return root._points.references(root._point.sources)
        if root._model is None:
            return []
        return list(_source_references_queryset_for_data_point(root._model))

    @sb.field(graphql_type=DataPointEvidenceType | None)
    @staticmethod
    def evidence(root: 'DataPointType') -> DataPointEvidenceType | None:
        if root._point is not None and root._points is not None:
            return root._points.evidence(root._point)
        if root._model is None:
            return None
        evidence = get_evidence(root._model)
        return DataPointEvidenceType.from_model(evidence) if evidence is not None else None

    @classmethod
    def from_model(cls, data_point: DataPointModel) -> DataPointType:
        obj = cls(
            id=sb.ID(str(data_point.uuid)),
            date=data_point.date,
            value=float(data_point.value) if data_point.value is not None else None,
            metric=DatasetMetricType.from_model(data_point.metric),
            dimension_categories=[
                DatasetDimensionCategoryType.from_model(category) for category in data_point.dimension_categories.all()
            ],
        )
        obj._model = data_point
        return obj


@strawberry_django.type(DatasetSchemaModel, name='DatasetSchema')
class DatasetSchemaType:
    description: auto

    @strawberry_django.field
    @staticmethod
    def id(root: sb.Parent[DatasetSchemaModel]) -> sb.ID:
        return sb.ID(str(root.uuid))

    @strawberry_django.field
    @staticmethod
    def name(root: sb.Parent[DatasetSchemaModel]) -> str:
        return root.name_i18n

    @strawberry_django.field
    @staticmethod
    def metrics(root: sb.Parent[DatasetSchemaModel]) -> list[DatasetMetricType]:
        prefetch_related_objects([root], 'metrics__validation_rules')
        return [DatasetMetricType.from_model(metric) for metric in root.metrics.all()]

    @strawberry_django.field
    @staticmethod
    def dimensions(root: sb.Parent[DatasetSchemaModel]) -> list[DatasetDimensionType]:
        prefetch_related_objects([root], 'dimensions__dimension__categories')
        return [DatasetDimensionType.from_schema_dimension(item) for item in root.dimensions.all()]


@register_strawberry_type
@sb.type(name='Dataset')
class DatasetType(UserPermissionsMixin):
    """
    A dataset as the selected instance source reads it.

    Structure and content come from the dataset's edition in that source: its pinned
    revision, or its live draft. What exists only on the live row -- permissions,
    editability -- is read-only or null in a frozen edition.
    """

    _edition: sb.Private[DatasetEdition]
    _forecast_from: sb.Private[int | None] = None
    """A binding's first forecast year, which overrides the dataset's own."""

    @property
    def _model(self) -> DatasetModel | None:
        """The row, where it is what this edition reads; `UserPermissionsMixin` checks it."""
        return self._edition.row if self._edition.is_live else None

    @property
    def _meta(self) -> DatasetMeta:
        return self._edition.meta

    @sb.field
    def id(self) -> sb.ID:
        return sb.ID(str(self._meta.id))

    @sb.field
    def identifier(self) -> str | None:
        return self._meta.identifier

    @sb.field(description='Whether the dataset object is only a placeholder without imported datapoints.')
    def is_external_placeholder(self) -> bool:
        return self._meta.is_external_placeholder

    @sb.field(
        graphql_type=Annotated['DatasetExternalRefType', sb.lazy('nodes.graphql.types.graph')] | None,
        description='External source reference for externally backed datasets.',
    )
    def external_ref(self) -> DatasetExternalRefType | None:
        from nodes.graphql.types.graph import dataset_external_ref_to_gql

        return dataset_external_ref_to_gql(self._meta.external_ref)

    @sb.field(description='The timestamp of the last modification; in a frozen edition, when it was saved.')
    def last_modified_at(self) -> datetime | None:
        edition = self._edition
        if edition.revision is not None:
            return edition.revision.created_at
        return edition.row.last_modified_at if edition.row is not None else None

    @sb.field(
        description='The user who last modified the dataset; in a frozen edition, who saved it.',
        graphql_type=UserType | None,
    )
    def last_modified_by(self) -> User | None:
        edition = self._edition
        if edition.revision is not None:
            return cast('User | None', edition.revision.user)
        return edition.row.last_modified_by if edition.row is not None else None

    @sb.field(description='The timestamp of the creation.')
    def created_at(self) -> datetime | None:
        return self._edition.row.created_at if self._edition.row is not None else None

    @sb.field(description='The user who created the dataset.', graphql_type=UserType | None)
    def created_by(self) -> User | None:
        return self._edition.row.created_by if self._edition.row is not None else None

    @sb.field(
        graphql_type=DatasetSchemaType | None,
        deprecation_reason='Read name, metrics and dimensions from the dataset; they follow the selected edition.',
    )
    def schema(self) -> DatasetSchemaModel | None:
        if self._model is None:
            return None
        if self._edition.owner is not None:
            self._edition.owner.prefetch_schemas()
        return self._model.schema

    @sb.field(description='Whether this dataset owns an editable schema definition.')
    def schema_is_editable(self) -> bool:
        model = self._model
        if model is None or model.schema is None:
            return False
        self._load_schema_editability(model)
        return model.schema.is_editable and not model._schema_is_shared and not model._schema_has_other_datasets

    @staticmethod
    def _load_schema_editability(model: DatasetModel) -> None:
        """Single-object mutations may return a model without list-query annotations."""
        if hasattr(model, '_schema_is_shared'):
            return
        annotated = DatasetModel.objects.with_schema_editability(Framework).get(pk=model.pk)
        model._schema_is_shared = annotated._schema_is_shared
        model._schema_has_other_datasets = annotated._schema_has_other_datasets

    @sb.field(description='UUID of the node that owns this dataset, or null for instance datasets.')
    def owner_node_id(self) -> sb.ID | None:
        row = self._edition.row
        owner = row.scope_node if row is not None else None
        return sb.ID(str(owner.uuid)) if owner is not None else None

    @sb.field
    def name(self) -> str:
        return str(self._meta.name or self._meta.identifier or '')

    @sb.field
    def is_editable(self, info: gql.Info) -> bool:
        model = self._model
        if model is None or model.schema is None:
            return False
        self._load_schema_editability(model)
        if not (model.schema.is_editable or model._schema_is_shared):
            return False
        instance = info.context.instance_config
        if instance is None:
            return True

        if model.scope_content_type_id != ContentType.objects.get_for_model(InstanceConfig).pk:
            return True  # Node-owned data follows its node editor's permissions.
        if model.scope_id != instance.pk:
            return False
        resources = info.context.instance_resources
        return bool(resources and resources.node_edit_context(info, instance, PreferredInstanceSource.DRAFT).can_change_instance)

    @sb.field(description='Default or effective first forecast year for this dataset.')
    def forecast_from(self) -> int | None:
        return self._forecast_from if self._forecast_from is not None else self._meta.forecast_from

    @sb.field
    def dimensions(self) -> list[DatasetDimensionType]:
        catalog = self._edition.dimensions
        return [DatasetDimensionType.from_meta(catalog[id]) for id in self._meta.declared_dimension_ids if id in catalog]

    @sb.field(
        graphql_type=ShapeType | None,
        description="The shape that defines the dataset's entry form, as the dataset's instance resolves it.",
    )
    def shape(self, info: gql.Info) -> ShapeType | None:
        return ShapeType.resolve(info, self._edition.config, self._meta.shape_id, self._edition.source)

    @sb.field(description="The dataset's entry form: the category combinations of its shape, as its instance resolves it.")
    def category_domain(self) -> DatasetCategoryDomainType:
        domain = self._meta.category_domain
        return DatasetCategoryDomainType(
            mode=domain.mode,
            combinations=[
                DatasetCategoryCombinationType(
                    id=combination.id,
                    identifier=combination.identifier,
                    coordinates=[
                        DatasetCategoryCoordinateType(dimension_id=dimension_id, category_id=category_id)
                        for dimension_id, category_id in combination.categories.items()
                    ],
                )
                for combination in domain.combinations
            ],
        )

    @sb.field
    def metrics(self) -> list[DatasetMetricType]:
        return [
            DatasetMetricType.from_meta(metric, previous_sibling=prev_id, next_sibling=next_id)
            for metric, prev_id, next_id in with_sibling_ids(list(self._meta.metrics), lambda metric: sb.ID(str(metric.id)))
        ]

    @sb.field(
        graphql_type=list[DataQualitySchemeType],
        description="Quality schemes whose grades may be assigned to this dataset's data points.",
    )
    def quality_schemes(self) -> list[DataQualitySchemeType]:
        row = self._edition.row
        if row is None:
            return []
        return [DataQualitySchemeType.from_model(scheme) for scheme in quality_schemes_for_dataset(row)]

    @cached_property
    def _points(self) -> EditionPoints:
        return EditionPoints(self._edition)

    @sb.field(graphql_type=list[DataPointType])
    def data_points(self) -> list[DataPointType]:
        return [self._points.point(point) for point in self._edition.snapshot.points]

    @sb.field(graphql_type=list[DataPointCommentType], description='All data point comments in this dataset, newest first.')
    def data_point_comments(self) -> list[DataPointCommentModel]:
        comments = [comment for thread in self._points.comments.values() for comment in thread]
        return sorted(comments, key=lambda comment: comment.created_at, reverse=True)

    @sb.field(
        graphql_type=list[Annotated['DatasetSourceReferenceType', sb.lazy('datasets.graphql.types')]],
        description=(
            'Source references inside this dataset. `target` selects refs attached '
            'directly to the dataset, refs attached to its data points, or both.'
        ),
    )
    def source_references(
        self,
        target: DatasetSourceReferenceTarget = DatasetSourceReferenceTarget.DATASET,
    ) -> list[DatasetSourceReferenceModel]:
        snapshot = self._edition.snapshot
        refs: list[SourceReferenceSnapshot] = []
        if target != DatasetSourceReferenceTarget.DATA_POINT:
            refs.extend(snapshot.source_references)
        if target != DatasetSourceReferenceTarget.DATASET:
            refs.extend(ref for point in snapshot.points for ref in point.sources)
        return self._points.references(refs)

    @sb.field(
        graphql_type=list[DataSourceType],
        description='DataSources referenced from this dataset (via refs on it or its data points).',
    )
    def data_sources(self) -> list[DataSourceModel]:
        ids = [source.id for source in self._edition.snapshot.data_sources]
        return list(DataSourceModel.objects.filter(uuid__in=ids).order_by('name'))

    @sb.field(graphql_type=list[Annotated['DimensionalMetricType', sb.lazy('nodes.graphql.types.metric')]])
    def data(self) -> list['DimensionalMetric']:
        """Load the full dataset as DimensionalMetric objects (one per metric column)."""
        df = self._edition.snapshot.to_frame(self._edition.dimensions)
        forecast_from = self.forecast_from()
        ds_id = self._meta.identifier or str(self._meta.id)
        return [
            metric_from_dataframe_standalone(
                df,
                metric_col=col,
                metric_id=f'{ds_id}:{col}',
                metric_name=col,
                forecast_from=forecast_from,
            )
            for col in df.get_meta().metric_cols
        ]

    @sb.field(
        graphql_type=list[Annotated['DatasetValidationViolationType', sb.lazy('nodes.graphql.types.problems')]],
        description=(
            'Validation-rule violations of this dataset. For the live draft, read from the '
            'persisted evaluation results, repairing a stale materialization first; for a '
            'frozen edition, evaluated on its content.'
        ),
    )
    def validation_violations(self) -> "list['DatasetValidationViolationType']":
        edition = self._edition
        if edition.is_live and edition.row is not None:
            materialization = ensure_dataset_materializations([edition.row]).get(edition.row.pk)
            violations = load_violations(materialization.validation_violations) if materialization is not None else []
        else:
            violations = evaluate_edition_rules(edition)
        return [DatasetValidationViolationType.from_violation(violation) for violation in violations]

    @sb.field(
        graphql_type=list[Annotated['DatasetPlausibilityFindingType', sb.lazy('nodes.graphql.types.problems')]],
        description='Advisory plausibility findings for entered cells; these do not block publication.',
    )
    def plausibility_findings(self) -> "list['DatasetPlausibilityFindingType']":
        return [DatasetPlausibilityFindingType.from_finding(finding) for finding in evaluate_edition_plausibility(self._edition)]

    @sb.field(
        description=(
            'Applicable advisory reference ranges for this dataset and instance: the curated ones, then those '
            'derived from its own history (source dataset-history).'
        ),
    )
    def plausibility_ranges(self) -> list[DatasetMetricPlausibilityRangeType]:
        row = self._edition.row
        if row is None:
            return []
        ranges = edition_plausibility_ranges(self._edition)
        if not ranges:
            return []
        coordinate_index = DatasetCoordinateIndex(row)
        return [DatasetMetricPlausibilityRangeType.from_model(rule, coordinate_index) for rule in ranges]

    @sb.field(graphql_type=list[Annotated['DatasetPortType', sb.lazy('nodes.graphql.types.graph')]])
    def port_bindings(self, info: gql.Info) -> list['DatasetPortType']:
        """Discover which node ports use this dataset."""
        from nodes.graphql.bindings import binding_to_gql
        from nodes.graphql.types.graph import DatasetPortType

        edition = self._edition
        resources = info.context.instance_resources
        if edition.graph is not None and resources is not None:
            editions = resources.require_dataset_editions(edition.config, edition.source)
            return [
                DatasetPortType.from_binding(binding, editions=editions)
                for binding in edition.graph.bindings
                if isinstance(binding, DatasetBindingDef) and binding.dataset_uuid == edition.id
            ]
        if edition.row is None:
            return []
        rows = (
            NodeInputPortBinding.objects
            .filter(dataset=edition.row)
            .select_related('node', 'dataset', 'metric')
            .order_by('node', 'port_id', 'position')
        )
        return [binding_to_gql(row) for row in rows]

    @classmethod
    def from_edition(cls, edition: DatasetEdition, *, forecast_from: int | None = None) -> DatasetType:
        return cls(_edition=edition, _forecast_from=forecast_from)

    @classmethod
    def from_model(cls, dataset: DatasetModel) -> DatasetType:
        """
        Read a dataset from its row alone, as its draft has it.

        For the result of a mutation, which always acts on the draft and must show what it wrote.
        """
        return cls(_edition=DatasetEdition.from_row(dataset))

    @classmethod
    def from_binding(cls, binding: DatasetBindingDef, editions: DatasetEditions | None = None) -> DatasetType | None:
        """Read a binding's dataset from the edition the binding belongs to, or from its row when there is none."""
        if binding.dataset_uuid is None:
            return None
        if editions is not None:
            edition = editions.get(binding.dataset_uuid)
            if edition is None:
                return None
            return cls.from_edition(edition, forecast_from=binding.forecast_from)
        row = (
            DatasetModel.objects
            .with_schema_editability(Framework)
            .filter(uuid=binding.dataset_uuid)
            .select_related('schema', 'created_by', 'last_modified_by')
            .first()
        )
        if row is None or row.schema is None:
            return None
        obj = cls.from_model(row)
        obj._forecast_from = binding.forecast_from
        return obj


# DatasetSourceReferenceType is defined after DataPointType and DatasetType
# because its `data_point` and `dataset` resolvers return those types — a
# circular reference that's awkward to forward-declare in the same module.
@register_strawberry_type
@strawberry_django.type(DatasetSourceReferenceModel, name='DatasetSourceReference')
class DatasetSourceReferenceType:
    """Link from a data point or a dataset to a `DataSource`."""

    data_source: DataSourceType
    created_at: auto
    created_by: Annotated['UserType', sb.lazy('users.schema')] | None
    last_modified_at: auto
    last_modified_by: Annotated['UserType', sb.lazy('users.schema')] | None

    @strawberry_django.field
    @staticmethod
    def id(root: sb.Parent[DatasetSourceReferenceModel]) -> sb.ID:
        return sb.ID(str(root.uuid))

    @strawberry_django.field(description='The data point this reference is attached to, if any.')
    @staticmethod
    def data_point(root: sb.Parent[DatasetSourceReferenceModel]) -> DataPointType | None:
        dp = root.data_point
        return DataPointType.from_model(dp) if dp is not None else None

    @strawberry_django.field(description='The dataset this reference is attached to directly, if any.')
    @staticmethod
    def dataset(root: sb.Parent[DatasetSourceReferenceModel]) -> DatasetType | None:
        ds = root.dataset
        return DatasetType.from_model(ds) if ds is not None else None
