"""Strawberry GraphQL types for DB-backed datasets."""

from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, cast
from uuid import UUID

import strawberry as sb
import strawberry_django
from strawberry import auto

from kausal_common.datasets.models import (
    DataPointComment as DataPointCommentModel,
    Dataset as DatasetModel,
    DatasetSourceReference as DatasetSourceReferenceModel,
    DataSource as DataSourceModel,
)
from kausal_common.strawberry.ordering import with_sibling_ids
from kausal_common.strawberry.permissions import UserPermissionsMixin
from kausal_common.strawberry.registry import register_strawberry_type

from datasets.validation_rules import (
    AllowedCombinationsRule,
    DimensionSumRule,
    NoGapsRule,
    RequiredCombinationsRule,
    ValidationRuleGQLInterface,
    ValueRangeRule,
    rule_to_gql,
    validation_rule_adapter,
)
from users.models import User
from users.schema import UserType

# The concrete rule types are only reachable through the ValidationRule
# interface, so they must be registered as extra schema types explicitly.
register_strawberry_type(ValueRangeRule.ObjectType)
register_strawberry_type(DimensionSumRule.ObjectType)
register_strawberry_type(NoGapsRule.ObjectType)
register_strawberry_type(RequiredCombinationsRule.ObjectType)
register_strawberry_type(AllowedCombinationsRule.ObjectType)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kausal_common.datasets.models import (
        DataPoint as DataPointModel,
        DatasetMetric as DatasetMetricModel,
        DatasetMetricValidationRule as DatasetMetricValidationRuleModel,
        DatasetSchemaDimension,
        Dimension as DimensionModel,
        DimensionCategory as DimensionCategoryModel,
    )

    from paths.graphql_types import UnitType  # used in lazy strawberry annotations

    from nodes.defs.binding_def import DatasetBindingDef
    from nodes.graphql.types.graph import DatasetExternalRefType, DatasetPortType
    from nodes.graphql.types.metric import DimensionalMetricType
    from nodes.graphql.types.node import QuantityKindType  # used in lazy strawberry annotations
    from nodes.graphql.types.problems import DatasetValidationViolationType
    from nodes.metric import DimensionalMetric


@sb.type(name='DatasetDimensionCategory')
class DatasetDimensionCategoryType:
    """A category within a dataset dimension (e.g. 'North', 'South')."""

    uuid: UUID
    identifier: str | None
    label: str

    @classmethod
    def from_model(cls, cat: DimensionCategoryModel) -> DatasetDimensionCategoryType:
        return cls(
            uuid=cat.uuid,
            identifier=cat.identifier,
            label=cat.label_i18n or str(cat.uuid),
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
        from nodes.units import unit_registry

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
        from nodes.quantities import get_registry

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
            _quantity_id=(metric.spec or {}).get('quantity'),
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
    from django.db.models import Q

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
    from django.db.models import Q

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


@register_strawberry_type
@sb.type(name='DataPoint')
class DataPointType:
    """A stored dataset data point."""

    id: sb.ID
    date: date
    value: float | None
    metric: DatasetMetricType
    dimension_categories: list[DatasetDimensionCategoryType]

    _model: sb.Private['DataPointModel | None'] = None

    @sb.field(description='Comments attached to this data point, newest first.')
    @staticmethod
    def comments(root: 'DataPointType') -> list[DataPointCommentType]:
        if root._model is None:
            return []
        return cast('list[DataPointCommentType]', list(_comments_queryset_for_data_point(root._model)))

    @sb.field(description='Source references attached directly to this data point, newest first.')
    @staticmethod
    def source_references(root: 'DataPointType') -> "list['DatasetSourceReferenceType']":
        if root._model is None:
            return []
        return cast(
            'list[DatasetSourceReferenceType]',
            list(_source_references_queryset_for_data_point(root._model)),
        )

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


@register_strawberry_type
@sb.type(name='Dataset')
class DatasetType(UserPermissionsMixin):
    """A DB-backed dataset with schema, dimensions, metrics and data."""

    id: sb.ID
    identifier: str | None
    is_external_placeholder: bool = sb.field(
        description='Whether the dataset object is only a placeholder without imported datapoints.'
    )
    external_ref: Annotated['DatasetExternalRefType', sb.lazy('nodes.graphql.types.graph')] | None = sb.field(
        description='External source reference for externally backed datasets.'
    )
    last_modified_at: datetime | None = sb.field(description='The timestamp of the last modification.')
    last_modified_by: User | None = sb.field(
        description='The user who last modified the dataset.',
        graphql_type=UserType | None,
    )
    created_at: datetime | None = sb.field(description='The timestamp of the creation.')
    created_by: User | None = sb.field(
        description='The user who created the dataset.',
        graphql_type=UserType | None,
    )

    _model: sb.Private['DatasetModel | None'] = None
    _forecast_from: sb.Private[int | None] = None

    @sb.field
    @staticmethod
    def name(root: 'DatasetType') -> str:
        if root._model is None or root._model.schema is None:
            return root.identifier or ''
        return root._model.schema.name_i18n or root.identifier or ''

    @sb.field
    @staticmethod
    def is_editable(root: 'DatasetType') -> bool:
        if root._model is None or root._model.schema is None:
            return False
        return root._model.schema.is_editable

    @sb.field(description='Default or effective first forecast year for this dataset.')
    @staticmethod
    def forecast_from(root: 'DatasetType') -> int | None:
        if root._forecast_from is not None:
            return root._forecast_from
        if root._model is None:
            return None
        return (root._model.spec or {}).get('forecast_from')

    @sb.field
    @staticmethod
    def dimensions(root: 'DatasetType') -> list[DatasetDimensionType]:
        if root._model is None or root._model.schema is None:
            return []
        return [
            DatasetDimensionType.from_schema_dimension(sd)
            for sd in root._model.schema.dimensions.select_related('dimension').prefetch_related('dimension__categories')
        ]

    @sb.field(description='Meaningful category combinations declared by this dataset schema.')
    @staticmethod
    def category_domain(root: 'DatasetType') -> DatasetCategoryDomainType:
        if root._model is None or root._model.schema is None:
            return DatasetCategoryDomainType(mode='open', combinations=[])
        domain = root._model.schema.category_domain
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
    @staticmethod
    def metrics(root: 'DatasetType') -> list[DatasetMetricType]:
        if root._model is None or root._model.schema is None:
            return []
        metrics = list(root._model.schema.metrics.all())
        return [
            DatasetMetricType.from_model(metric, previous_sibling=prev_id, next_sibling=next_id)
            for metric, prev_id, next_id in with_sibling_ids(metrics, lambda metric: sb.ID(str(metric.uuid)))
        ]

    @sb.field(graphql_type=list[DataPointType])
    @staticmethod
    def data_points(root: 'DatasetType') -> list[DataPointType]:
        if root._model is None:
            return []
        data_points = root._model.data_points.select_related('metric').prefetch_related('dimension_categories__dimension')
        return [DataPointType.from_model(data_point) for data_point in data_points]

    @sb.field(description='All data point comments in this dataset, newest first.')
    @staticmethod
    def data_point_comments(root: 'DatasetType') -> list[DataPointCommentType]:
        if root._model is None:
            return []
        return cast('list[DataPointCommentType]', list(_comments_queryset_for_dataset(root._model)))

    @sb.field(
        description=(
            'Source references inside this dataset. `target` selects refs attached '
            'directly to the dataset, refs attached to its data points, or both.'
        ),
    )
    @staticmethod
    def source_references(
        root: 'DatasetType',
        target: DatasetSourceReferenceTarget = DatasetSourceReferenceTarget.DATASET,
    ) -> "list['DatasetSourceReferenceType']":
        if root._model is None:
            return []
        return cast(
            'list[DatasetSourceReferenceType]',
            list(_source_references_queryset_for_dataset(root._model, target)),
        )

    @sb.field(description='DataSources referenced from this dataset (via refs on it or its data points).')
    @staticmethod
    def data_sources(root: 'DatasetType') -> list[DataSourceType]:
        if root._model is None:
            return []
        return cast('list[DataSourceType]', list(_data_sources_queryset_for_dataset(root._model)))

    @sb.field(graphql_type=list[Annotated['DimensionalMetricType', sb.lazy('nodes.graphql.types.metric')]])
    @staticmethod
    def data(root: 'DatasetType') -> list['DimensionalMetric']:
        """Load the full dataset as DimensionalMetric objects (one per metric column)."""
        if root._model is None:
            return []
        from nodes.datasets import DBDataset

        df = DBDataset.deserialize_df(root._model)

        forecast_from = DatasetType.forecast_from(root)
        meta = df.get_meta()
        results: list['DimensionalMetric'] = []
        from nodes.metric_gen import metric_from_dataframe_standalone

        for col in meta.metric_cols:
            ds_id = root.identifier or str(root.id)
            results.append(
                metric_from_dataframe_standalone(
                    df,
                    metric_col=col,
                    metric_id=f'{ds_id}:{col}',
                    metric_name=col,
                    forecast_from=forecast_from,
                )
            )
        return results

    @sb.field(
        graphql_type=list[Annotated['DatasetValidationViolationType', sb.lazy('nodes.graphql.types.problems')]],
        description=(
            'Current validation-rule violations of this dataset. Read from the '
            'persisted evaluation results, repairing a stale materialization first.'
        ),
    )
    @staticmethod
    def validation_violations(root: 'DatasetType') -> "list['DatasetValidationViolationType']":
        if root._model is None:
            return []
        from datasets.validation import load_violations
        from nodes.dataset_materialization import ensure_dataset_materializations
        from nodes.graphql.types.problems import DatasetValidationViolationType

        materializations = ensure_dataset_materializations([root._model])
        materialization = materializations.get(root._model.pk)
        if materialization is None:
            return []
        return [
            DatasetValidationViolationType.from_violation(violation)
            for violation in load_violations(materialization.validation_violations)
        ]

    @sb.field(graphql_type=list[Annotated['DatasetPortType', sb.lazy('nodes.graphql.types.graph')]])
    @staticmethod
    def port_bindings(root: 'DatasetType') -> list['DatasetPortType']:
        """Discover which node ports use this dataset."""
        from nodes.graphql.bindings import _to_gql
        from nodes.models import NodeInputPortBinding

        if root._model is None:
            return []
        rows = (
            NodeInputPortBinding.objects
            .filter(dataset=root._model)
            .select_related('node', 'dataset', 'metric')
            .order_by('node', 'port_id', 'position')
        )
        return [_to_gql(row) for row in rows]

    @classmethod
    def from_model(cls, dataset: DatasetModel) -> DatasetType:
        from nodes.graphql.types.graph import _dataset_external_ref_to_gql

        obj = cls(
            id=sb.ID(str(dataset.uuid)),
            identifier=dataset.identifier,
            is_external_placeholder=dataset.is_external_placeholder,
            external_ref=_dataset_external_ref_to_gql(dataset.external_ref),
            last_modified_at=dataset.last_modified_at,
            last_modified_by=dataset.last_modified_by,
            created_at=dataset.created_at,
            created_by=dataset.created_by,
        )
        obj._model = dataset
        obj._forecast_from = (dataset.spec or {}).get('forecast_from')
        return obj

    @classmethod
    def from_binding(
        cls,
        binding: DatasetBindingDef,
        dataset_models_by_uuid: Mapping[UUID, DatasetModel] | None = None,
    ) -> DatasetType | None:
        """Construct from a binding, using a bulk-loaded model map when available."""
        from nodes.graphql.types.graph import _dataset_external_ref_to_gql

        if binding.dataset_uuid is None:
            return None
        if dataset_models_by_uuid is None:
            model = (
                DatasetModel.objects
                .filter(uuid=binding.dataset_uuid)
                .select_related('schema', 'created_by', 'last_modified_by')
                .first()
            )
        else:
            model = dataset_models_by_uuid.get(binding.dataset_uuid)
        if model is not None:
            obj = cls.from_model(model)
            if binding.forecast_from is not None:
                obj._forecast_from = binding.forecast_from
            return obj
        # Fallback: construct without model (dimensions/data will be empty)
        return cls(
            id=sb.ID(str(binding.dataset_uuid)),
            identifier=binding.external_dataset_id,
            is_external_placeholder=binding.dataset_is_external_placeholder,
            external_ref=_dataset_external_ref_to_gql(binding.dataset_external_ref),
            created_at=None,
            created_by=None,
            last_modified_at=None,
            last_modified_by=None,
        )


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
