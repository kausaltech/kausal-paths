"""Strawberry GraphQL types for DB-backed datasets."""

from datetime import date
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

import strawberry as sb

from kausal_common.datasets.models import Dataset as DatasetModel
from kausal_common.strawberry.ordering import with_sibling_ids
from kausal_common.strawberry.registry import register_strawberry_type

from nodes.defs.binding_def import DatasetPortBindingDef  # noqa: TC001  # used in runtime annotation

if TYPE_CHECKING:
    from kausal_common.datasets.models import (
        DataPoint as DataPointModel,
        DatasetMetric as DatasetMetricModel,
        DatasetSchemaDimension,
        Dimension as DimensionModel,
        DimensionCategory as DimensionCategoryModel,
    )

    from nodes.graphql.types.graph import DatasetExternalRefType, DatasetPortType
    from nodes.graphql.types.metric import DimensionalMetricType


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


@register_strawberry_type
@sb.type(name='DatasetMetric')
class DatasetMetricType:
    """A metric (value column) defined in a dataset schema."""

    id: sb.ID
    name: str | None = sb.field(description='Column name used in DataFrames.')
    label: str = sb.field(description='Human-readable label.')
    unit: str
    previous_sibling: sb.ID | None
    next_sibling: sb.ID | None

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

    @classmethod
    def from_model(cls, data_point: DataPointModel) -> DataPointType:
        return cls(
            id=sb.ID(str(data_point.uuid)),
            date=data_point.date,
            value=float(data_point.value) if data_point.value is not None else None,
            metric=DatasetMetricType.from_model(data_point.metric),
            dimension_categories=[
                DatasetDimensionCategoryType.from_model(category) for category in data_point.dimension_categories.all()
            ],
        )


@register_strawberry_type
@sb.type(name='Dataset')
class DatasetType:
    """A DB-backed dataset with schema, dimensions, metrics and data."""

    id: sb.ID
    identifier: str | None
    is_external_placeholder: bool = sb.field(
        description='Whether the dataset object is only a placeholder without imported datapoints.'
    )
    external_ref: Annotated['DatasetExternalRefType', sb.lazy('nodes.graphql.types.graph')] | None = sb.field(
        description='External source reference for externally backed datasets.'
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
    def dimensions(root: 'DatasetType') -> list[DatasetDimensionType]:
        if root._model is None or root._model.schema is None:
            return []
        return [
            DatasetDimensionType.from_schema_dimension(sd)
            for sd in root._model.schema.dimensions.select_related('dimension').prefetch_related('dimension__categories')
        ]

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

    @sb.field(graphql_type=list[Annotated['DimensionalMetricType', sb.lazy('nodes.graphql.types.metric')]])
    @staticmethod
    def data(root: 'DatasetType') -> list[Any]:
        """Load the full dataset as DimensionalMetric objects (one per metric column)."""
        if root._model is None:
            return []
        from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN
        from nodes.datasets import DBDataset

        df = DBDataset.deserialize_df(root._model)

        if FORECAST_COLUMN not in df.columns and root._forecast_from is not None:
            import polars as pl

            from common import polars as ppl

            df = df.with_columns(
                pl
                .when(pl.col(YEAR_COLUMN) >= root._forecast_from)
                .then(pl.lit(value=True))
                .otherwise(pl.lit(value=False))
                .alias(FORECAST_COLUMN),
            )
            df = ppl.to_ppdf(df, meta=df.get_meta())

        meta = df.get_meta()
        results: list[Any] = []
        from nodes.metric_gen import metric_from_dataframe_standalone

        for col in meta.metric_cols:
            ds_id = root.identifier or str(root.id)
            results.append(
                metric_from_dataframe_standalone(
                    df,
                    metric_col=col,
                    metric_id=f'{ds_id}:{col}',
                    metric_name=col,
                )
            )
        return results

    @sb.field(graphql_type=list[Annotated['DatasetPortType', sb.lazy('nodes.graphql.types.graph')]])
    @staticmethod
    def port_bindings(root: 'DatasetType') -> list[Any]:
        """Discover which node ports use this dataset."""
        from nodes.graphql.types.graph import DatasetPortType, NodePortRef
        from nodes.models import DatasetPort

        if root._model is None:
            return []
        ports = DatasetPort.objects.filter(dataset=root._model).select_related('node', 'metric')
        result = []
        for dp in ports:
            port = DatasetPortType(
                id=sb.ID(str(dp.uuid)),
                uuid=dp.uuid,
                node_ref=NodePortRef(node_id=sb.ID(str(dp.node.identifier)), port_id=dp.port_id),
                metric=None,
                external_dataset_id=None,
                external_metric_id=dp.metric.name if dp.metric else None,
            )
            result.append(port)
        return result

    @classmethod
    def from_model(cls, dataset: DatasetModel) -> DatasetType:
        from nodes.graphql.types.graph import _dataset_external_ref_to_gql

        obj = cls(
            id=sb.ID(str(dataset.uuid)),
            identifier=dataset.identifier,
            is_external_placeholder=dataset.is_external_placeholder,
            external_ref=_dataset_external_ref_to_gql(dataset.external_ref),
        )
        obj._model = dataset
        return obj

    @classmethod
    def from_binding(cls, binding: DatasetPortBindingDef) -> DatasetType | None:
        """Construct from a DatasetPortBindingDef, loading the DB model by UUID."""
        from nodes.graphql.types.graph import _dataset_external_ref_to_gql

        if binding.dataset_uuid is None:
            return None
        model = DatasetModel.objects.filter(uuid=binding.dataset_uuid).select_related('schema').first()
        if model is not None:
            return cls.from_model(model)
        # Fallback: construct without model (dimensions/data will be empty)
        return cls(
            id=sb.ID(str(binding.dataset_uuid)),
            identifier=binding.external_dataset_id,
            is_external_placeholder=binding.dataset_is_external_placeholder,
            external_ref=_dataset_external_ref_to_gql(binding.dataset_external_ref),
        )
