"""Strawberry GraphQL types for DB-backed datasets."""

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import strawberry as sb

from kausal_common.strawberry.registry import register_strawberry_type

from nodes.defs.binding_def import DatasetPortBindingDef  # noqa: TC001  # used in runtime annotation
from nodes.graphql.types.graph import DatasetExternalRefType, DatasetPortType, _dataset_external_ref_to_gql
from nodes.graphql.types.metric import DimensionalMetricType
from nodes.metric import DimensionalMetric

if TYPE_CHECKING:
    from kausal_common.datasets.models import (
        Dataset as DatasetModel,
        DatasetMetric as DatasetMetricModel,
        DatasetSchemaDimension,
        Dimension as DimensionModel,
        DimensionCategory as DimensionCategoryModel,
    )


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

    @classmethod
    def from_model(cls, metric: DatasetMetricModel) -> DatasetMetricType:
        return cls(
            id=sb.ID(str(metric.uuid)),
            name=metric.name,
            label=metric.label_i18n or metric.name or '',
            unit=metric.unit or '',
        )


@register_strawberry_type
@sb.type(name='Dataset')
class DatasetType:
    """A DB-backed dataset with schema, dimensions, metrics and data."""

    id: sb.ID
    identifier: str | None
    is_external: bool = sb.field(description='Whether the dataset is backed by an external source.')
    external_ref: DatasetExternalRefType | None = sb.field(
        description='External source reference for externally backed datasets.'
    )

    _model: sb.Private['DatasetModel | None'] = None

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
        return [DatasetMetricType.from_model(m) for m in root._model.schema.metrics.all()]

    @sb.field(graphql_type=list[DimensionalMetricType])
    @staticmethod
    def data(root: 'DatasetType') -> list[DimensionalMetric]:
        """Load the full dataset as DimensionalMetric objects (one per metric column)."""
        if root._model is None:
            return []
        from nodes.datasets import DBDataset

        try:
            df = DBDataset.deserialize_df(root._model)
        except Exception:
            return []
        meta = df.get_meta()
        results: list[DimensionalMetric] = []
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
    def port_bindings(root: 'DatasetType') -> list[DatasetPortType]:
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
                node_ref=NodePortRef(node_id=sb.ID(str(dp.node.identifier)), port_id=dp.port_id),
                metric=None,
                external_dataset_id=None,
                external_metric_id=dp.metric.name if dp.metric else None,
            )
            result.append(port)
        return result

    @classmethod
    def from_model(cls, dataset: DatasetModel) -> DatasetType:
        obj = cls(
            id=sb.ID(str(dataset.uuid)),
            identifier=dataset.identifier,
            is_external=dataset.is_external_placeholder,
            external_ref=_dataset_external_ref_to_gql(dataset.external_ref),
        )
        obj._model = dataset
        return obj

    @classmethod
    def from_binding(cls, binding: DatasetPortBindingDef) -> DatasetType | None:
        """
        Construct from a DatasetPortBindingDef without loading the model.

        The lazy-resolved fields (name, dimensions, metrics, data) will
        return empty values unless the model is later attached.
        """
        if binding.dataset_uuid is None:
            return None
        return cls(
            id=sb.ID(str(binding.dataset_uuid)),
            identifier=binding.external_dataset_id,
            is_external=binding.dataset_is_external_placeholder,
            external_ref=_dataset_external_ref_to_gql(binding.dataset_external_ref),
        )
