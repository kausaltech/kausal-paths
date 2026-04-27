from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

import strawberry as sb

from kausal_common.strawberry.pydantic import pydantic_type

from paths.graphql_helpers import pass_context

from nodes.actions.action import ActionNode
from nodes.context import Context
from nodes.defs.binding_def import DatasetPortBindingDef
from nodes.defs.edge_def import (
    AssignCategoryTransformation,
    EdgeTransformation,
    FlattenTransformation,
    SelectCategoriesTransformation,
)
from nodes.defs.instance_defs import ActionGroup
from nodes.graphql.types.change_history import EditableEntity
from nodes.graphql.types.metric import DimensionalMetricType

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset as DatasetModel, DatasetMetric

    from nodes.defs.binding_def import EdgeBindingDef
    from nodes.graphql.types.change_history import InstanceModelLogEntryType
    from nodes.graphql.types.node import ActionNodeType
    from nodes.models import NodeEdge
    from nodes.node import Node

from nodes.metric import DimensionalMetric


@sb.type
class InstanceHostname:
    hostname: str
    base_path: str


@sb.type
class NodePortRef:
    node_id: sb.ID
    port_id: UUID


@pydantic_type(SelectCategoriesTransformation)
class SelectCategoriesTransformationType:
    """Filter or select categories within a dimension, optionally flattening."""

    dimension: sb.auto
    categories: sb.auto
    flatten: sb.auto
    exclude: sb.auto


@pydantic_type(AssignCategoryTransformation)
class AssignCategoryTransformationType:
    """Assign a fixed category to a (possibly new) dimension."""

    dimension: sb.auto
    category: sb.auto


@pydantic_type(FlattenTransformation)
class FlattenTransformationType:
    """Flatten (sum over) a dimension."""

    dimension: sb.auto


EdgeTransformationType = Annotated[
    SelectCategoriesTransformationType | AssignCategoryTransformationType | FlattenTransformationType,
    sb.union('EdgeTransformationUnion'),
]


@sb.type
class NodeEdgeType(EditableEntity):
    id: sb.ID
    uuid: UUID
    from_ref: NodePortRef
    to_ref: NodePortRef
    tags: list[str]

    _node: sb.Private['Node | None'] = None
    """The target node that this edge feeds into (set when resolving input port bindings)."""

    _transformations: sb.Private[list[EdgeTransformation] | None] = None

    @sb.field(graphql_type=list[EdgeTransformationType])
    @staticmethod
    def transformations(root: 'NodeEdgeType') -> list[EdgeTransformation]:
        return root._transformations or []

    @sb.field(graphql_type=list[DimensionalMetricType])
    @staticmethod
    def data(root: 'NodeEdgeType') -> list[DimensionalMetric]:
        """Compute the upstream node's output as seen through this edge."""
        if root._node is None:
            return []
        ctx = root._node.context
        source_node = ctx.nodes.get(str(root.from_ref.node_id))
        if source_node is None:
            return []
        for edge in source_node.edges:
            if edge.output_node == root._node and edge.input_node == source_node:
                break
        else:
            return []
        result = DimensionalMetric.from_edge_input(source_node, edge)
        return [result] if result is not None else []

    @classmethod
    def from_binding(cls, binding: EdgeBindingDef, node: Node | None = None) -> NodeEdgeType:
        edge = NodeEdgeType(
            id=sb.ID(str(binding.id)),
            uuid=binding.id if isinstance(binding.id, UUID) else UUID(str(binding.id)),
            from_ref=NodePortRef(node_id=sb.ID(str(binding.from_ref.node_id)), port_id=binding.from_ref.port_id),
            to_ref=NodePortRef(node_id=sb.ID(str(binding.to_ref.node_id)), port_id=binding.to_ref.port_id),
            tags=binding.tags,
        )
        edge._node = node
        return edge

    @classmethod
    def from_node_edge(cls, edge: NodeEdge) -> NodeEdgeType:
        obj = NodeEdgeType(
            id=sb.ID(str(edge.uuid)),
            uuid=edge.uuid,
            from_ref=NodePortRef(node_id=sb.ID(str(edge.from_node.identifier)), port_id=edge.from_port),
            to_ref=NodePortRef(node_id=sb.ID(str(edge.to_node.identifier)), port_id=edge.to_port),
            tags=edge.tags or [],
        )
        obj._transformations = list(edge.transformations)
        return obj

    @sb.field(
        graphql_type=list[Annotated['InstanceModelLogEntryType', sb.lazy('nodes.graphql.types.change_history')]],
        description='Row-level change history for this edge, newest first.',
    )
    @staticmethod
    def change_history(
        root: 'NodeEdgeType',
        limit: int = 50,
        before: 'datetime | None' = None,
    ) -> 'list[InstanceModelLogEntryType]':
        from nodes.graphql.types.change_history import fetch_entity_history
        from nodes.models import NodeEdge

        edge = NodeEdge.objects.filter(uuid=root.uuid).first()
        if edge is None:
            return []
        return fetch_entity_history(NodeEdge, edge.pk, limit=limit, before=before)


@sb.type
class DatasetExternalRefType:
    """Stable source reference for an externally backed dataset."""

    repo_url: str = sb.field(description='URL of the external dataset repository.')
    commit: str | None = sb.field(description='Repository commit used for this dataset snapshot.')
    dataset_id: str = sb.field(description='Path-like identifier of the dataset inside the external repository.')


@sb.type
class DatasetMetricRefType:
    """Lightweight reference to a dataset metric object bound in the model."""

    id: sb.ID = sb.field(description='Globally unique identifier of the dataset metric object.')
    name: str | None = sb.field(description='Stable identifier of the metric within its dataset schema.')
    label: str = sb.field(description='Human-readable label of the metric.')

    @classmethod
    def from_binding(cls, binding: DatasetPortBindingDef) -> DatasetMetricRefType | None:
        if binding.metric_uuid is None:
            return None
        return DatasetMetricRefType(
            id=sb.ID(str(binding.metric_uuid)),
            name=binding.external_metric_id,
            label=binding.external_metric_id or '',
        )

    @classmethod
    def from_model(cls, metric: DatasetMetric) -> DatasetMetricRefType:
        return DatasetMetricRefType(
            id=sb.ID(str(metric.uuid)),
            name=metric.name,
            label=metric.label_i18n,
        )


@pydantic_type(DatasetPortBindingDef)
class DatasetPortType(EditableEntity):
    """Binding of an external dataset metric to one node input port."""

    id: sb.ID = sb.field(description='Globally unique identifier of this dataset-port binding.')
    uuid: UUID
    node_ref: NodePortRef = sb.field(description='Reference to the node that owns the bound input port.')
    metric: DatasetMetricRefType | None = sb.field(description='Dataset metric object bound to this port.')
    external_dataset_id: str | None = sb.field(
        description='Stable identifier of the external dataset, usually the dataset repo path without extension.'
    )
    external_metric_id: str | None = sb.field(description='Stable identifier of the metric within the external dataset.')

    _node: sb.Private['Node | None'] = None
    """The node that owns the bound input port (set when resolving input port bindings)."""

    _dataset: sb.Private[Any] = None

    @sb.field(graphql_type=Annotated['DatasetType', sb.lazy('datasets.graphql.types')] | None)  # type: ignore[name-defined]  # noqa: F821
    @staticmethod
    def dataset(root: 'DatasetPortType') -> Any:
        """Dataset object bound to this port."""
        return root._dataset

    @sb.field(graphql_type=list[DimensionalMetricType])
    @staticmethod
    def data(root: 'DatasetPortType') -> list[DimensionalMetric]:
        """Return the dataset's data as DimensionalMetric objects (one per metric column), filtered as the node sees it."""
        if root._node is None:
            return []
        # Match by external_dataset_id (string identifier) or by dataset UUID
        ds_id = root.external_dataset_id
        ds_uuid = str(root._dataset.id) if root._dataset else None
        matched_ds = None
        for ds in root._node.input_dataset_instances:
            if ds_id is not None and ds.id == ds_id:
                matched_ds = ds
                break
            if ds_uuid is not None:
                from nodes.datasets import DBDataset

                if isinstance(ds, DBDataset) and ds.db_dataset_obj is not None and str(ds.db_dataset_obj.uuid) == ds_uuid:
                    matched_ds = ds
                    break
        if matched_ds is None:
            return []
        try:
            return DimensionalMetric.from_input_dataset(matched_ds, root._node.context)
        except Exception:
            return []

    @classmethod
    def from_binding(cls, binding: DatasetPortBindingDef, node: Node | None = None) -> DatasetPortType:
        from datasets.graphql.types import DatasetType

        port = DatasetPortType(
            id=sb.ID(str(binding.id)),
            uuid=binding.id if isinstance(binding.id, UUID) else UUID(str(binding.id)),
            node_ref=NodePortRef(node_id=sb.ID(str(binding.node_ref.node_id)), port_id=binding.node_ref.port_id),
            metric=DatasetMetricRefType.from_binding(binding),
            external_dataset_id=binding.external_dataset_id,
            external_metric_id=binding.external_metric_id,
        )
        dataset_type = DatasetType.from_binding(binding)
        if dataset_type is not None and binding.forecast_from is not None:
            dataset_type._forecast_from = binding.forecast_from
        port._dataset = dataset_type
        port._node = node
        return port

    @sb.field(
        graphql_type=list[Annotated['InstanceModelLogEntryType', sb.lazy('nodes.graphql.types.change_history')]],
        description='Row-level change history for this dataset-port binding, newest first.',
    )
    @staticmethod
    def change_history(
        root: 'DatasetPortType',
        limit: int = 50,
        before: datetime | None = None,
    ) -> 'list[InstanceModelLogEntryType]':
        from nodes.graphql.types.change_history import fetch_entity_history
        from nodes.models import DatasetPort

        dp = DatasetPort.objects.filter(uuid=root.uuid).first()
        if dp is None:
            return []
        return fetch_entity_history(DatasetPort, dp.pk, limit=limit, before=before)


InputPortBinding = Annotated[NodeEdgeType | DatasetPortType, sb.union('InputPortBindingUnion')]


@sb.type
class ActionGroupType:
    id: sb.ID
    name: str
    color: str | None

    @sb.field(graphql_type=list[Annotated['ActionNodeType', sb.lazy('nodes.schema')]])
    @pass_context
    @staticmethod
    def actions(root: ActionGroup, context: Context) -> list[ActionNode]:
        return [act for act in context.get_actions() if act.group == root]


def _dataset_external_ref_to_gql(external_ref: object) -> DatasetExternalRefType | None:
    if not isinstance(external_ref, dict):
        return None
    repo_url = external_ref.get('repo_url')
    dataset_id = external_ref.get('dataset_id')
    if not isinstance(repo_url, str) or not isinstance(dataset_id, str):
        return None
    commit = external_ref.get('commit')
    return DatasetExternalRefType(
        repo_url=repo_url,
        commit=commit if isinstance(commit, str) else None,
        dataset_id=dataset_id,
    )


def _external_dataset_id_from_dataset(dataset: DatasetModel | DatasetPortBindingDef) -> str | None:
    if isinstance(dataset, DatasetPortBindingDef):
        external_ref = dataset.dataset_external_ref
        if isinstance(external_ref, dict):
            dataset_id = external_ref.get('dataset_id')
            if isinstance(dataset_id, str):
                return dataset_id
        return dataset.external_dataset_id

    external_ref = dataset.external_ref
    if isinstance(external_ref, dict):
        dataset_id = external_ref.get('dataset_id')
        if isinstance(dataset_id, str):
            return dataset_id
    return dataset.identifier
