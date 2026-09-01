"""
Draft-graph constraint checks shared by the binding mutations.

Every mutation that creates or rewrites an input-port binding — edge or
dataset, create or update, with or without `replace` — builds a
:class:`~nodes.constraints.validation.BindingChange` and passes it through
:func:`check_binding_change` before writing. A rejected change returns the
introduced conflicts as a ``ConstraintViolations`` payload member and writes
nothing, so a failed replace always leaves the old binding intact.
"""

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from nodes.constraints.validation import BindingChange, validate_binding_change
from nodes.defs.binding_def import DatasetBindingDef, EdgeBindingDef, NodePortRef
from nodes.graphql.types.constraints import ConstraintViolationsType

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset as DatasetModel, DatasetMetric

    from paths import gql

    from nodes.defs.graph import DatasetMeta
    from nodes.defs.transform_def import PortTransformOp
    from nodes.instance_graph import InstanceGraph
    from nodes.models import InstanceConfig, NodeConfig


def require_draft_graph(info: gql.Info, ic: InstanceConfig) -> InstanceGraph:
    from nodes.models import PreferredInstanceSource

    return info.context.require_instance_graph(ic, source=PreferredInstanceSource.DRAFT)


def check_binding_change(info: gql.Info, ic: InstanceConfig, change: BindingChange) -> ConstraintViolationsType | None:
    """Validate a candidate binding change against the draft graph; ``None`` means acceptable."""
    from nodes.instance_graph_cache import resolve_instance_source
    from nodes.models import PreferredInstanceSource

    graph = require_draft_graph(info, ic)
    resolved_source = resolve_instance_source(ic, PreferredInstanceSource.DRAFT)
    validation = validate_binding_change(ic, graph, resolved_source, change)
    if validation.ok:
        return None
    return ConstraintViolationsType.from_conflicts(validation.new_conflicts)


def _next_position(graph: InstanceGraph, node_uuid: UUID, port_id: UUID) -> int:
    return len(graph.bindings_by_input.get((node_uuid, port_id), ()))


def edge_candidate(
    graph: InstanceGraph,
    *,
    from_node: NodeConfig,
    from_port: UUID,
    to_node: NodeConfig,
    to_port: UUID,
    transformations: list[PortTransformOp],
    tags: list[str] | None = None,
    binding_id: UUID | None = None,
    position: int | None = None,
) -> EdgeBindingDef:
    """Build the unbound definition for a hypothetical edge, as the graph builder would."""
    return EdgeBindingDef(
        id=binding_id or uuid4(),
        port_ref=NodePortRef(node_uuid=to_node.uuid, node_id=to_node.identifier, port_id=to_port),
        from_ref=NodePortRef(node_uuid=from_node.uuid, node_id=from_node.identifier, port_id=from_port),
        position=position if position is not None else _next_position(graph, to_node.uuid, to_port),
        tags=list(tags or []),
        transformations=list(transformations),
    )


def dataset_candidate(
    graph: InstanceGraph,
    *,
    nc: NodeConfig,
    port_id: UUID,
    dataset: DatasetModel,
    metric: DatasetMetric,
    transformations: list[PortTransformOp],
    tags: list[str] | None = None,
    binding_id: UUID | None = None,
    position: int | None = None,
    primary_language: str,
) -> tuple[DatasetBindingDef, tuple[DatasetMeta, ...]]:
    """
    Build the unbound definition for a hypothetical dataset binding.

    Also returns the catalog addition when the dataset is not yet bound
    anywhere in the instance: the graph's dataset catalog covers only bound
    datasets, so a first-time bind must inject the metadata it validates
    against.
    """
    from nodes.instance_serialization import dataset_meta_from_model

    binding = DatasetBindingDef(
        id=binding_id or uuid4(),
        port_ref=NodePortRef(node_uuid=nc.uuid, node_id=nc.identifier, port_id=port_id),
        position=position if position is not None else _next_position(graph, nc.uuid, port_id),
        tags=list(tags or []),
        transformations=list(transformations),
        dataset_uuid=dataset.uuid,
        metric_uuid=metric.uuid,
        dataset_is_external_placeholder=dataset.is_external_placeholder,
        dataset_external_ref=dataset.external_ref,
        external_dataset_id=dataset.identifier,
        external_metric_id=metric.name,
    )
    additions: tuple[DatasetMeta, ...] = ()
    if dataset.uuid not in graph.dataset_by_id:
        additions = (dataset_meta_from_model(dataset, primary_language=primary_language),)
    return binding, additions
