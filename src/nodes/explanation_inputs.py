"""
Typed input of the node explanation system.

The explanation rules describe a node's inputs -- edges, dataset bindings and
parameters -- and validate the formula they imply. They read this model, never
the runtime ``Node`` objects (a node that failed to construct still gets its
static explanation) and never YAML-shaped dicts. ``explained_nodes_from_graph``
builds it from ``InstanceGraph``, the structural aggregate that already
resolves every binding to UUIDs, plus the typed dataset definitions the loader
grouped for runtime construction, so both consumers describe the same groups.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nodes.constants import VALUE_COLUMN
from nodes.defs.binding_def import EdgeBindingDef

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from nodes.defs.node_defs import InputDatasetDef
    from nodes.defs.transform_def import PortTransformOp
    from nodes.instance_graph import InstanceGraph, NodeMeta
    from nodes.units import Unit
    from params.base import Parameter


@dataclass(frozen=True)
class ExplainedParam:
    id: str
    value: Any = None
    unit: str = ''
    ref: str | None = None
    """Global parameter id when this is a reference parameter; ``value`` is then ``None``."""


@dataclass(frozen=True)
class ExplainedEdge:
    """All bindings from one source node to the explained node, merged the way the runtime merges them into one edge."""

    source_id: str
    tags: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    """Source output columns delivered, named only when the source has several outputs."""
    transformations: tuple[PortTransformOp, ...] = ()
    """The binding pipeline in the current vocabulary."""
    declared_dimensions: tuple[str, ...] = ()
    """Dimensions the target port declares the delivered value to carry."""


@dataclass(frozen=True)
class ExplainedNode:
    id: str
    node_class: str = ''
    """Dotted class path; the rules only ever look at the class name."""
    unit: str | None = None
    """Unit of the node's single output; ``None`` for multi-output nodes."""
    quantity: str | None = None
    tags: tuple[str, ...] = ()
    params: tuple[ExplainedParam, ...] = ()
    inputs: tuple[ExplainedEdge, ...] = ()
    """Incoming edges in input-port declaration order, then position."""
    datasets: tuple[InputDatasetDef, ...] = ()
    output_dimensions: tuple[str, ...] | None = None

    @property
    def class_name(self) -> str:
        return self.node_class.rsplit('.', maxsplit=1)[-1]

    def param(self, param_id: str) -> ExplainedParam | None:
        return next((param for param in self.params if param.id == param_id), None)


def _unit_str(unit: Unit | None) -> str | None:
    """Serialize the unit the way the spec does, so explanations read the same from every config source."""
    if unit is None:
        return None
    return str(unit) or 'dimensionless'


def _explained_param(param: Parameter) -> ExplainedParam:
    from params.param import ReferenceParameter

    if isinstance(param, ReferenceParameter):
        return ExplainedParam(id=param.local_id, ref=param.target_id)
    # The parameter's own serialization: nested values (a shift plan, say) read
    # as plain data, and the unit as the parameter reports it.
    data = param.model_dump(exclude_none=True)
    unit = data.get('unit')
    return ExplainedParam(id=param.local_id, value=data.get('value'), unit='' if unit is None else str(unit))


def _explained_edges(meta: NodeMeta) -> tuple[ExplainedEdge, ...]:
    """Merge the node's edge bindings per source node, in target-port order."""
    port_order = {port.id: index for index, port in enumerate(meta.spec.input_ports)}
    edges = [binding for binding in meta.input_bindings if isinstance(binding, EdgeBindingDef)]
    edges.sort(key=lambda b: (port_order.get(b.port_ref.port_id, len(port_order)), b.position, str(b.id)))
    by_source: OrderedDict[UUID, list[EdgeBindingDef]] = OrderedDict()
    for binding in edges:
        source_uuid = binding.from_ref.node_uuid
        assert source_uuid is not None
        by_source.setdefault(source_uuid, []).append(binding)

    result: list[ExplainedEdge] = []
    for bindings in by_source.values():
        first = bindings[0]
        source = first.source_node
        metrics: tuple[str, ...] = ()
        if len(source.spec.output_ports) > 1:
            metrics = tuple(b.source_port.column_id or VALUE_COLUMN for b in bindings)
        declared = dict.fromkeys([*first.target_port.required_dimensions, *first.declared_dimensions])
        result.append(
            ExplainedEdge(
                source_id=source.identifier or str(source.id),
                tags=tuple(first.tags),
                metrics=metrics,
                transformations=tuple(first.transformations),
                declared_dimensions=tuple(declared),
            )
        )
    return tuple(result)


def explained_node_from_meta(meta: NodeMeta, datasets: Sequence[InputDatasetDef] = ()) -> ExplainedNode:
    if meta.identifier is None:
        raise ValueError(f'Node {meta.id} has no identifier; explanations are keyed by identifier')
    spec = meta.spec
    unit = quantity = None
    if len(spec.output_ports) == 1:
        port = spec.output_ports[0]
        unit = _unit_str(port.unit)
        quantity = port.quantity
    return ExplainedNode(
        id=meta.identifier,
        node_class=meta.node_class_path,
        unit=unit,
        quantity=quantity,
        tags=tuple(spec.extra.tags),
        params=tuple(_explained_param(param) for param in spec.params),
        inputs=_explained_edges(meta),
        datasets=tuple(datasets),
        output_dimensions=tuple(spec.output_dimensions) if spec.output_dimensions else None,
    )


def explained_nodes_from_graph(
    graph: InstanceGraph,
    datasets_by_node: Mapping[UUID, Sequence[InputDatasetDef]],
) -> list[ExplainedNode]:
    return [explained_node_from_meta(meta, datasets_by_node.get(meta.id, ())) for meta in graph.nodes]
