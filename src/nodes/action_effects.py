"""
The effects of an action on other nodes: hooks, and what may feed them.

An action acts on a node through a hook (`nodes.hooks`): its output port is
added to one of the node's output ports. The hook's contribution must have the
target port's dimensions and a unit compatible with its unit. This module
holds the rules the editor applies before writing a hook, and the same rule
applied one step earlier, to the sources that may feed an action's effect: a
dataset metric or another node's output port.

See docs/architecture/action-hooks.md and docs/plans/action-from-output-port.md.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nodes.constraints.values import PortValue
from nodes.defs.node_defs import ActionConfig, ActionHookDef

if TYPE_CHECKING:
    from collections.abc import Iterable
    from uuid import UUID

    from nodes.constraints.solver import ConstraintSolveResult
    from nodes.instance_graph import InstanceGraph, NodeMeta
    from nodes.units import Unit


class EffectError(ValueError):
    """An action effect (a hook or its source) that cannot hold; the message is for the modeller."""


@dataclass(frozen=True)
class ValueShape:
    """
    The part of a value's shape that decides whether it can act on a port.

    `dimensions` is None when the shape is not known. `categories` lists, per
    dimension, the categories the value is known to be limited to; a dimension
    missing from it may have any category.
    """

    dimensions: frozenset[UUID] | None
    unit: Unit | None
    quantity: str | None = None
    categories: dict[UUID, frozenset[UUID]] = field(default_factory=dict)


def declared_dimensions(graph: InstanceGraph, node: NodeMeta, port_id: UUID) -> frozenset[UUID] | None:
    """Return the dimensions an output port declares (or else its node does), or None if one is not in the instance."""
    port = node.spec.output_port_by_id[port_id]
    identifiers = port.dimensions or node.spec.output_dimensions
    dimensions = [graph.dimension_by_identifier.get(identifier) for identifier in identifiers]
    if any(dimension is None for dimension in dimensions):
        return None
    return frozenset(dimension.id for dimension in dimensions if dimension is not None)


def output_port_shape(graph: InstanceGraph, result: ConstraintSolveResult, node: NodeMeta, port_id: UUID) -> ValueShape:
    """
    Return the shape of what an output port produces.

    The solver's answer where it has one, since a port's declared dimensions
    can be empty while its node's rules fix them. Where the solve leaves the
    dimensions, unit or quantity open, the port's declaration fills them in.
    """
    port = node.spec.output_port_by_id[port_id]
    solved = result.shapes.get(PortValue(node.id, port_id, 'output'))
    if solved is None:
        return ValueShape(dimensions=declared_dimensions(graph, node, port_id), unit=port.unit, quantity=port.quantity)
    return ValueShape(
        dimensions=solved.dimensions if solved.dimensions is not None else declared_dimensions(graph, node, port_id),
        unit=solved.unit or port.unit,
        quantity=solved.quantity or port.quantity,
        categories=dict(solved.categories),
    )


def incompatibility(source: ValueShape, target: ValueShape) -> str | None:
    """
    Say why `source` cannot be added to `target`, or return None if it can.

    The rule is the hook's (`hook_contribution`): the same dimensions, and a
    compatible unit. Categories may be a subset of the target's, since a
    category the source leaves out contributes nothing. A source whose
    dimensions or unit are unknown is refused, because nothing could be
    promised about it.
    """
    if target.dimensions is None:
        return 'the shape of the target is not known'
    if source.dimensions is None:
        return 'its dimensions are not known'
    if source.dimensions != target.dimensions:
        return 'its dimensions differ from the target'
    if target.unit is None or source.unit is None:
        return 'its unit is not known'
    if not source.unit.is_compatible_with(target.unit):
        return f'its unit {source.unit:~P} cannot be converted to {target.unit:~P}'
    for dimension_id, allowed in target.categories.items():
        categories = source.categories.get(dimension_id)
        if categories is not None and not categories <= allowed:
            return 'it has categories the target does not have'
    return None


def _sole_output_port(node: NodeMeta, role: str) -> UUID:
    ports = node.spec.output_ports
    if len(ports) != 1:
        name = node.identifier or str(node.id)
        raise EffectError(f'{name} has {len(ports)} output ports; name the {role} port')
    return ports[0].id


def creates_cycle(graph: InstanceGraph, edges: Iterable[tuple[UUID, UUID]]) -> bool:
    """Whether adding `edges` (pairs of node uuids, from upstream to downstream) to the graph would close a cycle."""
    import networkx as nx

    candidate = graph.nx_graph.copy()
    candidate.add_edges_from(edges)
    return not nx.is_directed_acyclic_graph(candidate)


def resolve_hook(
    graph: InstanceGraph,
    *,
    action: NodeMeta,
    target: NodeMeta,
    target_port: UUID | None,
    from_port: UUID | None,
) -> ActionHookDef:
    """
    Build the definition of a new hook of `action` on `target`, with both ports explicit.

    Refuses a hook that cannot hold structurally: from a node that is not an
    action, on the action itself, on a port that does not exist, the same hook
    twice, or one that closes a cycle. Whether the shapes fit is a separate
    check (`hook_shape_problem`), since it needs a constraint solve.
    """
    type_config = action.spec.type_config
    if not isinstance(type_config, ActionConfig):
        raise EffectError('Only actions can act on other nodes')
    if target.id == action.id:
        raise EffectError('An action cannot act on itself')
    if target.identifier is None:
        raise EffectError('The target node has no identifier')
    target_port = target_port or _sole_output_port(target, 'target')
    if target_port not in target.spec.output_port_by_id:
        raise EffectError(f'{target.identifier} has no output port {target_port}')
    from_port = from_port or _sole_output_port(action, "action's")
    if from_port not in action.spec.output_port_by_id:
        raise EffectError(f'The action has no output port {from_port}')
    for hook in type_config.hooks:
        if hook.node == target.identifier and hook.port in (None, target_port) and hook.from_port in (None, from_port):
            raise EffectError(f'The action already acts on this port of {target.identifier}')
    if creates_cycle(graph, [(action.id, target.id)]):
        raise EffectError(f'{target.identifier} is upstream of the action, so acting on it would form a loop')
    return ActionHookDef(node=target.identifier, port=target_port, from_port=from_port)


def hook_shape_problem(
    graph: InstanceGraph, result: ConstraintSolveResult, *, action: NodeMeta, target: NodeMeta, hook: ActionHookDef
) -> str | None:
    """Say why the action's port cannot act on the target's port, or return None if it can."""
    assert hook.port is not None
    assert hook.from_port is not None
    target_shape = output_port_shape(graph, result, target, hook.port)
    source_shape = output_port_shape(graph, result, action, hook.from_port)
    problem = incompatibility(source_shape, target_shape)
    if problem is None:
        return None
    return f'The action cannot act on {target.identifier}: {problem}'
