"""
The rules for an action's effects on other nodes.

An effect reaches its target in one of two ways. Through a *hook* the action's
output port is added to one of the target's output ports, in the years after
the last historical year (`nodes.hooks`); the target needs no input for it, so
this is how an instance acts on a node it inherits from a framework. Through an
ordinary *edge* the action feeds the target's additive input, in every year.

The action wizard builds an `AdditiveAction` whose effects land on existing
nodes. Each effect is an ordinary edge from one of the action's output ports
into the target's *additive* input: the multi `additive` port of an additive
or multiplicative node, or a new input port of a formula node, whose formula
adds every input it does not name. Either way the action's value is added to
what the node computes from its other inputs, as the last step, so wiring an
action in does not change the node's own arithmetic.

This module holds the rules the wizard applies before it writes anything:
which nodes an action can act on, and which sources may feed an effect. See
docs/plans/action-from-output-port.md.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from nodes.constraints.values import PortValue
from nodes.defs.node_defs import ActionConfig, ActionHookDef, ActionHookMode, FormulaConfig
from nodes.defs.port_def import InputPortDef
from nodes.formula import FormulaNode
from nodes.instance_graph import node_class_for_spec

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nodes.constraints.solver import ConstraintSolveResult
    from nodes.instance_graph import InstanceGraph, NodeMeta
    from nodes.units import Unit


ADDITIVE_ROLE = 'additive'


class EffectError(ValueError):
    """An action effect that cannot hold; the message is for the modeller."""


@dataclass(frozen=True)
class ValueShape:
    """
    The part of a value's shape that decides whether it can be added to a node.

    `dimensions` is None when the shape is not known. `categories` lists, per
    dimension, the categories the value is known to be limited to; a dimension
    missing from it may have any category.
    """

    dimensions: frozenset[UUID] | None
    unit: Unit | None
    quantity: str | None = None
    categories: dict[UUID, frozenset[UUID]] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionTarget:
    """
    Where an action's effect on a node lands.

    `input_port` is the target's existing additive port, or None when the
    wizard has to add one; `new_port_role` then says which kind (`additive`,
    or None for a plain port on a formula node).
    """

    node: NodeMeta
    output_port: UUID
    """The target's single output port, whose shape the effect copies."""
    input_port: UUID | None
    new_port_role: str | None = None
    new_port_identifier: str | None = None

    def plan_new_port(self, shape: ValueShape, label: str | None) -> InputPortDef | None:
        """Return the input port to add to the target, or None if the effect uses an existing one."""
        if self.input_port is not None:
            return None
        if self.new_port_role is not None:
            return InputPortDef(id=uuid4(), identifier=self.new_port_identifier, role=self.new_port_role, multi=True)
        # A formula node's port: the formula does not name the action, so its value is added.
        return InputPortDef(id=uuid4(), label=label, quantity=shape.quantity, unit=shape.unit, multi=False)


def action_target(node: NodeMeta, *, is_local: bool) -> ActionTarget:
    """
    Return where an action's effect on `node` lands, or raise `EffectError` saying why it cannot.

    `is_local` says whether the instance owns the node. A node inherited from a
    framework cannot take a new input, so an action cannot be wired into it.
    """
    name = node.identifier or str(node.id)
    if isinstance(node.spec.type_config, ActionConfig):
        raise EffectError(f'{name} is an action')
    if len(node.spec.output_ports) != 1:
        raise EffectError(f'{name} has {len(node.spec.output_ports)} output ports; an action can act only on a node with one')
    if not is_local:
        raise EffectError(f'{name} belongs to the framework, so it cannot take new inputs')
    output_port = node.spec.output_ports[0].id
    node_class = node_class_for_spec(node.spec)

    declaration = next(
        (d for d in node_class.input_port_declarations if str(d.role) == ADDITIVE_ROLE and d.multi),
        None,
    )
    if declaration is not None:
        existing = next((port.id for port in node.spec.input_ports if port.role == ADDITIVE_ROLE), None)
        identifiers = {str(port.identifier) for port in node.spec.input_ports if port.identifier}
        identifier = str(declaration.instance_identifier)
        index = 2
        while identifier in identifiers:
            identifier, index = f'{declaration.instance_identifier}{index}', index + 1
        return ActionTarget(
            node=node,
            output_port=output_port,
            input_port=existing,
            new_port_role=ADDITIVE_ROLE,
            new_port_identifier=identifier,
        )
    if isinstance(node.spec.type_config, FormulaConfig) and issubclass(node_class, FormulaNode):
        return ActionTarget(node=node, output_port=output_port, input_port=None)
    raise EffectError(f'{name} has no additive input, so an action cannot be added to it')


def declared_dimensions(graph: InstanceGraph, node: NodeMeta, port_id: UUID) -> frozenset[UUID] | None:
    """Return the dimensions an output port declares, or None if one is not in the instance."""
    port = node.spec.output_port_by_id[port_id]
    dimensions = [graph.dimension_by_identifier.get(identifier) for identifier in port.dimensions or ()]
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

    The same dimensions and a compatible unit. Categories may be a subset of
    the target's, since a category the source leaves out adds nothing. A
    source whose dimensions or unit are unknown is refused, because nothing
    could be promised about it.
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


def creates_cycle(graph: InstanceGraph, edges: Iterable[tuple[UUID, UUID]]) -> bool:
    """Whether adding `edges` (pairs of node uuids, from upstream to downstream) to the graph would close a cycle."""
    import networkx as nx

    candidate = graph.nx_graph.copy()
    candidate.add_edges_from(edges)
    return not nx.is_directed_acyclic_graph(candidate)


# --- Hooks ------------------------------------------------------------------


def hook_target_problem(node: NodeMeta, *, is_local: bool) -> str | None:
    """
    Say why an action cannot act on `node` through a hook, or return None if it can.

    `is_local` says whether the instance owns the node. A node it inherits from
    a framework takes hooks only where the framework says so (`NodeSpec.action_hooks`).
    """
    name = node.identifier or str(node.id)
    if isinstance(node.spec.type_config, ActionConfig):
        return f'{name} is an action'
    if not node.spec.output_ports:
        return f'{name} has no output port'
    if not is_local and not node.spec.action_hooks:
        return f'{name} belongs to the framework, which does not let actions act on it'
    return None


def _sole_output_port(node: NodeMeta, role: str) -> UUID:
    ports = node.spec.output_ports
    if len(ports) != 1:
        name = node.identifier or str(node.id)
        raise EffectError(f'{name} has {len(ports)} output ports; name the {role} port')
    return ports[0].id


def resolve_hook(
    graph: InstanceGraph,
    *,
    action: NodeMeta,
    target: NodeMeta,
    target_port: UUID | None,
    from_port: UUID | None,
    mode: ActionHookMode = ActionHookMode.ABSOLUTE,
    is_local_target: bool,
) -> ActionHookDef:
    """
    Build the definition of a new hook of `action` on `target`, with both ports explicit.

    Refuses a hook that cannot hold structurally: from a node that is not an
    action, on the action itself, on a node that takes no hooks, on a port that
    does not exist, the same hook twice, or one that closes a cycle. Whether the
    shapes fit is a separate check (`hook_shape_problem`), since it needs a
    constraint solve.
    """
    type_config = action.spec.type_config
    if not isinstance(type_config, ActionConfig):
        raise EffectError('Only actions can act on other nodes')
    if target.id == action.id:
        raise EffectError('An action cannot act on itself')
    problem = hook_target_problem(target, is_local=is_local_target)
    if problem is not None:
        raise EffectError(problem)
    name = target.identifier or str(target.id)
    target_port = target_port or _sole_output_port(target, 'target')
    if target_port not in target.spec.output_port_by_id:
        raise EffectError(f'{name} has no output port {target_port}')
    from_port = from_port or _sole_output_port(action, "action's")
    if from_port not in action.spec.output_port_by_id:
        raise EffectError(f'The action has no output port {from_port}')
    for hook in type_config.hooks:
        if hook.node == target.id and hook.port in (None, target_port) and hook.from_port in (None, from_port):
            raise EffectError(f'The action already acts on this port of {name}')
    if creates_cycle(graph, [(action.id, target.id)]):
        raise EffectError(f'{name} is upstream of the action, so acting on it would form a loop')
    return ActionHookDef(node=target.id, port=target_port, from_port=from_port, mode=mode)


def hook_source_problem(source: ValueShape, target: ValueShape, mode: ActionHookMode) -> str | None:
    """
    Say why a value of shape `source` cannot act on `target` in the given mode, or return None if it can.

    An absolute effect is an amount the target's unit can take. A relative one is
    a share, so it has no unit; its dimensions and categories are held to the same rule.
    """
    if mode == ActionHookMode.ABSOLUTE:
        return incompatibility(source, target)
    if source.unit is None:
        return 'its unit is not known'
    if not source.unit.dimensionless:
        return f'a relative effect is a share without a unit, not {source.unit:~P}'
    return incompatibility(ValueShape(source.dimensions, target.unit, source.quantity, source.categories), target)


def hook_shape_problem(
    graph: InstanceGraph, result: ConstraintSolveResult, *, action: NodeMeta, target: NodeMeta, hook: ActionHookDef
) -> str | None:
    """Say why the action's port cannot act on the target's port, or return None if it can."""
    assert hook.port is not None
    assert hook.from_port is not None
    target_shape = output_port_shape(graph, result, target, hook.port)
    source_shape = output_port_shape(graph, result, action, hook.from_port)
    problem = hook_source_problem(source_shape, target_shape, hook.mode)
    if problem is None:
        return None
    return f'The action cannot act on {target.identifier}: {problem}'
