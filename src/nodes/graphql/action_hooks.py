"""
Actions acting on nodes' output ports (hooks) in the instance editor.

Reading: every hook of the graph, and the output ports an action may act on.
Writing: adding a hook to an action and removing one. The rules are in
`nodes.action_effects`; see docs/architecture/action-hooks.md.
"""

from typing import TYPE_CHECKING
from uuid import UUID

import strawberry as sb

from kausal_common.strawberry.errors import GraphQLValidationError, NotFoundError

from paths import gql
from paths.graphql_types import UnitType

from nodes.action_effects import (
    EffectError,
    ValueShape,
    hook_shape_problem,
    hook_target_problem,
    output_port_shape,
    resolve_hook,
)
from nodes.change_ops import gql_change_operation, record_change
from nodes.defs.node_defs import ActionConfig, ActionHookDef, ActionHookMode
from nodes.graphql.action_wizard import DimensionCategoriesType, output_categories
from nodes.graphql.constraint_checks import require_draft_graph
from nodes.models import InstanceConfig, NodeConfig, PreferredInstanceSource

if TYPE_CHECKING:
    from collections.abc import Callable

    from nodes.instance_graph import InstanceGraph, NodeMeta
    from nodes.units import Unit


ActionHookModeEnum = sb.enum(ActionHookMode, name='ActionHookMode')


@sb.type(description="An action acting on a node's output port: the action's port is added to it.")
class ActionHookEdgeType:
    action_id: UUID
    from_port_id: UUID | None = sb.field(description="The action's output port; null when it cannot be told which.")
    target_node_id: UUID
    target_port_id: UUID | None = sb.field(description="The target's output port; null when it cannot be told which.")
    mode: ActionHookModeEnum = sb.field(  # type: ignore[valid-type]
        description="Whether the action's output is an amount in the target's unit or a share of its un-hooked value."
    )


@sb.input(description="The action acts on another node's output port: its own output port is added to it.")
class ActionHookInput:
    target_node_id: UUID = sb.field(description='The node the action acts on.')
    target_port_id: UUID | None = sb.field(default=None, description='May be omitted when the node has a single output.')
    from_port_id: UUID | None = sb.field(
        default=None, description="The action's output port; may be omitted when the action has a single output."
    )
    mode: ActionHookModeEnum = sb.field(  # type: ignore[valid-type]
        default=ActionHookMode.ABSOLUTE,
        description=(
            "ABSOLUTE: the action gives an amount in the target's unit. RELATIVE: it gives a share without a unit, "
            "-0.3 for -30 %, of the target's value before any action. Ignored when removing a hook."
        ),
    )


@sb.type(description='An output port that an action of this instance may act on, and the shape an effect on it has.')
class ActionHookTargetType:
    node_id: UUID
    port_id: UUID
    quantity: str | None
    _config: sb.Private[InstanceConfig]
    _node: sb.Private['NodeMeta']
    _shape: sb.Private[ValueShape]

    @sb.field(graphql_type=UnitType | None, description='The unit of an absolute effect; a relative one has no unit.')
    @staticmethod
    def unit(root: 'ActionHookTargetType') -> 'Unit | None':
        return root._shape.unit

    @sb.field(
        graphql_type=list[DimensionCategoriesType],
        description=(
            'The dimensions of the port and, for each, the categories its output actually has. An effect names a '
            'subset of them. Computing this computes the node.'
        ),
    )
    @staticmethod
    def dimensions(root: 'ActionHookTargetType', info: gql.Info) -> list[DimensionCategoriesType]:
        graph = require_draft_graph(info, root._config)
        categories = output_categories(info, root._config, graph, root._node, root._shape)
        order = {category.id: index for dimension in graph.dimensions for index, category in enumerate(dimension.categories)}
        return [
            DimensionCategoriesType(dimension_id=dimension_id, category_ids=sorted(ids, key=order.__getitem__))
            for dimension_id, ids in categories.items()
        ]


def is_local_node(ic: InstanceConfig, node_id: UUID) -> bool:
    return ic.nodes.filter(uuid=node_id).exists()


def action_hooks(graph: InstanceGraph) -> list[ActionHookEdgeType]:
    """List every hook of the graph, with both ports resolved where the nodes have a single output."""
    edges: list[ActionHookEdgeType] = []
    for action in graph.nodes:
        type_config = action.spec.type_config
        if not isinstance(type_config, ActionConfig):
            continue
        for hook in type_config.hooks:
            target = graph.node_by_id.get(hook.node)
            if target is None:
                continue
            target_ports, action_ports = target.spec.output_ports, action.spec.output_ports
            edges.append(
                ActionHookEdgeType(
                    action_id=action.id,
                    from_port_id=hook.from_port or (action_ports[0].id if len(action_ports) == 1 else None),
                    target_node_id=target.id,
                    target_port_id=hook.port or (target_ports[0].id if len(target_ports) == 1 else None),
                    mode=hook.mode,
                )
            )
    return edges


def action_hook_targets(info: gql.Info, ic: InstanceConfig) -> list[ActionHookTargetType]:
    """
    List the output ports an action of this instance may act on.

    The instance's own nodes, and the nodes it inherits that the framework has opened to actions.
    A yaml-sourced instance has no draft to build an action in, so it has none.
    """
    if ic.config_source != 'database':
        return []
    graph = require_draft_graph(info, ic)
    solve = info.context.require_constraint_solve(ic, source=PreferredInstanceSource.DRAFT)
    local = set(ic.nodes.values_list('uuid', flat=True))
    targets: list[ActionHookTargetType] = []
    for node in graph.nodes:
        if hook_target_problem(node, is_local=node.id in local) is not None:
            continue
        for port in node.spec.output_ports:
            shape = output_port_shape(graph, solve, node, port.id)
            targets.append(
                ActionHookTargetType(
                    node_id=node.id, port_id=port.id, quantity=shape.quantity, _config=ic, _node=node, _shape=shape
                )
            )
    return targets


def write_hooks(nc: NodeConfig, change: Callable[[list[ActionHookDef]], list[ActionHookDef]]) -> None:
    """Replace the hooks of action `nc` with `change(hooks)`, inside an open change operation."""
    spec = nc.spec
    assert spec is not None
    assert isinstance(spec.type_config, ActionConfig)
    before = nc.serializable_data()
    spec.type_config.hooks = change(list(spec.type_config.hooks))
    # queryset.update(): NodeConfig.save() goes through ClusterableModel, which can revert i18n fields.
    NodeConfig.objects.filter(pk=nc.pk).update(spec=spec)
    nc.refresh_from_db()
    record_change(nc, action='node.update', before=before, after=nc.serializable_data())


def checked_hook(
    info: gql.Info, ic: InstanceConfig, graph: InstanceGraph, action: NodeMeta, input: ActionHookInput
) -> ActionHookDef:
    """Return the hook `input` asks for on `action`, or refuse it with the reason."""
    target = graph.node_by_id.get(input.target_node_id)
    if target is None:
        raise NotFoundError(info, 'Node not found')
    try:
        hook = resolve_hook(
            graph,
            action=action,
            target=target,
            target_port=input.target_port_id,
            from_port=input.from_port_id,
            mode=ActionHookMode(input.mode),
            is_local_target=is_local_node(ic, target.id),
        )
    except EffectError as exc:
        raise GraphQLValidationError(info, str(exc)) from exc
    solve = info.context.require_constraint_solve(ic, source=PreferredInstanceSource.DRAFT)
    problem = hook_shape_problem(graph, solve, action=action, target=target, hook=hook)
    if problem is not None:
        raise GraphQLValidationError(info, problem)
    return hook


def add_action_hook(info: gql.Info, ic: InstanceConfig, nc: NodeConfig, input: ActionHookInput) -> None:
    graph = require_draft_graph(info, ic)
    action = graph.node_by_id.get(nc.uuid)
    if action is None:
        raise NotFoundError(info, 'Node not found')
    hook = checked_hook(info, ic, graph, action, input)
    with gql_change_operation(info, ic, action='node.hooks.add'):
        write_hooks(nc, lambda hooks: [*hooks, hook])


def delete_action_hook(info: gql.Info, ic: InstanceConfig, nc: NodeConfig, input: ActionHookInput) -> None:
    type_config = nc.spec.type_config if nc.spec is not None else None
    if not isinstance(type_config, ActionConfig):
        raise NotFoundError(info, 'Hook not found')
    target_id = input.target_node_id

    # By the stored uuid, not the draft graph: a hook whose target a template release removed must stay removable.
    def matches(hook: ActionHookDef) -> bool:
        return (
            hook.node == target_id
            and (input.target_port_id is None or hook.port in (None, input.target_port_id))
            and (input.from_port_id is None or hook.from_port in (None, input.from_port_id))
        )

    if sum(1 for hook in type_config.hooks if matches(hook)) != 1:
        raise GraphQLValidationError(info, 'The input must name exactly one hook of this action')
    with gql_change_operation(info, ic, action='node.hooks.delete'):
        write_hooks(nc, lambda hooks: [hook for hook in hooks if not matches(hook)])
