"""
Creating an action from its effects (`createAction`).

Each effect names where it lands and how its numbers are given:

- on a node's **output port**, through a hook: the action's output is added to
  what the node gives its consumers, in the years after the last historical
  year. The node needs no input for it, so this is how an instance acts on a
  node it inherits from a framework. The effect is an amount in the target's
  unit, or a share of the target's value before any action.
- into a node's **additive input**, through an ordinary edge, in every year.
  This is what `createActionOnNodes` does; that mutation remains until the
  model editor has moved here.

The action is an `AdditiveAction` with one output port per effect. By default
its numbers are in a dataset of its own, with a zero in the start year and the
values the caller gives. See docs/architecture/action-hooks.md.
"""

from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID, uuid4

import strawberry as sb
from django.db import transaction
from strawberry import Maybe

from kausal_common.strawberry.errors import GraphQLValidationError, NotFoundError, PermissionDeniedError

from nodes.action_effects import (
    ActionTarget,
    EffectError,
    ValueShape,
    creates_cycle,
    hook_shape_problem,
    hook_target_problem,
    output_port_shape,
    resolve_hook,
)
from nodes.change_ops import gql_change_operation
from nodes.defs.node_defs import ActionHookMode
from nodes.graphql.action_hooks import ActionHookModeEnum, is_local_node, write_hooks
from nodes.graphql.action_wizard import (
    ActionIdentity,
    EffectSourceInput,
    EffectValue,
    OutputPortRefInput,
    _connect_targets,
    _create_action,
    _create_dataset,
    _Effect,
    _feed_effects,
    _instance_datasets,
    _Refused,
    _resolve_source,
    _source_port,
    _target,
    _text,
    output_categories,
)
from nodes.graphql.constraint_checks import require_draft_graph
from nodes.graphql.types.node import AnyNodeType
from nodes.models import InstanceConfig, NodeConfig, PreferredInstanceSource
from nodes.units import unit_registry

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset

    from paths import gql

    from datasets.graphql.types import DatasetType
    from nodes.constraints.solver import ConstraintSolveResult
    from nodes.graphql.types.constraints import ConstraintViolationsType
    from nodes.instance_graph import InstanceGraph
    from nodes.node import Node

SHARE_QUANTITY = 'fraction'


@sb.input(
    one_of=True,
    description=(
        'Where an effect lands. `outputPort`: on what a node gives its consumers, through a hook, in the years '
        'after the last historical year. `additiveInputOf`: into the additive input of a node of this instance, '
        'through an edge, in every year.'
    ),
)
class ActionEffectTargetInput:
    output_port: Maybe[OutputPortRefInput]
    additive_input_of: Maybe[UUID]


@sb.input(description='One category of one dimension.')
class EffectCategoryInput:
    dimension_id: UUID
    category_id: UUID


@sb.input(description="One cell of a new effect's data.")
class EffectValueInput:
    year: int
    categories: list[EffectCategoryInput] = sb.field(
        default_factory=list, description='One category for each dimension of the effect; empty when it has none.'
    )
    value: float


@sb.input(description='One effect of the new action: where it lands, and where its numbers come from.')
class CreateActionEffectInput:
    target: ActionEffectTargetInput
    mode: ActionHookModeEnum = sb.field(  # type: ignore[valid-type]
        default=ActionHookMode.ABSOLUTE,
        description=(
            "ABSOLUTE: a signed amount in the target's unit. RELATIVE: a share without a unit, -0.3 for -30 %, of "
            "the target's value before any action; only on an `outputPort` target."
        ),
    )
    label: str | None = sb.field(
        default=None, description="Label of the action's output port (and new metric); defaults to the target's name."
    )
    source: EffectSourceInput | None = sb.field(default=None, description='Omitted: a new dataset with all categories.')
    values: list[EffectValueInput] | None = sb.field(
        default=None,
        description=(
            'Cells to write into the new dataset, beside the zero it gets in the start year: a target value for one '
            'year, or a whole series. The years between are interpolated and the last value is held. Only with a '
            'new dataset.'
        ),
    )


@sb.input
class CreateActionInput:
    name: str
    description: str | None = None
    identifier: sb.ID | None = sb.field(default=None, description='Derived from the name when omitted.')
    group: UUID | None = sb.field(default=None, description='UUID of the action group.')
    start_year: int | None = sb.field(
        default=None,
        description="Year of the zeros in the new dataset; defaults to the instance's last historical year.",
    )
    effects: list[CreateActionEffectInput]


@sb.type
class CreateActionResult:
    _action: sb.Private['Node']
    _datasets: sb.Private[list['Dataset']]

    @sb.field(graphql_type=AnyNodeType)
    @staticmethod
    def action(root: 'CreateActionResult') -> Any:
        return root._action

    @sb.field(
        graphql_type=list[Annotated['DatasetType', sb.lazy('datasets.graphql.types')]],
        description="The action's own dataset, if it has one: where its numbers are edited.",
    )
    @staticmethod
    def datasets(root: 'CreateActionResult') -> list[Any]:
        from datasets.graphql.types import DatasetType

        return [DatasetType.from_model(dataset) for dataset in root._datasets]


def _hook_target(info: gql.Info, ic: InstanceConfig, graph: InstanceGraph, ref: OutputPortRefInput) -> ActionTarget:
    """Return the output port a hook is to act on, refusing a node that takes none."""
    node, port_id = _source_port(info, graph, ref)
    problem = hook_target_problem(node, is_local=is_local_node(ic, node.id))
    if problem is not None:
        raise GraphQLValidationError(info, problem)
    return ActionTarget(node=node, output_port=port_id, input_port=None)


def _effect_shape(
    info: gql.Info, ic: InstanceConfig, graph: InstanceGraph, solve: ConstraintSolveResult, target: ActionTarget, relative: bool
) -> ValueShape:
    """Return the shape of the action's output for an effect on `target`: the target's, or a share with its dimensions."""
    shape = output_port_shape(graph, solve, target.node, target.output_port)
    shape = replace(shape, categories=output_categories(info, ic, graph, target.node, shape))
    if shape.dimensions is None or shape.unit is None:
        raise GraphQLValidationError(info, f'The shape of {target.node.identifier} is not known, so no action can be built on it')
    if relative:
        shape = replace(shape, unit=unit_registry.dimensionless, quantity=SHARE_QUANTITY)
    return shape


def _values(info: gql.Info, graph: InstanceGraph, effect: _Effect, given: list[EffectValueInput]) -> list[EffectValue]:
    """Check the cells given for a new dataset: each names one kept category of every dimension, once."""
    assert effect.categories is not None
    name = effect.target.node.identifier
    last_year = graph.spec.years.model_end or graph.spec.years.target
    cells: dict[tuple[int, tuple[tuple[UUID, UUID], ...]], EffectValue] = {}
    for entry in given:
        categories = {category.dimension_id: category.category_id for category in entry.categories}
        if len(categories) != len(entry.categories) or set(categories) != set(effect.categories):
            raise GraphQLValidationError(info, f'A value for {name} names one category of each of its dimensions')
        if any(category not in effect.categories[dimension] for dimension, category in categories.items()):
            raise GraphQLValidationError(info, f'A value for {name} is in a category the effect does not have')
        if last_year is not None and entry.year > last_year:
            raise GraphQLValidationError(info, f'{entry.year} is after the last year of the model, {last_year}')
        key = (entry.year, tuple(sorted(categories.items())))
        if key in cells:
            raise GraphQLValidationError(info, f'{name} is given two values for one cell in {entry.year}')
        cells[key] = EffectValue(year=entry.year, categories=categories, value=entry.value)
    return list(cells.values())


def _resolve_effects(
    info: gql.Info, ic: InstanceConfig, graph: InstanceGraph, solve: ConstraintSolveResult, input: CreateActionInput
) -> list[_Effect]:
    """Resolve every effect and check its source against its target; nothing is written."""
    if not input.effects:
        raise GraphQLValidationError(info, 'An action needs at least one effect')
    datasets = {dataset.uuid: dataset for dataset in _instance_datasets(info, ic)}
    effects: list[_Effect] = []
    for entry in input.effects:
        mode = ActionHookMode(entry.mode)
        if entry.target.output_port is not None:
            target, hook_mode = _hook_target(info, ic, graph, entry.target.output_port.value), mode
        else:
            assert entry.target.additive_input_of is not None
            if mode == ActionHookMode.RELATIVE:
                # It would have to read the output of the node it feeds, which is a cycle.
                raise GraphQLValidationError(info, 'A relative effect acts on an output port, not through an additive input')
            target, hook_mode = _target(info, ic, graph, entry.target.additive_input_of.value), None
        effect = _Effect(
            target=target,
            shape=_effect_shape(info, ic, graph, solve, target, relative=mode == ActionHookMode.RELATIVE),
            label=entry.label or _text(target.node.name, target.node.identifier or ''),
            output_port=uuid4(),
            kind='new_dataset',
            hook_mode=hook_mode,
        )
        _resolve_source(info, graph, solve, datasets, effect, entry.source)
        if entry.values:
            if effect.kind != 'new_dataset':
                raise GraphQLValidationError(info, 'Values can be given only for a new dataset')
            effect.values = _values(info, graph, effect, entry.values)
        effects.append(effect)

    ports = [(effect.target.node.id, effect.target.output_port) for effect in effects]
    if len(set(ports)) != len(ports):
        raise GraphQLValidationError(info, 'An action can act on each output only once')
    # The action does not exist yet; check its connections with a stand-in.
    stand_in = uuid4()
    edges = [(stand_in, effect.target.node.id) for effect in effects]
    edges += [(effect.source_node.id, stand_in) for effect in effects if effect.source_node is not None]
    if creates_cycle(graph, edges):
        raise GraphQLValidationError(info, 'A source node depends on a node the action acts on, so the action would form a loop')
    return effects


def _hook_targets(info: gql.Info, ic: InstanceConfig, nc: NodeConfig, effects: list[_Effect]) -> None:
    """Give the action a hook on the target of each of its effects that acts through one."""
    members = [effect for effect in effects if effect.hook_mode is not None]
    if not members:
        return
    graph = require_draft_graph(info, ic)
    solve = info.context.require_constraint_solve(ic, source=PreferredInstanceSource.DRAFT)
    action = graph.node_by_id.get(nc.uuid)
    if action is None:
        raise NotFoundError(info, 'The new action is not in the draft graph')
    hooks = []
    for effect in members:
        assert effect.hook_mode is not None
        target = graph.node_by_id[effect.target.node.id]
        try:
            hook = resolve_hook(
                graph,
                action=action,
                target=target,
                target_port=effect.target.output_port,
                from_port=effect.output_port,
                mode=effect.hook_mode,
                is_local_target=is_local_node(ic, target.id),
            )
        except EffectError as exc:
            raise GraphQLValidationError(info, str(exc)) from exc
        problem = hook_shape_problem(graph, solve, action=action, target=target, hook=hook)
        if problem is not None:
            raise GraphQLValidationError(info, problem)
        hooks.append(hook)
    nc = NodeConfig.objects.with_spec().get(pk=nc.pk)
    write_hooks(nc, lambda existing: [*existing, *hooks])
    ic.invalidate_cache()


def create_action(info: gql.Info, ic: InstanceConfig, input: CreateActionInput) -> CreateActionResult | ConstraintViolationsType:
    from nodes.graphql.editor import _resolve_runtime_node

    if ic.config_source != 'database':
        raise GraphQLValidationError(info, 'Cannot edit YAML-sourced instances')
    if not NodeConfig.gql_create_allowed(info, ic):
        raise PermissionDeniedError(info, 'Permission denied for create')
    graph = require_draft_graph(info, ic)
    solve = info.context.require_constraint_solve(ic, source=PreferredInstanceSource.DRAFT)
    effects = _resolve_effects(info, ic, graph, solve, input)
    identity = ActionIdentity(
        name=input.name,
        identifier=str(input.identifier) if input.identifier else None,
        group=input.group,
        start_year=input.start_year,
        description=input.description,
    )

    try:
        with transaction.atomic(), gql_change_operation(info, ic, action='action.create'):
            nc = _create_action(info, ic, graph, identity, effects)
            nc = NodeConfig.objects.with_spec().get(pk=nc.pk)
            # Each step below reads the draft graph, which must now contain the new action.
            ic.invalidate_cache()
            dataset, metrics = _create_dataset(info, ic, nc, identity, effects)
            ic.invalidate_cache()
            _feed_effects(info, ic, nc, effects, dataset, metrics)
            ic.invalidate_cache()
            _connect_targets(info, ic, nc, effects)
            _hook_targets(info, ic, nc, effects)
    except _Refused as refused:
        return refused.violations
    node = _resolve_runtime_node(info, ic, nc.pk)
    return CreateActionResult(_action=node, _datasets=[dataset] if dataset is not None else [])
