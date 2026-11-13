"""
Creating an action from the nodes it acts on.

The modeller picks existing nodes; each becomes one *effect* of a new
`AdditiveAction`: an output port of the action with the target's quantity,
unit and dimensions, an edge from that port into the target's additive input,
and a source feeding the action's paired input port. The source is the
modeller's choice per effect, made before anything is created:

- a new dataset owned by the action, with an explicit zero in the start year
  and an empty cell in the target year for each kept category, interpolated,
  backfilled and extended;
- an existing instance dataset metric, bound the same way; or
- another node's output port, by an edge.

All new-dataset effects share one dataset, with the union of their
dimensions. A metric's cells name only its own dimensions and categories; the
rest stay empty, and the binding drops a dimension its metric does not use.

Everything happens in one change operation, so a refusal leaves nothing
behind. See docs/plans/action-from-output-port.md.
"""

import contextlib
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID, uuid4

import strawberry as sb
from django.db import transaction
from strawberry import Maybe

import networkx as nx

from kausal_common.datasets.models import DataPointDimensionCategory, Dataset, DatasetMetric
from kausal_common.strawberry.errors import GraphQLValidationError, NotFoundError, PermissionDeniedError
from kausal_common.users import user_or_none

from paths.identifiers import validate_identifier

from datasets.materialization import refresh_dataset_materialization
from datasets.snapshot import metric_column_id
from nodes.action_effects import (
    ActionTarget,
    EffectError,
    ValueShape,
    action_target,
    creates_cycle,
    incompatibility,
    output_port_shape,
)
from nodes.change_ops import gql_change_operation
from nodes.constants import VALUE_COLUMN
from nodes.defs.node_defs import InputDatasetDef, NodeKind
from nodes.exceptions import NodeError
from nodes.graphql.bindings import bind_dataset_metric
from nodes.graphql.constraint_checks import require_draft_graph
from nodes.graphql.types.constraints import ConstraintViolationsType
from nodes.graphql.types.node import AnyNodeType
from nodes.models import NodeConfig, PreferredInstanceSource
from nodes.owned_datasets import OwnedDatasetError, OwnedMetric, create_owned_dataset, write_year
from nodes.units import unit_registry

if TYPE_CHECKING:
    from paths import gql

    from datasets.graphql.types import DatasetType  # used in lazy strawberry annotation
    from nodes.constraints.solver import ConstraintSolveResult
    from nodes.defs.port_def import InputPortDef
    from nodes.instance_graph import InstanceGraph, NodeMeta
    from nodes.models import InstanceConfig
    from nodes.node import Node
    from nodes.units import Unit

ADDITIVE_ACTION_CLASS = 'nodes.actions.simple.AdditiveAction'


# --- Inputs -----------------------------------------------------------------


@sb.input(description='An output port of a node.')
class OutputPortRefInput:
    node_id: UUID
    port_id: UUID | None = sb.field(default=None, description='May be omitted when the node has a single output.')


@sb.input(description='The categories of one dimension that an effect touches.')
class EffectCategoriesInput:
    dimension_id: UUID
    category_ids: list[UUID]


@sb.input(description='A new dataset owned by the action, with a zero in the start year for every kept category.')
class NewDatasetSourceInput:
    categories: list[EffectCategoriesInput] | None = sb.field(
        default=None,
        description="Narrowing of the target's categories, per dimension; a dimension not named keeps all of them.",
    )


@sb.input(description='A metric of an existing instance dataset. It must hold changes, not levels.')
class DatasetSourceInput:
    dataset_id: UUID
    metric_id: UUID | None = sb.field(default=None, description='May be omitted when the dataset has a single metric.')


@sb.input(
    one_of=True,
    description="Where an effect's numbers come from. They are added to the target, so they must be changes, not levels.",
)
class EffectSourceInput:
    new_dataset: Maybe[NewDatasetSourceInput]
    dataset: Maybe[DatasetSourceInput]
    node: Maybe[OutputPortRefInput]


@sb.input(description='One effect of the new action: the node it acts on, and the source of its numbers.')
class ActionEffectInput:
    target_node_id: UUID = sb.field(description='The node the action acts on. It must have a single output port.')
    label: str | None = sb.field(
        default=None, description="Label of the action's output port (and new metric); defaults to the target's name."
    )
    source: EffectSourceInput | None = sb.field(default=None, description='Omitted: a new dataset with all categories.')


@sb.input
class CreateActionOnNodesInput:
    name: str
    identifier: sb.ID | None = sb.field(default=None, description='Derived from the name when omitted.')
    group: UUID | None = sb.field(default=None, description='UUID of the action group.')
    start_year: int | None = sb.field(
        default=None,
        description="Year of the zeros in the new dataset; defaults to the instance's last historical year.",
    )
    effects: list[ActionEffectInput]


@sb.type
class CreateActionOnNodesResult:
    _action: sb.Private['Node']
    _datasets: sb.Private[list[Dataset]]

    @sb.field(graphql_type=AnyNodeType)
    @staticmethod
    def action(root: 'CreateActionOnNodesResult') -> Any:
        return root._action

    @sb.field(
        graphql_type=list[Annotated['DatasetType', sb.lazy('datasets.graphql.types')]],
        description='The dataset created for the action, if any, for the modeller to fill in.',
    )
    @staticmethod
    def datasets(root: 'CreateActionOnNodesResult') -> list[Any]:
        from datasets.graphql.types import DatasetType

        return [DatasetType.from_model(dataset) for dataset in root._datasets]


# --- Which nodes an action can act on ----------------------------------------


@sb.type(description='Whether the action wizard can act on a node, and if not, why.')
class ActionTargetType:
    node_id: UUID
    available: bool
    reason: str | None = sb.field(description='Why an action cannot act on the node; null when it can.')


def _local_node_ids(ic: InstanceConfig) -> set[UUID]:
    return set(ic.nodes.values_list('uuid', flat=True))


def action_targets(info: gql.Info, ic: InstanceConfig) -> list[ActionTargetType]:
    """
    Say, for every node of the draft, whether the wizard can act on it.

    A yaml-sourced instance has no draft graph to build an action in, and may not
    even be able to build one (its stored spec can be the minimal one). The node
    graph query asks for this field on every instance, so it must answer rather
    than fail: every node is then unavailable, with the reason.
    """
    if ic.config_source != 'database':
        reason = 'The instance is defined in YAML, so the model editor cannot change it'
        return [
            ActionTargetType(node_id=node_id, available=False, reason=reason)
            for node_id in ic.nodes.values_list('uuid', flat=True)
        ]
    graph = require_draft_graph(info, ic)
    local = _local_node_ids(ic)
    result: list[ActionTargetType] = []
    for node in graph.nodes:
        try:
            action_target(node, is_local=node.id in local)
        except EffectError as exc:
            result.append(ActionTargetType(node_id=node.id, available=False, reason=str(exc)))
        else:
            result.append(ActionTargetType(node_id=node.id, available=True, reason=None))
    return result


# --- Candidate sources ------------------------------------------------------


@sb.type(description="The categories of one dimension that a node's output has.")
class DimensionCategoriesType:
    dimension_id: UUID
    category_ids: list[UUID]


def output_categories(
    info: gql.Info, ic: InstanceConfig, graph: InstanceGraph, target: ActionTarget, shape: ValueShape
) -> dict[UUID, frozenset[UUID]]:
    """
    Return, per dimension of the target's output, the categories an action on it can have.

    The answer must be categories the runtime knows: the stored category catalogue can
    lag the model and hold categories it no longer declares (see
    `reconcile_dimension_categories`), and a dataset cell in such a category makes the
    action, and everything downstream of it, fail with "Unknown categories".

    The solver narrows categories only where a rule fixes them, so for most nodes its
    answer is "any category of the dimension". The output itself is what the modeller
    expects to be offered, so the target is computed from the draft and the categories
    its output has are taken, intersected with the solver's answer where it has one.
    A target that fails to compute falls back to its runtime dimension's categories:
    those the node validates its output against, and so those the action's must be in.
    """
    categories = dict(shape.categories)
    if not shape.dimensions:
        return categories
    identifier = target.node.identifier or ''
    instance = info.context.require_instance(ic, source=PreferredInstanceSource.DRAFT)
    node = instance.context.nodes.get(identifier)
    if node is None:
        # A runtime built before the node was created; the draft graph already has it.
        instance = info.context.require_instance(ic, source=PreferredInstanceSource.DRAFT, refresh=True)
        node = instance.context.nodes.get(identifier)
    if node is None:
        return {dimension_id: frozenset() for dimension_id in shape.dimensions}
    df = None
    with contextlib.suppress(NodeError):
        df = node.get_output_pl()
    for dimension_id in shape.dimensions:
        dimension = graph.dimension_by_id[dimension_id]
        runtime = node.output_dimensions.get(dimension.identifier)
        known = set(runtime.get_cat_ids()) if runtime is not None else set()
        if df is not None and dimension.identifier in df.columns:
            known &= {str(value) for value in df[dimension.identifier].drop_nulls().unique().to_list()}
        found = frozenset(category.id for category in dimension.categories if category.identifier in known)
        solved = categories.get(dimension_id)
        categories[dimension_id] = found & solved if solved is not None else found
    return categories


def action_target_categories(info: gql.Info, ic: InstanceConfig, target_node_id: UUID) -> list[DimensionCategoriesType]:
    """List, per dimension, the categories an action acting on the node can change."""
    graph = require_draft_graph(info, ic)
    solve = info.context.require_constraint_solve(ic, source=PreferredInstanceSource.DRAFT)
    target = _target(info, ic, graph, target_node_id)
    shape = output_port_shape(graph, solve, target.node, target.output_port)
    categories = output_categories(info, ic, graph, target, shape)
    order = {category.id: index for dimension in graph.dimensions for index, category in enumerate(dimension.categories)}
    return [
        DimensionCategoriesType(dimension_id=dimension_id, category_ids=sorted(ids, key=order.__getitem__))
        for dimension_id, ids in categories.items()
    ]


@sb.type(description='A dataset metric that can feed an effect on the given node.')
class EffectDatasetCandidate:
    dataset_id: UUID
    dataset_name: str
    metric_id: UUID
    metric_label: str


@sb.type(description='An output port that can feed an effect on the given node.')
class EffectNodeCandidate:
    node_id: UUID
    node_name: str
    port_id: UUID


@sb.type(description='The existing sources whose shape fits an effect on a node: same dimensions, compatible unit.')
class EffectSourceCandidates:
    datasets: list[EffectDatasetCandidate]
    nodes: list[EffectNodeCandidate]


def _text(value: object, fallback: str) -> str:
    return str(value) if value else fallback


def _dataset_metric_shapes(datasets: list[Dataset]) -> dict[tuple[int, int], ValueShape]:
    """Return the shape of each (dataset pk, metric pk), with the categories its data uses."""
    used: dict[tuple[int, int], dict[UUID, set[UUID]]] = defaultdict(lambda: defaultdict(set))
    rows = (
        DataPointDimensionCategory.objects
        .filter(data_point__dataset__in=datasets)
        .values_list(
            'data_point__dataset_id', 'data_point__metric_id', 'dimension_category__dimension__uuid', 'dimension_category__uuid'
        )
        .distinct()
    )
    for dataset_pk, metric_pk, dimension_id, category_id in rows:
        used[(dataset_pk, metric_pk)][dimension_id].add(category_id)
    shapes: dict[tuple[int, int], ValueShape] = {}
    for dataset in datasets:
        assert dataset.schema is not None
        dimensions = frozenset(sd.dimension.uuid for sd in dataset.schema.dimensions.all())
        for metric in dataset.schema.metrics.all():
            if 'quality_of' in (metric.spec or {}):
                continue
            try:
                unit: Unit | None = unit_registry.parse_units(metric.unit) if metric.unit else None
            except Exception:
                unit = None
            categories = used.get((dataset.pk, metric.pk), {})
            shapes[(dataset.pk, metric.pk)] = ValueShape(
                dimensions=dimensions,
                unit=unit,
                categories={dimension_id: frozenset(ids) for dimension_id, ids in categories.items()},
            )
    return shapes


def _instance_datasets(info: gql.Info, ic: InstanceConfig) -> list[Dataset]:
    return list(
        Dataset.objects
        .get_queryset()
        .for_instance_config(ic)
        .viewable_by(info.context.user)
        .filter(schema__isnull=False)
        .select_related('schema')
        .prefetch_related('schema__metrics', 'schema__dimensions__dimension')
        .order_by('identifier', 'pk')
    )


def _target(info: gql.Info, ic: InstanceConfig, graph: InstanceGraph, node_id: UUID) -> ActionTarget:
    node = graph.node_by_id.get(node_id)
    if node is None:
        raise NotFoundError(info, f'Node "{node_id}" not found')
    try:
        return action_target(node, is_local=ic.nodes.filter(uuid=node_id).exists())
    except EffectError as exc:
        raise GraphQLValidationError(info, str(exc)) from exc


def effect_source_candidates(info: gql.Info, ic: InstanceConfig, target_node_id: UUID) -> EffectSourceCandidates:
    """List the instance datasets and node output ports that could feed an effect on the target node."""
    graph = require_draft_graph(info, ic)
    solve = info.context.require_constraint_solve(ic, source=PreferredInstanceSource.DRAFT)
    target = _target(info, ic, graph, target_node_id)
    target_shape = output_port_shape(graph, solve, target.node, target.output_port)

    datasets = _instance_datasets(info, ic)
    shapes = _dataset_metric_shapes(datasets)
    dataset_candidates = [
        EffectDatasetCandidate(
            dataset_id=dataset.uuid,
            dataset_name=_text(dataset.schema.name if dataset.schema else None, dataset.identifier or str(dataset.uuid)),
            metric_id=metric.uuid,
            metric_label=_text(metric.label, metric.name or str(metric.uuid)),
        )
        for dataset in datasets
        for metric in (dataset.schema.metrics.all() if dataset.schema else [])
        if (shape := shapes.get((dataset.pk, metric.pk))) is not None and incompatibility(shape, target_shape) is None
    ]

    # A node downstream of the target would feed on the effect it is the source of.
    downstream = nx.descendants(graph.nx_graph, target.node.id) | {target.node.id}
    node_candidates = [
        EffectNodeCandidate(node_id=candidate.id, node_name=_text(candidate.name, candidate.identifier or ''), port_id=port.id)
        for candidate in graph.nodes
        if candidate.id not in downstream
        for port in candidate.spec.output_ports
        if incompatibility(output_port_shape(graph, solve, candidate, port.id), target_shape) is None
    ]
    return EffectSourceCandidates(datasets=dataset_candidates, nodes=node_candidates)


# --- The mutation -----------------------------------------------------------


@dataclass
class _Effect:
    target: ActionTarget
    shape: ValueShape
    label: str
    output_port: UUID
    kind: str
    """'new_dataset', 'dataset' or 'node'."""
    categories: dict[UUID, list[UUID]] | None = None
    dataset: Dataset | None = None
    metric: DatasetMetric | None = None
    source_node: NodeMeta | None = None
    source_port: UUID | None = None


class _Refused(Exception):  # noqa: N818
    def __init__(self, violations: ConstraintViolationsType) -> None:
        self.violations = violations


def _source_port(info: gql.Info, graph: InstanceGraph, ref: OutputPortRefInput) -> tuple[NodeMeta, UUID]:
    node = graph.node_by_id.get(ref.node_id)
    if node is None:
        raise NotFoundError(info, f'Node "{ref.node_id}" not found')
    if ref.port_id is not None:
        if ref.port_id not in node.spec.output_port_by_id:
            raise GraphQLValidationError(info, f'{node.identifier} has no output port {ref.port_id}')
        return node, ref.port_id
    if len(node.spec.output_ports) != 1:
        raise GraphQLValidationError(info, f'{node.identifier} has several output ports; name one')
    return node, node.spec.output_ports[0].id


def _identifier(info: gql.Info, ic: InstanceConfig, graph: InstanceGraph, input: CreateActionOnNodesInput) -> str:
    taken = {node.identifier for node in graph.nodes} | set(ic.nodes.values_list('identifier', flat=True))
    if input.identifier:
        identifier = validate_identifier(str(input.identifier))
        if identifier in taken:
            raise GraphQLValidationError(info, f'Node with identifier {identifier} already exists')
        return identifier
    base = re.sub(r'[^a-z0-9]+', '_', input.name.lower()).strip('_') or 'action'
    if not base[0].isalpha():
        base = f'action_{base}'
    identifier, n = base, 2
    while identifier in taken:
        identifier, n = f'{base}_{n}', n + 1
    return validate_identifier(identifier)


def _narrowed(
    info: gql.Info, graph: InstanceGraph, effect: _Effect, given: list[EffectCategoriesInput] | None
) -> dict[UUID, list[UUID]]:
    """Return the categories to write cells for, per dimension: the target's, narrowed by `given`."""
    assert effect.shape.dimensions is not None
    name = effect.target.node.identifier
    categories: dict[UUID, list[UUID]] = {}
    for dimension_id in effect.shape.dimensions:
        dimension = graph.dimension_by_id[dimension_id]
        allowed = effect.shape.categories.get(dimension_id)
        categories[dimension_id] = [category.id for category in dimension.categories if allowed is None or category.id in allowed]
    for entry in given or []:
        if entry.dimension_id not in categories:
            raise GraphQLValidationError(info, f'{name} has no dimension {entry.dimension_id}')
        unknown = set(entry.category_ids) - set(categories[entry.dimension_id])
        if unknown or not entry.category_ids:
            raise GraphQLValidationError(info, f'The categories kept for {name} must be some of its own categories')
        keep = set(entry.category_ids)
        categories[entry.dimension_id] = [category for category in categories[entry.dimension_id] if category in keep]
    empty = [graph.dimension_by_id[dimension_id].identifier for dimension_id, ids in categories.items() if not ids]
    if empty:
        raise GraphQLValidationError(
            info, f'{name} has no categories the action could change in: {", ".join(empty)}. Does the node compute?'
        )
    return categories


def _resolve_source(
    info: gql.Info,
    graph: InstanceGraph,
    solve: ConstraintSolveResult,
    datasets: dict[UUID, Dataset],
    effect: _Effect,
    source: EffectSourceInput | None,
) -> None:
    """Fill in where `effect`'s numbers come from, refusing a source whose shape does not fit the target."""
    problem: str | None = None
    if source is None or source.new_dataset is not None:
        given = source.new_dataset.value.categories if source is not None and source.new_dataset is not None else None
        effect.categories = _narrowed(info, graph, effect, given)
    elif source.dataset is not None:
        ref = source.dataset.value
        dataset = datasets.get(ref.dataset_id)
        if dataset is None or dataset.schema is None:
            raise GraphQLValidationError(info, 'The source dataset must be one of the instance datasets')
        metrics = [m for m in dataset.schema.metrics.all() if 'quality_of' not in (m.spec or {})]
        metric = next((m for m in metrics if m.uuid == ref.metric_id), None) if ref.metric_id else None
        if metric is None and ref.metric_id is None and len(metrics) == 1:
            metric = metrics[0]
        if metric is None:
            raise GraphQLValidationError(info, 'Name one metric of the source dataset')
        effect.kind, effect.dataset, effect.metric = 'dataset', dataset, metric
        problem = incompatibility(_dataset_metric_shapes([dataset])[(dataset.pk, metric.pk)], effect.shape)
    elif source.node is not None:
        source_node, source_port = _source_port(info, graph, source.node.value)
        effect.kind, effect.source_node, effect.source_port = 'node', source_node, source_port
        problem = incompatibility(output_port_shape(graph, solve, source_node, source_port), effect.shape)
    if problem is not None:
        raise GraphQLValidationError(info, f'The source cannot feed the effect on {effect.target.node.identifier}: {problem}')


def _resolve_effects(
    info: gql.Info,
    ic: InstanceConfig,
    graph: InstanceGraph,
    solve: ConstraintSolveResult,
    input: CreateActionOnNodesInput,
) -> list[_Effect]:
    """Resolve every effect and check its source against the target; nothing is written."""
    if not input.effects:
        raise GraphQLValidationError(info, 'An action needs at least one effect')
    targets = [entry.target_node_id for entry in input.effects]
    if len(set(targets)) != len(targets):
        raise GraphQLValidationError(info, 'An action can act on each node only once')
    datasets = {dataset.uuid: dataset for dataset in _instance_datasets(info, ic)}
    effects: list[_Effect] = []
    for entry in input.effects:
        target = _target(info, ic, graph, entry.target_node_id)
        shape = output_port_shape(graph, solve, target.node, target.output_port)
        shape = replace(shape, categories=output_categories(info, ic, graph, target, shape))
        if shape.dimensions is None or shape.unit is None:
            raise GraphQLValidationError(
                info, f'The shape of {target.node.identifier} is not known, so no action can be built on it'
            )
        effect = _Effect(
            target=target,
            shape=shape,
            label=entry.label or _text(target.node.name, target.node.identifier or ''),
            output_port=uuid4(),
            kind='new_dataset',
        )
        _resolve_source(info, graph, solve, datasets, effect, entry.source)
        effects.append(effect)

    # The action does not exist yet; check its edges with a stand-in.
    stand_in = uuid4()
    edges = [(stand_in, effect.target.node.id) for effect in effects]
    edges += [(effect.source_node.id, stand_in) for effect in effects if effect.source_node is not None]
    if creates_cycle(graph, edges):
        raise GraphQLValidationError(info, 'A source node depends on a node the action acts on, so the action would form a loop')
    return effects


def _ordered(graph: InstanceGraph, dimension_ids: set[UUID] | frozenset[UUID]) -> list[UUID]:
    """Return `dimension_ids` in the instance's order of dimensions."""
    order = {dimension.id: index for index, dimension in enumerate(graph.dimensions)}
    return sorted(dimension_ids, key=order.__getitem__)


def _create_action(
    info: gql.Info, ic: InstanceConfig, graph: InstanceGraph, input: CreateActionOnNodesInput, effects: list[_Effect]
) -> NodeConfig:
    from nodes.graphql.editor import ActionConfigInput, CreateNodeInput, NodeConfigInput, OutputPortInput, create_node_config

    output_ports = [
        OutputPortInput(
            id=effect.output_port,
            identifier=f'effect_{index + 1}',
            label=effect.label,
            quantity=effect.shape.quantity,
            unit=str(effect.shape.unit),
            # A single output's column is the runtime default, as the editor's own nodes have it.
            column_id=VALUE_COLUMN if len(effects) == 1 else f'effect_{index + 1}',
            # Each effect keeps its target's dimensions; AdditiveAction leaves the
            # dimensions a port does not have empty, and each edge drops them.
            dimensions=[graph.dimension_by_id[d].identifier for d in _ordered(graph, effect.shape.dimensions or frozenset())],
        )
        for index, effect in enumerate(effects)
    ]
    all_dimensions = set().union(*(effect.shape.dimensions or frozenset() for effect in effects))
    config = ActionConfigInput(node_class=ADDITIVE_ACTION_CLASS, group=input.group)  # type: ignore[call-arg]
    return create_node_config(
        info,
        ic,
        CreateNodeInput(  # type: ignore[call-arg]
            identifier=sb.ID(_identifier(info, ic, graph, input)),
            name=input.name,
            kind=NodeKind.ACTION,
            output_ports=output_ports,
            input_dimensions=[graph.dimension_by_id[d].identifier for d in _ordered(graph, all_dimensions)],
            config=NodeConfigInput(action=sb.Some(config)),  # type: ignore[call-arg]
        ),
    )


def _paired_input_ports(nc: NodeConfig) -> dict[UUID, UUID]:
    """Return the action's input port paired with each of its output ports."""
    assert nc.spec is not None
    return {port.paired_output_port_id: port.id for port in nc.spec.input_ports if port.paired_output_port_id is not None}


def _create_dataset(
    info: gql.Info, ic: InstanceConfig, nc: NodeConfig, input: CreateActionOnNodesInput, effects: list[_Effect]
) -> tuple[Dataset | None, dict[UUID, DatasetMetric]]:
    """Create the action's own dataset for its new-dataset effects: one metric each, start-year zeros, empty target-year cells."""
    members = [effect for effect in effects if effect.kind == 'new_dataset']
    if not members:
        return None, {}
    paired = _paired_input_ports(nc)
    graph = require_draft_graph(info, ic)
    start_year = input.start_year or graph.spec.years.max_historical
    if start_year is None:
        raise GraphQLValidationError(info, 'The instance has no last historical year; give the start year')
    target_year = graph.spec.years.target
    user = user_or_none(info.context.user)
    dimensions = set().union(*(effect.shape.dimensions or frozenset() for effect in members))
    try:
        owned = create_owned_dataset(
            nc,
            name=input.name,
            metrics=[
                OwnedMetric(
                    port_id=paired[effect.output_port],
                    label=effect.label,
                    unit=effect.shape.unit,  # type: ignore[arg-type]
                    quantity=effect.shape.quantity,
                )
                for effect in members
            ],
            dimension_ids=_ordered(graph, dimensions),
            user=user,
        )
        for effect in members:
            assert effect.categories is not None
            metric = owned.metrics[paired[effect.output_port]]
            write_year(owned.dataset, metric, start_year, effect.categories, value=0.0)
            # The target year's cells are left empty, to show the modeller where the numbers go:
            # the zeros are the baseline, and the change is what they enter for the target year.
            # Empty cells do not count until filled, since the binding drops null rows.
            if target_year is not None and target_year > start_year:
                write_year(owned.dataset, metric, target_year, effect.categories, value=None)
    except OwnedDatasetError as exc:
        raise GraphQLValidationError(info, str(exc)) from exc
    refresh_dataset_materialization(owned.dataset, user=user)
    return owned.dataset, {effect.output_port: owned.metrics[paired[effect.output_port]] for effect in members}


def _feed_effects(
    info: gql.Info,
    ic: InstanceConfig,
    nc: NodeConfig,
    effects: list[_Effect],
    owned_dataset: Dataset | None,
    owned_metrics: dict[UUID, DatasetMetric],
) -> None:
    """Feed each effect's paired input port: bind a dataset metric, or draw an edge from a node."""
    from nodes.graphql.editor import CreateEdgeInput, NodePortRefInput, create_edge_binding

    paired = _paired_input_ports(nc)
    for effect in effects:
        port_id = paired[effect.output_port]
        result: Any
        if effect.kind == 'node':
            assert effect.source_node is not None
            result = create_edge_binding(
                info,
                CreateEdgeInput(
                    instance_id=sb.ID(ic.identifier),
                    from_ref=NodePortRefInput(node_uuid=effect.source_node.id, port_id=effect.source_port),
                    port_ref=NodePortRefInput(node_uuid=nc.uuid, port_id=port_id),
                ),
            )
        else:
            if effect.kind == 'new_dataset':
                dataset, metric = owned_dataset, owned_metrics[effect.output_port]
            else:
                dataset, metric = effect.dataset, effect.metric
            assert dataset is not None
            assert metric is not None
            # Actions rarely have a number for every year: fill between the years given, hold the
            # last one to the model end, and hold the first one back to the start of the model, so
            # the effect neither vanishes after its target year nor leaves the history without rows.
            # The action's own dataset starts from a zero, so its history is zero.
            transformations = InputDatasetDef(
                id='placeholder',
                column=metric_column_id(metric),
                interpolate=True,
                backfill=effect.kind == 'new_dataset',
                extend=True,
            ).to_transformations()
            result = bind_dataset_metric(
                info,
                ic,
                nc,
                port_id=port_id,
                dataset=dataset,
                metric=metric,
                transformations=transformations,
                displaced=[],
                replace=False,
            )
        if isinstance(result, ConstraintViolationsType):
            raise _Refused(result)


def _connect_targets(info: gql.Info, ic: InstanceConfig, nc: NodeConfig, effects: list[_Effect]) -> None:
    """Draw an edge from each of the action's output ports into its target's additive input."""
    from nodes.graphql.editor import CreateEdgeInput, NodePortRefInput, create_edge_binding

    for effect in effects:
        new_port: InputPortDef | None = effect.target.plan_new_port(effect.shape, effect.label)
        to_port = effect.target.input_port if new_port is None else None
        result = create_edge_binding(
            info,
            CreateEdgeInput(
                instance_id=sb.ID(ic.identifier),
                from_ref=NodePortRefInput(node_uuid=nc.uuid, port_id=effect.output_port),
                port_ref=NodePortRefInput(node_uuid=effect.target.node.id, port_id=to_port),
            ),
            new_port=new_port,
        )
        if isinstance(result, ConstraintViolationsType):
            raise _Refused(result)
        ic.invalidate_cache()


def create_action_on_nodes(
    info: gql.Info, ic: InstanceConfig, input: CreateActionOnNodesInput
) -> CreateActionOnNodesResult | ConstraintViolationsType:
    from nodes.graphql.editor import _resolve_runtime_node

    if ic.config_source != 'database':
        raise GraphQLValidationError(info, 'Cannot edit YAML-sourced instances')
    if not NodeConfig.gql_create_allowed(info, ic):
        raise PermissionDeniedError(info, 'Permission denied for create')
    graph = require_draft_graph(info, ic)
    solve = info.context.require_constraint_solve(ic, source=PreferredInstanceSource.DRAFT)
    effects = _resolve_effects(info, ic, graph, solve, input)

    try:
        with transaction.atomic(), gql_change_operation(info, ic, action='action.create_on_nodes'):
            nc = _create_action(info, ic, graph, input, effects)
            nc = NodeConfig.objects.with_spec().get(pk=nc.pk)
            # Each step below reads the draft graph, which must now contain the new action.
            ic.invalidate_cache()
            dataset, metrics = _create_dataset(info, ic, nc, input, effects)
            ic.invalidate_cache()
            _feed_effects(info, ic, nc, effects, dataset, metrics)
            ic.invalidate_cache()
            _connect_targets(info, ic, nc, effects)
    except _Refused as refused:
        return refused.violations
    node = _resolve_runtime_node(info, ic, nc.pk)
    return CreateActionOnNodesResult(_action=node, _datasets=[dataset] if dataset is not None else [])
