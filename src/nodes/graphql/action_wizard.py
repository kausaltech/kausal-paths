"""
Creating an action from the output ports it acts on.

The modeller picks output ports of existing nodes; each becomes one *effect* of
a new `AdditiveAction`: an output port of the action with the target port's
quantity, unit and dimensions, a hook on the target port, and a source feeding
the action's paired input port. The source is the modeller's choice per
effect, made before anything is created:

- a new dataset owned by the action, with an explicit zero in the start year
  for each kept category, interpolated and extended;
- an existing instance dataset metric, bound the same way; or
- another node's output port, by an edge.

Everything happens in one change operation, so a refusal leaves nothing
behind. See docs/plans/action-from-output-port.md.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID, uuid4

import strawberry as sb
from django.db import transaction
from strawberry import Maybe

from kausal_common.datasets.models import DataPointDimensionCategory, Dataset, DatasetMetric
from kausal_common.strawberry.errors import GraphQLValidationError, NotFoundError, PermissionDeniedError
from kausal_common.users import user_or_none

from paths.identifiers import validate_identifier

from nodes.action_effects import EffectError, ValueShape, creates_cycle, incompatibility, output_port_shape, resolve_hook
from nodes.change_ops import gql_change_operation
from nodes.constants import VALUE_COLUMN
from nodes.defs.node_defs import InputDatasetDef, NodeKind
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


@sb.input(description='One effect of the new action: the output port it acts on, and the source of its numbers.')
class ActionEffectInput:
    target: OutputPortRefInput
    label: str | None = sb.field(
        default=None, description="Label of the action's output port (and new metric); defaults to the target's name."
    )
    source: EffectSourceInput | None = sb.field(default=None, description='Omitted: a new dataset with all categories.')


@sb.input
class CreateActionFromPortsInput:
    name: str
    identifier: sb.ID | None = sb.field(default=None, description='Derived from the name when omitted.')
    group: UUID | None = sb.field(default=None, description='UUID of the action group.')
    start_year: int | None = sb.field(
        default=None,
        description="Year of the zeros in new datasets; defaults to the instance's last historical year.",
    )
    effects: list[ActionEffectInput]


@sb.type
class CreateActionFromPortsResult:
    _action: sb.Private['Node']
    _datasets: sb.Private[list[Dataset]]

    @sb.field(graphql_type=AnyNodeType)
    @staticmethod
    def action(root: 'CreateActionFromPortsResult') -> Any:
        return root._action

    @sb.field(
        graphql_type=list[Annotated['DatasetType', sb.lazy('datasets.graphql.types')]],
        description='The datasets created for the action, for the modeller to fill in.',
    )
    @staticmethod
    def datasets(root: 'CreateActionFromPortsResult') -> list[Any]:
        from datasets.graphql.types import DatasetType

        return [DatasetType.from_model(dataset) for dataset in root._datasets]


# --- Candidate sources ------------------------------------------------------


@sb.type(description='A dataset metric that can feed an effect on the given port.')
class EffectDatasetCandidate:
    dataset_id: UUID
    dataset_name: str
    metric_id: UUID
    metric_label: str


@sb.type(description='An output port that can feed an effect on the given port.')
class EffectNodeCandidate:
    node_id: UUID
    node_name: str
    port_id: UUID


@sb.type(description='The existing sources whose shape fits an effect on a port: same dimensions, compatible unit.')
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


def effect_source_candidates(info: gql.Info, ic: InstanceConfig, target: OutputPortRefInput) -> EffectSourceCandidates:
    """List the instance datasets and node output ports that could feed an effect on `target`."""
    import networkx as nx

    graph = require_draft_graph(info, ic)
    solve = info.context.require_constraint_solve(ic, source=PreferredInstanceSource.DRAFT)
    node, port_id = _target_port(info, graph, target)
    target_shape = output_port_shape(graph, solve, node, port_id)

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
    downstream = nx.descendants(graph.nx_graph, node.id) | {node.id}
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
    target: NodeMeta
    target_port: UUID
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


def _target_port(info: gql.Info, graph: InstanceGraph, ref: OutputPortRefInput) -> tuple[NodeMeta, UUID]:
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


def _identifier(info: gql.Info, ic: InstanceConfig, graph: InstanceGraph, input: CreateActionFromPortsInput) -> str:
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
    categories: dict[UUID, list[UUID]] = {}
    for dimension_id in effect.shape.dimensions:
        dimension = graph.dimension_by_id[dimension_id]
        allowed = effect.shape.categories.get(dimension_id)
        categories[dimension_id] = [category.id for category in dimension.categories if allowed is None or category.id in allowed]
    for entry in given or []:
        if entry.dimension_id not in categories:
            raise GraphQLValidationError(info, f'{effect.target.identifier} has no dimension {entry.dimension_id}')
        unknown = set(entry.category_ids) - set(categories[entry.dimension_id])
        if unknown or not entry.category_ids:
            raise GraphQLValidationError(
                info, f'The categories kept for {effect.target.identifier} must be some of its own categories'
            )
        keep = set(entry.category_ids)
        categories[entry.dimension_id] = [category for category in categories[entry.dimension_id] if category in keep]
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
        source_node, source_port = _target_port(info, graph, source.node.value)
        effect.kind, effect.source_node, effect.source_port = 'node', source_node, source_port
        problem = incompatibility(output_port_shape(graph, solve, source_node, source_port), effect.shape)
    if problem is not None:
        raise GraphQLValidationError(info, f'The source cannot feed the effect on {effect.target.identifier}: {problem}')


def _resolve_effects(
    info: gql.Info,
    ic: InstanceConfig,
    graph: InstanceGraph,
    solve: ConstraintSolveResult,
    input: CreateActionFromPortsInput,
) -> list[_Effect]:
    """Resolve every effect and check its source against the target; nothing is written."""
    if not input.effects:
        raise GraphQLValidationError(info, 'An action needs at least one effect')
    datasets = {dataset.uuid: dataset for dataset in _instance_datasets(info, ic)}
    effects: list[_Effect] = []
    for entry in input.effects:
        target, target_port = _target_port(info, graph, entry.target)
        shape = output_port_shape(graph, solve, target, target_port)
        if shape.dimensions is None or shape.unit is None:
            raise GraphQLValidationError(info, f'The shape of {target.identifier} is not known, so no action can be built on it')
        port = target.spec.output_port_by_id[target_port]
        default_label = _text(target.name, target.identifier or '')
        if len(target.spec.output_ports) > 1:
            default_label = f'{default_label}: {_text(port.label, port.identifier or "")}'
        effect = _Effect(
            target=target,
            target_port=target_port,
            shape=shape,
            label=entry.label or default_label,
            output_port=uuid4(),
            kind='new_dataset',
        )
        _resolve_source(info, graph, solve, datasets, effect, entry.source)
        effects.append(effect)

    # The runtime gives a node one set of dimensions for all its outputs (`Node.validate_dims`).
    if len({effect.shape.dimensions for effect in effects}) > 1:
        raise GraphQLValidationError(
            info, 'All effects of one action must have the same dimensions; make the others a separate action'
        )

    # The action does not exist yet; check its edges with a stand-in.
    stand_in = uuid4()
    edges = [(stand_in, effect.target.id) for effect in effects]
    edges += [(effect.source_node.id, stand_in) for effect in effects if effect.source_node is not None]
    if creates_cycle(graph, edges):
        raise GraphQLValidationError(info, 'A source node depends on a node the action acts on, so the action would form a loop')
    return effects


def _ordered_dimensions(graph: InstanceGraph, effect: _Effect) -> list[str]:
    """Return the identifiers of the effect's dimensions, in the instance's order."""
    order = {dimension.id: index for index, dimension in enumerate(graph.dimensions)}
    return [graph.dimension_by_id[d].identifier for d in sorted(effect.shape.dimensions or (), key=order.__getitem__)]


def _create_action(
    info: gql.Info, ic: InstanceConfig, graph: InstanceGraph, input: CreateActionFromPortsInput, effects: list[_Effect]
) -> NodeConfig:
    from nodes.graphql.editor import ActionConfigInput, CreateNodeInput, NodeConfigInput, OutputPortInput, create_node_config

    dimensions = _ordered_dimensions(graph, effects[0])
    output_ports = [
        OutputPortInput(
            id=effect.output_port,
            identifier=f'effect_{index + 1}',
            label=effect.label,
            quantity=effect.shape.quantity,
            unit=str(effect.shape.unit),
            # A single output's column is the runtime default, as the editor's own nodes have it.
            column_id=VALUE_COLUMN if len(effects) == 1 else f'effect_{index + 1}',
            dimensions=dimensions,
        )
        for index, effect in enumerate(effects)
    ]
    config = ActionConfigInput(node_class=ADDITIVE_ACTION_CLASS, group=input.group)  # type: ignore[call-arg]
    return create_node_config(
        info,
        ic,
        CreateNodeInput(
            identifier=sb.ID(_identifier(info, ic, graph, input)),
            name=input.name,
            kind=NodeKind.ACTION,
            output_ports=output_ports,
            input_dimensions=dimensions,
            output_dimensions=dimensions,
            config=NodeConfigInput(action=sb.Some(config)),  # type: ignore[call-arg]
        ),
    )


def _write_hooks(info: gql.Info, ic: InstanceConfig, nc: NodeConfig, effects: list[_Effect]) -> None:
    from nodes.graphql.editor import write_hooks

    graph = require_draft_graph(info, ic)
    action = graph.node_by_id[nc.uuid]
    hooks = []
    for effect in effects:
        try:
            hooks.append(
                resolve_hook(
                    graph, action=action, target=effect.target, target_port=effect.target_port, from_port=effect.output_port
                )
            )
        except EffectError as exc:
            raise GraphQLValidationError(info, str(exc)) from exc
    write_hooks(nc, lambda existing: [*existing, *hooks])


def _paired_input_ports(nc: NodeConfig) -> dict[UUID, UUID]:
    """Return the action's input port paired with each of its output ports."""
    assert nc.spec is not None
    return {port.paired_output_port_id: port.id for port in nc.spec.input_ports if port.paired_output_port_id is not None}


def _create_dataset(
    info: gql.Info, ic: InstanceConfig, nc: NodeConfig, input: CreateActionFromPortsInput, effects: list[_Effect]
) -> tuple[Dataset | None, dict[UUID, tuple[Dataset, DatasetMetric]]]:
    """Create the action's own dataset for its new-dataset effects, one metric each, with the start-year zeros."""
    members = [effect for effect in effects if effect.kind == 'new_dataset']
    if not members:
        return None, {}
    paired = _paired_input_ports(nc)
    graph = require_draft_graph(info, ic)
    start_year = input.start_year or graph.spec.years.max_historical
    if start_year is None:
        raise GraphQLValidationError(info, 'The instance has no last historical year; give the start year')
    user = user_or_none(info.context.user)
    by_identifier = {dimension.identifier: dimension.id for dimension in graph.dimensions}
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
            dimension_ids=[by_identifier[identifier] for identifier in _ordered_dimensions(graph, members[0])],
            user=user,
        )
        for effect in members:
            assert effect.categories is not None
            metric = owned.metrics[paired[effect.output_port]]
            write_year(owned.dataset, [metric], start_year, effect.categories, value=0.0, user=user)
    except OwnedDatasetError as exc:
        raise GraphQLValidationError(info, str(exc)) from exc
    return owned.dataset, {effect.output_port: (owned.dataset, owned.metrics[paired[effect.output_port]]) for effect in members}


def _connect_sources(
    info: gql.Info, ic: InstanceConfig, nc: NodeConfig, effects: list[_Effect], owned: dict[UUID, tuple[Dataset, DatasetMetric]]
) -> None:
    """Feed each effect's paired input port: bind a dataset metric, or draw an edge from a node."""
    from datasets.snapshot import metric_column_id
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
                    instance_id=sb.ID(str(ic.pk)),
                    from_ref=NodePortRefInput(node_uuid=effect.source_node.id, port_id=effect.source_port),
                    port_ref=NodePortRefInput(node_uuid=nc.uuid, port_id=port_id),
                ),
            )
        else:
            dataset, metric = owned[effect.output_port] if effect.kind == 'new_dataset' else (effect.dataset, effect.metric)
            assert dataset is not None
            assert metric is not None
            # Actions rarely have a number for every year: fill between the years given, and hold
            # the last one to the model end, or the effect would vanish after its target year.
            transformations = InputDatasetDef(
                id='placeholder', column=metric_column_id(metric), interpolate=True, extend=True
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


def create_action_from_ports(
    info: gql.Info, ic: InstanceConfig, input: CreateActionFromPortsInput
) -> CreateActionFromPortsResult | ConstraintViolationsType:
    from nodes.graphql.editor import _resolve_runtime_node

    if ic.config_source != 'database':
        raise GraphQLValidationError(info, 'Cannot edit YAML-sourced instances')
    if not NodeConfig.gql_create_allowed(info, ic):
        raise PermissionDeniedError(info, 'Permission denied for create')
    graph = require_draft_graph(info, ic)
    solve = info.context.require_constraint_solve(ic, source=PreferredInstanceSource.DRAFT)
    effects = _resolve_effects(info, ic, graph, solve, input)

    try:
        with transaction.atomic(), gql_change_operation(info, ic, action='action.create_from_ports'):
            nc = _create_action(info, ic, graph, input, effects)
            nc = NodeConfig.objects.with_spec().get(pk=nc.pk)
            # Each step below reads the draft graph, which must now contain the new action.
            ic.invalidate_cache()
            _write_hooks(info, ic, nc, effects)
            ic.invalidate_cache()
            dataset, owned = _create_dataset(info, ic, nc, input, effects)
            ic.invalidate_cache()
            _connect_sources(info, ic, nc, effects, owned)
    except _Refused as refused:
        return refused.violations
    node = _resolve_runtime_node(info, ic, nc.pk)
    return CreateActionFromPortsResult(_action=node, _datasets=[dataset] if dataset is not None else [])
