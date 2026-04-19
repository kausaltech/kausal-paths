"""
Strawberry GraphQL schema for the model editor.

Provides queries and mutations for reading and editing DB-sourced
model instances (NodeConfig, NodeEdge, ActionGroup, Scenario).
"""

from typing import TYPE_CHECKING, Annotated, Any, TypeGuard, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import strawberry as sb
from django.db import transaction
from django.utils.module_loading import import_string
from graphql import GraphQLError
from pydantic import TypeAdapter, ValidationError as PydanticValidationError
from strawberry import Maybe, Some, auto

from kausal_common.strawberry.errors import GraphQLValidationError, NotFoundError, PermissionDeniedError
from kausal_common.strawberry.helpers import get_or_error
from kausal_common.strawberry.ordering import SiblingPositionInputMixin
from kausal_common.strawberry.pydantic import StrawberryPydanticType, pydantic_input

from paths import gql

from nodes.defs import FormulaConfig, SimpleConfig
from nodes.defs.node_defs import ActionConfig, InputDatasetDef, NodeKind, NodeSpec, PipelineConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.models import InstanceConfig, NodeConfig, NodeKindChoices
from nodes.node import Node
from nodes.units import unit_registry
from params.param import BoolParameter, NumberParameter, StringParameter

from .types.dimension import DimensionType
from .types.graph import NodeEdgeType
from .types.instance import InstanceType
from .types.node import AnyNodeType, NodeInterface
from .types.scenario import ScenarioType

if TYPE_CHECKING:
    from kausal_common.datasets.models import DimensionCategory as DimensionCategoryModel, DimensionScope
    from kausal_common.models.ordered import OrderedModel

    from datasets.graphql.editor import DatasetEditorMutation


def _get_instance_config(info: gql.Info, instance_id: sb.ID) -> InstanceConfig:
    qs = InstanceConfig.objects.qs.modifiable_by(info.context.user).by_all_identifiers(str(instance_id))
    try:
        return qs.defer(None).get()
    except InstanceConfig.DoesNotExist:
        raise GraphQLError(f'Instance "{instance_id}" not found') from None


def _resolve_model_instance(ic: InstanceConfig) -> InstanceType:
    instance = ic._initialize_instance(node_refs=True)
    node_configs = ic.nodes_for_serialization
    node_config_by_identifier = {nc.identifier: nc for nc in node_configs}
    instance._annotated_node_configs_by_identifier = node_config_by_identifier  # type: ignore[attr-defined]
    for node_id, node in instance.context.nodes.items():
        nc = node_config_by_identifier.get(node_id)
        if nc is not None:
            node.db_obj = nc
    return cast('InstanceType', instance)


def _resolve_runtime_node(ic: InstanceConfig, node_id: int) -> Node:
    try:
        nc = NodeConfig.objects.select_related('instance').get(pk=node_id)
    except NodeConfig.DoesNotExist:
        raise GraphQLError('Node not found') from None

    instance = nc.instance._initialize_instance(node_refs=True)
    node = instance.context.nodes.get(nc.identifier)
    if node is None:
        raise GraphQLError(f'Node "{node_id}" not found in runtime instance "{ic.identifier}"')
    return node


def _parse_port_id(info: gql.Info, raw_port_id: str, *, field_name: str) -> UUID:
    try:
        return UUID(raw_port_id)
    except ValueError as exc:
        raise GraphQLValidationError(info, f'"{field_name}" must be a UUID, got "{raw_port_id}"') from exc


def _get_output_port(nc: NodeConfig, port_id: UUID) -> OutputPortDef | None:
    assert nc.spec is not None
    for port in nc.spec.output_ports:
        if port.id == port_id:
            return port
    return None


def _get_input_port(nc: NodeConfig, port_id: UUID) -> InputPortDef | None:
    assert nc.spec is not None
    for port in nc.spec.input_ports:
        if port.id == port_id:
            return port
    return None


def _resolve_target_port(info: gql.Info, to_node: NodeConfig, to_port: str | None) -> UUID:
    if to_port is not None:
        port_id = _parse_port_id(info, to_port, field_name='toPort')
        if _get_input_port(to_node, port_id) is not None:
            return port_id
        raise GraphQLValidationError(info, f'Input port "{to_port}" does not exist on node "{to_node.identifier}"')
    assert to_node.spec is not None
    input_ports = to_node.spec.input_ports
    if len(input_ports) == 1:
        return input_ports[0].id
    raise GraphQLValidationError(
        info,
        f'Target node "{to_node.identifier}" has {len(input_ports)} input ports; toPort must be specified explicitly',
    )


def _resolve_source_port(info: gql.Info, from_node: NodeConfig, from_port: str) -> UUID:
    assert from_node.spec is not None
    output_ports = from_node.spec.output_ports
    if from_port == 'output' and len(output_ports) == 1:
        return output_ports[0].id
    port_id = _parse_port_id(info, from_port, field_name='fromPort')
    if _get_output_port(from_node, port_id) is not None:
        return port_id
    raise GraphQLValidationError(
        info,
        f'Output port "{from_port}" does not exist on node "{from_node.identifier}"',
    )


def _validate_edge_ports(info: gql.Info, from_node: NodeConfig, from_port: UUID, to_node: NodeConfig, to_port: UUID) -> None:
    from nodes.models import DatasetPort, NodeEdge

    source_port = _get_output_port(from_node, from_port)
    if source_port is None:
        raise GraphQLValidationError(info, f'Output port "{from_port}" does not exist on node "{from_node.identifier}"')

    target_port = _get_input_port(to_node, to_port)
    if target_port is None:
        raise GraphQLValidationError(info, f'Input port "{to_port}" does not exist on node "{to_node.identifier}"')

    if source_port.quantity is not None and target_port.quantity is not None and source_port.quantity != target_port.quantity:
        raise GraphQLValidationError(
            info,
            f'Quantity mismatch: output port "{from_node.identifier}.{from_port}" has quantity '
            + f'"{source_port.quantity}", input port "{to_node.identifier}.{to_port}" expects "{target_port.quantity}"',
        )

    source_dims = set(source_port.dimensions)
    required_dims = set(target_port.required_dimensions)
    supported_dims = set(target_port.supported_dimensions)

    missing_dims = sorted(required_dims - source_dims)
    if missing_dims:
        raise GraphQLValidationError(
            info,
            f'Input port "{to_node.identifier}.{to_port}" requires dimensions {missing_dims}, '
            + f'but output port "{from_node.identifier}.{from_port}" provides {sorted(source_dims)}',
        )

    if supported_dims and not source_dims.issubset(supported_dims):
        raise GraphQLValidationError(
            info,
            f'Input port "{to_node.identifier}.{to_port}" supports only dimensions {sorted(supported_dims)}, '
            + f'but output port "{from_node.identifier}.{from_port}" provides {sorted(source_dims)}',
        )

    if (
        source_port.unit is not None
        and target_port.unit is not None
        and source_port.unit.dimensionality != target_port.unit.dimensionality
    ):
        raise GraphQLValidationError(
            info,
            f'Unit dimensionality mismatch: output port "{from_node.identifier}.{from_port}" has '
            + f'{source_port.unit.dimensionality}, input port "{to_node.identifier}.{to_port}" expects '
            + f'{target_port.unit.dimensionality}',
        )

    if not target_port.multi:
        has_edge_binding = NodeEdge.objects.filter(to_node=to_node, to_port=to_port).exists()
        has_dataset_binding = DatasetPort.objects.filter(node=to_node, port_id=to_port).exists()
        if has_edge_binding or has_dataset_binding:
            raise GraphQLValidationError(
                info,
                f'Input port "{to_node.identifier}.{to_port}" already has a binding and does not allow multiple inputs',
            )


@pydantic_input(model=FormulaConfig)
class FormulaConfigInput(StrawberryPydanticType[FormulaConfig]):
    formula: str


@pydantic_input(model=SimpleConfig)
class SimpleConfigInput(StrawberryPydanticType[SimpleConfig]):
    node_class: str


@pydantic_input(model=ActionConfig)
class ActionConfigInput(StrawberryPydanticType[ActionConfig]):
    node_class: str
    decision_level: auto
    group: auto
    parent: auto
    no_effect_value: auto


@sb.input
class InputPortInput:
    id: UUID | None = None
    label: str | None = None
    quantity: str | None = None
    unit: str | None = None
    multi: bool = False
    required_dimensions: list[str] | None = None
    supported_dimensions: list[str] | None = None


@sb.input
class OutputPortInput:
    id: UUID | None = None
    label: str | None = None
    quantity: str | None = None
    unit: str
    column_id: str | None = None
    is_editable: bool = True
    dimensions: list[str] | None = None


@sb.input
class OutputMetricInput:
    id: str
    label: str | None = None
    quantity: str | None = None
    unit: str
    column_id: str | None = None
    port_id: UUID | None = None


@sb.input
class PipelineOperationInput:
    operation: str


@pydantic_input(model=PipelineConfig)
class PipelineConfigInput(StrawberryPydanticType[PipelineConfig]):
    operations: list[PipelineOperationInput]


@sb.input(one_of=True)
class NodeConfigInput:
    formula: Maybe[FormulaConfigInput]
    simple: Maybe[SimpleConfigInput]
    action: Maybe[ActionConfigInput]
    pipeline: Maybe[PipelineConfigInput]


type AnyTypeConfig = Some[FormulaConfigInput] | Some[SimpleConfigInput] | Some[ActionConfigInput] | Some[PipelineConfigInput]


@sb.input(name='CreateNodeInput')
class CreateNodeInput:
    identifier: sb.ID
    name: str = ''
    kind: NodeKind = NodeKind.FORMULA
    color: str | None = None
    order: int | None = None
    is_visible: bool = True
    is_outcome: bool = False
    short_name: str | None = None
    description: str | None = None
    node_group: sb.ID | None = None
    allow_nulls: bool = False
    minimum_year: int | None = None
    input_ports: list[InputPortInput] | None = None
    output_ports: list[OutputPortInput] | None = None
    output_metrics: list[OutputMetricInput] | None = None
    input_dimensions: list[str] | None = None
    output_dimensions: list[str] | None = None
    input_datasets: sb.scalars.JSON | None = None
    params: sb.scalars.JSON | None = None
    tags: list[str] | None = None
    i18n: sb.scalars.JSON | None = None

    config: NodeConfigInput


@sb.input
class UpdateNodeInput:
    name: Maybe[str]
    color: Maybe[str]
    is_visible: Maybe[bool]
    is_outcome: Maybe[bool]


@sb.input
class CreateEdgeInput:
    instance_id: sb.ID
    from_node_id: str
    to_node_id: str
    from_port: str = 'output'
    to_port: str | None = None
    transformations: sb.scalars.JSON | None = None


@sb.input
class CreateScenarioInput:
    instance_id: sb.ID
    identifier: str
    name: str
    kind: str = ''
    all_actions_enabled: bool = False


@sb.input
class UpdateScenarioInput:
    scenario_id: sb.ID
    name: str | None = sb.UNSET
    description: str | None = sb.UNSET
    kind: str | None = sb.UNSET
    all_actions_enabled: bool | None = sb.UNSET
    parameter_overrides: sb.scalars.JSON | None = sb.UNSET


@sb.input
class CreateDimensionCategoryInput(SiblingPositionInputMixin):
    dimension_id: UUID
    label: str
    id: Maybe[UUID]
    identifier: Maybe[str]


@sb.input
class UpdateDimensionCategoryInput(SiblingPositionInputMixin):
    category_id: UUID
    identifier: Maybe[str]
    label: Maybe[str]


@sb.input
class UpdateDimensionInput:
    dimension_id: UUID
    name: Maybe[str]


@sb.type(name='ModelNodePayload')
class NodePayload:
    ok: bool
    node: NodeInterface | None = sb.field(graphql_type=Annotated['NodeInterface', sb.lazy('nodes.schema')])


@sb.type(name='ModelEdgePayload')
class EdgePayload:
    ok: bool
    edge: NodeEdgeType | None = None


@sb.type(name='ModelScenarioPayload')
class ScenarioPayload:
    ok: bool
    scenario: ScenarioType | None = None


@sb.type(name='ModelDeletePayload')
class DeletePayload:
    ok: bool


@sb.type
class ModelEditorQuery:
    @sb.field(description='Fetch a complete model instance with all nodes, edges, and scenarios')
    @staticmethod
    def model_instance(info: gql.Info, instance_id: sb.ID) -> InstanceType:
        ic = _get_instance_config(info, instance_id)
        return _resolve_model_instance(ic)


def is_maybe_set[T](maybe: Some[T] | None) -> TypeGuard[Some[T]]:
    return maybe is not None and maybe is not sb.UNSET


def _generated_port_id(node_identifier: str, direction: str, key: str) -> UUID:
    return uuid5(NAMESPACE_URL, f'kausal-paths:node-port:{node_identifier}:{direction}:{key}')


def _input_port_to_def(node_identifier: str, index: int, port: InputPortInput) -> InputPortDef:
    key = port.label or port.quantity or str(index)
    return InputPortDef(
        id=port.id or _generated_port_id(node_identifier, 'input', key),
        label=port.label,
        quantity=port.quantity,
        unit=unit_registry.parse_units(port.unit) if port.unit is not None else None,
        multi=port.multi,
        required_dimensions=port.required_dimensions or [],
        supported_dimensions=port.supported_dimensions or [],
    )


def _output_port_to_def(node_identifier: str, index: int, port: OutputPortInput) -> OutputPortDef:
    key = port.column_id or port.label or port.quantity or str(index)
    return OutputPortDef(
        id=port.id or _generated_port_id(node_identifier, 'output', key),
        label=port.label,
        quantity=port.quantity,
        unit=unit_registry.parse_units(port.unit),
        column_id=port.column_id,
        is_editable=port.is_editable,
        dimensions=port.dimensions or [],
    )


def _output_metric_to_port_def(node_identifier: str, metric: OutputMetricInput, dimensions: list[str]) -> OutputPortDef:
    column_id = metric.column_id or metric.id
    return OutputPortDef(
        id=metric.port_id or _generated_port_id(node_identifier, 'output', metric.id),
        label=metric.label,
        quantity=metric.quantity,
        unit=unit_registry.parse_units(metric.unit),
        column_id=column_id,
        dimensions=dimensions,
    )


def _parse_input_datasets(info: gql.Info, raw: Any) -> list[InputDatasetDef]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise GraphQLValidationError(info, 'inputDatasets must be a list')
    adapter = TypeAdapter(list[InputDatasetDef])
    try:
        return adapter.validate_python(raw)
    except PydanticValidationError as exc:
        raise GraphQLValidationError(info, f'Invalid inputDatasets: {exc}') from exc


def _node_class_path(node_class: str, kind: NodeKind) -> str:
    if node_class.startswith('nodes.'):
        return node_class
    if kind == NodeKind.ACTION:
        return f'nodes.actions.{node_class}'
    return f'nodes.{node_class}'


def _allowed_parameter_templates(node_class: str, kind: NodeKind) -> dict[str, Any]:
    try:
        cls = import_string(_node_class_path(node_class, kind))
    except ImportError:
        return {}
    return {p.local_id: p for p in getattr(cls, 'allowed_parameters', [])}


def _normalize_param_config(raw_param: dict[str, Any]) -> dict[str, Any]:
    param = dict(raw_param)
    if 'localId' in param and 'local_id' not in param:
        param['local_id'] = param.pop('localId')
    if 'id' in param and 'local_id' not in param:
        param['local_id'] = param.pop('id')
    for camel, snake in (
        ('isCustomized', 'is_customized'),
        ('isCustomizable', 'is_customizable'),
        ('isVisible', 'is_visible'),
        ('minValue', 'min_value'),
        ('maxValue', 'max_value'),
    ):
        if camel in param and snake not in param:
            param[snake] = param.pop(camel)
    return param


def _raw_param_configs(info: gql.Info, raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [{'local_id': key, 'value': value} for key, value in raw.items()]
    if isinstance(raw, list):
        if not all(isinstance(raw_param, dict) for raw_param in raw):
            raise GraphQLValidationError(info, 'Each parameter config must be an object')
        return cast('list[dict[str, Any]]', raw)
    raise GraphQLValidationError(info, 'params must be an object or a list')


def _infer_param_type(info: gql.Info, param_id: str, value: Any) -> str:
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, int | float):
        return 'number'
    if isinstance(value, str):
        return 'string'
    raise GraphQLValidationError(info, f'Could not infer type for parameter "{param_id}"')


def _build_custom_param(
    info: gql.Info,
    param_id: str,
    param_config: dict[str, Any],
    value: Any,
    value_provided: bool,
) -> BoolParameter | NumberParameter | StringParameter:
    if 'type' not in param_config:
        param_config['type'] = _infer_param_type(info, param_id, value)
    param_type = param_config['type']
    param_cls = {'bool': BoolParameter, 'number': NumberParameter, 'string': StringParameter}.get(param_type)
    if param_cls is None:
        raise GraphQLValidationError(info, f'Unsupported parameter type "{param_type}" for "{param_id}"')
    param = param_cls(**param_config)
    if value_provided:
        param.set(value, notify=False)
    return param


def _parse_params(info: gql.Info, raw: Any, type_config: Any, kind: NodeKind) -> list[Any]:
    node_class = getattr(type_config, 'node_class', None)
    templates = _allowed_parameter_templates(node_class, kind) if isinstance(node_class, str) else {}
    params = []
    for raw_param in _raw_param_configs(info, raw):
        param_config = _normalize_param_config(raw_param)
        param_id = param_config.get('local_id')
        if not isinstance(param_id, str):
            raise GraphQLValidationError(info, 'Each parameter config must contain id or localId')
        value_provided = 'value' in param_config
        value = param_config.pop('value', None)

        template = templates.get(param_id)
        if template is not None:
            param = template.copy(**param_config)
            if value_provided:
                param.set(value, notify=False)
            params.append(param)
            continue

        params.append(_build_custom_param(info, param_id, param_config, value, value_provided))
    return params


@sb.type
class InstanceEditorMutation:
    instance: sb.Private[InstanceConfig]
    type Me = InstanceEditorMutation

    @gql.mutation(description='Create a new node in the model', graphql_type=AnyNodeType)
    @staticmethod
    def create_node(info: gql.Info, root: sb.Parent[Me], input: CreateNodeInput) -> Node:
        from nodes.change_ops import change_operation, record_change

        ic = root.instance
        if not NodeConfig.gql_create_allowed(info, ic):
            raise PermissionDeniedError(info, 'Permission denied for create')

        maybe_type_config: AnyTypeConfig | None
        if input.kind == NodeKind.FORMULA:
            maybe_type_config = input.config.formula
        elif input.kind == NodeKind.SIMPLE:
            maybe_type_config = input.config.simple
        elif input.kind == NodeKind.ACTION:
            maybe_type_config = input.config.action
        elif input.kind == NodeKind.PIPELINE:
            maybe_type_config = input.config.pipeline
        else:
            raise GraphQLError(f'Invalid node kind: {input.kind}')

        if maybe_type_config is None:
            raise GraphQLValidationError(info, 'Invalid node type config')

        type_config = maybe_type_config.value

        input_dimensions = input.input_dimensions or []
        output_dimensions = input.output_dimensions or []
        input_ports = [
            _input_port_to_def(str(input.identifier), index, port) for index, port in enumerate(input.input_ports or [])
        ]
        output_ports = [
            _output_port_to_def(str(input.identifier), index, port) for index, port in enumerate(input.output_ports or [])
        ]
        output_ports.extend(
            _output_metric_to_port_def(str(input.identifier), metric, output_dimensions) for metric in input.output_metrics or []
        )
        if not output_ports:
            raise GraphQLValidationError(info, 'At least one outputPort or outputMetric must be provided')

        spec = NodeSpec(
            kind=input.kind,
            type_config=type_config.to_pydantic(),
            short_name=input.short_name,
            description=input.description,
            is_outcome=input.is_outcome,
            node_group=input.node_group,
            allow_nulls=input.allow_nulls,
            minimum_year=input.minimum_year,
            input_ports=input_ports,
            output_ports=output_ports,
            input_datasets=_parse_input_datasets(info, input.input_datasets),
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
        )
        spec.params = _parse_params(info, input.params, spec.type_config, input.kind)
        spec.extra.tags = input.tags or []

        nc = ic.nodes.filter(identifier=input.identifier).first()
        if nc is not None:
            raise GraphQLValidationError(info, 'Node with identifier %s already exists' % input.identifier)

        user = getattr(info.context, 'user', None)
        with change_operation(ic, user=user, action='node.create'):
            nc = ic.nodes.create(
                identifier=input.identifier,
                name=input.name or input.identifier,
                color=input.color or '',
                order=input.order,
                is_visible=input.is_visible,
                description=input.description,
                node_type=NodeKindChoices(input.kind.value),
                i18n=input.i18n or {},
                spec=spec,
            )
            record_change(nc, action='node.create', before=None, after=nc.serializable_data())

        return _resolve_runtime_node(ic, nc.pk)

    @gql.mutation(description='Update an existing node', graphql_type=AnyNodeType)
    @staticmethod
    def update_node(info: gql.Info, root: sb.Parent[Me], node_id: sb.ID, input: UpdateNodeInput) -> Node:
        from nodes.change_ops import change_operation, record_change

        ic = root.instance
        nc = get_or_error(info, ic.nodes.get_queryset(), id=node_id)

        user = getattr(info.context, 'user', None)
        with change_operation(ic, user=user, action='node.update'):
            before = nc.serializable_data()

            spec = nc.spec
            updates: dict[str, object] = {}
            assert spec is not None
            if is_maybe_set(input.name):
                spec.name = input.name.value
                updates['name'] = input.name.value
            if is_maybe_set(input.color):
                spec.color = input.color.value
                updates['color'] = input.color.value
            if is_maybe_set(input.is_visible):
                spec.is_visible = input.is_visible.value
                updates['is_visible'] = input.is_visible.value
            if is_maybe_set(input.is_outcome):
                spec.is_outcome = input.is_outcome.value
            nc.spec = spec
            updates['spec'] = spec
            NodeConfig.objects.filter(pk=nc.pk).update(**updates)
            # QuerySet.update() bypasses the instance; refresh so snapshot_data()
            # reflects the committed state.
            nc.refresh_from_db()
            record_change(nc, action='node.update', before=before, after=nc.serializable_data())

        return _resolve_runtime_node(nc.instance, nc.pk)

    @gql.mutation(description='Delete a node and its edges')
    @staticmethod
    def delete_node(root: sb.Parent[Me], info: gql.Info, node_id: sb.ID) -> None:
        from django.db.models import Q

        from nodes.change_ops import change_operation, record_change
        from nodes.models import DatasetPort, NodeEdge

        ic = root.instance
        nc = get_or_error(info, ic.nodes.get_queryset(), id=node_id)
        if not nc.gql_action_allowed(info, 'delete'):
            raise PermissionDeniedError(info, 'Permission denied for delete')

        user = getattr(info.context, 'user', None)
        with change_operation(ic, user=user, action='node.delete'):
            # Log cascade-delete entries BEFORE the DB CASCADE wipes the rows,
            # while pks are still valid. After the ``nc.delete()`` call below,
            # only the IMLE rows carry the pre-state.
            affected_edges = list(
                NodeEdge.objects.filter(Q(from_node=nc) | Q(to_node=nc)).select_related('from_node', 'to_node'),
            )
            for edge in affected_edges:
                record_change(
                    edge,
                    action='node.edges.delete',
                    before=edge.serializable_data(),
                    after=None,
                )

            affected_ports = list(
                DatasetPort.objects.filter(node=nc).select_related('node', 'dataset', 'metric'),
            )
            for port in affected_ports:
                record_change(
                    port,
                    action='node.dataset_ports.delete',
                    before=port.serializable_data(),
                    after=None,
                )

            record_change(
                nc,
                action='node.delete',
                before=nc.serializable_data(),
                after=None,
            )

            nc.delete()

    @sb.field(description='Edit a DB-backed dataset that belongs to this instance')
    @staticmethod
    def dataset_editor(
        info: gql.Info, root: sb.Parent[Me], dataset_id: sb.ID
    ) -> Annotated[
        'DatasetEditorMutation',
        sb.lazy('datasets.graphql.editor'),
    ]:
        from kausal_common.datasets.models import Dataset

        from datasets.graphql.editor import DatasetEditorMutation

        ic = root.instance
        dataset = get_or_error(
            info,
            Dataset.objects.get_queryset().for_instance_config(ic),
            uuid=str(dataset_id),
            for_action='change',
        )
        return DatasetEditorMutation(dataset=dataset, instance=ic)

    @gql.mutation(description='Create a new edge between nodes')
    @staticmethod
    def create_edge(info: gql.Info, input: CreateEdgeInput) -> NodeEdgeType:
        from nodes.models import NodeConfig, NodeEdge

        ic = _get_instance_config(info, input.instance_id)
        if ic.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        try:
            from_node = NodeConfig.objects.get(instance=ic, identifier=input.from_node_id)
            to_node = NodeConfig.objects.get(instance=ic, identifier=input.to_node_id)
        except NodeConfig.DoesNotExist:
            raise GraphQLError('Source or target node not found') from None

        from_port = _resolve_source_port(info, from_node, input.from_port)
        to_port = _resolve_target_port(info, to_node, input.to_port)
        _validate_edge_ports(info, from_node, from_port, to_node, to_port)

        with transaction.atomic():
            edge = NodeEdge.objects.create(
                instance=ic,
                from_node=from_node,
                from_port=from_port,
                to_node=to_node,
                to_port=to_port,
                transformations=input.transformations or [],
            )

        return NodeEdgeType.from_node_edge(edge)

    @gql.mutation(description='Delete an edge')
    @staticmethod
    def delete_edge(root: sb.Parent[Me], info: gql.Info, edge_id: sb.ID) -> None:
        from nodes.models import NodeEdge

        ic = root.instance
        try:
            edge = NodeEdge.objects.get(instance=ic, pk=edge_id)
        except NodeEdge.DoesNotExist:
            raise GraphQLError('Edge not found') from None

        if ic.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        with transaction.atomic():
            edge.delete()

    # -- Dimension mutations --------------------------------------------------

    @staticmethod
    def _get_dimension_scope(info: gql.Info, ic: InstanceConfig, dimension_id: UUID) -> DimensionScope:
        from kausal_common.datasets.models import DimensionScope

        scope = (
            DimensionScope.objects
            .get_queryset()
            .for_instance_config(ic)
            .filter(dimension__uuid=dimension_id)
            .select_related('dimension')
            .first()
        )
        if scope is None:
            raise NotFoundError(info, f'Dimension "{dimension_id}" not found in instance "{ic.identifier}"')
        return scope

    @staticmethod
    def _set_sibling_hints(info: gql.Info, target: OrderedModel, input: SiblingPositionInputMixin) -> None:
        """Transfer previousSibling/nextSibling from a mutation input to an OrderedModel instance."""
        prev = input.previous_sibling if is_maybe_set(input.previous_sibling) else None
        nxt = input.next_sibling if is_maybe_set(input.next_sibling) else None
        if prev is not None:
            target.previous_sibling = UUID(prev.value)
        if nxt is not None:
            target.next_sibling = UUID(nxt.value)

    @gql.mutation(description='Update a dimension (e.g. rename)')
    def update_dimension(self, info: gql.Info, root: sb.Parent[Me], input: UpdateDimensionInput) -> DimensionType:
        ic = root.instance
        scope = self._get_dimension_scope(info, ic, input.dimension_id)
        dim = scope.dimension

        updates: dict[str, object] = {}
        if is_maybe_set(input.name):
            updates['name'] = input.name.value
        if updates:
            type(dim).objects.filter(pk=dim.pk).update(**updates)
            dim.refresh_from_db()
        return DimensionType.from_scope(scope)

    @gql.mutation(description='Add categories to a dimension')
    def create_dimension_categories(
        self, info: gql.Info, root: sb.Parent[Me], input: list[CreateDimensionCategoryInput]
    ) -> DimensionType:
        from kausal_common.datasets.models import DimensionCategory

        if not input:
            raise GraphQLValidationError(info, 'At least one category input is required')

        ic = root.instance
        # All inputs must target the same dimension
        dim_ids = {item.dimension_id for item in input}
        if len(dim_ids) > 1:
            raise GraphQLValidationError(info, 'All categories in a batch must target the same dimension')

        scope = self._get_dimension_scope(info, ic, input[0].dimension_id)
        dim = scope.dimension

        created: list[DimensionCategory] = []
        with transaction.atomic():
            for item in input:
                cat_uuid = item.id.value if is_maybe_set(item.id) else uuid4()
                identifier = item.identifier.value if is_maybe_set(item.identifier) else None

                if identifier is not None and dim.categories.filter(identifier=identifier).exists():
                    raise GraphQLValidationError(
                        info,
                        f'Category with identifier "{identifier}" already exists in dimension "{scope.identifier}"',
                    )

                cat = DimensionCategory(
                    dimension=dim,
                    uuid=cat_uuid,
                    identifier=identifier,
                    label=item.label,
                )
                cat.save()
                self._set_sibling_hints(info, cat, item)
                created.append(cat)

            try:
                DimensionCategory.finalize_sibling_order(dim.categories.all(), hinted=created)
            except ValueError as e:
                raise GraphQLValidationError(info, str(e)) from e

        return DimensionType.from_scope(scope)

    @staticmethod
    def _apply_category_update(info: gql.Info, item: UpdateDimensionCategoryInput) -> DimensionCategoryModel:
        """Look up and apply field updates for a single category. Returns the updated instance."""
        from kausal_common.datasets.models import DimensionCategory

        cat = DimensionCategory.objects.filter(uuid=item.category_id).select_related('dimension').first()
        if cat is None:
            raise NotFoundError(info, f'Category "{item.category_id}" not found')

        updates: dict[str, object] = {}
        if is_maybe_set(item.identifier):
            updates['identifier'] = item.identifier.value
        if is_maybe_set(item.label):
            updates['label'] = item.label.value
        if updates:
            DimensionCategory.objects.filter(pk=cat.pk).update(**updates)
        return cat

    @gql.mutation(description='Update dimension categories')
    def update_dimension_categories(
        self, info: gql.Info, root: sb.Parent[Me], input: list[UpdateDimensionCategoryInput]
    ) -> DimensionType:
        from kausal_common.datasets.models import DimensionCategory, DimensionScope

        if not input:
            raise GraphQLValidationError(info, 'At least one category input is required')

        ic = root.instance
        dim = None
        updated: list[DimensionCategory] = []

        for item in input:
            cat = self._apply_category_update(info, item)

            if dim is None:
                dim = cat.dimension
                scope = DimensionScope.objects.for_instance_config(ic).filter(dimension=dim).first()
                if scope is None:
                    raise NotFoundError(info, 'Category does not belong to this instance')
            elif cat.dimension_id != dim.pk:
                raise GraphQLValidationError(info, 'All categories in a batch must belong to the same dimension')

            self._set_sibling_hints(info, cat, item)
            updated.append(cat)

        assert dim is not None
        try:
            DimensionCategory.finalize_sibling_order(dim.categories.all(), hinted=updated)
        except ValueError as e:
            raise GraphQLValidationError(info, str(e)) from e

        scope = self._get_dimension_scope(info, ic, dim.uuid)
        return DimensionType.from_scope(scope)

    @gql.mutation(description='Delete a dimension category')
    def delete_dimension_category(self, info: gql.Info, root: sb.Parent[Me], category_id: UUID) -> None:
        from kausal_common.datasets.models import DimensionCategory, DimensionScope

        cat = DimensionCategory.objects.filter(uuid=category_id).select_related('dimension').first()
        if cat is None:
            raise NotFoundError(info, f'Category "{category_id}" not found')

        ic = root.instance
        scope = DimensionScope.objects.for_instance_config(ic).filter(dimension=cat.dimension).first()
        if scope is None:
            raise GraphQLError('Category does not belong to this instance')

        cat.delete()

    @gql.mutation(description='Create a new scenario')
    @staticmethod
    def create_scenario(info: gql.Info, input: CreateScenarioInput) -> ScenarioType:
        raise GraphQLError('Scenario mutations not yet implemented')

    @gql.mutation(description='Update a scenario')
    @staticmethod
    def update_scenario(info: gql.Info, input: UpdateScenarioInput) -> ScenarioType:
        raise GraphQLError('Scenario mutations not yet implemented')

    @gql.mutation(description='Delete a scenario')
    @staticmethod
    def delete_scenario(info: gql.Info, scenario_id: sb.ID) -> None:
        raise GraphQLError('Scenario mutations not yet implemented')

    @gql.mutation(description='Publish the current model state as a new revision')
    @staticmethod
    def publish_model_instance(info: gql.Info, instance_id: sb.ID) -> InstanceType:
        ic = _get_instance_config(info, instance_id)
        if ic.config_source != 'database':
            raise GraphQLError('Cannot publish YAML-sourced instances')

        user = getattr(info.context, 'user', None)
        ic.publish_instance(user=user)
        ic.refresh_from_db()
        return _resolve_model_instance(ic)

    @sb.mutation(description='Revert draft to the last published revision')
    @staticmethod
    def revert_model_instance(info: gql.Info, instance_id: sb.ID) -> InstanceType:
        ic = _get_instance_config(info, instance_id)
        if ic.config_source != 'database':
            raise GraphQLError('Cannot revert YAML-sourced instances')

        with transaction.atomic():
            ic.revert_to_published()
        return _resolve_model_instance(ic)


@sb.type
class ModelEditorMutation:
    @sb.field(description='Edit the nodes and edges of an instance')
    @staticmethod
    def instance_editor(info: gql.Info, instance_id: sb.ID) -> InstanceEditorMutation:
        ic = _get_instance_config(info, instance_id)
        if not ic.gql_action_allowed(info, 'change'):
            raise PermissionDeniedError(info, 'Model editor access denied')
        if ic.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')
        return InstanceEditorMutation(instance=ic)
