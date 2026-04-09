"""
Strawberry GraphQL schema for the model editor.

Provides queries and mutations for reading and editing DB-sourced
model instances (NodeConfig, NodeEdge, ActionGroup, Scenario).
"""

from typing import TYPE_CHECKING, Annotated, TypeGuard, cast
from uuid import UUID

import strawberry as sb
from django.db import transaction
from graphql import GraphQLError
from strawberry import Maybe, Some, auto

from kausal_common.strawberry.errors import GraphQLValidationError, PermissionDeniedError
from kausal_common.strawberry.helpers import get_or_error
from kausal_common.strawberry.pydantic import StrawberryPydanticType, pydantic_input

from paths import gql

from nodes.defs import FormulaConfig, SimpleConfig
from nodes.defs.node_defs import ActionConfig, NodeKind, NodeSpec, PipelineConfig
from nodes.models import InstanceConfig, NodeConfig
from nodes.node import Node

from .types.graph import NodeEdgeType, NodePortRef
from .types.instance import InstanceType
from .types.node import AnyNodeType, NodeInterface
from .types.scenario import ScenarioType

if TYPE_CHECKING:
    from nodes.defs.port_def import InputPortDef, OutputPortDef


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
    for port in nc.spec.output_ports:
        if port.id == port_id:
            return port
    return None


def _get_input_port(nc: NodeConfig, port_id: UUID) -> InputPortDef | None:
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
    input_ports = to_node.spec.input_ports
    if len(input_ports) == 1:
        return input_ports[0].id
    raise GraphQLValidationError(
        info,
        f'Target node "{to_node.identifier}" has {len(input_ports)} input ports; toPort must be specified explicitly',
    )


def _resolve_source_port(info: gql.Info, from_node: NodeConfig, from_port: str) -> UUID:
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
    decision_level: auto
    group: auto
    parent: auto
    no_effect_value: auto


@sb.input
class PipelineOperationInput:
    kind: str


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


@pydantic_input(model=NodeSpec, name='CreateNodeInput')
class CreateNodeInput:
    identifier: sb.ID
    name: auto
    kind: auto = NodeKind.FORMULA
    color: auto
    is_outcome: auto
    node_group: sb.ID | None = None
    allow_nulls: auto = False

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


@sb.type
class InstanceEditorMutation:
    instance: sb.Private[InstanceConfig]
    type Me = InstanceEditorMutation

    @gql.mutation(description='Create a new node in the model', graphql_type=AnyNodeType)
    @staticmethod
    def create_node(info: gql.Info, root: sb.Parent[Me], input: CreateNodeInput) -> Node:
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

        spec = NodeSpec(
            type_config=type_config.to_pydantic(),
            is_outcome=input.is_outcome,
            node_group=input.node_group,
            allow_nulls=input.allow_nulls,
        )

        nc = ic.nodes.filter(identifier=input.identifier).first()
        if nc is not None:
            raise GraphQLValidationError(info, 'Node with identifier %s already exists' % input.identifier)
        nc = ic.nodes.create(
            identifier=input.identifier,
            name=input.name or input.identifier,
            color=input.color,
            spec=spec,
        )

        return _resolve_runtime_node(ic, nc.pk)

    @gql.mutation(description='Update an existing node', graphql_type=AnyNodeType)
    @staticmethod
    def update_node(info: gql.Info, root: sb.Parent[Me], node_id: sb.ID, input: UpdateNodeInput) -> Node:
        ic = root.instance
        nc = get_or_error(info, ic.nodes.get_queryset(), id=node_id)

        spec = nc.spec
        updates: dict[str, object] = {}
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
        return _resolve_runtime_node(nc.instance, nc.pk)

    @gql.mutation(description='Delete a node and its edges')
    @staticmethod
    def delete_node(root: sb.Parent[Me], info: gql.Info, node_id: sb.ID) -> None:
        ic = root.instance
        nc = get_or_error(info, ic.nodes.get_queryset(), id=node_id)
        if not nc.gql_action_allowed(info, 'delete'):
            raise PermissionDeniedError(info, 'Permission denied for delete')
        nc.delete()

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
    def delete_edge(info: gql.Info, edge_id: sb.ID) -> None:
        from nodes.models import NodeEdge

        try:
            edge = NodeEdge.objects.get(pk=edge_id)
        except NodeEdge.DoesNotExist:
            raise GraphQLError('Edge not found') from None

        if edge.instance.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        with transaction.atomic():
            edge.delete()

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
