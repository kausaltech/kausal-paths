"""
Strawberry GraphQL schema for the Trailhead model editor.

Provides queries and mutations for reading and editing DB-sourced
model instances (NodeConfig, NodeEdge, ActionGroup, Scenario).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import strawberry as sb
from django.db import transaction
from graphql import GraphQLError

from paths import gql

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


# --- Types ---


@sb.type(name='ModelActionGroup')
class ActionGroupType:
    id: sb.ID
    identifier: str
    name: str
    color: str
    order: int


@sb.type(name='ModelEdge')
class NodeEdgeType:
    id: sb.ID
    from_node_id: str
    from_port: str
    to_node_id: str
    to_port: str
    transformations: sb.scalars.JSON
    tags: list[str]


@sb.type(name='ModelDatasetPort')
class DatasetPortType:
    id: sb.ID
    node_id: str
    port_id: str
    dataset_id: str
    metric_id: str | None


@sb.type(name='ModelScenario')
class ScenarioType:
    id: sb.ID
    identifier: str
    name: str
    description: str
    kind: str
    all_actions_enabled: bool
    parameter_overrides: sb.scalars.JSON


@sb.type(name='ModelNode')
class NodeConfigType:
    id: sb.ID
    identifier: str
    name: str | None
    node_type: str | None
    unit: str | None
    quantity: str | None
    color: str
    order: int | None
    is_visible: bool
    is_outcome: bool
    node_group: str | None
    input_ports: sb.scalars.JSON
    output_ports: sb.scalars.JSON
    pipeline: sb.scalars.JSON | None
    formula: str | None
    decision_level: str | None
    action_group_id: str | None
    params: sb.scalars.JSON | None
    extra: sb.scalars.JSON


@sb.type(name='ModelInstance')
class ModelInstanceType:
    id: sb.ID
    identifier: str
    config_source: str
    target_year: int | None
    reference_year: int | None
    minimum_historical_year: int | None
    maximum_historical_year: int | None
    model_end_year: int | None
    emission_unit: str | None
    features: sb.scalars.JSON
    parameters: sb.scalars.JSON
    nodes: list[NodeConfigType]
    edges: list[NodeEdgeType]
    action_groups: list[ActionGroupType]
    scenarios: list[ScenarioType]
    dataset_ports: list[DatasetPortType]


# --- Resolvers ---


def _get_instance_config(info: gql.Info, instance_id: sb.ID) -> InstanceConfig:
    return gql.get_ic_or_error(info, str(instance_id))


def _resolve_model_instance(ic: InstanceConfig) -> ModelInstanceType:
    from nodes.models import ActionGroup, DatasetPort, NodeConfig, NodeEdge, Scenario

    nodes = list(NodeConfig.objects.filter(instance=ic).select_related('action_group'))
    edges = list(NodeEdge.objects.filter(instance=ic))
    action_groups = list(ActionGroup.objects.filter(instance=ic).order_by('order'))
    scenarios = list(Scenario.objects.filter(instance=ic))
    dataset_ports = list(DatasetPort.objects.filter(instance=ic))

    return ModelInstanceType(
        id=sb.ID(str(ic.pk)),
        identifier=ic.identifier,
        config_source=ic.config_source,
        target_year=ic.target_year,
        reference_year=ic.reference_year,
        minimum_historical_year=ic.minimum_historical_year,
        maximum_historical_year=ic.maximum_historical_year,
        model_end_year=ic.model_end_year,
        emission_unit=ic.emission_unit,
        features=ic.features,
        parameters=ic.parameters,
        nodes=[
            NodeConfigType(
                id=sb.ID(str(nc.pk)),
                identifier=nc.identifier,
                name=nc.name,
                node_type=nc.node_type,
                unit=nc.unit,
                quantity=nc.quantity,
                color=nc.color,
                order=nc.order,
                is_visible=nc.is_visible,
                is_outcome=nc.is_outcome,
                node_group=nc.node_group,
                input_ports=nc.input_ports,
                output_ports=nc.output_ports,
                pipeline=nc.pipeline,
                formula=nc.formula,
                decision_level=nc.decision_level,
                action_group_id=nc.action_group.identifier if nc.action_group else None,
                params=nc.params,
                extra=nc.extra,
            )
            for nc in nodes
        ],
        edges=[
            NodeEdgeType(
                id=sb.ID(str(e.pk)),
                from_node_id=e.from_node.identifier,
                from_port=e.from_port,
                to_node_id=e.to_node.identifier,
                to_port=e.to_port,
                transformations=e.transformations,
                tags=e.tags or [],
            )
            for e in edges
        ],
        action_groups=[
            ActionGroupType(
                id=sb.ID(str(ag.pk)),
                identifier=ag.identifier,
                name=ag.name,
                color=ag.color,
                order=ag.order,
            )
            for ag in action_groups
        ],
        scenarios=[
            ScenarioType(
                id=sb.ID(str(s.pk)),
                identifier=s.identifier,
                name=s.name,
                description=s.description,
                kind=s.kind,
                all_actions_enabled=s.all_actions_enabled,
                parameter_overrides=s.parameter_overrides,
            )
            for s in scenarios
        ],
        dataset_ports=[
            DatasetPortType(
                id=sb.ID(str(dp.pk)),
                node_id=dp.node.identifier,
                port_id=dp.port_id,
                dataset_id=str(dp.dataset_id),
                metric_id=str(dp.metric_id) if dp.metric_id else None,
            )
            for dp in dataset_ports
        ],
    )


# --- Input types ---


@sb.input
class CreateNodeInput:
    instance_id: sb.ID
    identifier: str
    name: str | None = None
    node_type: str = 'formula'
    unit: str | None = None
    quantity: str | None = None
    color: str = ''
    is_outcome: bool = False
    node_group: str | None = None


@sb.input
class UpdateNodeInput:
    node_id: sb.ID
    name: str | None = sb.UNSET
    unit: str | None = sb.UNSET
    quantity: str | None = sb.UNSET
    color: str | None = sb.UNSET
    is_visible: bool | None = sb.UNSET
    is_outcome: bool | None = sb.UNSET
    node_group: str | None = sb.UNSET
    pipeline: sb.scalars.JSON | None = sb.UNSET
    formula: str | None = sb.UNSET
    params: sb.scalars.JSON | None = sb.UNSET


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


# --- Mutation payloads ---


@sb.type(name='ModelNodePayload')
class NodePayload:
    ok: bool
    node: NodeConfigType | None = None


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


# --- Query ---


@sb.type
class SBTrailheadQuery:
    @sb.field(description='Fetch a complete model instance with all nodes, edges, and scenarios')
    @staticmethod
    def model_instance(info: gql.Info, instance_id: sb.ID) -> ModelInstanceType:
        ic = _get_instance_config(info, instance_id)
        return _resolve_model_instance(ic)


# --- Mutations ---


@sb.type
class SBTrailheadMutation:
    @sb.mutation(description='Create a new node in the model')
    @staticmethod
    def create_node(info: gql.Info, input: CreateNodeInput) -> NodePayload:
        from nodes.models import NodeConfig

        ic = _get_instance_config(info, input.instance_id)
        if ic.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        with transaction.atomic():
            nc = NodeConfig.objects.create(
                instance=ic,
                identifier=input.identifier,
                name=input.name,
                node_type=input.node_type,
                unit=input.unit,
                quantity=input.quantity,
                color=input.color,
                is_outcome=input.is_outcome,
                node_group=input.node_group or '',
            )

        return NodePayload(
            ok=True,
            node=NodeConfigType(
                id=sb.ID(str(nc.pk)),
                identifier=nc.identifier,
                name=nc.name,
                node_type=nc.node_type,
                unit=nc.unit,
                quantity=nc.quantity,
                color=nc.color,
                order=nc.order,
                is_visible=nc.is_visible,
                is_outcome=nc.is_outcome,
                node_group=nc.node_group,
                input_ports=nc.input_ports,
                output_ports=nc.output_ports,
                pipeline=nc.pipeline,
                formula=nc.formula,
                decision_level=nc.decision_level,
                action_group_id=None,
                params=nc.params,
                extra=nc.extra,
            ),
        )

    @sb.mutation(description='Update an existing node')
    @staticmethod
    def update_node(info: gql.Info, input: UpdateNodeInput) -> NodePayload:
        from nodes.models import NodeConfig

        try:
            nc = NodeConfig.objects.select_related('action_group').get(pk=input.node_id)
        except NodeConfig.DoesNotExist:
            raise GraphQLError('Node not found') from None

        if nc.instance.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        with transaction.atomic():
            for field_name in (
                'name',
                'unit',
                'quantity',
                'color',
                'is_visible',
                'is_outcome',
                'node_group',
                'pipeline',
                'formula',
                'params',
            ):
                val = getattr(input, field_name)
                if val is not sb.UNSET:
                    setattr(nc, field_name, val)
            nc.save()

        return NodePayload(
            ok=True,
            node=NodeConfigType(
                id=sb.ID(str(nc.pk)),
                identifier=nc.identifier,
                name=nc.name,
                node_type=nc.node_type,
                unit=nc.unit,
                quantity=nc.quantity,
                color=nc.color,
                order=nc.order,
                is_visible=nc.is_visible,
                is_outcome=nc.is_outcome,
                node_group=nc.node_group,
                input_ports=nc.input_ports,
                output_ports=nc.output_ports,
                pipeline=nc.pipeline,
                formula=nc.formula,
                decision_level=nc.decision_level,
                action_group_id=nc.action_group.identifier if nc.action_group else None,
                params=nc.params,
                extra=nc.extra,
            ),
        )

    @sb.mutation(description='Delete a node and its edges')
    @staticmethod
    def delete_node(info: gql.Info, node_id: sb.ID) -> DeletePayload:
        from nodes.models import NodeConfig

        try:
            nc = NodeConfig.objects.get(pk=node_id)
        except NodeConfig.DoesNotExist:
            raise GraphQLError('Node not found') from None

        if nc.instance.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        with transaction.atomic():
            nc.delete()

        return DeletePayload(ok=True)

    @sb.mutation(description='Create a new edge between nodes')
    @staticmethod
    def create_edge(info: gql.Info, input: CreateEdgeInput) -> EdgePayload:
        from nodes.models import NodeConfig, NodeEdge

        ic = _get_instance_config(info, input.instance_id)
        if ic.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        try:
            from_node = NodeConfig.objects.get(instance=ic, identifier=input.from_node_id)
            to_node = NodeConfig.objects.get(instance=ic, identifier=input.to_node_id)
        except NodeConfig.DoesNotExist:
            raise GraphQLError('Source or target node not found') from None

        to_port = input.to_port or f'from_{input.from_node_id}'

        with transaction.atomic():
            edge = NodeEdge.objects.create(
                instance=ic,
                from_node=from_node,
                from_port=input.from_port,
                to_node=to_node,
                to_port=to_port,
                transformations=input.transformations or [],
            )

        return EdgePayload(
            ok=True,
            edge=NodeEdgeType(
                id=sb.ID(str(edge.pk)),
                from_node_id=from_node.identifier,
                from_port=edge.from_port,
                to_node_id=to_node.identifier,
                to_port=edge.to_port,
                transformations=edge.transformations,
                tags=edge.tags or [],
            ),
        )

    @sb.mutation(description='Delete an edge')
    @staticmethod
    def delete_edge(info: gql.Info, edge_id: sb.ID) -> DeletePayload:
        from nodes.models import NodeEdge

        try:
            edge = NodeEdge.objects.get(pk=edge_id)
        except NodeEdge.DoesNotExist:
            raise GraphQLError('Edge not found') from None

        if edge.instance.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        with transaction.atomic():
            edge.delete()

        return DeletePayload(ok=True)

    @sb.mutation(description='Create a new scenario')
    @staticmethod
    def create_scenario(info: gql.Info, input: CreateScenarioInput) -> ScenarioPayload:
        from nodes.models import Scenario

        ic = _get_instance_config(info, input.instance_id)
        if ic.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        with transaction.atomic():
            s = Scenario.objects.create(
                instance=ic,
                identifier=input.identifier,
                name=input.name,
                kind=input.kind,
                all_actions_enabled=input.all_actions_enabled,
            )

        return ScenarioPayload(
            ok=True,
            scenario=ScenarioType(
                id=sb.ID(str(s.pk)),
                identifier=s.identifier,
                name=s.name,
                description=s.description,
                kind=s.kind,
                all_actions_enabled=s.all_actions_enabled,
                parameter_overrides=s.parameter_overrides,
            ),
        )

    @sb.mutation(description='Update a scenario')
    @staticmethod
    def update_scenario(info: gql.Info, input: UpdateScenarioInput) -> ScenarioPayload:
        from nodes.models import Scenario

        try:
            s = Scenario.objects.get(pk=input.scenario_id)
        except Scenario.DoesNotExist:
            raise GraphQLError('Scenario not found') from None

        if s.instance.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        with transaction.atomic():
            for field_name in ('name', 'description', 'kind', 'all_actions_enabled', 'parameter_overrides'):
                val = getattr(input, field_name)
                if val is not sb.UNSET:
                    setattr(s, field_name, val)
            s.save()

        return ScenarioPayload(
            ok=True,
            scenario=ScenarioType(
                id=sb.ID(str(s.pk)),
                identifier=s.identifier,
                name=s.name,
                description=s.description,
                kind=s.kind,
                all_actions_enabled=s.all_actions_enabled,
                parameter_overrides=s.parameter_overrides,
            ),
        )

    @sb.mutation(description='Delete a scenario')
    @staticmethod
    def delete_scenario(info: gql.Info, scenario_id: sb.ID) -> DeletePayload:
        from nodes.models import Scenario

        try:
            s = Scenario.objects.get(pk=scenario_id)
        except Scenario.DoesNotExist:
            raise GraphQLError('Scenario not found') from None

        if s.instance.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        with transaction.atomic():
            s.delete()

        return DeletePayload(ok=True)
