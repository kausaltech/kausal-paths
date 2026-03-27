"""
Strawberry GraphQL schema for the model editor.

Provides queries and mutations for reading and editing DB-sourced
model instances (NodeConfig, NodeEdge, ActionGroup, Scenario).
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import strawberry as sb
from django.db import transaction
from graphql import GraphQLError

from paths import gql

from nodes.schema_spec import (
    InstanceSpecType,
    NodeSpecType,
    instance_spec_to_gql,
    node_spec_to_gql,
)

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
    color: str
    order: int | None
    is_visible: bool
    spec: NodeSpecType


@sb.type(name='ModelInstance')
class ModelInstanceType:
    id: sb.ID
    identifier: str
    config_source: str
    live: bool
    has_unpublished_changes: bool
    first_published_at: datetime.datetime | None
    last_published_at: datetime.datetime | None
    spec: InstanceSpecType
    nodes: list[NodeConfigType]
    edges: list[NodeEdgeType]
    dataset_ports: list[DatasetPortType]


# --- Resolvers ---


def _get_instance_config(info: gql.Info, instance_id: sb.ID) -> InstanceConfig:
    from nodes.models import InstanceConfig

    # Direct lookup by all identifier types, bypassing permission filter
    # since model editor access will be gated by its own auth check.
    qs = InstanceConfig.objects.qs.by_all_identifiers(str(instance_id))
    try:
        return qs.get()
    except InstanceConfig.DoesNotExist:
        raise GraphQLError(f'Instance "{instance_id}" not found') from None


def _node_config_to_gql(nc: Any) -> NodeConfigType:
    return NodeConfigType(
        id=sb.ID(str(nc.pk)),
        identifier=nc.identifier,
        name=nc.name,
        color=nc.color,
        order=nc.order,
        is_visible=nc.is_visible,
        spec=node_spec_to_gql(nc.spec),
    )


def _resolve_model_instance(ic: InstanceConfig) -> ModelInstanceType:
    from nodes.models import DatasetPort, NodeConfig, NodeEdge

    nodes = list(NodeConfig.objects.filter(instance=ic))
    edges = list(NodeEdge.objects.filter(instance=ic).select_related('from_node', 'to_node'))
    dataset_ports = list(DatasetPort.objects.filter(instance=ic).select_related('node'))

    return ModelInstanceType(
        id=sb.ID(str(ic.pk)),
        identifier=ic.identifier,
        config_source=ic.config_source,
        live=ic.live,
        has_unpublished_changes=ic.has_unpublished_changes,
        first_published_at=ic.first_published_at,
        last_published_at=ic.last_published_at,
        spec=instance_spec_to_gql(ic.spec),
        nodes=[_node_config_to_gql(nc) for nc in nodes],
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
class ModelEditorQuery:
    @sb.field(description='Fetch a complete model instance with all nodes, edges, and scenarios')
    @staticmethod
    def model_instance(info: gql.Info, instance_id: sb.ID) -> ModelInstanceType:
        ic = _get_instance_config(info, instance_id)
        return _resolve_model_instance(ic)


# --- Mutations ---


@sb.type
class ModelEditorMutation:
    @sb.mutation(description='Create a new node in the model')
    @staticmethod
    def create_node(info: gql.Info, input: CreateNodeInput) -> NodePayload:
        from nodes.models import NodeConfig

        ic = _get_instance_config(info, input.instance_id)
        if ic.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        from nodes.defs.node_defs import FormulaConfig, NodeSpec, SimpleConfig

        spec = NodeSpec(
            type_config=FormulaConfig(formula='') if input.node_type == 'formula' else SimpleConfig(),
            is_outcome=input.is_outcome,
        )

        with transaction.atomic():
            nc = NodeConfig.objects.create(
                instance=ic,
                identifier=input.identifier,
                name=input.name,
                node_type=input.node_type,
                color=input.color,
                spec=spec,
            )

        return NodePayload(ok=True, node=_node_config_to_gql(nc))

    @sb.mutation(description='Update an existing node')
    @staticmethod
    def update_node(info: gql.Info, input: UpdateNodeInput) -> NodePayload:
        from nodes.models import NodeConfig

        try:
            nc = NodeConfig.objects.get(pk=input.node_id)
        except NodeConfig.DoesNotExist:
            raise GraphQLError('Node not found') from None

        if nc.instance.config_source != 'database':
            raise GraphQLError('Cannot edit YAML-sourced instances')

        with transaction.atomic():
            # Direct model fields
            for field_name in ('name', 'color', 'is_visible'):
                val = getattr(input, field_name)
                if val is not sb.UNSET:
                    setattr(nc, field_name, val)
            # Spec fields
            spec = nc.spec
            for field_name in ('is_outcome', 'pipeline', 'params'):
                val = getattr(input, field_name)
                if val is not sb.UNSET:
                    setattr(spec, field_name, val)
            if input.formula is not sb.UNSET:
                from nodes.defs.node_defs import FormulaConfig

                spec.type_config = FormulaConfig(formula=input.formula or '')
            nc.spec = spec
            nc.save()

        return NodePayload(ok=True, node=_node_config_to_gql(nc))

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
        raise GraphQLError('Scenario mutations not yet implemented')

    @sb.mutation(description='Update a scenario')
    @staticmethod
    def update_scenario(info: gql.Info, input: UpdateScenarioInput) -> ScenarioPayload:
        raise GraphQLError('Scenario mutations not yet implemented')

    @sb.mutation(description='Delete a scenario')
    @staticmethod
    def delete_scenario(info: gql.Info, scenario_id: sb.ID) -> DeletePayload:
        raise GraphQLError('Scenario mutations not yet implemented')

    @sb.mutation(description='Publish the current model state as a new revision')
    @staticmethod
    def publish_model_instance(info: gql.Info, instance_id: sb.ID) -> ModelInstanceType:
        ic = _get_instance_config(info, instance_id)
        if ic.config_source != 'database':
            raise GraphQLError('Cannot publish YAML-sourced instances')

        user = getattr(info.context, 'user', None)
        ic.publish_instance(user=user)
        ic.refresh_from_db()
        return _resolve_model_instance(ic)

    @sb.mutation(description='Revert draft to the last published revision')
    @staticmethod
    def revert_model_instance(info: gql.Info, instance_id: sb.ID) -> ModelInstanceType:
        ic = _get_instance_config(info, instance_id)
        if ic.config_source != 'database':
            raise GraphQLError('Cannot revert YAML-sourced instances')

        with transaction.atomic():
            ic.revert_to_published()
        return _resolve_model_instance(ic)
