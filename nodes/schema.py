import graphene
from graphql.error import GraphQLError
from wagtail.core.rich_text import expand_db_html

from paths.graphql_helpers import GQLInfo
from pages.models import NodePage
from pages.base import Metric

from . import Node
from .actions import ActionNode
from .constants import BASELINE_VALUE_COLUMN, FORECAST_COLUMN, IMPACT_COLUMN, VALUE_COLUMN
from .scenario import Scenario


class InstanceType(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    target_year = graphene.Int()
    reference_year = graphene.Int()
    minimum_historical_year = graphene.Int()
    maximum_historical_year = graphene.Int()


class YearlyValue(graphene.ObjectType):
    year = graphene.Int()
    value = graphene.Float()


class ForecastMetricType(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    # output_node will be set if the node outputs multiple time-series
    output_node = graphene.Field(lambda: NodeType)
    unit = graphene.Field('paths.schema.UnitType')
    historical_values = graphene.List(YearlyValue)
    forecast_values = graphene.List(YearlyValue)
    baseline_forecast_values = graphene.List(YearlyValue)

    def resolve_historical_values(root: Metric, info):
        return root.get_historical_values()

    def resolve_forecast_values(root: Metric, info):
        return root.get_forecast_values()

    def resolve_baseline_forecast_values(root: Metric, info):
        return root.get_baseline_forecast_values()


class NodeType(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    color = graphene.String()
    unit = graphene.Field('paths.schema.UnitType')
    quantity = graphene.String()
    target_year_goal = graphene.Float()
    is_action = graphene.Boolean()
    input_nodes = graphene.List(lambda: NodeType)
    output_nodes = graphene.List(lambda: NodeType)
    descendant_nodes = graphene.List(lambda: NodeType, proper=graphene.Boolean())
    upstream_actions = graphene.List(lambda: NodeType)

    # TODO: Many nodes will output multiple time series. Remove metric
    # and handle a single-metric node as a special case in the UI??
    metric = graphene.Field(ForecastMetricType)
    output_metrics = graphene.List(ForecastMetricType)
    # If resolving through `descendant_nodes`, `impact_metric` will be
    # by default be calculated from the ancestor node.
    impact_metric = graphene.Field(ForecastMetricType, target_node_id=graphene.ID(required=False))

    # TODO: input_datasets, baseline_values, context
    description = graphene.String()
    parameters = graphene.List('params.schema.ParameterInterface')

    # These are potentially plucked from pages.models.NodeContent
    short_description = graphene.String()
    body = graphene.String()

    def resolve_color(root, info):
        if root.color:
            return root.color
        if root.quantity == 'emissions':
            for parent in root.output_nodes:
                if parent.color:
                    root.color = parent.color
                    return root.color

    def resolve_is_action(root, info):
        return isinstance(root, ActionNode)

    def resolve_descendant_nodes(root: Node, info: GQLInfo, proper=False):
        info.context._upstream_node = root
        return root.get_descendant_nodes(proper)

    def resolve_upstream_actions(root: Node, info):
        return root.get_upstream_nodes(filter=lambda x: isinstance(x, ActionNode))

    def resolve_metric(root: Node, info):
        df = root.get_output()
        if df is None:
            return None
        if VALUE_COLUMN not in df.columns:
            return None
        if root.baseline_values is not None:
            df[BASELINE_VALUE_COLUMN] = root.baseline_values[VALUE_COLUMN]
        return Metric(id=root.id, name=root.name, node=root, df=df)

    def resolve_impact_metric(root: Node, info, target_node_id: str = None):
        context = info.context.instance.context
        upstream_node = getattr(info.context, '_upstream_node', None)
        if target_node_id is not None:
            if target_node_id not in context.nodes:
                raise GraphQLError("Node %s not found" % target_node_id, [info])
            source_node = root
            target_node = context.get_node(target_node_id)
        elif upstream_node is not None:
            source_node = upstream_node
            target_node = root
        else:
            # FIXME: Determine a "default" target node from instance
            source_node = root
            target_node = context.get_node('net_emissions')

        if not isinstance(source_node, ActionNode):
            return None

        df = source_node.compute_impact(target_node)
        df = df[[IMPACT_COLUMN, FORECAST_COLUMN]]
        df = df.rename(columns={IMPACT_COLUMN: VALUE_COLUMN})
        return Metric(id='%s-%s-impact' % (root.id, target_node.id), name='Impact', df=df)

    def resolve_description(root, info):
        try:
            page = NodePage.objects.get(node=root.id)
        except NodePage.DoesNotExist:
            if root.description:
                return '<p>%s</p>' % root.description
            else:
                return None
        return expand_db_html(page.description)

    def resolve_parameters(root, info):
        return root.params.values()


class ScenarioType(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    is_active = graphene.Boolean()
    is_default = graphene.Boolean()

    def resolve_is_active(root: Scenario, info: GQLInfo):
        context = info.context.instance.context
        return context.active_scenario == root

    def resolve_is_default(root: Scenario, info: GQLInfo):
        return root.default


class Query(graphene.ObjectType):
    instance = graphene.Field(InstanceType)
    nodes = graphene.List(NodeType)
    node = graphene.Field(
        NodeType, id=graphene.ID(required=True)
    )
    actions = graphene.List(NodeType)
    scenarios = graphene.List(ScenarioType)

    def resolve_instance(root, info: GQLInfo):
        instance = info.context.instance
        return dict(
            id=instance.id,
            name=instance.name,
            target_year=instance.context.target_year
        )

    def resolve_scenarios(root, info: GQLInfo):
        context = info.context.instance.context
        return list(context.scenarios.values())

    def resolve_node(root, info, id):
        instance = info.context.instance
        return instance.context.nodes.get(id)

    def resolve_nodes(root, info):
        instance = info.context.instance
        return instance.context.nodes.values()

    def resolve_actions(root, info):
        instance = info.context.instance
        return [n for n in instance.context.nodes.values() if isinstance(n, ActionNode)]
