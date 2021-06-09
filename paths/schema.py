from nodes.scenario import Scenario
from nodes.instance import Instance
from nodes.context import Context
from params.param import ValidationError
from nodes.constants import BASELINE_VALUE_COLUMN, FORECAST_COLUMN, IMPACT_COLUMN, VALUE_COLUMN
import graphene
from pages.base import ActionPage, EmissionPage, Metric
from graphql.type import (
    DirectiveLocation, GraphQLArgument, GraphQLDirective, GraphQLNonNull,
    GraphQLString, specified_directives
)
from graphql.error import GraphQLError
from wagtail.core.rich_text import expand_db_html

from params import BoolParameter, NumberParameter, StringParameter
from nodes import Node
from nodes.actions import ActionNode
from pages.models import NodePage


# Helper classes for typing
class GQLContext:
    instance: Instance


class GQLInfo:
    context: GQLContext


class UnitType(graphene.ObjectType):
    short = graphene.String()
    long = graphene.String()
    html_short = graphene.String()
    html_long = graphene.String()

    def resolve_short(root, info):
        return root.format_babel('~P')

    def resolve_long(root, info):
        return root.format_babel('P')

    def resolve_html_short(root, info):
        return root.format_babel('~H')

    def resolve_html_long(root, info):
        return root.format_babel('H')


class YearlyValue(graphene.ObjectType):
    year = graphene.Int()
    value = graphene.Float()


class ForecastMetricType(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    # output_node will be set if the node outputs multiple time-series
    output_node = graphene.Field(lambda: NodeType)
    unit = graphene.Field(UnitType)
    historical_values = graphene.List(YearlyValue)
    forecast_values = graphene.List(YearlyValue)
    baseline_forecast_values = graphene.List(YearlyValue)

    def resolve_historical_values(root: Metric, info):
        return root.get_historical_values()

    def resolve_forecast_values(root: Metric, info):
        return root.get_forecast_values()

    def resolve_baseline_forecast_values(root: Metric, info):
        return root.get_baseline_forecast_values()


class CardType(graphene.ObjectType):
    id = graphene.String()
    name = graphene.String()
    metrics = graphene.List(ForecastMetricType)
    upstream_cards = graphene.List(lambda: CardType)
    downstream_cards = graphene.List(lambda: CardType)


class PageInterface(graphene.Interface):
    id = graphene.ID()
    path = graphene.String()
    name = graphene.String()

    @classmethod
    def resolve_type(cls, page, info):
        if isinstance(page, EmissionPage):
            return EmissionPageType
        elif isinstance(page, ActionPage):
            return ActionPageType
        raise Exception(f"{page} has invalid type")


class EmissionSector(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    color = graphene.String()
    parent = graphene.Field(lambda: EmissionSector)
    metric = graphene.Field(ForecastMetricType)


class EmissionPageType(graphene.ObjectType):
    emission_sectors = graphene.List(
        EmissionSector, id=graphene.ID()
    )

    class Meta:
        interfaces = (PageInterface,)

    def resolve_emission_sectors(root: EmissionPage, info, id=None):
        all_sectors = root.get_sectors()
        if id is not None:
            all_sectors = list(filter(lambda x: x.id == id, all_sectors))
        return all_sectors


class ActionPageType(graphene.ObjectType):
    action = graphene.Field(lambda: NodeType)

    class Meta:
        interfaces = (PageInterface,)


class InstanceType(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    target_year = graphene.Int()


class ParameterInterface(graphene.Interface):
    id = graphene.ID()  # global id
    node_relative_id = graphene.ID()  # can be null if node is null
    node = graphene.Field(lambda: NodeType)  # can be null for global params
    is_customized = graphene.Boolean()

    @classmethod
    def resolve_type(cls, parameter, info):
        if isinstance(parameter, BoolParameter):
            return BoolParameterType
        elif isinstance(parameter, NumberParameter):
            return NumberParameterType
        elif isinstance(parameter, StringParameter):
            return StringParameterType
        raise Exception(f"{parameter} has invalid type")


class BoolParameterType(graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)

    value = graphene.Boolean()
    default_value = graphene.Boolean()


class NumberParameterType(graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)

    value = graphene.Float()
    min_value = graphene.Float()
    max_value = graphene.Float()

    default_value = graphene.Float()


class StringParameterType(graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)

    value = graphene.String()
    default_value = graphene.String()


class NodeType(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    color = graphene.String()
    unit = graphene.Field(UnitType)
    quantity = graphene.String()
    is_action = graphene.Boolean()
    input_nodes = graphene.List(lambda: NodeType)
    output_nodes = graphene.List(lambda: NodeType)
    descendant_nodes = graphene.List(lambda: NodeType, proper=graphene.Boolean())
    upstream_actions = graphene.List(lambda: NodeType)

    # TODO: Remove metric??
    metric = graphene.Field(ForecastMetricType)
    output_metrics = graphene.List(ForecastMetricType)

    impact_metric = graphene.Field(ForecastMetricType, target_node_id=graphene.ID(required=False))

    # TODO: input_datasets, parameters, baseline_values, context
    description = graphene.String()
    parameters = graphene.List(ParameterInterface)

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
    # TODO: Put (some of) the below in a separate app (like pages)?
    instance = graphene.Field(InstanceType)
    pages = graphene.List(PageInterface)
    page = graphene.Field(
        PageInterface, path=graphene.String(required=False),
        id=graphene.String(required=False)
    )
    nodes = graphene.List(NodeType)
    node = graphene.Field(
        NodeType, id=graphene.ID(required=True)
    )
    actions = graphene.List(NodeType)
    parameters = graphene.List(ParameterInterface)
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

    def resolve_pages(root, info):
        instance = info.context.instance
        return list(instance.pages.values())

    def resolve_page(root, info, path=None, id=None):
        instance = info.context.instance
        all_pages = list(instance.pages.values())
        if path:
            for page in all_pages:
                if page.path == path:
                    return page
        return None

    def resolve_node(root, info, id):
        instance = info.context.instance
        return instance.context.nodes.get(id)

    def resolve_nodes(root, info):
        instance = info.context.instance
        return instance.context.nodes.values()

    def resolve_actions(root, info):
        instance = info.context.instance
        return [n for n in instance.context.nodes.values() if isinstance(n, ActionNode)]

    def resolve_parameters(root, info):
        instance = info.context.instance
        return instance.context.params.values()


class SetParameterMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)
        number_value = graphene.Float()
        bool_value = graphene.Boolean()
        string_value = graphene.String()

    ok = graphene.Boolean()
    parameter = graphene.Field(ParameterInterface)

    def mutate(root, info: GQLInfo, id, number_value=None, bool_value=None, string_value=None):
        context = info.context.instance.context
        try:
            param = context.params[id]
        except KeyError:
            raise GraphQLError("Parameter %s does not exist", [info])

        parameter_values = {
            NumberParameter: (number_value, 'numberValue'),
            BoolParameter: (bool_value, 'boolValue'),
            StringParameter: (string_value, 'stringValue'),
        }
        p = parameter_values.pop(type(param), None)
        if p is None:
            raise Exception("Attempting to mutate an unsupported parameter class: %s" % type(param))
        value, attr_name = p
        if value is None:
            raise GraphQLError("You must specify '%s' for '%s'" % (attr_name, param.id))

        for v, _ in parameter_values.values():
            if v is not None:
                raise GraphQLError("Only one type of value allowed", [info])

        try:
            value = param.clean(value)
        except ValidationError as e:
            raise GraphQLError(str(e), [info])

        session = info.context.session
        session_params = session.setdefault('params', {})
        session_params[id] = value

        custom_scenario = context.scenarios['custom']
        custom_scenario.set_session(session)
        context.activate_scenario(custom_scenario)
        session['active_scenario'] = 'custom'
        # Explicitly mark session as modified because we might only have modified `session['params']`, not `session`
        session.modified = True

        return SetParameterMutation(ok=True, parameter=param)


class ResetParameterMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID()

    ok = graphene.Boolean()

    def mutate(root, info: GQLInfo, id=None):
        context = info.context.instance.context
        session = info.context.session
        if id is None:
            # Reset all parameters to defaults
            session.pop('params', None)
        else:
            params = session.get('params', {})
            params.pop(id, None)

        params = session.get('params', {})
        if not params:
            session['active_scenario'] = context.get_default_scenario().id

        info.context.session.modified = True
        return ResetParameterMutation(ok=True)


class Mutations(graphene.ObjectType):
    set_parameter = SetParameterMutation.Field()
    reset_parameter = ResetParameterMutation.Field()


class LocaleDirective(GraphQLDirective):
    def __init__(self):
        super().__init__(
            name='locale',
            description='Select locale in which to return data',
            args={
                'lang': GraphQLArgument(
                    type_=GraphQLNonNull(GraphQLString),
                    description='Selected language'
                )
            },
            locations=[DirectiveLocation.QUERY]
        )


schema = graphene.Schema(
    query=Query,
    directives=specified_directives + [LocaleDirective()],
    types=[
        ActionPageType,
        BoolParameterType,
        EmissionPageType,
        NumberParameterType,
        StringParameterType,
    ],
    mutation=Mutations,
)
