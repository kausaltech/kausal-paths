import graphene
from pages.base import ActionPage, EmissionPage, Metric
from graphql.type import (
    DirectiveLocation, GraphQLArgument, GraphQLDirective, GraphQLNonNull,
    GraphQLString, specified_directives
)
from wagtail.core.rich_text import expand_db_html

from params.base import BoolParameter, NumberParameter, StringParameter
from nodes.actions import ActionNode
from pages.models import NodePage


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
    unit = graphene.Field(UnitType)
    historical_values = graphene.List(YearlyValue)
    forecast_values = graphene.List(YearlyValue)
    baseline_forecast_values = graphene.List(YearlyValue)

    def resolve_historical_values(root, info):
        instance = info.context.instance
        return root.get_historical_values(instance.context)

    def resolve_forecast_values(root, info):
        instance = info.context.instance
        return root.get_forecast_values(instance.context)

    def resolve_baseline_forecast_values(root, info):
        instance = info.context.instance
        return root.get_baseline_forecast_values(instance.context)


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

    def resolve_is_customized(root, info):
        context = info.context.instance.context
        return root.is_customized(info.context.session.get('params'))


class BoolParameterType(graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)

    value = graphene.Boolean()
    default_value = graphene.Boolean()

    def resolve_value(root, info):
        return root.get(info.context.session.get('params'))


class NumberParameterType(graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)

    value = graphene.Float()
    min_value = graphene.Float()
    max_value = graphene.Float()

    default_value = graphene.Float()

    def resolve_value(root, info):
        return root.get(info.context.session.get('params'))


class StringParameterType(graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)

    value = graphene.String()
    default_value = graphene.String()

    def resolve_value(root, info):
        return root.get(info.context.session.get('params'))


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
    metric = graphene.Field(ForecastMetricType)
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

    def resolve_descendant_nodes(root, info, proper=False):
        return root.get_descendant_nodes(proper)

    def resolve_metric(root, info):
        return Metric(id=root.id, name=root.name)

    def resolve_description(root, info):
        try:
            page = NodePage.objects.get(node=root.id)
        except NodePage.DoesNotExist:
            return None
        return expand_db_html(page.description)

    def resolve_parameters(root, info):
        return root.parameters.values()


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
    parameters = graphene.List(ParameterInterface)

    def resolve_instance(root, info):
        instance = info.context.instance
        return dict(
            id=instance.id,
            name=instance.name,
            target_year=instance.context.target_year
        )

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

    def mutate(root, info, id, number_value=None, bool_value=None, string_value=None):
        value = None
        for v in (number_value, bool_value, string_value):
            if v is not None:
                if value is not None:
                    raise Exception("Only one type of value allowed")
                value = v
        if value is None:
            raise Exception("No value specified")

        session = info.context.session
        instance = info.context.instance
        context = instance.context
        # TODO: Ensure parameter id actually exists, validate input value

        params = session.setdefault('params', {})
        params[id] = value
        # Explicitly mark session as modified because we might only have modified `session['params']`, not `session`
        session.active_scenario = 'custom'
        session.modified = True
        instance = info.context.instance
        # TODO: Get the parameter from the 'custom' scenario
        return SetParameterMutation(ok=True, parameter=instance.context.params.get(id))


class ResetParameterMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID()

    ok = graphene.Boolean()

    def mutate(root, info, id=None):
        session = info.context.session
        if id is None:
            # Reset all parameters to defaults
            session.pop('params', None)
            session.pop('active_scenario', None)
        else:
            params = info.context.session.get('params', {})
            params.pop(id, None)
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
