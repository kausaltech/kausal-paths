import graphene
from pages.base import EmissionPage, Metric, Page
from graphql.type import (
    DirectiveLocation, GraphQLArgument, GraphQLDirective, GraphQLNonNull,
    GraphQLString, specified_directives
)
from wagtail.core.rich_text import expand_db_html

from . import loader
from nodes.actions import Action
from pages.models import NodePage


class UnitNode(graphene.ObjectType):
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


class ForecastMetricNode(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    unit = graphene.String()
    historical_values = graphene.List(YearlyValue)
    forecast_values = graphene.List(YearlyValue)
    baseline_forecast_values = graphene.List(YearlyValue)

    def resolve_historical_values(root, info):
        return root.get_historical_values(loader.context)

    def resolve_forecast_values(root, info):
        return root.get_forecast_values(loader.context)

    def resolve_baseline_forecast_values(root, info):
        return root.get_baseline_forecast_values(loader.context)


class CardNode(graphene.ObjectType):
    id = graphene.String()
    name = graphene.String()
    metrics = graphene.List(ForecastMetricNode)
    upstream_cards = graphene.List(lambda: CardNode)
    downstream_cards = graphene.List(lambda: CardNode)


class PageInterface(graphene.Interface):
    id = graphene.ID()
    path = graphene.String()
    name = graphene.String()

    @classmethod
    def resolve_type(cls, page, info):
        if isinstance(page, EmissionPage):
            return EmissionPageNode
        raise Exception()


class EmissionSector(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    color = graphene.String()
    parent = graphene.Field(lambda: EmissionSector)
    metric = graphene.Field(ForecastMetricNode)


class EmissionPageNode(graphene.ObjectType):
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


class InstanceNode(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    target_year = graphene.Int()


class ParameterInterface(graphene.Interface):
    id = graphene.ID()
    node = graphene.Field(lambda x: NodeNode)


class NumberParameterNode(graphene.ObjectType):
    value = graphene.Float()


class BoolParameterNode(graphene.ObjectType):
    value = graphene.Boolean()


class StringParameterNode(graphene.ObjectType):
    value = graphene.String()


class NodeNode(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    color = graphene.String()
    unit = graphene.Field(UnitNode)
    quantity = graphene.String()
    is_action = graphene.Boolean()
    input_nodes = graphene.List(lambda: NodeNode)
    output_nodes = graphene.List(lambda: NodeNode)
    metric = graphene.Field(ForecastMetricNode)
    # TODO: input_datasets, parameters, baseline_values, context
    description = graphene.String()

    def resolve_color(root, info):
        if root.color:
            return root.color
        if root.quantity == 'emissions':
            for parent in root.output_nodes:
                if parent.color:
                    root.color = parent.color
                    return root.color

    def resolve_is_action(root, info):
        return isinstance(root, Action)

    def resolve_description(root, info):
        try:
            page = NodePage.objects.get(node=root.id)
        except NodePage.DoesNotExist:
            return None
        return expand_db_html(page.description)

    def resolve_metric(root, info):
        return Metric(id=root.id, name=root.name)


def get_page_node(page: Page):
    if isinstance(page, EmissionPage):
        return EmissionPageNode(page)


class Query(graphene.ObjectType):
    # TODO: Put (some of) the below in a separate app (like pages)?
    instance = graphene.Field(InstanceNode)
    pages = graphene.List(PageInterface)
    page = graphene.Field(
        PageInterface, path=graphene.String(required=False),
        id=graphene.String(required=False)
    )
    nodes = graphene.List(NodeNode)
    node = graphene.Field(
        NodeNode, id=graphene.ID(required=True)
    )

    def resolve_instance(root, info):
        instance = loader.instance
        return dict(
            id=instance.id,
            name=instance.name,
            target_year=loader.context.target_year
        )

    def resolve_pages(root, info):
        return list(loader.pages.values())

    def resolve_page(root, info, path=None, id=None):
        all_pages = list(loader.pages.values())
        if path:
            for page in all_pages:
                if page.path == path:
                    return page
        return None

    def resolve_node(root, info, id):
        return loader.context.nodes.get(id)

    def resolve_nodes(root, info):
        return loader.context.nodes.values()


class SetParameterMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)
        number_value = graphene.Float()
        bool_value = graphene.Boolean()
        string_value = graphene.String()

    def mutate(self, id, number_value=None, bool_value=None, string_value=None):
        pass


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
    types=[EmissionPageNode]
)
