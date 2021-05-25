import graphene
from pages.base import EmissionPage, Page
from graphql.type import (
    DirectiveLocation, GraphQLArgument, GraphQLDirective, GraphQLNonNull,
    GraphQLString, specified_directives
)
from wagtail.core.rich_text import expand_db_html

from . import loader
from nodes.actions import Action
from pages.models import NodePage


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
    description = graphene.String()

    @classmethod
    def resolve_type(cls, page, info):
        if isinstance(page, EmissionPage):
            return EmissionPageNode
        raise Exception()

    def resolve_description(root, info):
        try:
            page = NodePage.objects.get(node=root.id)
        except NodePage.DoesNotExist:
            return None
        return expand_db_html(page.description)


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


class NodeNode(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    color = graphene.String()
    unit = graphene.String()
    quantity = graphene.String()
    is_action = graphene.Boolean()
    input_nodes = graphene.List(lambda: NodeNode)
    output_nodes = graphene.List(lambda: NodeNode)
    # TODO: input_datasets, parameters, baseline_values, context

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

    def resolve_nodes(root, info):
        return loader.context.nodes.values()


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
