import os
from django.conf import settings
import graphene
from nodes.instance import InstanceLoader


loader = InstanceLoader(os.path.join(settings.BASE_DIR, 'configs/tampere.yaml'))
loader.print_graph()


class YearlyValue(graphene.ObjectType):
    year = graphene.Int()
    value = graphene.Float()


class ForecastMetricNode(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    historical_values = graphene.List(YearlyValue)
    forecast_values = graphene.List(YearlyValue)

    def resolve_historical_values(root, info):
        return root.get_historical_values(loader.context)

    def resolve_forecast_values(root, info):
        return root.get_forecast_values(loader.context)


class CardNode(graphene.ObjectType):
    id = graphene.String()
    name = graphene.String()
    metrics = graphene.List(ForecastMetricNode)
    upstream_cards = graphene.List(lambda: CardNode)
    downstream_cards = graphene.List(lambda: CardNode)


class PageNode(graphene.ObjectType):
    id = graphene.ID()
    path = graphene.String()
    name = graphene.String()
    cards = graphene.List(CardNode)


class Query(graphene.ObjectType):
    pages = graphene.List(PageNode)
    page = graphene.Field(PageNode, path=graphene.String(required=False), id=graphene.String(required=False))

    def resolve_pages(root, info):
        return list(loader.pages.values())

    def resolve_page(root, info, path=None, id=None):
        all_pages = list(loader.pages.values())
        if path:
            for page in all_pages:
                if page.path == path:
                    return page
        return None


schema = graphene.Schema(
    query=Query,
)
