import os
from pages.base import EmissionPage, Page
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
    emission_sectors = graphene.List(EmissionSector)

    class Meta:
        interfaces = (PageInterface,)

    def resolve_emission_sectors(root, info):
        return root.get_sectors()


class InstanceNode(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    target_year = graphene.Int()


def get_page_node(page: Page):
    if isinstance(page, EmissionPage):
        return EmissionPageNode(page)


class Query(graphene.ObjectType):
    instance = graphene.Field(InstanceNode)
    pages = graphene.List(PageInterface)
    page = graphene.Field(
        PageInterface, path=graphene.String(required=False),
        id=graphene.String(required=False)
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

    def resolve_instance(root, info):
        instance = loader.instance
        return dict(
            id=instance.id,
            name=instance.name,
            target_year=loader.context.target_year
        )


schema = graphene.Schema(
    query=Query,
    types=[EmissionPageNode]
)
