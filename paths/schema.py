import graphene
from pages.base import ActionPage, EmissionPage
from graphql.type import (
    DirectiveLocation, GraphQLArgument, GraphQLDirective, GraphQLNonNull,
    GraphQLString, specified_directives
)
from nodes.schema import Query as NodesQuery
from params.schema import (
    Query as ParamsQuery,
    Mutations as ParamsMutations,
    types as params_types
)


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
    metric = graphene.Field('nodes.schema.ForecastMetricType')
    node = graphene.Field('nodes.schema.NodeType')


class EmissionPageType(graphene.ObjectType):
    emission_sectors = graphene.List(
        EmissionSector, id=graphene.ID()
    )

    class Meta:
        interfaces = (PageInterface,)

    @staticmethod
    def resolve_emission_sectors(root: EmissionPage, info, id=None):
        context = info.context.instance.context
        all_sectors = root.get_sectors(context)
        if id is not None:
            all_sectors = list(filter(lambda x: x.id == id, all_sectors))
        return all_sectors


class ActionPageType(graphene.ObjectType):
    action = graphene.Field('nodes.schema.NodeType')

    class Meta:
        interfaces = (PageInterface,)


class Query(NodesQuery, ParamsQuery):
    pages = graphene.List(PageInterface)
    page = graphene.Field(
        PageInterface, path=graphene.String(required=False),
        id=graphene.String(required=False)
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


class Mutations(ParamsMutations):
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
    types=[
        ActionPageType,
        EmissionPageType,
        *params_types,
    ],
    mutation=Mutations,
)
