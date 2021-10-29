import graphene
from grapple.types.structures import QuerySetList
from grapple.types.pages import Page as GrapplePageType, PageInterface
from graphql.error import GraphQLLocatedError
from grapple.utils import resolve_queryset

from wagtail.core.models import Page as WagtailPage, Site

from nodes.models import InstanceConfig
from nodes.schema import NodeType
from paths.graphql_helpers import GQLInstanceContext, GQLInstanceInfo, ensure_instance

from .models import OutcomePage, ActionListPage, PathsPage


def resolve_parent(self, info: GQLInstanceInfo, **kwargs):
    if self.depth <= 2:
        return None
    try:
        return self.get_parent().specific
    except GraphQLLocatedError:
        return WagtailPage.objects.none()


def resolve_ancestors(self, info: GQLInstanceInfo, **kwargs):
    return resolve_queryset(
        self.get_ancestors().live().public().specific().filter(depth__gte=2), info, **kwargs
    )


class PathsPageType(GrapplePageType):
    show_in_footer = graphene.Boolean()

    class Meta:
        model = PathsPage
        interfaces = (PageInterface,)
        name = 'PathsPage'


class OutcomePageType(PathsPageType):
    outcome_node = graphene.Field(NodeType, required=True)

    @ensure_instance
    def resolve_outcome_node(root: OutcomePage, info: GQLInstanceInfo):
        return info.context.instance.context.get_node(root.outcome_node.identifier)

    class Meta:
        model = OutcomePage
        interfaces = (PageInterface,)
        name = 'OutcomePage'


class Query:
    pages = graphene.List(PageInterface)
    page = graphene.Field(PageInterface, path=graphene.String(required=True))

    @ensure_instance
    def resolve_pages(query, info: GQLInstanceInfo, **kwargs):
        instance_config = InstanceConfig.objects.get(identifier=info.context.instance.id)
        root_page = instance_config.site.root_page
        qs = root_page.get_descendants(inclusive=True).live().public().specific()

        return qs

    @ensure_instance
    def resolve_page(query, info: GQLInstanceInfo, path: str, **kwargs):
        qs = Query.resolve_pages(query, info, **kwargs)
        if not path.endswith('/'):
            path = path + '/'
        # Prepend the instance identifier
        path = '/%s%s' % (info.context.instance.id, path)
        qs = qs.filter(url_path=path)
        return qs.first()


def monkeypatch_grapple():
    from grapple.registry import registry
    # Monkeypatch resolvers to ensure we don't traverse outside
    # of site pages.
    PageInterface.resolve_parent = resolve_parent
    PageInterface.resolve_ancestors = resolve_ancestors
    # Replace Grapple-generated PageTypes with our own
    registry.pages[OutcomePage] = OutcomePageType
    #registry.pages[ActionListPage] = ActionListPageType
