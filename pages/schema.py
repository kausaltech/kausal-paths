import graphene
import re
from grapple.types.pages import Page as GrapplePageType, PageInterface
from graphql.error import GraphQLLocatedError
from grapple.utils import resolve_queryset

from wagtail.core.models import Page as WagtailPage

from nodes.models import InstanceConfig
from nodes.schema import NodeType
from paths.graphql_helpers import GQLInstanceInfo, ensure_instance

from .models import OutcomePage, PathsPage


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


def resolve_url_path(self, info: GQLInstanceInfo, **kwargs):
    url_path = self.url_path
    # FIXME: This is a dirty way to work around the issue of the slug having the form <instance>-1 or so for translated
    # pages.
    # Replace instance ID, optionally followed by a `-` and a number, if it is surrounded by slashes, by a single slash
    url_path = re.sub('^/%s(-[0-9]+)?/' % re.escape(info.context.instance.id), '/', self.url_path)
    if len(url_path) > 1:
        url_path = url_path.rstrip('/')
    return url_path


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
    pages = graphene.List(PageInterface, in_menu=graphene.Boolean(required=False))
    page = graphene.Field(PageInterface, path=graphene.String(required=True))

    @ensure_instance
    def resolve_pages(query, info: GQLInstanceInfo, in_menu: bool = False, **kwargs):
        instance_config = InstanceConfig.objects.get(identifier=info.context.instance.id)
        root_page = instance_config.get_translated_root_page()
        qs = root_page.get_descendants(inclusive=True).live().public().specific()
        if in_menu:
            qs = qs.filter(show_in_menus=True)

        return qs

    @ensure_instance
    def resolve_page(query, info: GQLInstanceInfo, path: str, **kwargs):
        qs = Query.resolve_pages(query, info, **kwargs)
        if not path.endswith('/'):
            path = path + '/'
        # Prepend the url_path of the translated root page
        instance_config = InstanceConfig.objects.get(identifier=info.context.instance.id)
        root_page = instance_config.get_translated_root_page()
        path = root_page.url_path.rstrip('/') + path
        qs = qs.filter(url_path=path)
        return qs.first()


def monkeypatch_grapple():
    from grapple.registry import registry
    # Monkeypatch resolvers to ensure we don't traverse outside
    # of site pages.
    PageInterface.resolve_parent = resolve_parent
    PageInterface.resolve_ancestors = resolve_ancestors
    PageInterface.resolve_url_path = resolve_url_path
    # Replace Grapple-generated PageTypes with our own
    registry.pages[OutcomePage] = OutcomePageType
    #registry.pages[ActionListPage] = ActionListPageType
