from __future__ import annotations

from typing import TYPE_CHECKING

import graphene

from grapple.types.pages import Page as GrapplePageType

from paths.graphql_helpers import GQLInstanceInfo, ensure_instance

from nodes.models import InstanceConfig
from nodes.schema import NodeType
from pages.page_interface import PageInterface

from .models import OutcomePage, Page, PathsPage
from .perms import PagePermissionPolicy

if TYPE_CHECKING:
    from nodes.node import Node

policy = PagePermissionPolicy()

class PathsPageType(GrapplePageType):
    show_in_footer = graphene.Boolean()

    class Meta:  # pyright: ignore
        model = PathsPage
        interfaces = (PageInterface,)
        name = 'PathsPage'


class OutcomePageType(PathsPageType):
    outcome_node = graphene.Field(NodeType, required=True)

    @staticmethod
    @ensure_instance
    def resolve_outcome_node(root: OutcomePage, info: GQLInstanceInfo) -> Node:
        return info.context.instance.context.get_node(root.outcome_node.identifier)

    class Meta:  # pyright: ignore
        model = OutcomePage
        interfaces = (PageInterface,)
        name = 'OutcomePage'


class Query:
    pages = graphene.List(graphene.NonNull(PageInterface), in_menu=graphene.Boolean(required=False), required=True)
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
    def resolve_page(query: Query, info: GQLInstanceInfo, path: str, **kwargs) -> Page | None:
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
    # Replace Grapple-generated PageTypes with our own
    registry.pages[OutcomePage] = OutcomePageType
    #registry.pages[ActionListPage] = ActionListPageType
