from __future__ import annotations

import re
from typing import TYPE_CHECKING

import graphene
from django.contrib.contenttypes.models import ContentType
from django.utils.module_loading import import_string

from grapple.registry import registry
from grapple.settings import grapple_settings
from grapple.types.structures import QuerySetList
from grapple.utils import resolve_queryset

if TYPE_CHECKING:
    from paths.graphql_helpers import GQLInstanceInfo

    #from .models import PathsPage


def get_page_interface() -> type[PageInterface]:
    return import_string(grapple_settings.PAGE_INTERFACE)


class PageInterface(graphene.Interface):
    id = graphene.ID()
    title = graphene.String(required=True)
    slug = graphene.String(required=True)
    content_type = graphene.String(required=True)
    page_type = graphene.String()
    live = graphene.Boolean(required=True)

    url = graphene.String()
    url_path = graphene.String(required=True)

    depth = graphene.Int()
    seo_title = graphene.String(required=True)
    search_description = graphene.String()
    show_in_menus = graphene.Boolean(required=True)

    locked = graphene.Boolean()

    first_published_at = graphene.DateTime()
    last_published_at = graphene.DateTime()

    parent = graphene.Field(get_page_interface)
    children = QuerySetList(
        graphene.NonNull(get_page_interface), enable_search=True, required=True
    )
    siblings = QuerySetList(
        graphene.NonNull(get_page_interface), enable_search=True, required=True
    )
    descendants = QuerySetList(
        graphene.NonNull(get_page_interface), enable_search=True, required=True
    )
    ancestors = QuerySetList(
        graphene.NonNull(get_page_interface), enable_search=True, required=True
    )

    search_score = graphene.Float()

    @classmethod
    def resolve_type(cls, instance, info, **kwargs):
        """
        If model has a custom Graphene Node type in registry then use it,
        otherwise use base page type.
        """
        from grapple.types.pages import Page
        return registry.pages.get(type(instance), Page)

    @staticmethod
    def resolve_content_type(root, info, **kwargs):
        root.content_type = ContentType.objects.get_for_model(root)
        return (
            f"{root.content_type.app_label}.{root.content_type.model_class().__name__}"
        )

    @staticmethod
    def resolve_page_type(root, info, **kwargs):
        return get_page_interface().resolve_type(root.specific, info, **kwargs)

    @staticmethod
    def resolve_children(root, info, **kwargs):
        """
        Resolves a list of live children of this page.
        Docs: https://docs.wagtail.io/en/stable/reference/pages/queryset_reference.html#examples
        """
        return resolve_queryset(
            root.get_children().live().public().specific(), info, **kwargs
        )

    @staticmethod
    def resolve_descendants(root, info, **kwargs):
        """
        Resolves a list of nodes pointing to the current page’s descendants.
        Docs: https://docs.wagtail.io/en/stable/reference/pages/model_reference.html#wagtail.models.Page.get_descendants
        """
        return resolve_queryset(
            root.get_descendants().live().public().specific(), info, **kwargs
        )

    def resolve_seo_title(self, info, **kwargs):
        """
        Get page's SEO title. Fallback to a normal page's title if absent.
        """
        return self.seo_title or self.title

    def resolve_search_score(self, info, **kwargs):
        """
        Get page's search score, will be None if not in a search context.
        """
        return getattr(self, "search_score", None)

    @staticmethod
    def resolve_parent(root, info: GQLInstanceInfo, **kwargs):
        if root.depth <= 2:
            return None
        parent = root.get_parent()
        if parent is None:
            return None
        return parent.specific

    @staticmethod
    def resolve_ancestors(root, info: GQLInstanceInfo, **kwargs):
        return resolve_queryset(
            root.get_ancestors().live().public().specific().filter(depth__gte=2), info, **kwargs
        )

    @staticmethod
    def resolve_siblings(root, info: GQLInstanceInfo, **kwargs):
        return resolve_queryset(
            root.get_siblings().exclude(pk=root.pk).filter(depth__gte=3).live().public().specific(),
            info,
            **kwargs,
        )

    @staticmethod
    def resolve_url_path(root, info: GQLInstanceInfo, **kwargs):
        url_path = root.url_path
        # FIXME: This is a dirty way to work around the issue of the slug having the form <instance>-1 or so for translated
        # pages.
        # Replace instance ID, optionally followed by a `-` and a number, if it is surrounded by slashes, by a single slash
        url_path = re.sub('^/%s(-[0-9]+)?/' % re.escape(info.context.instance.id), '/', root.url_path)
        if len(url_path) > 1:
            url_path = url_path.rstrip('/')
        return url_path
