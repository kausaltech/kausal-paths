from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import graphene

from grapple.types.interfaces import PageInterface as BasePageInterface, get_page_interface

if TYPE_CHECKING:
    from wagtail.models import Page

    from paths.context import InstanceSpecificCache
    from paths.types import GQLInstanceInfo, PathsGQLInfo as GQLInfo

    from pages.models import PathsPage


@dataclass
class VisibleSpecificPage:
    page: PathsPage
    cache: InstanceSpecificCache

    @classmethod
    def from_page(cls, page: Page, info: GQLInfo) -> Self | None:
        cache = info.context.cache.for_page(page)
        if cache is None:
            return None
        for visible_page in cache.visible_pages:
            if visible_page.path == page.path:
                return cls(page=visible_page, cache=cache)
        return None


class PageInterface(BasePageInterface):
    children = graphene.List(graphene.NonNull(get_page_interface), required=True)
    siblings = graphene.List(graphene.NonNull(get_page_interface), required=True)
    next_siblings = graphene.List(graphene.NonNull(get_page_interface), required=True)
    previous_siblings = graphene.List(graphene.NonNull(get_page_interface), required=True)
    ancestors = graphene.List(graphene.NonNull(get_page_interface), required=True)
    content_type = None

    @staticmethod
    def resolve_parent(root: Page, info: GQLInfo) -> Page | None:
        specific = VisibleSpecificPage.from_page(root, info)
        if specific is None:
            return None
        return specific.page.get_visible_parent(specific.cache)

    @staticmethod
    def resolve_children(root: Page, info: GQLInfo) -> list[PathsPage]:
        specific = VisibleSpecificPage.from_page(root, info)
        if specific is None:
            return []
        return specific.page.get_visible_children(specific.cache)

    @staticmethod
    def resolve_siblings(root: Page, info: GQLInfo) -> list[PathsPage]:
        return []

    resolve_next_siblings = resolve_siblings  # pyright: ignore[reportAssignmentType]
    resolve_previous_siblings = resolve_siblings  # pyright: ignore[reportAssignmentType]

    @staticmethod
    def resolve_ancestors(root: Page, info: GQLInfo) -> list[PathsPage]:
        specific = VisibleSpecificPage.from_page(root, info)
        if specific is None:
            return []
        return specific.page.get_visible_ancestors(specific.cache)

    @staticmethod
    def resolve_url_path(root, info: GQLInstanceInfo) -> str:
        url_path = root.url_path
        # FIXME: This is a dirty way to work around the issue of the slug having the form <instance>-1 or so for translated
        # pages.
        # Replace instance ID, optionally followed by a `-` and a number, if it is surrounded by slashes, by a single slash
        url_path = re.sub('^/%s(-[0-9]+)?/' % re.escape(info.context.instance.id), '/', root.url_path)
        if len(url_path) > 1:
            url_path = url_path.rstrip('/')
        return url_path
