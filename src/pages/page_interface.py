from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import graphene

from grapple.types.interfaces import PageInterface as BasePageInterface, get_page_interface

from pages.url_paths import to_instance_path

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
    menu_label = graphene.String()
    content_type = None

    @staticmethod
    def resolve_menu_label(root: Page, info: GQLInfo) -> str | None:
        return getattr(root.specific, 'menu_label', None) or None

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
    def resolve_url_path(root: Page, info: GQLInstanceInfo) -> str:
        """
        Expose the page's path relative to its instance's root page.

        The prefix comes from the root page's own ``url_path``, which is the same value
        ``Query.resolve_page`` prepends when the front end hands the result back, so the
        round trip holds whatever the root page's slug happens to be. See
        ``pages.url_paths`` for why that matters.

        ``for_page`` locates the instance by treebeard ``path`` prefix rather than by slug,
        so it cannot drift either.
        """
        cache = info.context.cache.for_page(root)
        root_page = cache.translated_root_page if cache is not None else None
        if root_page is None:
            return root.url_path
        return to_instance_path(root.url_path, root_page.url_path)
