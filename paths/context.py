from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol, cast

from django.core.exceptions import ObjectDoesNotExist
from django.utils import translation

from kausal_common.context.single import SingleValueContext
from kausal_common.models.object_cache import CacheableModel

from paths.types import PathsModel

if TYPE_CHECKING:
    from django.db.models import Model
    from wagtail.models import Page

    from kausal_common.users import UserOrAnon

    from frameworks.models import (
        Framework,
    )
    from frameworks.object_cache import FrameworkCache
    from nodes.models import InstanceConfig
    from pages.models import ActionListPage, PathsPage
    from users.models import User


@dataclass
class RealmContext:
    realm: InstanceConfig
    user: User | None


class CacheablePathsModel[CacheT](CacheableModel[CacheT], PathsModel):
    class Meta:
        abstract = True



class HasParent(Protocol):
    def __init__(self, parent: Any, /): ...


@dataclass
class InstanceSpecificCache:
    instance: InstanceConfig

    @cached_property
    def translated_root_page(self) -> Page | None:
        try:
            root = self.instance.get_translated_root_page()
        except ObjectDoesNotExist:
            # If the current active language is not supported, fall back to
            # the instance's default language.
            with translation.override(self.instance.default_language):
                root = self.instance.get_translated_root_page()
        return root.specific

    @cached_property
    def visible_pages(self) -> list[PathsPage]:
        if self.translated_root_page is None:
            return []
        pages = self.translated_root_page.get_descendants(inclusive=True).live().public().specific()

        # We store the parent object in the page object itself to avoid extra DB hits.
        # MP_Node.get_parent() will use this cached parent object if it exists.
        # Also, we filter out pages that don't have a visible parent.
        pages_by_path = {page.path: page for page in pages}
        visible_pages = []
        for page in pages:
            parent_path = page._get_basepath(page.path, depth=page.depth - 1)
            parent = pages_by_path.get(parent_path)
            setattr(page, '_cached_parent_obj', parent)  # noqa: B010
            if page != self.translated_root_page and parent is None:
                continue
            visible_pages.append(page)

        return cast('list[PathsPage]', visible_pages)

    @cached_property
    def action_list_page(self) -> ActionListPage | None:
        from pages.models import ActionListPage
        for page in self.visible_pages:
            if isinstance(page, ActionListPage):
                return page
        return None


@dataclass
class PathsObjectCache:
    user: UserOrAnon | None = None

    frameworks: FrameworkCache = field(init=False)
    instance_caches: dict[int, InstanceSpecificCache] = field(default_factory=dict)

    def __post_init__(self):
        from frameworks.object_cache import FrameworkCache
        self.frameworks = FrameworkCache(None, self.user)

    def _get_or_create[C: HasParent](self, obj: Model, klass: type[C], caches: dict[int, C]) -> C:
        cache = caches.get(obj.pk)
        if cache is None:
            cache = klass(obj)
            caches[obj.pk] = cache
        return cache

    def for_framework_id(self, fw_id: int) -> Framework | None:
        return self.frameworks.get(fw_id)

    def for_framework(self, fw: Framework) -> Framework | None:
        return self.frameworks.get(fw.pk)

    def for_instance(self, instance: InstanceConfig) -> InstanceSpecificCache:
        if instance.pk in self.instance_caches:
            return self.instance_caches[instance.pk]
        cache = InstanceSpecificCache(instance)
        self.instance_caches[instance.pk] = cache
        return cache

    def for_page(self, page: Page) -> InstanceSpecificCache | None:
        page_path = page.path
        for instance_cache in self.instance_caches.values():
            root_page = instance_cache.translated_root_page
            if root_page is None:
                continue
            if page_path.startswith(root_page.path):
                return instance_cache

        ic = cast('PathsPage',page.specific).instance_config
        return self.for_instance(ic)


realm_context = SingleValueContext[RealmContext]('RealmContext', RealmContext)
"""The currently active realm scope.

This will be set for:
  * admin requests (through a middleware)
  * GraphQL resolvers (in the GraphQL executor)
"""

paths_object_cache = SingleValueContext[PathsObjectCache]('PathsObjectCache', PathsObjectCache)
