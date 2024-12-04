from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from kausal_common.context.single import SingleValueContext
from kausal_common.models.object_cache import CacheableModel

from paths.types import PathsModel

if TYPE_CHECKING:
    from django.db.models import Model

    from kausal_common.users import UserOrAnon

    from frameworks.models import (
        Framework,
    )
    from frameworks.object_cache import FrameworkCache
    from nodes.models import InstanceConfig
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
class PathsObjectCache:
    user: UserOrAnon | None = None

    frameworks: FrameworkCache = field(init=False)

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


realm_context = SingleValueContext[RealmContext]('RealmContext', RealmContext)
"""The currently active realm scope.

This will be set for:
  * admin requests (through a middleware)
  * GraphQL resolvers (in the GraphQL executor)
"""

paths_object_cache = SingleValueContext[PathsObjectCache]('PathsObjectCache', PathsObjectCache)
