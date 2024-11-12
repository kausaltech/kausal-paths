from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

from django.db.models import Model, Q

from kausal_common.context.single import SingleValueContext

from paths.types import PathsModel, PathsQuerySet

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kausal_common.models.permission_policy import ObjectSpecificAction
    from kausal_common.users import UserOrAnon

    from frameworks.models import (
        Framework,
    )
    from frameworks.object_cache import CFramework, FrameworkCache
    from nodes.models import InstanceConfig
    from users.models import User


@dataclass
class RealmContext:
    realm: InstanceConfig
    user: User | None
    realm_user_roles: set[str] = field(default_factory=set)

    def has_realm_role(self, role: str, superusers_have_all_roles: bool = True) -> bool:
        if self.user is None:
            return False
        if self.user.is_superuser and superusers_have_all_roles:
            return True
        return role in self.realm_user_roles


@dataclass
class ObjectCacheGroup[CachedM: PathsModel]:
    cache: ModelObjectCache[Any, CachedM, Any, Any]
    get_group: Callable[[CachedM], int]
    objs: dict[int, dict[int, CachedM]] = field(init=False, default_factory=dict)

    def add(self, obj: CachedM):
        group_id = self.get_group(obj)
        group_objs = self.objs.setdefault(group_id, {})
        if obj.pk not in group_objs:
            group_objs[obj.pk] = obj

    def get_list(self, group_id: int) -> list[CachedM]:
        self.cache.full_populate()
        group_objs = list(self.objs.get(group_id, {}).values())
        return group_objs


@dataclass
class ModelObjectCache[M: PathsModel, CachedM: PathsModel, QS: PathsQuerySet, ParentM: Model | None](ABC):
    parent: ParentM
    user: UserOrAnon | None
    _by_id: dict[int, CachedM] = field(init=False, default_factory=dict)
    _is_fully_prefetched: bool = field(init=False, default=False)
    _groups: dict[str, ObjectCacheGroup[CachedM]] = field(init=False, default_factory=dict)

    @property
    @abstractmethod
    def model(self) -> type[M]: ...

    def __post_init__(self) -> None:
        return

    def populate(self, qs: QS) -> Iterable[CachedM]:
        """Populate the object cache."""

        obj_list = self._as_list(qs)
        for obj in obj_list:
            self.add_obj(obj)
        return obj_list

    def add_obj(self, obj: CachedM) -> None:
        """Add an object to cache."""

        for grp in self._groups.values():
            grp.add(obj)

    def get_list_by_group(self, group_type: str, group_id: int) -> list[CachedM]:
        return self._groups[group_type].get_list(group_id)

    def get_base_qs(self, qs: QS | None = None, action: ObjectSpecificAction = 'view'):
        if qs is None:
            qs = cast(QS, self.model._default_manager.get_queryset())
        return self.filter_for_user(qs, action)

    def _as_list(self, qs: QS) -> list[CachedM]:
        return list(qs)

    def filter_for_user(self, qs: QS, action: ObjectSpecificAction = 'view') -> QS:
        if self.user is None:
            return qs
        if action == 'view':
            return qs.viewable_by(self.user)
        if action == 'change':
            return qs.modifiable_by(self.user)
        if action == 'delete':
            return qs.deletable_by(self.user)
        raise KeyError("Invalid action: %s" % action)

    def _do_populate(self, qs_filter: Q | None = None) -> list[CachedM]:
        qs = self.get_base_qs()
        if qs_filter is not None:
            qs = qs.filter(qs_filter)
        objs: list[CachedM] = []
        for obj in self.populate(qs):
            self._by_id[obj.pk] = obj
            objs.append(obj)
        return objs

    def full_populate(self):
        if self._is_fully_prefetched:
            return
        self._do_populate()
        self._is_fully_prefetched = True

    def get_list(self, filter_func: Callable[[CachedM], bool] | None = None) -> list[CachedM]:
        self.full_populate()
        obj_list = list(self._by_id.values())
        if filter_func:
            return list(filter(filter_func, obj_list))
        return obj_list

    def get(self, obj_id: int, qs_filter: Q | None = None) -> CachedM | None:
        if obj_id in self._by_id:
            return self._by_id[obj_id]
        self._do_populate(qs_filter)
        return self._by_id.get(obj_id)

    def first(self, qs_filter: Q) -> CachedM | None:
        obj_list = self._do_populate(qs_filter)
        if len(obj_list) == 0:
            return None
        return obj_list[0]


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

    def for_framework(self, fw: Framework) -> CFramework | None:
        return self.frameworks.get(fw.pk)


realm_context = SingleValueContext[RealmContext]('RealmContext', RealmContext)
"""The currently active realm scope.

This will be set for:
  * admin requests (through a middleware)
  * GraphQL resolvers (in the GraphQL executor)
"""

paths_object_cache = SingleValueContext[PathsObjectCache]('PathsObjectCache', PathsObjectCache)
