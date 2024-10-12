from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Self

from django.db import models
from django.db.models.query import QuerySet
from django.http import HttpRequest
from graphql import GraphQLError

from paths.permissions import PathsPermissionPolicy

if TYPE_CHECKING:
    from django.contrib.auth.models import AnonymousUser
    from wagtail.models import Site

    from kausal_common.graphene import GQLInfo

    from paths.cache import PathsObjectCache
    from paths.permissions import ObjectSpecificAction, PathsPermissionPolicy

    from nodes.models import InstanceConfig
    from users.models import User


type UserOrAnon = 'User | AnonymousUser'


class PathsRequest(HttpRequest):
    user: UserOrAnon
    cache: PathsObjectCache
    correlation_id: str
    """Randomly generated ID for correlation."""


class PathsAuthenticatedRequest(PathsRequest):
    user: 'User'


class PathsAdminRequest(PathsAuthenticatedRequest):
    admin_instance: InstanceConfig
    _wagtail_site: Site | None


class PathsAPIRequest(PathsAuthenticatedRequest):
    wildcard_domains: list[str] | None


class PathsModel(models.Model):  # noqa: DJ008
    if TYPE_CHECKING:
        Meta: Any
    else:
        class Meta:
            abstract = True

    @classmethod
    @abc.abstractmethod
    def permission_policy(cls) -> PathsPermissionPolicy[Self, Any]: ...

    def gql_action_allowed(self, info: GQLInfo, action: ObjectSpecificAction) -> bool:
        return self.permission_policy().gql_action_allowed(info, action, self)

    def ensure_gql_action_allowed(self, info: GQLInfo, action: ObjectSpecificAction) -> None:
        if not self.gql_action_allowed(info, action):
            raise GraphQLError("Permission denied for action '%s'" % action, nodes=info.field_nodes)


class PathsQuerySet[M: PathsModel](QuerySet[M]):
    @property
    def _pp(self) -> PathsPermissionPolicy[M, Self]:
        return self.model.permission_policy()

    def viewable_by(self, user: UserOrAnon) -> Self:
        return self._pp.filter_by_perm(self, user, 'view')
    def deletable_by(self, user: UserOrAnon) -> Self:
        return self._pp.filter_by_perm(self, user, 'delete')
    def modifiable_by(self, user: UserOrAnon) -> Self:
        return self._pp.filter_by_perm(self, user, 'change')
