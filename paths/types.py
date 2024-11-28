from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from django.core.exceptions import FieldDoesNotExist
from django.db.models import ForeignKey
from django.http import HttpRequest

from kausal_common.models.object_cache import CacheableModel
from kausal_common.models.permissions import PermissionedModel, PermissionedQuerySet

if TYPE_CHECKING:
    from collections import OrderedDict
    from typing import type_check_only

    from django.contrib.auth.models import AnonymousUser
    from wagtail.models import Site

    from kausal_common.graphene import GQLContext as CommonGQLContext, GQLInfo as CommonGQLInfo

    from paths.context import PathsObjectCache
    from paths.graphql_helpers import GraphQLPerfNode

    from nodes.instance import Instance
    from nodes.models import InstanceConfig
    from nodes.perf import PerfContext
    from users.models import User


type UserOrAnon = 'User | AnonymousUser'


class PathsRequest(HttpRequest):
    user: UserOrAnon  # type: ignore[override]
    cache: PathsObjectCache
    correlation_id: str
    """Randomly generated ID for correlation."""


class PathsAuthenticatedRequest(PathsRequest):
    user: User  # type: ignore[override]


class PathsAdminRequest(PathsAuthenticatedRequest):
    admin_instance: InstanceConfig
    _wagtail_site: Site | None


class PathsAPIRequest(PathsAuthenticatedRequest):
    wildcard_domains: list[str] | None


class PathsModel(PermissionedModel):
    if TYPE_CHECKING:
        Meta: Any
    else:
        class Meta:
            abstract = True


class PathsQuerySet[M: PathsModel](PermissionedQuerySet[M]):
    def within_realm(self, realm: InstanceConfig) -> Self:
        from nodes.models import InstanceConfig

        try:
            field = self.model._meta.get_field('instance')
        except FieldDoesNotExist:
            field = None
        if field is None:
            return self
        if isinstance(field, ForeignKey) and field.related_model is InstanceConfig:
            return self.filter(instance=realm)
        return self


if TYPE_CHECKING:
    @type_check_only
    class PathsGQLContext(CommonGQLContext):  # pyright: ignore[reportGeneralTypeIssues]
        graphql_operation_name: str | None
        graphql_perf: PerfContext[GraphQLPerfNode]
        oauth2_error: OrderedDict[str, str]
        cache: PathsObjectCache
        _referer: str | None


    @type_check_only
    class PathsGQLInfo(CommonGQLInfo):  # pyright: ignore[reportGeneralTypeIssues]
        context: PathsGQLContext  # pyright: ignore[reportIncompatibleVariableOverride]

    @type_check_only
    class GQLInstanceContext(PathsGQLContext):  # pyright: ignore
        instance: Instance
        wildcard_domains: list[str]

    @type_check_only
    class GQLInstanceInfo(PathsGQLInfo):  # pyright: ignore
        context: GQLInstanceContext  # type: ignore[assignment]


class CacheablePathsModel[CacheT](CacheableModel[CacheT], PathsModel):
    class Meta:
        abstract = True
