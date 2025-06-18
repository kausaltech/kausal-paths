from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from django.core.exceptions import FieldDoesNotExist
from django.db.models import ForeignKey

from kausal_common.models.object_cache import CacheableModel
from kausal_common.models.permissions import PermissionedModel, PermissionedQuerySet

if TYPE_CHECKING:
    from collections import OrderedDict
    from typing import type_check_only

    from graphql import GraphQLResolveInfo
    from wagtail.models import Site

    from kausal_common.deployment.types import LoggedHttpRequest
    from kausal_common.graphene import GQLContext as CommonGQLContext, GQLInfo as CommonGQLInfo
    from kausal_common.perf.perf_context import PerfContext
    from kausal_common.users import UserOrAnon

    from paths.context import PathsObjectCache
    from paths.graphql_helpers import GraphQLPerfNode
    from paths.graphql_types import PathsGraphQLContext

    from common.cache import CacheResult
    from nodes.instance import Instance
    from nodes.models import InstanceConfig
    from users.models import User


if TYPE_CHECKING:
    class PathsRequest(LoggedHttpRequest):
        user: UserOrAnon  # type: ignore[override, assignment]
        cache: PathsObjectCache


    class PathsAuthenticatedRequest(PathsRequest):
        user: User  # type: ignore[override]


    class PathsAdminRequest(PathsAuthenticatedRequest):
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
        graphql_perf: PerfContext[GraphQLPerfNode, CacheResult]
        oauth2_error: OrderedDict[str, str]
        cache: PathsObjectCache
        _referer: str | None

    @type_check_only
    class PathsGQLInfo(CommonGQLInfo):  # pyright: ignore[reportGeneralTypeIssues]
        context: PathsGraphQLContext

    @type_check_only
    class GQLInstanceContext(PathsGQLContext):  # pyright: ignore
        instance: Instance
        wildcard_domains: list[str]

    @type_check_only
    class GQLInstanceInfo(GraphQLResolveInfo):
        context: PathsGraphQLContext[Instance]


class CacheablePathsModel[CacheT](CacheableModel[CacheT], PathsModel):
    class Meta:
        abstract = True
