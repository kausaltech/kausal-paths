from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Concatenate, OrderedDict, ParamSpec, TypeVar, cast

from django.core.handlers.wsgi import WSGIRequest
from graphql.error import GraphQLError

from nodes.instance import Instance

if TYPE_CHECKING:
    from typing import type_check_only

    from kausal_common.graphene import GQLInfo

    from paths.types import UserOrAnon

    from nodes.context import Context
    from nodes.instance import Instance
    from nodes.perf import PerfContext


@dataclass(slots=True)
class GraphQLPerfNode:
    id: str


if TYPE_CHECKING:
    # Helper classes for typing
    @type_check_only
    class GQLContext(WSGIRequest):
        user: UserOrAnon  # type: ignore[override]
        graphql_query_language: str

    @type_check_only
    class GQLInstanceContext(GQLContext):  # pyright: ignore
        instance: Instance
        _referer: str | None
        graphql_operation_name: str | None
        graphql_perf: PerfContext[GraphQLPerfNode]
        wildcard_domains: list[str]
        oauth2_error: OrderedDict[str, str]

    @type_check_only
    class GQLInstanceInfo(GQLInfo):  # pyright: ignore
        context: GQLInstanceContext  # type: ignore[override]


def _instance_or_bust(info: GQLInstanceInfo) -> Instance:
    if getattr(info.context, 'instance', None) is None:
        raise GraphQLError(
            "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            info.field_nodes,
        )
    return info.context.instance


C = TypeVar('C', bound=Callable)

def ensure_instance(method: C) -> C:
    """Wrap a class method to ensure instance is specified when the method is called."""

    @functools.wraps(method)
    def method_wrapper(self, info: GQLInstanceInfo, *args, **kwargs):
        _instance_or_bust(info)
        return method(self, info, *args, **kwargs)

    return cast(C, method_wrapper)


P = ParamSpec('P')
R = TypeVar('R')

def pass_context(
    method: Callable[Concatenate[Any, GQLInstanceInfo, Context, P], R]
) -> Callable[Concatenate[Any, GQLInstanceInfo, P], R]:
    """Wrap a resolver function to provide Context as an argument."""

    @functools.wraps(method)
    def method_wrapper(root, info: GQLInstanceInfo, *args: P.args, **kwargs: P.kwargs) -> R:
        instance = _instance_or_bust(info)
        return method(root, info, instance.context, *args, **kwargs)

    return method_wrapper
