from __future__ import annotations
from dataclasses import dataclass

import functools
from typing import Any, Callable, Concatenate, ParamSpec, TypeAlias, TypeVar, TYPE_CHECKING, cast

from django.core.handlers.wsgi import WSGIRequest
from graphql.type import GraphQLResolveInfo
from graphql.language.ast import OperationDefinitionNode
from graphql.error import GraphQLError
from common.perf import PerfCounter

from nodes.instance import Instance
from nodes.perf import PerfContext

if TYPE_CHECKING:
    from nodes.context import Context
    from paths.types import UserOrAnon


# Helper classes for typing
class GQLContext(WSGIRequest):
    user: UserOrAnon
    graphql_query_language: str


@dataclass(slots=True)
class GraphQLPerfNode:
    id: str


class GQLInstanceContext(GQLContext):
    instance: Instance
    _referer: str | None
    graphql_operation_name: str | None
    graphql_perf: PerfContext[GraphQLPerfNode]
    wildcard_domains: list[str]


class GQLInfo(GraphQLResolveInfo):
    context: GQLContext
    operation: OperationDefinitionNode


class GQLInstanceInfo(GQLInfo):
    context: GQLInstanceContext


def _instance_or_bust(info: GQLInstanceInfo):
    if getattr(info.context, 'instance', None) is None:
        raise GraphQLError(
            "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            info.field_nodes
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