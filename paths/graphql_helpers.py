import functools
from typing import Any, Callable, Concatenate, ParamSpec, TypeAlias, TypeVar, TYPE_CHECKING

from django.core.handlers.wsgi import WSGIRequest
from graphql.type import GraphQLResolveInfo
from graphql.language.ast import OperationDefinitionNode
from graphql.error import GraphQLError

from nodes.instance import Instance

if TYPE_CHECKING:
    from nodes.context import Context


# Helper classes for typing
class GQLContext(WSGIRequest):
    graphql_query_language: str


class GQLInstanceContext(GQLContext):
    instance: Instance
    _referer: str | None


class GQLInfo(GraphQLResolveInfo):
    context: GQLContext
    operation: OperationDefinitionNode


class GQLInstanceInfo(GQLInfo):
    context: GQLInstanceContext


def _instance_or_bust(info: GQLInstanceInfo):
    if info.context.instance is None:
        raise GraphQLError(
            "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            info.field_nodes
        )
    return info.context.instance



def ensure_instance(method):
    """Wrap a class method to ensure instance is specified when the method is called."""

    @functools.wraps(method)
    def method_wrapper(self, info: GQLInstanceInfo, *args, **kwargs):
        _instance_or_bust(info)
        return method(self, info, *args, **kwargs)

    return method_wrapper

P = ParamSpec('P')
R = TypeVar('R')

def pass_context(
    method: Callable[Concatenate[Any, GQLInstanceInfo, P], R]
) -> Callable[Concatenate[Any, GQLInstanceInfo, P], R]:
    """Wrap a resolver function to provide Context as an argument."""

    @functools.wraps(method)
    def method_wrapper(root, info: GQLInstanceInfo, *args: P.args, **kwargs: P.kwargs) -> R:
        instance = _instance_or_bust(info)
        return method(root, info, instance.context, *args, **kwargs) # type: ignore

    return method_wrapper
