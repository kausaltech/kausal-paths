from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Concatenate, ParamSpec, Protocol, TypeVar, cast

from graphql.error import GraphQLError

from nodes.instance import Instance

if TYPE_CHECKING:
    from paths.types import GQLInstanceInfo

    from nodes.context import Context
    from nodes.instance import Instance


@dataclass(slots=True)
class GraphQLPerfNode:
    id: str


def _instance_or_bust(info: GQLInstanceInfo) -> Instance:
    if getattr(info.context, 'instance', None) is None:
        raise GraphQLError(
            "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            info.field_nodes,
        )
    context = info.context
    return context.instance


class Resolver(Protocol):
    @staticmethod
    def __call__(root: Any, info: GQLInstanceInfo, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401

C = TypeVar('C', bound=Resolver)

def ensure_instance(method: C) -> C:
    """Wrap a class method to ensure instance is specified when the method is called."""

    @functools.wraps(method)
    def method_wrapper(self: Any, info: GQLInstanceInfo, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        _instance_or_bust(info)
        return method(self, info, *args, **kwargs)

    return cast(C, method_wrapper)


P = ParamSpec('P')
R = TypeVar('R')

def pass_context(
    method: Callable[Concatenate[Any, GQLInstanceInfo, Context, P], R],
) -> Callable[Concatenate[Any, GQLInstanceInfo, P], R]:
    """Wrap a resolver function to provide Context as an argument."""

    @functools.wraps(method)
    def method_wrapper(root, info: GQLInstanceInfo, *args: P.args, **kwargs: P.kwargs) -> R:
        instance = _instance_or_bust(info)
        return method(root, info, instance.context, *args, **kwargs)

    return method_wrapper
