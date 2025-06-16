from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar, cast, overload

from graphql.error import GraphQLError
from strawberry.types.field import StrawberryField

from nodes.instance import Instance

if TYPE_CHECKING:
    from paths.types import GQLInstanceInfo

    from nodes.context import Context
    from nodes.instance import Instance

    from .graphql_types import SBInfo


@dataclass(slots=True)
class GraphQLPerfNode:
    id: str


def _instance_or_bust(info: InfoType) -> Instance:
    if getattr(info.context, 'instance', None) is None:
        raise GraphQLError(
            "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            info.field_nodes,
        )
    context = info.context
    return context.instance


type InfoType = GQLInstanceInfo | SBInfo

type AnyResolver[**P, R, I: InfoType] = Callable[Concatenate[Any, I, P], R]

def ensure_instance[**P, R, I: InfoType](method: AnyResolver[P, R, I]) -> AnyResolver[P, R, I]:
    """Wrap a class method to ensure instance is specified when the method is called."""

    @functools.wraps(method)
    def method_wrapper(self: Any, info: I, *args: P.args, **kwargs: P.kwargs) -> R:
        _instance_or_bust(info)
        return method(self, info, *args, **kwargs)

    return method_wrapper


P = ParamSpec('P')
R = TypeVar('R')

type ResolverWithContext[**P, R, I: InfoType] = Callable[Concatenate[Any, I, Context, P], R]


@overload
def pass_context(method_or_field: StrawberryField) -> StrawberryField: ...

@overload
def pass_context[**P, R, I: InfoType](
    method_or_field: ResolverWithContext[P, R, I],
) -> Callable[Concatenate[Any, I, P], R]: ...

def pass_context[**P, R, I: InfoType](
    method_or_field: ResolverWithContext[P, R, I] | StrawberryField,
) -> Callable[Concatenate[Any, I, P], R] | StrawberryField:
    """Wrap a resolver function to provide Context as an argument."""

    if isinstance(method_or_field, StrawberryField):
        field = method_or_field
        field.arguments = [arg for arg in field.arguments if arg.python_name != 'context']
        base_resolver = method_or_field.base_resolver
        assert base_resolver is not None
        method = cast('ResolverWithContext[P, R, I]', base_resolver.wrapped_func)
        is_field = True
    else:
        method = method_or_field
        is_field = False

    @functools.wraps(method)
    def method_wrapper(root, info: I, *args: P.args, **kwargs: P.kwargs) -> R:
        instance = _instance_or_bust(info)
        return method(root, info, instance.context, *args, **kwargs)

    if not is_field:
        # The signature of the wrapper method must be changed to remove the context parameter
        # for strawberry's signature reflection to work.
        s = inspect.signature(method)
        params = [param for param in s.parameters.values() if param.name != 'context']
        setattr(method_wrapper, '__signature__', s.replace(parameters=params))  # noqa: B010
        return method_wrapper

    assert base_resolver is not None
    base_resolver.wrapped_func = method_wrapper
    return field


def get_instance_context(info: InfoType) -> Context:
    instance = _instance_or_bust(info)
    return instance.context


def get_instance(info: InfoType) -> Instance:
    instance = _instance_or_bust(info)
    return instance
