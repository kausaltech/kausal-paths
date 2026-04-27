from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar, cast, overload

import graphene
from django.utils.module_loading import import_string
from graphql.error import GraphQLError
from strawberry.types.field import StrawberryField
from strawberry.types.info import Info as StrawberryInfo

from paths.graphql_types import AdminButton
from paths.schema_context import PathsGraphQLContext

from nodes.instance import Instance

if TYPE_CHECKING:
    from django.db.models import Model

    from paths.types import GQLInstanceInfo, PathsGQLInfo

    from admin_site.viewsets import PathsViewSet
    from nodes.context import Context
    from nodes.instance import Instance

    from .graphql_types import SBInfo


@dataclass
class GraphQLPerfNode:
    id: str


def _instance_or_bust(info: InfoType) -> Instance:
    if (instance := getattr(info.context, 'instance', None)) is None:
        raise GraphQLError(
            "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            info.field_nodes,
        )
    if instance is None:
        raise GraphQLError(
            "Instance is not set in the context. Use the 'instance' directive or HTTP headers.",
            info.field_nodes,
        )
    return instance


type InfoType = GQLInstanceInfo | SBInfo | StrawberryInfo[PathsGraphQLContext[Any]]

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
type ResolverWithRootAndContext[**P, R] = Callable[Concatenate[Any, Context, P], R]
type ResolverWithInfoAndContext[**P, R, I: InfoType] = Callable[Concatenate[I, Context, P], R]
type ResolverContextOnly[**P, R] = Callable[Concatenate[Context, P], R]

type ContextResolver[**P, R, I: InfoType] = (
    ResolverWithContext[P, R, I]
    | ResolverWithRootAndContext[P, R]
    | ResolverWithInfoAndContext[P, R, I]
    | ResolverContextOnly[P, R]
)


def _get_public_context_resolver_signature(sig: inspect.Signature) -> inspect.Signature:
    public_params = [param for param in sig.parameters.values() if param.name != 'context']
    if any(param.name == 'info' for param in public_params):
        return sig.replace(parameters=public_params)

    insert_at = 1 if public_params and public_params[0].name in {'self', 'cls', 'root'} else 0
    public_params.insert(
        insert_at,
        inspect.Parameter(
            'info',
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=StrawberryInfo,
        ),
    )
    return sig.replace(parameters=public_params)


def _call_context_resolver[R](
    method: Callable[..., R],
    sig: inspect.Signature,
    public_sig: inspect.Signature,
    *args: Any,
    **kwargs: Any,
) -> R:
    bound = public_sig.bind_partial(*args, **kwargs)
    info = cast('InfoType', bound.arguments['info'])
    instance = _instance_or_bust(info)
    call_args: list[Any] = []
    call_kwargs: dict[str, Any] = {}

    for param in sig.parameters.values():
        if param.name == 'context':
            value = instance.context
        elif param.name in bound.arguments:
            value = bound.arguments[param.name]
        else:
            continue

        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            call_args.append(value)
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            call_args.extend(cast('tuple[Any, ...]', value))
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            call_kwargs[param.name] = value
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            call_kwargs.update(cast('dict[str, Any]', value))

    return method(*call_args, **call_kwargs)


@overload
def pass_context[**P, R, I: InfoType](method_or_field: ResolverWithContext[P, R, I]) -> Callable[Concatenate[Any, I, P], R]: ...


@overload
def pass_context[**P, R](method_or_field: ResolverWithRootAndContext[P, R]) -> Callable[Concatenate[Any, P], R]: ...


@overload
def pass_context[**P, R, I: InfoType](
    method_or_field: ResolverWithInfoAndContext[P, R, I],
) -> Callable[Concatenate[I, P], R]: ...


@overload
def pass_context[**P, R](
    method_or_field: ResolverContextOnly[P, R],
) -> Callable[P, R]: ...


def pass_context[**P, R, I: InfoType](
    method_or_field: object,
) -> Callable[..., Any]:
    """Wrap a resolver function to provide Context as an argument."""

    if isinstance(method_or_field, StrawberryField) or not callable(method_or_field):
        msg = 'pass_context must wrap the resolver function before @sb.field'
        raise TypeError(msg)

    method = cast('Callable[..., R]', method_or_field)

    sig = inspect.signature(method)
    public_sig = _get_public_context_resolver_signature(sig)

    @functools.wraps(cast('Callable[..., Any]', method))
    def method_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _call_context_resolver(method, sig, public_sig, *args, **kwargs)

    del method_wrapper.__wrapped__
    setattr(method_wrapper, '__signature__', public_sig)  # noqa: B010
    return method_wrapper


def get_instance_context(info: InfoType) -> Context:
    instance = _instance_or_bust(info)
    return instance.context


def get_instance(info: InfoType) -> Instance:
    instance = _instance_or_bust(info)
    return instance


class AdminButtonsMixin:
    admin_buttons = graphene.List(graphene.NonNull(AdminButton), required=True)

    @staticmethod
    def resolve_admin_buttons(root: Model, info: PathsGQLInfo) -> list[AdminButton]:
        if not info.context.user.is_staff:
            return []

        view_set_class: type[PathsViewSet] = import_string(root.VIEWSET_CLASS)  # type: ignore
        view_set = view_set_class()

        # if isinstance(view_set.permission_policy, InstanceConfigPermissionPolicy):
        #     view_set.permission_policy.disable_admin_plan_check()

        if not hasattr(view_set, 'get_index_view_buttons'):
            raise ValueError(f'get_index_view_buttons method not found for view set {view_set.__class__.__name__}')
        user = info.context.user
        active_instance = info.context.instance_config
        buttons = view_set.get_index_view_buttons(user, root, active_instance)  # type: ignore[attr-defined]

        # TODO: Temporary workaround to support both the new and old attribute
        # name for icon, making the code work for modeladmin code as well. The
        # GraphQL queries should be updated to use the new attribute name once
        # actions have migrated from modeladmin.
        for button in buttons:
            button.icon = button.icon_name

        return buttons
