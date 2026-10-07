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

from nodes.context import Context

if TYPE_CHECKING:
    from django.db.models import Model

    from paths.types import PathsGQLInfo

    from admin_site.viewsets import PathsViewSet
    from nodes.instance import Instance

    from .graphql_types import SBInfo


@dataclass
class GraphQLPerfNode:
    id: str


type InfoType = PathsGQLInfo | SBInfo | StrawberryInfo[PathsGraphQLContext]


def graphql_error_nodes(info: InfoType):
    """Return graphql-core AST nodes from Graphene or Strawberry resolver info."""
    raw_info: Any = getattr(info, '_raw_info', info)
    return raw_info.field_nodes


ROOT_PARAMETER_NAMES = frozenset({'self', 'root'})

P = ParamSpec('P')
R = TypeVar('R')

type ResolverWithContext[**P, R, I: InfoType] = Callable[Concatenate[Any, I, Context, P], R]
type ResolverWithRootAndContext[**P, R] = Callable[Concatenate[Any, Context, P], R]


def _get_public_context_resolver_signature(sig: inspect.Signature) -> inspect.Signature:
    public_params = [param for param in sig.parameters.values() if param.name != 'context']
    if not public_params or public_params[0].name not in ROOT_PARAMETER_NAMES:
        msg = 'pass_context resolvers take the runtime from their root; the first parameter must be `root` or `self`'
        raise TypeError(msg)
    return sig.replace(parameters=public_params)


def _call_context_resolver[R](
    method: Callable[..., R],
    sig: inspect.Signature,
    public_sig: inspect.Signature,
    *args: Any,
    **kwargs: Any,
) -> R:
    bound = public_sig.bind_partial(*args, **kwargs)
    root_name = next(iter(public_sig.parameters))
    context = root_context(bound.arguments[root_name])
    call_args: list[Any] = []
    call_kwargs: dict[str, Any] = {}

    for param in sig.parameters.values():
        if param.name == 'context':
            value = context
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


def pass_context[**P, R, I: InfoType](
    method_or_field: object,
) -> Callable[..., Any]:
    """
    Wrap a resolver function to provide the runtime Context of its root object.

    The context comes from the root (`root.context`), never from the request, so a
    field resolves against the same model runtime as the object it is a field of.
    A request can hold several runtimes of one instance -- `InstanceType.model`
    with parameter overrides builds its own -- and a resolver that read the
    request's default runtime instead would silently mix their states.
    """

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


def root_context(root: object) -> Context:
    """Return the model runtime a GraphQL root object is bound to."""
    context = getattr(root, 'context', None)
    if not isinstance(context, Context):
        msg = f'{type(root).__name__} is not bound to a model runtime'
        raise TypeError(msg)
    return context


def default_instance(info: InfoType) -> Instance:
    """
    Return the runtime of the instance the operation names, without overrides.

    For entry points that have no root object to take a runtime from: top-level
    query fields, mutations and pages. A field of a runtime-bound object must use
    `root_context` (or `pass_context`) instead.
    """
    context = cast('PathsGraphQLContext', info.context)
    if context.instance_resources is None or context.instance_resources.default_config is None:
        raise GraphQLError(
            "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            graphql_error_nodes(info),
        )
    return context.instance_resources.require_instance()


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
