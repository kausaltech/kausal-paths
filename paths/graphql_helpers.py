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

from paths.graphql_types import AdminButton

from admin_site.viewsets import PathsViewSet, admin_req
from nodes.instance import Instance

if TYPE_CHECKING:
    from django.db.models import Model

    from kausal_common.graphene import GQLInfo

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


class AdminButtonsMixin:
    admin_buttons = graphene.List(graphene.NonNull(AdminButton), required=True)

    @staticmethod
    def resolve_admin_buttons(root: Model, info: GQLInfo) -> list[AdminButton]:
        if not info.context.user.is_staff:
            return []

        view_set_class: type[PathsViewSet] = import_string(root.VIEWSET_CLASS)  # type: ignore
        view_set = view_set_class()

        # if isinstance(view_set.permission_policy, InstanceConfigPermissionPolicy):
        #     view_set.permission_policy.disable_admin_plan_check()

        if not hasattr(view_set, 'get_index_view_buttons'):
            raise ValueError(f'get_index_view_buttons method not found for view set {view_set.__class__.__name__}')
        user = admin_req(info.context).user
        instance_config = user.get_active_instance()
        buttons = view_set.get_index_view_buttons(user, root, instance_config)  # type: ignore[attr-defined]

        # TODO: Temporary workaround to support both the new and old attribute
        # name for icon, making the code work for modeladmin code as well. The
        # GraphQL queries should be updated to use the new attribute name once
        # actions have migrated from modeladmin.
        for button in buttons:
            button.icon = button.icon_name

        return buttons
