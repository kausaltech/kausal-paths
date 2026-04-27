from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, overload

import strawberry
from strawberry.types.field import StrawberryField

from kausal_common.models.types import copy_signature
from kausal_common.strawberry.fields import field as base_field
from kausal_common.strawberry.grapple import grapple_field
from kausal_common.strawberry.helpers import get_or_error
from kausal_common.strawberry.mutations import (
    mutation as base_mutation,
    parse_input,
    prepare_create_update,
    prepare_instance,
)

from .graphql_types import SBInfo as Info

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from strawberry.extensions import FieldExtension
    from strawberry_django.mutations.fields import DjangoMutationBase

    from kausal_common.strawberry.mutations import (
        ResolverFunc,
    )

    from paths.schema_context import PathsGraphQLContext

    from nodes.instance import Instance
    from nodes.models import InstanceConfig


type InstanceInfo = strawberry.Info['PathsGraphQLContext[Instance]']


def get_ic_or_error(info: Info, ic_id: str) -> InstanceConfig:
    """
    Get an instance config by id, identifier or UUID.

    Raises a GraphQL error if the instance config is not found or not visible for the user.
    """
    from nodes.models import InstanceConfig

    # Permission/visibility check is done in `get_or_error()`
    qs = InstanceConfig.objects.qs.by_all_identifiers(ic_id)
    ic = get_or_error(info, qs)

    return ic


@overload
def mutation(
    *,
    name: str | None = None,
    description: str | None = None,
    permission_classes: list[type[strawberry.BasePermission]] | None = None,
    deprecation_reason: str | None = None,
    default: Any = dataclasses.MISSING,
    default_factory: Callable[..., object] | object = dataclasses.MISSING,
    metadata: Mapping[Any, Any] | None = None,
    directives: Sequence[object] | None = (),
    extensions: list[FieldExtension] | None = None,
    graphql_type: Any | None = None,
) -> DjangoMutationBase | Callable[[ResolverFunc], DjangoMutationBase]: ...


@overload
def mutation(
    resolver: ResolverFunc,
    *,
    name: str | None = None,
    description: str | None = None,
    permission_classes: list[type[strawberry.BasePermission]] | None = None,
    deprecation_reason: str | None = None,
    default: Any = dataclasses.MISSING,
    default_factory: Callable[..., object] | object = dataclasses.MISSING,
    metadata: Mapping[Any, Any] | None = None,
    directives: Sequence[object] | None = (),
    extensions: list[FieldExtension] | None = None,
    graphql_type: Any | None = None,
) -> DjangoMutationBase: ...


def mutation(  # noqa: PLR0913
    resolver: ResolverFunc | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    permission_classes: list[type[strawberry.BasePermission]] | None = None,
    deprecation_reason: str | None = None,
    default: Any = dataclasses.MISSING,
    default_factory: Callable[..., object] | object = dataclasses.MISSING,
    metadata: Mapping[Any, Any] | None = None,
    directives: Sequence[object] | None = (),
    extensions: list[FieldExtension] | None = None,
    graphql_type: Any | None = None,
) -> DjangoMutationBase | Callable[[ResolverFunc], DjangoMutationBase]:
    """
    Create a mutation type with Django-specific bells and whistles.

    It will wrap the resolver in a Django transaction and handle Django-specific errors.
    The GraphQL return type will be a union of the resolver return type and the OperationInfo type
    for error handling. ValidationError exceptions will be converted to GraphQL validation errors.

    Args:
        resolver: The resolver for the field. It can be a sync or async function.
        name: The GraphQL name of the field.
        description: The GraphQL description of the field.
        permission_classes: The permission classes required to access the field.
        deprecation_reason: The deprecation reason for the field.
        default: The default value for the field.
        default_factory: The default factory for the field.
        metadata: The metadata for the field.
        directives: The directives for the field.
        extensions: The extensions for the field.
        graphql_type: The GraphQL type for the field, useful when you want to use a
            different type in the resolver than the one in the schema.
        init: This parameter is used by PyRight to determine whether this field is
            added in the constructor or not. It is not used to change any behavior at
            the moment.

    """
    ret = base_mutation(
        name=name,
        description=description,
        permission_classes=permission_classes,
        deprecation_reason=deprecation_reason,
        default=default,
        default_factory=default_factory,
        metadata=metadata,
        directives=directives,
        extensions=extensions,
        graphql_type=graphql_type,
    )
    if resolver is not None:
        return ret(resolver)
    return ret


class InstanceField(StrawberryField):
    def get_result(self, source: Any, info: Info | None, args: list[Any], kwargs: Any) -> Awaitable[Any] | Any:
        return super().get_result(source, info, args, kwargs)


@copy_signature(strawberry.field)
def field(*args, **kwargs) -> Any:
    return grapple_field(*args, **kwargs)


@copy_signature(strawberry.field)
def instance_field(*args, **kwargs) -> Any:
    from nodes.schema import InstanceType

    if 'graphql_type' not in kwargs:
        kwargs['graphql_type'] = InstanceType

    return base_field(*args, custom_field_class=InstanceField, **kwargs)


__all__ = [
    'Info',
    'field',
    'get_ic_or_error',
    'instance_field',
    'mutation',
    'parse_input',
    'prepare_create_update',
    'prepare_instance',
]
