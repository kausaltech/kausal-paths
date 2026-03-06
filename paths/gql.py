from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack, overload

import strawberry

from kausal_common.models.types import copy_signature
from kausal_common.strawberry.grapple import grapple_field
from kausal_common.strawberry.helpers import get_or_error
from kausal_common.strawberry.mutations import (
    mutation as base_mutation,
    parse_input,
    prepare_create_update,
    prepare_instance,
)

from nodes.models import InstanceConfig

from .graphql_types import SBInfo as Info

if TYPE_CHECKING:
    from strawberry.extensions import FieldExtension
    from strawberry_django.mutations.fields import DjangoMutationBase

    from kausal_common.strawberry.mutations import (
        MutationArgs,
        ResolverFunc,
    )


def get_ic_or_error(info: Info, ic_id: str) -> InstanceConfig:
    """
    Get an instance config by id, identifier or UUID.

    Raises a GraphQL error if the instance config is not found or not visible for the user.
    """

    # Permission/visibility check is done in `get_or_error()`
    qs = InstanceConfig.objects.qs.by_all_identifiers(ic_id)
    ic = get_or_error(info, qs)

    return ic


@overload
def mutation(*, extensions: list[FieldExtension] | None = None, **kwargs: Unpack[MutationArgs]) -> DjangoMutationBase: ...

@overload
def mutation(resolver: ResolverFunc, **kwargs: Unpack[MutationArgs]) -> DjangoMutationBase: ...


def mutation(
    resolver: ResolverFunc | None = None, *, extensions: list[FieldExtension] | None = None, **kwargs: Unpack[MutationArgs]
) -> DjangoMutationBase:
    ret = base_mutation(extensions=extensions, **kwargs)
    if resolver is not None:
        return ret(resolver)
    return ret


@copy_signature(strawberry.field)
def field(*args, **kwargs) -> Any:
    return grapple_field(*args, **kwargs)

__all__ = ['Info', 'field', 'get_ic_or_error', 'mutation', 'parse_input', 'prepare_create_update', 'prepare_instance']
