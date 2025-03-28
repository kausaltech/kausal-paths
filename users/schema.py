from __future__ import annotations

import dataclasses

import graphene
import strawberry
from strawberry.types.field import StrawberryField

from kausal_common.strawberry.registry import register_strawberry_type

from frameworks.roles import FrameworkRoleDef
from users.models import User  # noqa: TC001

# Instead of defining a new class for the Strawberry type UserFrameworkRole and copying the fields of FrameworkRoleDef,
# register FrameworkRoleDef with a different name. This way we don't need to juggle data around in
# `UserType.framework_roles()`.
strawberry.type(FrameworkRoleDef, name='UserFrameworkRole')
register_strawberry_type(FrameworkRoleDef)


@register_strawberry_type
@strawberry.type
class UserType:
    id: int
    email: str
    first_name: str
    last_name: str

    _user: strawberry.Private[User]

    def __init__(self, user: User):
        proper_fields = [
            field.name for field in dataclasses.fields(self)
            if not isinstance(field, StrawberryField) and field.name != '_user'
        ]
        for field in proper_fields:
            setattr(self, field, getattr(user, field))
        self._user = user

    @strawberry.field
    def framework_roles(self) -> list[FrameworkRoleDef]:
        return list(self._user.extra.framework_roles)


class Query(graphene.ObjectType):
    me = graphene.Field(UserType)

    def resolve_me(self, info) -> UserType | None:
        user = info.context.user
        if user.is_authenticated:
            return UserType(user)
        return None
