from __future__ import annotations

import strawberry as sb
import strawberry_django
from strawberry import auto

from kausal_common.users import user_or_none

from paths import gql

from frameworks.roles import FrameworkRoleDef

from .models import User


@sb.type(name='UserFrameworkRole')
class UserFrameworkRoleType:
    framework_id: sb.ID
    role_id: str | None
    org_slug: str | None
    org_id: str | None


@strawberry_django.type(User, name='User')
class UserType:
    email: auto
    first_name: auto
    last_name: auto

    @strawberry_django.field
    @staticmethod
    def id(root: sb.Parent[User]) -> sb.ID:
        return sb.ID(str(root.uuid))

    @strawberry_django.field(graphql_type=list[UserFrameworkRoleType])
    @staticmethod
    def framework_roles(root: sb.Parent[User]) -> list[FrameworkRoleDef]:
        # FrameworkRoleDef shares the same field names as UserFrameworkRoleType,
        # so Strawberry serializes the pydantic instances directly by attribute.
        return list(root.extra.framework_roles)


@sb.type
class UsersQuery:
    @sb.field(graphql_type=UserType | None)
    @staticmethod
    def me(info: gql.Info) -> User | None:
        return user_or_none(info.context.user)
