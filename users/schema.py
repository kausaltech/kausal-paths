from __future__ import annotations

from typing import TYPE_CHECKING

import graphene
from graphene_django import DjangoObjectType

from kausal_common.users import user_or_none

from .models import User

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kausal_common.graphene import GQLInfo

    from frameworks.roles import FrameworkRoleDef


class UserFrameworkRole(graphene.ObjectType):
    framework_id = graphene.ID(required=True)
    role_id = graphene.String(required=False)
    org_slug = graphene.String(required=False)
    org_id = graphene.String(required=False)


class UserType(DjangoObjectType):
    framework_roles = graphene.List(graphene.NonNull(UserFrameworkRole))

    class Meta:
        model = User
        fields = ('id', 'email', 'first_name', 'last_name')

    @staticmethod
    def resolve_framework_roles(root: User, info: GQLInfo) -> Sequence[FrameworkRoleDef]:
        return root.extra.framework_roles


class Query(graphene.ObjectType):
    me = graphene.Field(UserType)

    def resolve_me(self, info: GQLInfo) -> User | None:
        user = info.context.get_user()
        return user_or_none(user)
