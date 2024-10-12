from __future__ import annotations

from typing import TYPE_CHECKING

import graphene
from graphene_django import DjangoObjectType
from graphql import GraphQLError

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

    def resolve_me(self, info):
        user = info.context.user
        if user.is_authenticated:
            return user
        return None


class RegisterUser(graphene.Mutation):
    class Arguments:
        email = graphene.String(required=True)
        password = graphene.String(required=True)

    user = graphene.Field(UserType)

    def mutate(self, info: GQLInfo, email: str, password: str):
        email = email.strip().lower()
        if User.objects.filter(email=email).exists():
            raise GraphQLError("User with email already exists", nodes=info.field_nodes)
        user = User(email=email)
        user.set_password(password)
        user.save()
        return RegisterUser(user=user)


class Mutations(graphene.ObjectType):
    register_user = RegisterUser.Field()
