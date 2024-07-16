from __future__ import annotations

import graphene
from graphene_django import DjangoObjectType
from graphql import GraphQLError

from kausal_common.graphene import GQLInfo

from .models import User


class UserType(DjangoObjectType):
    class Meta:
        model = User
        fields = ('id', 'email',)


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
