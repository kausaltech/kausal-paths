from __future__ import annotations

import graphene
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _
from graphql.error import GraphQLError
from graphql.type.definition import GraphQLArgument, GraphQLNonNull
from graphql.type.directives import DirectiveLocation, GraphQLDirective, specified_directives
from graphql.type.scalars import GraphQLID, GraphQLString
from grapple.registry import registry as grapple_registry

from frameworks.schema import Mutations as FrameworksMutations, Query as FrameworksQuery
from kausal_common.graphene.version_query import Query as ServerVersionQuery
from nodes.schema import Mutations as NodesMutations, Query as NodesQuery
from pages.schema import Query as PagesQuery
from users.schema import Query as UsersQuery, Mutations as UsersMutations
from params.schema import Mutations as ParamsMutations, Query as ParamsQuery, types as params_types
from paths.graphql_helpers import GQLInfo
from paths.graphql_types import UnitType
from paths.utils import validate_unit

CO2E = 'CO<sub>2</sub>e'


class Query(NodesQuery, ParamsQuery, PagesQuery, FrameworksQuery, ServerVersionQuery, UsersQuery):
    unit = graphene.Field(UnitType, value=graphene.String(required=True))

    @staticmethod
    def resolve_unit(root: 'Query', info: GQLInfo, value: str):
        try:
            unit = validate_unit(value)
        except ValidationError:
            raise GraphQLError(_("Invalid unit"), info.field_nodes)
        return unit


class Mutations(ParamsMutations, NodesMutations, FrameworksMutations, UsersMutations):
    pass


class LocaleDirective(GraphQLDirective):
    def __init__(self):
        super().__init__(
            name='locale',
            description='Select locale in which to return data',
            args={
                'lang': GraphQLArgument(
                    type_=GraphQLNonNull(GraphQLString),
                    description='Selected language'
                )
            },
            locations=[DirectiveLocation.QUERY, DirectiveLocation.MUTATION]
        )


class InstanceDirective(GraphQLDirective):
    def __init__(self):
        super().__init__(
            name='instance',
            description='Select the Paths instance for the request',
            args={
                'hostname': GraphQLArgument(
                    type_=GraphQLString,
                    description='Hostname'
                ),
                'identifier': GraphQLArgument(
                    type_=GraphQLID,
                    description='Instance identifier'
                ),
                'token': GraphQLArgument(
                    type_=GraphQLString,
                    description='Token for accessing the instance'
                ),
            },
            locations=[DirectiveLocation.QUERY, DirectiveLocation.MUTATION]
        )


schema = graphene.Schema(
    query=Query,
    directives=list(specified_directives) + [LocaleDirective(), InstanceDirective()],
    types=params_types + list(grapple_registry.models.values()),
    mutation=Mutations,
)
