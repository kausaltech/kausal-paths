from __future__ import annotations

from typing import TYPE_CHECKING

import graphene
import strawberry as sb
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _
from graphql import DirectiveLocation
from graphql.error import GraphQLError
from graphql.type.definition import GraphQLArgument, GraphQLNonNull
from graphql.type.directives import GraphQLDirective, specified_directives
from graphql.type.scalars import GraphQLID, GraphQLString

from grapple.registry import registry as grapple_registry

from kausal_common.graphene.strawberry_schema import CombinedSchema
from kausal_common.graphene.version_query import Query as ServerVersionQuery

from paths.graphql_types import UnitType
from paths.utils import validate_unit

from frameworks.schema import Mutations as FrameworksMutations, Query as FrameworksQuery
from nodes.schema import Mutations as NodesMutations, Query as NodesQuery
from pages.schema import Query as PagesQuery
from params.schema import Mutations as ParamsMutations, Query as ParamsQuery, types as params_types
from users.schema import Mutations as UsersMutations, Query as UsersQuery

if TYPE_CHECKING:
    from kausal_common.graphene import GQLInfo

    from nodes.units import Unit


CO2E = 'CO<sub>2</sub>e'


class LocaleDirective(GraphQLDirective):
    def __init__(self):
        super().__init__(
            name='locale',
            description='Select locale in which to return data',
            args={
                'lang': GraphQLArgument(
                    type_=GraphQLNonNull(GraphQLString),
                    description='Selected language',
                ),
            },
            locations=[DirectiveLocation.QUERY, DirectiveLocation.MUTATION],
        )


class InstanceDirective(GraphQLDirective):
    def __init__(self):
        super().__init__(
            name='instance',
            description='Select the Paths instance for the request',
            args={
                'hostname': GraphQLArgument(
                    type_=GraphQLString,
                    description='Hostname',
                ),
                'identifier': GraphQLArgument(
                    type_=GraphQLID,
                    description='Instance identifier',
                ),
                'token': GraphQLArgument(
                    type_=GraphQLString,
                    description='Token for accessing the instance',
                ),
            },
            locations=[DirectiveLocation.QUERY, DirectiveLocation.MUTATION],
        )


class Query(NodesQuery, ParamsQuery, PagesQuery, FrameworksQuery, ServerVersionQuery, UsersQuery):
    unit = graphene.Field(UnitType, value=graphene.String(required=True))

    @staticmethod
    def resolve_unit(root: Query, info: GQLInfo, value: str) -> Unit:
        try:
            unit = validate_unit(value)
        except ValidationError:
            raise GraphQLError(_("Invalid unit"), info.field_nodes) from None
        return unit


class Mutations(ParamsMutations, NodesMutations, FrameworksMutations, UsersMutations):
    pass


class InstanceSelectionType(graphene.InputObjectType):
    identifier = graphene.ID(required=False)
    hostname = graphene.String(required=False)


@sb.type
class StrawberryQuery:
    id: str


def generate_schema() -> CombinedSchema:
    from kausal_common.strawberry.registry import strawberry_types

    # We generate the Strawberry schema just to be able to utilize the
    # resolved GraphQL types directly in the Graphene schema.
    sb_schema = sb.Schema(query=StrawberryQuery, types=strawberry_types)

    schema = CombinedSchema(
        sb_schema=sb_schema,
        query=Query,
        mutation=Mutations,
        directives=list(specified_directives) + [LocaleDirective(), InstanceDirective()],
        types=params_types + list(grapple_registry.models.values()),
    )
    return schema

schema = generate_schema()
