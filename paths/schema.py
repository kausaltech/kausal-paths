from __future__ import annotations

from typing import TYPE_CHECKING, cast

import graphene
import strawberry as sb
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _
from graphql import DirectiveLocation
from graphql.error import GraphQLError
from graphql.type.definition import GraphQLArgument, GraphQLNonNull
from graphql.type.directives import GraphQLDirective, specified_directives
from graphql.type.scalars import GraphQLID, GraphQLString
from strawberry.tools import merge_types

from grapple.registry import registry as grapple_registry

from kausal_common.deployment import test_mode_enabled
from kausal_common.graphene.strawberry_schema import CombinedSchema
from kausal_common.graphene.version_query import Query as ServerVersionQuery
from kausal_common.testing.schema import TestModeMutations

from paths.graphql_types import UnitType
from paths.utils import validate_unit

from frameworks.schema import Mutations as FrameworksMutations, Query as FrameworksQuery
from nodes.schema import Mutations as NodesMutations, Query as NodesQuery
from pages.schema import Query as PagesQuery
from params.schema import Mutations as ParamsMutations, Query as ParamsQuery, types as params_types
from users.schema import Query as UsersQuery

if TYPE_CHECKING:
    from kausal_common.graphene import GQLInfo

    from paths.types import GQLInstanceContext

    from nodes.units import Unit


CO2E = 'CO<sub>2</sub>e'


LocaleDirective = GraphQLDirective(
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


InstanceDirective = GraphQLDirective(
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
            raise GraphQLError(_('Invalid unit'), info.field_nodes) from None
        return unit


class Mutations(ParamsMutations, NodesMutations, FrameworksMutations):
    pass


type SBInfo = sb.Info['GQLInstanceContext']

@sb.input(name='InstanceContext')
class InstanceContextInput:
    hostname: str | None
    identifier: sb.ID | None
    locale: str | None


@sb.directive(
    locations=[DirectiveLocation.QUERY, DirectiveLocation.MUTATION],
    name='context',
    description='Paths instance context, including the selected locale',
)
def context_directive(info: SBInfo, input: InstanceContextInput):
    return


@sb.type(name='NodeType')
class SBNode:
    id: sb.ID


@sb.type(name='Query')
class SBQuery:  # FIXME this does not seem to have any effect at the moment
    @sb.field
    def node(self, info: SBInfo, id: str) -> SBNode:
        context = info.context.instance.context
        node = context.get_node(id)
        return SBNode(id=cast(sb.ID, node.id))


SB_MUTATION_TYPES: list[type] = []
if test_mode_enabled():
    SB_MUTATION_TYPES.append(TestModeMutations)

SBMutation: type | None = None
if SB_MUTATION_TYPES:
    SBMutation = merge_types('Mutation', tuple(SB_MUTATION_TYPES))


def generate_strawberry_schema() -> sb.Schema:
    from kausal_common.strawberry.registry import strawberry_types

    sb_schema = sb.Schema(
        query=SBQuery, mutation=SBMutation, types=strawberry_types, directives=[context_directive]
    )
    return sb_schema


def generate_schema() -> tuple[sb.Schema, CombinedSchema]:
    # We generate the Strawberry schema just to be able to utilize the
    # resolved GraphQL types directly in the Graphene schema.
    sb_schema = generate_strawberry_schema()

    schema = CombinedSchema(
        sb_schema=sb_schema,
        query=Query,
        mutation=Mutations,
        directives=list(specified_directives) + [LocaleDirective, InstanceDirective],
        types=params_types + list(grapple_registry.models.values()),
    )
    return sb_schema, schema


sb_schema, schema = generate_schema()
