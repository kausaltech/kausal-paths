from __future__ import annotations

from typing import Annotated

import strawberry as sb
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _
from graphql import DirectiveLocation
from graphql.error import GraphQLError
from strawberry.schema import Schema as StrawberrySchema
from strawberry.tools import merge_types

from grapple.registry import registry as grapple_registry

from kausal_common import graphql_gis  # noqa: F401  # pyright: ignore[reportUnusedImport]
from kausal_common.deployment import test_mode_enabled
from kausal_common.graphene.version_query import Query as ServerVersionQuery
from kausal_common.models.types import copy_signature
from kausal_common.strawberry.extensions import LoggingTracingExtension
from kausal_common.strawberry.schema import Schema as UnifiedSchema
from kausal_common.testing.schema import TestModeMutations

from paths import gql
from paths.context import realm_context
from paths.graphql_types import UnitType
from paths.schema_context import PathsGraphQLContext
from paths.utils import validate_unit

from frameworks.mutations import FrameworkMutation
from frameworks.schema import Mutations as FrameworksMutations, Query as FrameworksQuery
from nodes.models import InstanceConfig
from nodes.schema import (
    Mutation as NodesMutation,
    SBQuery as SBNodesQuery,
    Subscription as NodesSubscription,
)
from nodes.schema_model_editor import ModelEditorMutation, ModelEditorQuery
from nodes.units import Unit
from orgs.models import Organization
from orgs.schema import OrganizationNode, Query as OrgsQuery
from pages.schema import Query as PagesQuery
from params.schema import SBMutation as SBParamsMutation, SBQuery as SBParamsQuery, types as params_types
from users.schema import Query as UsersQuery

CO2E = 'CO<sub>2</sub>e'


@sb.directive(
    locations=[DirectiveLocation.QUERY, DirectiveLocation.MUTATION],
    name='instance',
    description='Select the Paths instance for the operation',
)
def instance_directive(
    info: gql.Info,
    hostname: Annotated[str | None, sb.argument(description='Hostname')],
    identifier: Annotated[sb.ID | None, sb.argument(description='Instance identifier')],
    token: Annotated[str | None, sb.argument(description='Token for accessing the instance')],
):
    pass


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
def context_directive(info: gql.Info, input: InstanceContextInput):
    return


@sb.type
class CommonQuery:
    @sb.field(graphql_type=UnitType)
    def unit(self, info: gql.Info, value: str) -> Unit:
        try:
            unit = validate_unit(value)
        except ValidationError:
            raise GraphQLError(_('Invalid unit'), info.field_nodes) from None
        return unit

    @sb.field(graphql_type=list[OrganizationNode])
    @staticmethod
    def instance_organizations(
        instance: sb.ID | None = None,
        with_ancestors: bool = False,
    ) -> list[Organization]:
        if not instance:
            instance_obj = realm_context.get().realm
        else:
            instance_obj = InstanceConfig.objects.get(identifier=instance)
        return list(Organization.objects.qs.available_for_instance(instance_obj))


class GrapheneQuery(PagesQuery, FrameworksQuery, ServerVersionQuery, UsersQuery, OrgsQuery):
    class Meta:
        name = 'Query'


class GrapheneMutations(FrameworksMutations):
    pass


SBQuery = merge_types('Query', (SBNodesQuery, ModelEditorQuery, SBParamsQuery, CommonQuery))


@sb.type
class Query(GrapheneQuery, SBQuery):  # type: ignore[valid-type, misc]
    pass


SB_MUTATION_TYPES: list[type] = [
    NodesMutation,
    ModelEditorMutation,
    SBParamsMutation,
    FrameworkMutation,
]
if test_mode_enabled():
    SB_MUTATION_TYPES.append(TestModeMutations)

SBMutation = merge_types('Mutation', tuple(SB_MUTATION_TYPES))


@sb.type
class Mutation(GrapheneMutations, SBMutation):  # type: ignore[valid-type, misc]
    pass


Subscription = merge_types('Subscription', (NodesSubscription,))


class PathsSchema(UnifiedSchema):
    @copy_signature(StrawberrySchema.__init__)
    def __init__(self, *args, **kwargs):
        from .schema_context import (
            ActivateInstanceContextExtension,
            DetermineInstanceContextExtension,
            PathsAuthenticationExtension,
            PathsExecutionCacheExtension,
        )

        extensions = kwargs.pop('extensions', [])
        extensions.extend([
            LoggingTracingExtension(context_class=PathsGraphQLContext),
            DetermineInstanceContextExtension,
            PathsExecutionCacheExtension,
            ActivateInstanceContextExtension,
            PathsAuthenticationExtension,
        ])
        kwargs['extensions'] = extensions
        super().__init__(*args, **kwargs)


def generate_strawberry_schema() -> sb.Schema:
    from kausal_common.strawberry.registry import strawberry_types

    all_types = set(strawberry_types)
    all_types.update(params_types)
    all_types.update(list(grapple_registry.models.values()))
    schema = PathsSchema(
        query=Query,
        mutation=Mutation,
        types=all_types,
        directives=[context_directive, instance_directive],
        subscription=Subscription,
    )
    return schema


schema = generate_strawberry_schema()
