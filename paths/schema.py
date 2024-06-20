from babel import Locale
from django.core.exceptions import ValidationError
import graphene
from graphql.type.definition import GraphQLArgument, GraphQLNonNull
from graphql.type.directives import (
    DirectiveLocation, GraphQLDirective, specified_directives
)
from graphql.error import GraphQLError
from graphql.type.scalars import GraphQLID, GraphQLString
from django.utils.translation import get_language, gettext as _

from grapple.registry import registry as grapple_registry

from nodes.schema import Query as NodesQuery, Mutations as NodesMutations
from nodes.units import Unit, unit_registry
from pages.schema import Query as PagesQuery
from params.schema import (
    Mutations as ParamsMutations, Query as ParamsQuery, types as params_types
)
from paths.graphql_helpers import GQLInfo
from paths.utils import validate_unit

CO2E = 'CO<sub>2</sub>e'

locale_cache: dict[str, Locale] = {}


def format_unit(unit: Unit, long: bool = False, html: bool = False) -> str:
    if dict(unit._units) == dict(percent=1):
        return '%'

    full_lang = get_language()
    locale = locale_cache.get(full_lang, None)
    if not locale:
        locale = Locale.parse(full_lang, sep='-')
        locale_cache[full_lang] = locale
    lang = locale.language
    fmt = '~P' if not html else '~Z'
    f = unit_registry.formatter.format_unit_babel(
        unit, spec=fmt, length='long' if long else 'short', locale=locale
    )
    if not long:
        if f == 't/a/cap':
            if lang == 'de':
                return 't/a/Einw.'
            elif lang == 'en':
                return 't/a/inh.'
    else:
        if f == 't/a/cap':
            if lang == 'de':
                return 't CO₂e/Jahr/Einw.'
            elif lang == 'en':
                return 't CO₂e/year/inh.'

    return f


class UnitType(graphene.ObjectType):
    short = graphene.String(required=True)
    long = graphene.String(required=True)
    html_short = graphene.String(required=True)
    html_long = graphene.String(required=True)

    @staticmethod
    def resolve_short(root: Unit, info):
        val = format_unit(root, html=False)
        return val

    @staticmethod
    def resolve_long(root: Unit, info):
        return format_unit(root, long=True, html=False)

    @staticmethod
    def resolve_html_short(root: Unit, info):
        return format_unit(root, long=False, html=True)

    @staticmethod
    def resolve_html_long(root: Unit, info):
        return format_unit(root, long=True, html=True)


class Query(NodesQuery, ParamsQuery, PagesQuery):
    unit = graphene.Field(UnitType, value=graphene.String(required=True))

    @staticmethod
    def resolve_unit(root: 'Query', info: GQLInfo, value: str):
        try:
            unit = validate_unit(value)
        except ValidationError:
            raise GraphQLError(_("Invalid unit"), info.field_nodes)
        return unit


class Mutations(ParamsMutations, NodesMutations):
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
