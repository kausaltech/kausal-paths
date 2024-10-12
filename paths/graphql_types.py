from __future__ import annotations

from typing import TYPE_CHECKING

import graphene
from django.utils.translation import get_language
from graphene_django.converter import convert_django_field, get_django_field_description

from babel import Locale

from paths.utils import UnitField

from nodes.units import Unit, unit_registry

if TYPE_CHECKING:
    from kausal_common.graphene import GQLInfo


locale_cache: dict[str, Locale] = {}


def format_unit(unit: Unit, long: bool = False, html: bool = False) -> str:  # noqa: C901
    if dict(unit._units) == dict(percent=1):
        return '%'

    full_lang = get_language()
    locale = locale_cache.get(full_lang)
    if not locale:
        locale = Locale.parse(full_lang, sep='-')
        locale_cache[full_lang] = locale
    lang = locale.language
    if html:
        formatter = unit_registry.html_formatter
    else:
        formatter = unit_registry.pretty_formatter  # type: ignore
    #fmt = 'Zh' if html else 'Zp'
    f = formatter.format_unit(
        unit, uspec='Z', sort_func=None, use_plural=not long, length='long' if long else 'short', locale=locale,
    )
    if not long:
        if f == 't/a/cap':
            if lang == 'de':
                return 't/a/Einw.'
            if lang == 'en':
                return 't/a/inh.'
    elif f == 't/a/cap':
        if lang == 'de':
            return 't CO₂e/Jahr/Einw.'
        if lang == 'en':
            return 't CO₂e/year/inh.'

    return f


def resolve_unit(root: Unit | str, info: GQLInfo) -> Unit:
    if isinstance(root, Unit):
        return root
    if isinstance(root, str):
        return unit_registry.parse_units(root)
    raise ValueError("Invalid type for unit: %s (expecting 'str' or 'Unit')" % type(root))


class UnitType(graphene.ObjectType):
    short = graphene.String(required=True)
    long = graphene.String(required=True)
    html_short = graphene.String(required=True)
    html_long = graphene.String(required=True)

    @staticmethod
    def resolve_short(root: Unit, info) -> str:
        val = format_unit(root, html=False)
        return val

    @staticmethod
    def resolve_long(root: Unit, info) -> str:
        return format_unit(root, long=True, html=False)

    @staticmethod
    def resolve_html_short(root: Unit, info) -> str:
        return format_unit(root, long=False, html=True)

    @staticmethod
    def resolve_html_long(root: Unit, info) -> str:
        return format_unit(root, long=True, html=True)


@convert_django_field.register(UnitField)  # pyright: ignore
def convert_unit_field(field, registry=None):
    return graphene.Field(UnitType, description=get_django_field_description(field), required=not field.null)
