from __future__ import annotations

from typing import TYPE_CHECKING

import graphene
import strawberry as sb
from django.utils.translation import get_language
from graphene_django.converter import convert_django_field, get_django_field_description

from babel import Locale

from kausal_common.strawberry.registry import register_strawberry_type

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


@register_strawberry_type
@sb.type(name='UnitType')
class UnitType:
    @sb.field
    def short(self, parent: sb.Parent[Unit]) -> str:
        return format_unit(parent, html=False)

    @sb.field
    def long(self, parent: sb.Parent[Unit]) -> str:
        return format_unit(parent, long=True, html=False)

    @sb.field
    def html_short(self, parent: sb.Parent[Unit]) -> str:
        return format_unit(parent, long=False, html=True)

    @sb.field
    def html_long(self, parent: sb.Parent[Unit]) -> str:
        return format_unit(parent, long=True, html=True)


@convert_django_field.register(UnitField)  # pyright: ignore
def convert_unit_field(field, registry=None):
    return graphene.Field(
        UnitType, description=get_django_field_description(field), required=not field.null,
    )
