from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack, cast

from pydantic_core import core_schema

import pint
import platformdirs
from loguru import logger
from pint import UnitRegistry, facets
from pint.babel_names import _babel_units
from pint.delegates.formatter._compound_unit_helpers import (
    BabelKwds,
    SortFunc,
    prepare_compount_unit,
)
from pint.delegates.formatter._format_helpers import formatter
from pint.delegates.formatter.html import HTMLFormatter
from pint.delegates.formatter.plain import PrettyFormatter

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django_stubs_ext import StrPromise
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema

    from pint.facets.plain import PlainUnit


#Unit = PlainUnit
#Quantity = PlainQuantity

def split_specifier(name: str) -> tuple[str, str | None]:
    m = re.match(r'(.+)\[(\w+)\]', name)
    if not m:
        m = re.match(r'(.+)__(\w+)', name)
    if m:
        name, specifier = m.groups()
    else:
        specifier = None
    return (name, specifier)


class Unit(pint.registry.Unit):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        assert source is Unit
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.union_schema([
                core_schema.is_instance_schema(Unit),
                core_schema.str_schema(),
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
                info_arg=False,
                return_schema=core_schema.str_schema(),
            ),
        )

    @staticmethod
    def _validate(value: str | Unit) -> Unit:
        if isinstance(value, Unit):
            return value
        return unit_registry.parse_units(value)

    @staticmethod
    def _serialize(value: Unit) -> str:
        return str(value)


type PathsUnit = Unit


class Quantity(pint.registry.Quantity):
    pass


@dataclass(frozen=True)
class SpecifierUnitDefinition(facets.plain.UnitDefinition):
    specifier: str | None = None


class CachingUnitRegistry(  # type: ignore[misc]
    facets.GenericSystemRegistry[Quantity, Unit],
    facets.GenericContextRegistry[Quantity, Unit],
    facets.GenericNumpyRegistry[Quantity, Unit],
    facets.GenericMeasurementRegistry[Quantity, Unit],
    facets.GenericNonMultiplicativeRegistry[Quantity, Unit],
    facets.GenericPlainRegistry[Quantity, Unit],
):
    unit_cache: dict[str, Unit] | None = None
    html_formatter: PathsHTMLFormatter
    pretty_formatter: PathsPrettyFormatter
    Unit = Unit
    Quantity = Quantity

    def parse_units(self, input_string: str, as_delta: bool | None = None, case_sensitive: bool | None = None) -> PathsUnit:
        if self.unit_cache is None:
            self.unit_cache = dict()
        cached_unit = self.unit_cache.get(input_string)
        if cached_unit is not None:
            return cached_unit
        ret = super().parse_units(input_string, as_delta, case_sensitive)
        self.unit_cache[input_string] = ret
        return ret

    def get_name(self, name_or_alias: str, case_sensitive: bool | None = None) -> str:
        name, specifier = split_specifier(name_or_alias)
        name = super().get_name(name, case_sensitive)
        if specifier:
            ud = self._units[name]
            possible = {f.name for f in fields(SpecifierUnitDefinition)}
            f = {key: val for key, val in asdict(ud).items() if key in possible}
            f['converter'] = ud.converter
            f['specifier'] = specifier
            ud = SpecifierUnitDefinition(**f)
            name = '%s[%s]' % (name, specifier)
            self._units[name] = ud
        return name


def create_unit_registry():
    cache_dir = os.getenv('PINT_CACHE_DIR', None)
    if cache_dir is None:
        cache_dir = platformdirs.user_cache_dir(appname="pint", appauthor=False)

    def try_create(cache_path: Path) -> CachingUnitRegistry:
        return CachingUnitRegistry(
            preprocessors=[
                lambda s: s.replace('%', ' percent '),
            ],
            on_redefinition='raise',
            cache_folder=str(cache_path),
        )

    cache_path = Path(cache_dir)
    try:
        reg = try_create(cache_path)
    except FileNotFoundError:
        logger.exception("Unit registry creation failed; removing pint cache and trying again")
        # This can sometimes happen with stale cache
        for fn in list(cache_path.glob('*.json')) + list(cache_path.glob('*.pickle')):
            fn.unlink()
        reg = try_create(cache_path)
    return reg


unit_registry = create_unit_registry()


def prepare_units_for_babel(unit: Unit, html: bool = False):
    items = unit._units.items()
    out = {}
    for key, val in items:
        name, specifier = split_specifier(key)
        if name in _babel_units:
            name = _babel_units[name]
        if specifier is not None and specifier == 'CO2e':
            if html:
                name = name + ' CO<sub>2</sub>-e.'
            else:
                name = name + ' CO₂-e.'
        out[name] = val
    return out.items()


class PathsHTMLFormatter(HTMLFormatter):
    def format_unit(
        self,
        unit: PlainUnit | Iterable[tuple[str, Any]],
        uspec: str = "",
        sort_func: SortFunc | None = None,
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        if isinstance(unit, Unit):
            unit = prepare_units_for_babel(unit, html=True)
        if not unit:  # this is the case for dimensionless units
            return ''
        numerator, denominator = prepare_compount_unit(  # FIXME It is possible to have units without nominator like 1/a.
            unit,
            uspec,
            sort_func=sort_func,
            **babel_kwds,
            registry=self._registry,
        )
        return formatter(
            numerator,
            denominator,
            as_ratio=True,
            single_denominator=True,
            product_fmt=r"·",
            division_fmt=r"{}∕{}",  # noqa: RUF001
            power_fmt=r"{}<sup>{}</sup>",
            parentheses_fmt=r"({})",
        )


class PathsPrettyFormatter(PrettyFormatter):
    def format_unit(  # type: ignore
        self,
        unit: PlainUnit | Iterable[tuple[str, Any]],
        uspec: str = "",
        sort_func: SortFunc | None = None,
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        if isinstance(unit, Unit):
            unit = prepare_units_for_babel(unit, html=False)
        if not unit:  # this is the case for dimensionless units
            return ''
        return super().format_unit(unit, uspec, sort_func, **babel_kwds)


unit_registry.html_formatter = PathsHTMLFormatter(registry=cast('UnitRegistry', unit_registry))
unit_registry.pretty_formatter = PathsPrettyFormatter(registry=cast('UnitRegistry', unit_registry))


"""
def _format_paths_html(unit, registry, short: bool, **options):
    print('formatting unit "%s"' % unit)

    return formatter(
        out.items(),
        as_ratio=True,
        single_denominator=True,
        product_fmt=r"·",
        division_fmt=r"{}∕{}",
        power_fmt=r"{}<sup>{}</sup>",
        parentheses_fmt=r"({})",
        **options,
    )

@register_unit_format("Z")
def format_paths_long(unit, registry, **options):
    return _format_paths_html(unit, registry, short=False, **options)


@register_unit_format("~Z")
def format_paths_short(unit, registry, **options):
    return _format_paths_html(unit, registry, short=True, **options)
"""  # noqa: RUF001

def define_custom_units(unit_registry: CachingUnitRegistry):
    # By default, kt is knots, but here kilotonne is the most common
    # usage.
    del unit_registry._units['kt']
    del unit_registry._units['ton']  # The default is 2000 pounds and we don't want to accidentally use that.
    DEFINITIONS = """
    kt = kilotonne
    ton = tonne
    # Mega-kilometers is often used for mileage
    Mkm = gigameters
    EUR = [currency] = €
    CAD = 0.7 EUR
    USD = nan EUR
    SEK = 0.1 EUR
    PLN = 0.2 EUR
    zł = 0.2 EUR
    pcs = [number] = pieces = number
    capita = [population] = cap = inh = inhabitant = person
    MMBtu = 1e6 Btu
    standard_cubic_feet = 0.01 therm = scf
    diesel_gallon_equivalent = 0.1385 MMBtu = DGE
    vehicle = [vehicle] = v = car
    passenger = [passenger] = p = pass = trip
    vkm = vehicle * kilometer
    pkm = passenger * kilometer
    tkm = tonne * kilometer
    VMT = vehicle * mile
    @alias vkm = vkt = v_km
    @alias pkm = pkt = p_km
    job = [employment] = fte = full_time_equivalent
    million_square_meters = 1e6 * meter ** 2 = Msqm
    thousand_square_meters = 1e3 * meter ** 2 = ksqm
    thousand_hectares = 1e7 * meter ** 2 = t_ha
    solid_cubic_meter = 1 m**3 = m3_solid
    Mpkm = 1e6 * pkm
    CO2e = [co2e]
    kt_co2e = kilotonne * CO2e
    t_co2e = tonne * CO2e
    kg_co2e = kg * CO2e
    g_co2e = g * CO2e
    utility = [utility] = Ut
    m3_natural_gas = 36.4 MJ  # https://en.wikipedia.org/wiki/Energy_density
    kg_propane = 49.6 MJ
    l_heating_oil = 37.3 MJ
    l_kerosene = 35 MJ
    l_diesel = 38.6 MJ
    l_gasoline = 34.2 MJ
    l_biodiesel = 33 MJ
    l_gasohol_E10 = 33.18 MJ
    l_ethanol = 24 MJ
    l_methanol = 15.6 MJ
    kg_wood = 18.0 MJ
    kg_peat = 12.8 MJ
    """  # noqa: N806

    for line in DEFINITIONS.strip().splitlines():
        s = line.strip()
        if s.startswith('#'):
            continue
        unit_registry.define(s)

    health_units_path = Path(__file__).parent / 'health_impact_units.txt'
    unit_registry.load_definitions(str(health_units_path))


define_custom_units(unit_registry)
unit_registry.formatter.default_format = '~P'
app_registry = pint.get_application_registry()
app_registry._registry = unit_registry  # pyright: ignore

_translations_added = False

def add_unit_translations():  # noqa: C901
    """
    Add translations for some commonly used units.

    Called from Django's App.ready() handler.
    """
    global _translations_added  # noqa: PLW0603
    if _translations_added:
        return
    _translations_added = True

    from babel import Locale as Loc
    from pint.babel_names import _babel_units
    try:
        from django.conf import settings
    except Exception:
        return
    from typing import TypedDict

    from django.utils import translation
    from django.utils.translation import gettext_lazy as _, pgettext_lazy

    # Define a typed dictionary structure for unit definitions
    class UnitDefinition(TypedDict):
        unit: str
        long: str | _  # type: ignore
        short: str | _ | None # type: ignore

    def set_one(u: str, long: str | StrPromise, short: str | StrPromise | None = None) -> None:
        bu = 'kausal-%s' % u
        if u not in _babel_units:
            _babel_units[u] = bu
        for lang in [la[0] for la in settings.LANGUAGES]:
            loc = Loc(lang.replace('-', '_'))
            loc_data: dict = loc._data  # type: ignore
            all_pats = loc_data['unit_patterns']
            with translation.override(lang):
                unit_pats = dict(long=dict(one=str(long)))
                if short is not None:
                    unit_pats['short'] = dict(one=str(short))
                all_pats[bu] = unit_pats

            # Work around a bug in pint
            cup = loc_data.get('compound_unit_patterns', {})
            if 'per' in cup:
                del cup['per']

    _babel_units['%'] = 'concentr-percent'
    _babel_units['percent'] = 'concentr-percent'

    # Define all units in a properly typed list
    kt_str = pgettext_lazy('kilotonne short', 'kt')

    unit_definitions: list[UnitDefinition] = [
        {'unit': 'capita', 'long': _('capita'), 'short': pgettext_lazy('capita short', 'cap')},
        {'unit': 'kt', 'long': kt_str, 'short': kt_str},
        {'unit': 'EUR', 'long': _('euros'), 'short': '€'},
        {'unit': 'kiloEUR', 'long': _('thousand euros'), 'short': 'k€'},
        {'unit': 'megaEUR', 'long': _('million euros'), 'short': 'M€'},
        {'unit': 'CAD', 'long': _('Canadian dollars'), 'short': '$'},
        {'unit': 'megaCAD', 'long': _('million Canadian dollars'), 'short': 'M$'},
        {'unit': 'kiloSEK', 'long': _('thousand kronor'), 'short': 'kSEK'},
        {'unit': 'megaSEK', 'long': _('million kronor'), 'short': 'MSEK'},
        {'unit': 'terawatt_hour', 'long': _('terawatt hours'), 'short': 'TWh'},
        {'unit': 'gigawatt_hour', 'long': _('gigawatt hours'), 'short': 'GWh'},
        {'unit': 'megawatt_hour', 'long': _('megawatt hour'), 'short': _('MWh')},
        {'unit': 'kilowatt_hour', 'long': _('kilowatt hour'), 'short': 'kWh'},
        {'unit': 'incident', 'long': _('number of cases'), 'short': _('#')},
        {'unit': 'passenger', 'long': _('trip'), 'short': _('trip')},
        {'unit': 'minute', 'long': _('minute'), 'short': _('min')},
        {'unit': 'per_100000py', 'long': _('cases per 100,000 person-years'), 'short': _('#/100000 py')},
        {'unit': 'personal_activity', 'long': _('minutes per day per person'), 'short': _('min/d/cap')},
        {'unit': 'gigaEUR', 'long': _('billion euros'), 'short': _('B€')},
        {'unit': 'solid_cubic_meter', 'long': _('solid m³'), 'short': _('m³ (solid)')},
        {'unit': 'megasolid_cubic_meter', 'long': _('million solid m³'), 'short': _('M m³ (solid)')},
        {'unit': 'g_co2e', 'long': _('grams CO₂e'), 'short': 'gCO₂e'},
        {'unit': 't_co2e', 'long': _('tonnes CO₂e'), 'short': 'tCO₂e'},
        {'unit': 'kt_co2e', 'long': _('ktCO₂e'), 'short': 'ktCO₂e'},
        {'unit': 'metric_ton', 'long': _('tonnes'), 'short': 't'},
        {'unit': 'megametric_ton', 'long': _('megatonnes'), 'short': _('Mt')},
        {'unit': 't_ha', 'long': _('1000 hectares'), 'short': '1000 ha'},
    ]
    #set_one('cap', pgettext_lazy('capita short', 'cap'))

    #set_one('a', pgettext_lazy('year short', 'yr.'))
    #set_one('percent', pgettext_lazy('percent', 'percent'))
    #set_one('%', '%')

    # Special handling for kilowatt_hour and other units in babel
    del_units = ['kilowatt_hour', 'metric_ton']
    for u in del_units:
        if u in _babel_units:
            del _babel_units[u]  # Otherwise fails with compound units.

    # Apply all unit definitions with explicit casting
    for definition in unit_definitions:
        set_one(
            str(definition['unit']),
            definition['long'],
            definition.get('short')
        )

    # Special locale-specific customizations
    loc = Loc('de')
    loc._data['unit_patterns']['duration-year']['short'] = dict(one='a')
