from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple, TypeAlias, Unpack, cast

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

    from django.utils.functional import _StrPromise as StrPromise  # pyright: ignore

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
    pass


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
    Unit: TypeAlias = Unit
    Quantity: TypeAlias = Quantity
    html_formatter: PathsHTMLFormatter
    pretty_formatter: PathsPrettyFormatter

    def parse_units(self, input_string: str, as_delta: bool | None = None, case_sensitive: bool | None = None) -> Unit:
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
    def format_unit(  # type: ignore
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


unit_registry.html_formatter = PathsHTMLFormatter(registry=cast(UnitRegistry, unit_registry))
unit_registry.pretty_formatter = PathsPrettyFormatter(registry=cast(UnitRegistry, unit_registry))


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
"""

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
    EUR = [currency]
    CAD = nan EUR
    USD = nan EUR
    SEK = 0.1 EUR
    PLN = 0.2 EUR
    pcs = [number] = pieces = number
    capita = [population] = cap = inh = inhabitant = person
    MMBtu = 1e6 Btu
    standard_cubic_feet = 0.01 therm = scf
    diesel_gallon_equivalent = 0.1385 MMBtu = DGE
    vehicle = [vehicle] = v
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
    solid_cubic_meter = 1 m**3 = m3_solid
    Mpkm = 1e6 * pkm
    CO2e = [co2e]
    kt_co2e = kilotonne * CO2e
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
#pint.set_application_registry(unit_registry)
app_registry = pint.get_application_registry()
app_registry._registry = unit_registry  # pyright: ignore
#pint_pandas.PintType.ureg = unit_registry  # type: ignore

def add_unit_translations():
    """
    Add translations for some commonly used units.

    Called from Django's App.ready() handler.
    """
    from babel import Locale as Loc
    from pint.babel_names import _babel_units
    try:
        from django.conf import settings
    except Exception:
        return
    from django.utils import translation
    from django.utils.translation import gettext_lazy as _, pgettext_lazy

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

    _babel_units['metric_ton'] = 'mass-tonne'
    _babel_units['%'] = 'concentr-percent'
    _babel_units['percent'] = 'concentr-percent'
    set_one('capita', _('capita'), short=pgettext_lazy('capita short', 'cap'))
    #set_one('cap', pgettext_lazy('capita short', 'cap'))
    kt_str = pgettext_lazy('kilotonne short', 'kt')
    set_one('kt', kt_str, kt_str)

    loc = Loc('de')
    loc._data['unit_patterns']['duration-year']['short'] = dict(one='a')

    #set_one('a', pgettext_lazy('year short', 'yr.'))
    #set_one('percent', pgettext_lazy('percent', 'percent'))
    #set_one('metric_ton', pgettext_lazy('metric_ton', 'metric ton'))
    #set_one('%', '%')
