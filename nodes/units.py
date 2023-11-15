from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import re
from typing import TYPE_CHECKING, Any, Optional, Tuple, TypeAlias
import os
import pint
from pint._typing import UnitLike
import pint_pandas
from pint import facets
from pint.formatting import register_unit_format, formatter

if TYPE_CHECKING:
    from django.utils.functional import _StrPromise as StrPromise  # pyright: ignore


#Unit = PlainUnit
#Quantity = PlainQuantity

def split_specifier(name: str) -> Tuple[str, str | None]:
    m = re.match(r'(.+)\[(\w+)\]', name)
    if not m:
        m = re.match(r'(.+)__(\w+)', name)
    if m:
        name, specifier = m.groups()
    else:
        name, specifier = name, None
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
    facets.GenericFormattingRegistry[Quantity, Unit],
    facets.GenericNonMultiplicativeRegistry[Quantity, Unit],
    facets.GenericPlainRegistry[Quantity, Unit],
):
    unit_cache: dict[str, Unit] | None = None
    Unit: TypeAlias = Unit
    Quantity: TypeAlias = Quantity

    def parse_units(self, input_string: str, as_delta: Optional[bool] = None, case_sensitive: Optional[bool] = None) -> Unit:
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


unit_registry = CachingUnitRegistry(
    preprocessors=[
        lambda s: s.replace('%', ' percent '),
    ],
    on_redefinition='raise',
    cache_folder=":auto:",
)


def define_custom_units(unit_registry: CachingUnitRegistry):
    # By default, kt is knots, but here kilotonne is the most common
    # usage.
    del unit_registry._units['kt']
    del unit_registry._units['ton']  # The default is 2000 pounds and we don't want to accidentally use that.
    DEFINITIONS = '''
    kt = kilotonne
    ton = tonne
    # Mega-kilometers is often used for mileage
    Mkm = gigameters
    EUR = [currency]
    USD = nan EUR
    SEK = 0.1 EUR
    pcs = [number] = pieces
    capita = [population] = cap = inh = inhabitant = person
    MMBtu = 1e6 Btu
    standard_cubic_feet = 0.01 therm = scf
    diesel_gallon_equivalent = 0.1385 MMBtu = DGE
    vehicle = [vehicle] = v
    passenger = [passenger] = p = pass = trip
    vkm = vehicle * kilometer
    pkm = passenger * kilometer
    VMT = vehicle * mile
    @alias vkm = vkt = v_km
    @alias pkm = pkt = p_km
    million_square_meters = 1e6 * meter ** 2 = Msqm
    CO2e = [co2e]
    kt_co2e = kilotonne * CO2e
    kg_co2e = kg * CO2e
    g_co2e = g * CO2e
    '''

    for line in DEFINITIONS.strip().splitlines():
        line = line.strip()
        if line.startswith('#'):
            continue
        unit_registry.define(line)

    unit_registry.load_definitions(os.path.join(os.path.dirname(__file__), 'health_impact_units.txt'))


define_custom_units(unit_registry)
unit_registry.default_format = '~P'
app_registry = pint.get_application_registry()
app_registry._registry = unit_registry  # pyright: ignore
pint_pandas.PintType.ureg = unit_registry  # type: ignore


def add_unit_translations():
    """Add translations for some commonly used units.

    Called from Django's App.ready() handler."""
    from pint.babel_names import _babel_units
    from babel import Locale as Loc
    try:
        from django.conf import settings
    except Exception:
        return
    from django.utils import translation
    from django.utils.translation import gettext_lazy as _, pgettext_lazy

    def set_one(u: str, t: str | StrPromise):
        bu = 'kausal-%s' % u
        if u not in _babel_units:
            _babel_units[u] = bu
        for lang in [la[0] for la in settings.LANGUAGES]:
            loc = Loc(lang.replace('-', '_'))
            loc_data: dict = loc._data  # type: ignore
            pats = loc_data['unit_patterns']
            with translation.override(lang):
                pats[bu] = dict(long=dict(one=str(t)))

            # Work around a bug in pint
            cup = loc_data.get('compound_unit_patterns', {})
            if 'per' in cup:
                del cup['per']

    set_one('capita', _('capita'))
    set_one('cap', pgettext_lazy('capita short', 'cap'))
    set_one('kt', pgettext_lazy('kilotonne short', 'kt'))
    set_one('a', pgettext_lazy('year short', 'yr.'))
    set_one('percent', pgettext_lazy('percent', 'percent'))
    set_one('%', '%')


@register_unit_format("Z")
def format_paths_html(unit, registry, **options):
    out = {}
    for key, val in unit.items():
        name, specifier = split_specifier(key)
        if specifier is not None and specifier == 'CO2e':
            name = name + ' CO<sub>2</sub>-e.'
        out[name] = val
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
