from typing import Optional, Type
import os
import pint
import pint_pandas
from pint.facets.plain.unit import PlainUnit
from pint.facets.plain.quantity import PlainQuantity
from pint.formatting import register_unit_format, formatter


Unit = PlainUnit
Quantity = PlainQuantity


class CachingUnitRegistry(pint.UnitRegistry):
    unit_cache: dict[str, pint.Unit] | None = None
    Unit: Type[PlainUnit]

    def parse_units(self, input_string: str, as_delta: Optional[bool] = None, case_sensitive: Optional[bool] = None) -> PlainUnit:
        if self.unit_cache is None:
            self.unit_cache = dict()
        cached_unit = self.unit_cache.get(input_string)
        if cached_unit is not None:
            return cached_unit
        ret = super().parse_units(input_string, as_delta, case_sensitive)
        self.unit_cache[input_string] = ret
        return ret


unit_registry = CachingUnitRegistry(
    preprocessors=[
        lambda s: s.replace('%', ' percent '),
    ],
    on_redefinition='raise',
    cache_folder=":auto:",
)


def define_custom_units(unit_registry: CachingUnitRegistry):
    from pint.facets import plain as plain_facets

    # By default, kt is knots, but here kilotonne is the most common
    # usage.
    del unit_registry._units['kt']
    unit_registry.define('kt = kilotonne')
    del unit_registry._units['ton']  # The default is 2000 pounds and we don't want to accidentally use that.
    unit_registry.define('ton = tonne')
    # Mega-kilometers is often used for mileage
    unit_registry.define('Mkm = gigameters')
    unit_registry.define(plain_facets.UnitDefinition(
        'percent', '%', (), plain_facets.ScaleConverter(0.01), reference=unit_registry.UnitsContainer({})
    ))
    unit_registry.define('EUR = [currency]')
    unit_registry.define('USD = nan EUR')
    unit_registry.define('SEK = 0.1 EUR')
    unit_registry.define('pcs = [number] = pieces')
    unit_registry.define('capita = [population] = cap = inh = inhabitant = person')
    unit_registry.define('MMBtu = 1e6 Btu')

    unit_registry.load_definitions(os.path.join(os.path.dirname(__file__), 'health_impact_units.txt'))


define_custom_units(unit_registry)
unit_registry.default_format = '~P'
pint.set_application_registry(unit_registry)
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

    def set_one(u: str, t: str):
        bu = 'kausal-%s' % u
        if u not in _babel_units:
            _babel_units[u] = bu
        for lang in [l[0] for l in settings.LANGUAGES]:
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
    return formatter(
        unit.items(),
        as_ratio=True,
        single_denominator=True,
        product_fmt=r"·",
        division_fmt=r"{}∕{}",
        power_fmt=r"{}<sup>{}</sup>",
        parentheses_fmt=r"({})",
        **options,
    )
