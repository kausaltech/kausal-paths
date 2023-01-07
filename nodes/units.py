from typing import Optional, Type
import pint
from pint.facets.plain.unit import PlainUnit
from pint.facets.plain.quantity import PlainQuantity


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
