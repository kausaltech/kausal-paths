from __future__ import annotations

from typing import TYPE_CHECKING, cast

from nodes.units import Quantity, Unit

if TYPE_CHECKING:
    from pint_pandas import PintType


def sep_unit(val: Quantity, output_unit: Unit | None = None) -> tuple[float, Unit]:
    """Return as tuple the magnitude and units of a Pint Quantity."""
    if output_unit is not None:
        val = cast("Quantity", val.to(output_unit))
    return float(val.m), cast("Unit", val.units)


def sep_unit_pt(val: Quantity) -> tuple[float, PintType]:
    """Return as tuple the magnitude and units (as Pandas PintType) of a Pint Quantity."""
    from pint_pandas import PintType

    m, units = sep_unit(val)
    return m, PintType(units)
