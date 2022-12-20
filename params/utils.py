from typing import Tuple

from pint_pandas import PintType

from nodes.units import Unit, Quantity


def sep_unit(val: Quantity) -> Tuple[float, Unit]:
    """Returns as tuple the magnitude and units of a Pint Quantity."""
    return float(val.m), val.units


def sep_unit_pt(val: Quantity) -> Tuple[float, PintType]:
    """Returns as tuple the magnitude and units (as Pandas PintType) of a Pint Quantity."""
    m, units = sep_unit(val)
    return m, PintType(units)
