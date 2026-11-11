"""Shared synthetic BISKO inventory conventions."""

from typing import TYPE_CHECKING

from nodes.defs.instance_defs import YearsSpec

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


DEMO_MUNICIPALITIES = ('034040000000', '071370203203', '120630252252')
LEGACY_SOURCE_NAME = 'Synthetic BISKO demo cells from Klimaschutz-Planer dashboards'


DEMO_GRADES = {
    'electricity': 'A',
    'heating_electricity': 'A',
    'natural_gas': 'A',
    'district_heating': 'A',
    'local_heating': 'B',
    'environmental_heat': 'B',
    'heating_oil': 'C',
    'biomass': 'C',
    'solar_thermal': 'C',
    'hard_coal': 'D',
    'brown_coal': 'D',
    'propane': 'D',
    'biogas': 'D',
    'other_conventional': 'D',
    'other_renewables': 'D',
}
"""
Invented grade per carrier, following how such a cell is typically sourced.

Grid-bound carriers come from network operators, which is also what the BISKO grade-A
requirement on them expects; municipal facilities are graded A in every carrier, since the
municipality holds its own bills. A cell's grade does not vary by year.
"""


def demo_grade(sector: str, carrier: str) -> str:
    return 'A' if sector == 'municipal_facilities' else DEMO_GRADES[carrier]


def configure_demo_definition(instance: InstanceConfig, inventory_years: list[int]) -> None:
    """Declare the demo's actual inventory calendar without assigning quality grades."""
    if instance.is_locked:
        raise ValueError(f'{instance.identifier}: instance is locked')
    if not inventory_years:
        raise ValueError(f'{instance.identifier}: demo has no inventory years')
    spec = instance.ensure_spec().model_copy(deep=True)
    years = spec.years.model_dump(mode='python')
    first, last = min(inventory_years), max(inventory_years)
    years.update(min_historical=first, max_historical=last, skipped=sorted(set(range(first, last + 1)) - set(inventory_years)))
    if years['reference'] not in inventory_years:
        years['reference'] = first
    spec.years = YearsSpec.model_validate(years)
    if spec.model_dump(mode='json') != instance.ensure_spec().model_dump(mode='json'):
        instance.spec = spec
        instance.save(update_fields=['spec'])
        instance.invalidate_cache()
