"""A demo without provider estimates can use eligible neighbours without using excluded source rows."""

import polars as pl
import pytest

from frameworks.bisko.demo.energy import regional_energy

pytestmark = pytest.mark.django_db


def test_regional_energy_uses_population_weights_and_the_requested_year() -> None:
    energy = pl.DataFrame({
        'ags': ['12063001', '12063002', '12063001', '12072001'],
        'Year': [2023, 2023, 2022, 2023],
        'sector': ['private_households'] * 4,
        'energy_carrier': ['electricity'] * 4,
        'Value': [100.0, 900.0, 99999.0, 99999.0],
    })
    population = pl.DataFrame({
        'ags': ['12063252', '12063001', '12063002', '12072001'],
        'Year': [2023] * 4,
        'population': [50, 100, 300, 1000],
    })
    cells = regional_energy(energy, population, ags='12063252', year=2023)
    assert len(cells) == 1
    assert cells[0].value == 125
    assert cells[0].provenance.source_region_code == '12063'
    assert cells[0].provenance.is_estimate


def test_regional_energy_falls_back_to_state_and_requires_population() -> None:
    energy = pl.DataFrame({
        'ags': ['12072001'],
        'Year': [2023],
        'sector': ['industry'],
        'energy_carrier': ['electricity'],
        'Value': [100.0],
    })
    population = pl.DataFrame({'ags': ['12063252', '12072001'], 'Year': [2023, 2023], 'population': [50, 100]})
    cells = regional_energy(energy, population, ags='12063252', year=2023)
    assert cells[0].value == 50
    assert cells[0].provenance.source_region_code == '12'
    with pytest.raises(ValueError, match='positive population'):
        regional_energy(energy, population, ags='03404000', year=2023)
