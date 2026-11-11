"""Synthetic energy for demo municipalities excluded from the prepared provider estimates."""

import hashlib
from typing import TYPE_CHECKING

import polars as pl

from datasets.placeholders import build_dataset_repo
from frameworks.bisko.default_sources import (
    ENERGY_SOURCE,
    POPULATION_SOURCE,
    DefaultProvenance,
    EnergyDefault,
)
from frameworks.bisko.defaults import seed_energy_defaults
from nodes.template_graph import template_snapshot

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset

    from nodes.models import InstanceConfig


def regional_energy(energy: pl.DataFrame, population: pl.DataFrame, *, ags: str, year: int) -> list[EnergyDefault]:
    """Scale the regional per-capita carrier grid to the demo town, using one common source year."""
    population = population.filter(pl.col('Year') == year).select('ags', 'population')
    own = population.filter(pl.col('ags') == ags)['population']
    if len(own) != 1 or own[0] <= 0:
        raise ValueError(f'{ags}: demo fallback needs a positive population for {year}.')
    energy = energy.filter(pl.col('Year') == year)
    for region in (ags[:5], ags[:2]):
        donors = energy.filter(pl.col('ags').str.starts_with(region) & (pl.col('ags') != ags))
        if donors.is_empty():
            continue
        joined = donors.join(population, on='ags', how='left', validate='m:1')
        if joined['population'].null_count() or (joined['population'] <= 0).any():
            raise ValueError(f'{ags}: regional energy donors lack population in {year}.')
        grid = joined.group_by('sector', 'energy_carrier').agg(
            (pl.col('Value').sum() * own[0] / pl.col('population').sum()).alias('Value')
        )
        provenance = DefaultProvenance(
            source_dataset=ENERGY_SOURCE,
            source_sha256=hashlib.sha256(grid.sort('sector', 'energy_carrier').write_json().encode()).hexdigest(),
            method='regional_population_scaling',
            source_region_code=region,
            source_year=year,
            is_estimate=True,
        )
        return [
            EnergyDefault(ags=ags, year=year, sector=sector, energy_carrier=carrier, value=value, provenance=provenance)
            for sector, carrier, value in grid.iter_rows()
        ]
    raise ValueError(f'{ags}: no regional energy donors are available for the demo.')


def seed_regional_energy(instance: InstanceConfig, dataset: Dataset, ags: str, year: int) -> None:
    """Read both inputs at the municipality's published template pin; never use staging files."""
    repo_spec = template_snapshot(instance).spec.dataset_repo
    if repo_spec is None or repo_spec.commit is None:
        raise ValueError('Demo energy fallback requires a pinned dataset repository.')
    repo = build_dataset_repo(repo_spec)
    energy = repo.load_dataset(ENERGY_SOURCE).df
    population = repo.load_dataset(POPULATION_SOURCE).df
    if energy is None or population is None:
        raise ValueError('Demo energy fallback needs energy and population provider datasets.')
    seed_energy_defaults(
        instance, dataset, regional_energy(energy, population, ags=ags, year=year), source_revision=repo_spec.commit
    )
