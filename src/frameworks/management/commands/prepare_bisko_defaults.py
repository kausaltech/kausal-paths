"""Validate staging inputs and write provider datasets for DVC; never modify the database."""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from django.core.management.base import BaseCommand, CommandError

import polars as pl
import yaml

from frameworks.bisko.default_sources import (
    ENERGY_SOURCE,
    POPULATION_DATASET,
    POPULATION_SOURCE,
    energy_frame,
    population_frame,
    prepare_energy_defaults,
    prepare_population_defaults,
)

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Options(TypedDict):
    history: str
    forecasts: str
    geography: str
    energy: str
    prior: str
    output_dir: str
    horizon: int


def _digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Command(BaseCommand):
    help = 'Prepare population forecasts and carrier-split activation defaults from importer staging files.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        for name in ('history', 'forecasts', 'geography', 'energy', 'prior', 'output-dir'):
            parser.add_argument(f'--{name}', required=True)
        parser.add_argument('--horizon', type=int, default=2045)

    def handle(self, *args: object, **options: object) -> None:
        opts = cast('Options', options)
        try:
            population = prepare_population_defaults(
                pl.read_parquet(opts['history']),
                pl.read_parquet(opts['forecasts']),
                pl.read_parquet(opts['geography']),
                historical_sha256=_digest(opts['history']),
                horizon=opts['horizon'],
            )
            energy, excluded = prepare_energy_defaults(
                pl.read_parquet(opts['energy']),
                pl.read_parquet(opts['prior']),
                prior_sha256=_digest(opts['prior']),
                input_sha256=_digest(opts['energy']),
            )
        except (ValueError, OSError, pl.exceptions.PolarsError) as error:
            raise CommandError(str(error)) from error
        output = Path(opts['output_dir'])
        files = {
            POPULATION_SOURCE: (population_frame(population), 'population', 'cap', 'Bevölkerungs-Vorgabewerte'),
            ENERGY_SOURCE: (energy_frame(energy), 'Value', 'MWh/a', 'Stationäre Endenergie-Vorgabewerte'),
            POPULATION_DATASET: (
                pl.DataFrame({'Year': [2023], 'population': [None]}, schema={'Year': pl.Int64, 'population': pl.Float64}),
                'population',
                'cap',
                'Einwohnerzahl',
            ),
        }
        targets = [output / f'{identifier}.parquet' for identifier in files]
        if any(path.exists() or path.with_suffix('.metadata.yaml').exists() for path in targets):
            raise CommandError('Refusing to overwrite prepared datasets; choose a new output directory.')
        for identifier, (frame, metric, unit, name) in files.items():
            path = output / f'{identifier}.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(path)
            metadata = {
                'meta': {
                    'identifier': identifier,
                    'name': {'de': name, 'en': name},
                    'metrics': [{'id': metric, 'column_id': metric, 'unit': unit}],
                    'units': {metric: unit},
                    'index_columns': (
                        ['ags', 'sector', 'energy_carrier', 'Year']
                        if identifier == ENERGY_SOURCE
                        else ['ags', 'Year']
                        if identifier == POPULATION_SOURCE
                        else ['Year']
                    ),
                }
            }
            path.with_suffix('.metadata.yaml').write_text(yaml.safe_dump(metadata, allow_unicode=True), encoding='utf-8')
        (output / 'excluded-energy-municipalities.txt').write_text('\n'.join(sorted(excluded)) + '\n', encoding='utf-8')
        self.stdout.write(
            f'{len(population)} population values; {len(energy)} energy cells; '
            f'{len(excluded)} municipalities excluded from energy defaults. Output: {output}'
        )
