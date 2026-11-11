"""Export the reference carrier grid used to prepare production BISKO activation defaults."""

import math
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand, CommandError

import polars as pl

from kausal_common.datasets.models import Dataset

from frameworks.bisko.default_sources import SECTORS
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = 'Export a sector/carrier energy grid as a prior for prepare_bisko_defaults.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instance')
        parser.add_argument('dataset')
        parser.add_argument('--year', type=int, required=True)
        parser.add_argument('--output', type=Path, required=True)

    def handle(self, *args: object, **options: object) -> None:
        path = Path(str(options['output']))
        year = int(str(options['year']))
        if path.exists() or path.suffix not in ('.csv', '.parquet'):
            raise CommandError('Choose a new output file ending in .csv or .parquet.')
        instance = InstanceConfig.objects.get_queryset_all().get(identifier=options['instance'])
        dataset = Dataset.objects.for_instance_config(instance).get(identifier=options['dataset'])
        points = (
            dataset.data_points
            .filter(date=date(year, 1, 1), metric__name='Value')
            .select_related('metric')
            .prefetch_related('dimension_categories__dimension')
        )
        rows = []
        keys = set()
        for point in points:
            coordinates = {category.dimension.name: category.identifier for category in point.dimension_categories.all()}
            sector, carrier = coordinates.get('Sektoren'), coordinates.get('Energieträger')
            if sector not in SECTORS or carrier is None or point.value is None or point.metric.unit != 'MWh/a':
                raise CommandError('Prior must contain sector/carrier values in MWh/a.')
            value = float(point.value)
            if not math.isfinite(value) or value < 0 or (sector, carrier) in keys:
                raise CommandError('Prior contains invalid or duplicate cells.')
            keys.add((sector, carrier))
            rows.append({'sector': sector, 'energy_carrier': carrier, 'value_mwh_per_year': value, 'year': year, 'unit': 'MWh/a'})
        carriers = {carrier for _sector, carrier in keys}
        if not rows or keys != {(sector, carrier) for sector in SECTORS for carrier in carriers}:
            raise CommandError('Prior must contain a complete sector/carrier grid.')
        frame = pl.DataFrame(rows).sort('sector', 'energy_carrier')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.csv':
            frame.write_csv(path)
        else:
            frame.write_parquet(path)
        self.stdout.write(f'Exported {len(rows)} prior cells to {path}')
