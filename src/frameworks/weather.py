"""Seed editable BISKO municipal weather factors from pinned regional degree days."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from django.db import transaction

import polars as pl

from kausal_common.datasets.models import DataPoint, DataPointDimensionCategory, Dataset

from datasets.materialization import refresh_dataset_materialization
from datasets.placeholders import build_dataset_repo

if TYPE_CHECKING:
    from frameworks.models import Framework
    from nodes.models import InstanceConfig


WEATHER_DATASET = 'kommune/witterungsbereinigung'
SOURCE_DATASET = 'weather/heating_degree_days'
REFERENCE_YEARS = range(1980, 2015)
HEATING_SECTORS = ('private_households', 'commerce_trade_services', 'municipal_facilities')
NEUTRAL_SECTORS = ('industry', 'transport')
VALUE_PRECISION = Decimal('0.0001')


def load_weather_source(framework: Framework) -> tuple[pl.DataFrame, str]:
    template = framework.template_instance
    if template is None:
        raise ValueError('BISKO has no template instance.')
    repo_spec = template.ensure_spec().dataset_repo
    if repo_spec is None or repo_spec.commit is None:
        raise ValueError('The BISKO template needs a pinned dataset repository commit for weather defaults.')
    dataset = build_dataset_repo(repo_spec).load_dataset(SOURCE_DATASET)
    if dataset.df is None:
        raise ValueError(f'{SOURCE_DATASET} has no dataframe.')
    return dataset.df, repo_spec.commit


def regional_weather_factors(frame: pl.DataFrame, nuts3: str, *, first_year: int, last_year: int) -> dict[int, Decimal]:
    """Return mean 1980-2014 Eurostat HDD divided by each observed year's HDD."""
    required = {'nuts', 'Year', 'hdd_eurostat'}
    if not required <= set(frame.columns):
        raise ValueError(f'{SOURCE_DATASET} lacks columns: {sorted(required - set(frame.columns))}')
    if first_year < min(REFERENCE_YEARS) or last_year < first_year:
        raise ValueError(f'Invalid weather default years {first_year}-{last_year}.')
    rows = frame.filter(pl.col('nuts') == nuts3).select('Year', 'hdd_eurostat')
    by_year: dict[int, Decimal] = {}
    for raw_year, raw_hdd in rows.iter_rows():
        year = int(raw_year)
        if raw_hdd is None:
            raise ValueError(f'Missing Eurostat heating degree days for {nuts3} in {year}.')
        hdd = Decimal(str(raw_hdd))
        if year in by_year or not hdd.is_finite() or hdd <= 0:
            raise ValueError(f'Invalid or duplicate Eurostat heating degree days for {nuts3} in {year}.')
        by_year[year] = hdd
    missing_reference = set(REFERENCE_YEARS) - by_year.keys()
    if missing_reference:
        raise ValueError(f'{nuts3} lacks reference HDD years: {sorted(missing_reference)}')
    last_year = min(last_year, max(by_year, default=first_year - 1))
    missing_output = set(range(first_year, last_year + 1)) - by_year.keys()
    if missing_output or last_year < first_year:
        raise ValueError(f'{nuts3} lacks weather default years: {sorted(missing_output)}')
    reference = sum(by_year[year] for year in REFERENCE_YEARS) / Decimal(len(REFERENCE_YEARS))
    return {year: (reference / by_year[year]).quantize(VALUE_PRECISION) for year in range(first_year, last_year + 1)}


@transaction.atomic
def seed_weather_defaults(
    instance: InstanceConfig, dataset: Dataset, frame: pl.DataFrame, *, nuts3: str, source_revision: str
) -> bool:
    """Seed only a wholly empty local slot; municipal edits always take precedence."""
    dataset = Dataset.objects.select_for_update().get(pk=dataset.pk)
    if dataset.identifier != WEATHER_DATASET or dataset.scope_instance.pk != instance.pk:
        raise ValueError("Weather defaults require the municipality's own weather dataset.")
    if dataset.data_points.exists():
        return False
    if dataset.schema is None:
        raise ValueError(f'{instance.identifier}: weather dataset has no schema.')
    dimensions = list(dataset.schema.dimensions.select_related('dimension'))
    if len(dimensions) != 1:
        raise ValueError(f'{instance.identifier}: weather dataset must have one sector dimension.')
    categories = {
        category.identifier: category
        for category in dimensions[0].dimension.categories.filter(identifier__in=(*HEATING_SECTORS, *NEUTRAL_SECTORS))
    }
    sectors = (*HEATING_SECTORS, *NEUTRAL_SECTORS)
    if set(categories) != set(sectors):
        raise ValueError(f'{instance.identifier}: weather dataset lacks sector categories: {sorted(set(sectors) - categories)}')
    metric = dataset.schema.metrics.get(name='default')
    if metric.unit not in ('', 'dimensionless'):
        raise ValueError(f'{instance.identifier}: weather default metric must be dimensionless.')
    spec = instance.ensure_spec()
    first_year = spec.years.min_historical or spec.years.reference
    last_year = spec.years.model_end or spec.years.target or spec.years.max_historical
    if first_year is None or last_year is None:
        raise ValueError(f'{instance.identifier}: historical and target years are required for weather defaults.')
    factors = regional_weather_factors(frame, nuts3, first_year=first_year, last_year=last_year)
    if spec.years.max_historical is not None and max(factors) < spec.years.max_historical:
        raise ValueError(f'{instance.identifier}: Eurostat weather data do not cover the historical model years.')

    coordinates = [(year, sector) for year in factors for sector in sectors]
    points = [
        DataPoint(dataset=dataset, metric=metric, date=date(year, 1, 1), value=factors[year] if sector in HEATING_SECTORS else 1)
        for year, sector in coordinates
    ]
    created = DataPoint.objects.bulk_create(points)
    DataPointDimensionCategory.objects.bulk_create([
        DataPointDimensionCategory(data_point=point, dimension_category=categories[sector])
        for point, (_year, sector) in zip(created, coordinates, strict=True)
    ])
    dataset.spec = {
        **(dataset.spec or {}),
        'bisko_weather_default': {
            'source_dataset': SOURCE_DATASET,
            'source_revision': source_revision,
            'nuts3': nuts3,
            'reference_years': [min(REFERENCE_YEARS), max(REFERENCE_YEARS)],
            'formula': 'reference_mean_hdd / annual_hdd',
        },
    }
    dataset.save(update_fields=['spec'])
    refresh_dataset_materialization(dataset, touch=False)
    return True
