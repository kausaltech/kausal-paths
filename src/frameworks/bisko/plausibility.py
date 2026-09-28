"""
Advisory plausibility ranges for BISKO stationary final energy.

The ranges are sums over cells of ``kommune/endenergieverbrauch``, because a
single sector-and-carrier cell varies too much between municipalities to say
much on its own, while sector and carrier totals are informative. Each is checked
per inhabitant against the spread across municipalities that publish a
Klimaschutz-Planer dashboard, and year over year against the changes in their
published series.

The bounds are derived, not hand-set: ``kausal-importers germany ksp-dashboards
--bands`` prints them from a dashboard snapshot, together with the percentiles and
sample sizes behind them. To update, refresh the snapshot there, copy its rows
into ``BANDS``, and move ``SOURCE_REVISION`` to the snapshot date.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from django.db import transaction

from kausal_common.datasets.models import Dataset, DatasetSchemaDimension, DimensionCategory

from datasets.models import (
    DatasetMetricPlausibilityRange,
    PlausibilityAggregation,
    PlausibilityDenominator,
    PlausibilityReference,
    PlausibilitySource,
)

if TYPE_CHECKING:
    from frameworks.models import Framework

ENERGY_DATASET = 'kommune/endenergieverbrauch'
VALUE_METRIC = 'Value'

STATIONARY_SECTORS = ('private_households', 'commerce_trade_services', 'industry', 'municipal_facilities')
CARRIERS = (
    'electricity',
    'natural_gas',
    'district_heating',
    'heating_oil',
    'biomass',
    'solar_thermal',
    'environmental_heat',
)

SOURCE_IDENTIFIER = 'klimaschutz-planer-dashboards'
SOURCE_REVISION = '2026-09-28'
SOURCE_METHOD = """\
Published Klimaschutz-Planer dashboards of 90 German municipalities (districts and
associations excluded), final energy without weather correction. Population is
recovered from each dashboard's household total and per-inhabitant indicator.

Per inhabitant: the 2.5th-97.5th percentile of each municipality's latest year,
widened by a factor of 1.25 on both sides. Where more than 1% of municipalities
report none of a quantity, the lower bound is 0 and the range only guards against
order-of-magnitude errors.

Year over year: the 1st-99th percentile of the ratios between consecutive
published years, widened by 10% and at least 0.8-1.25. Many published years are
defaults scaled from national trends, which makes the series smoother than data a
municipality enters itself.

Carrier totals are the dashboards' sums over the four stationary sectors and match
the model's cells; Strom and Heizstrom together correspond to electricity, and
Fernwärme and Nahwärme to district heating. The dashboards' sector totals also
include carriers the model does not ask for (LPG, coal, biogas), so they run
slightly above the model's sum over its seven carriers."""


@dataclass(frozen=True)
class Band:
    identifier: str
    sectors: tuple[str, ...]
    carriers: tuple[str, ...]
    reference: PlausibilityReference
    lower: float
    upper: float
    sample_size: int

    @property
    def denominator(self) -> PlausibilityDenominator:
        if self.reference == PlausibilityReference.ABSOLUTE:
            return PlausibilityDenominator.POPULATION
        return PlausibilityDenominator.NONE


def _sector(identifier: str, lower: float, upper: float, n: int, yoy: tuple[float, float, int] | None) -> list[Band]:
    bands = [Band(identifier, (identifier,), CARRIERS, PlausibilityReference.ABSOLUTE, lower, upper, n)]
    if yoy is not None:
        bands.append(Band(f'{identifier}-yoy', (identifier,), CARRIERS, PlausibilityReference.PREVIOUS_YEAR, *yoy))
    return bands


def _carrier(identifier: str, lower: float, upper: float, n: int, yoy: tuple[float, float, int] | None) -> list[Band]:
    bands = [Band(identifier, STATIONARY_SECTORS, (identifier,), PlausibilityReference.ABSOLUTE, lower, upper, n)]
    if yoy is not None:
        bands.append(Band(f'{identifier}-yoy', STATIONARY_SECTORS, (identifier,), PlausibilityReference.PREVIOUS_YEAR, *yoy))
    return bands


# Absolute bounds are MWh/a per inhabitant; year-over-year bounds are ratios.
BANDS: tuple[Band, ...] = (
    *_sector('private_households', 4.34, 17.0, 90, (0.703, 1.34, 337)),
    *_sector('commerce_trade_services', 0.739, 9.31, 90, (0.45, 2.64, 337)),
    *_sector('municipal_facilities', 0.0, 0.99, 90, (0.377, 2.29, 292)),
    *_carrier('electricity', 1.71, 10.2, 90, (0.8, 1.25, 337)),
    *_carrier('natural_gas', 1.87, 16.0, 90, (0.705, 1.42, 337)),
    *_carrier('heating_oil', 0.266, 10.2, 90, (0.542, 1.79, 337)),
    *_carrier('district_heating', 0.0, 4.53, 90, None),
    *_carrier('biomass', 0.0, 4.32, 90, None),
    *_carrier('solar_thermal', 0.0, 0.445, 90, None),
    *_carrier('environmental_heat', 0.0, 0.974, 90, None),
    Band('stationary_total', STATIONARY_SECTORS, CARRIERS, PlausibilityReference.ABSOLUTE, 7.45, 33.8, 90),
    Band('stationary_total-yoy', STATIONARY_SECTORS, CARRIERS, PlausibilityReference.PREVIOUS_YEAR, 0.736, 1.32, 337),
)


def _category_uuids(dataset: Dataset) -> tuple[str, dict[str, str], str, dict[str, str]]:
    """Find the sector and carrier dimensions by their categories; return their UUIDs."""
    assert dataset.schema is not None
    found: dict[str, tuple[str, dict[str, str]]] = {}
    for schema_dimension in DatasetSchemaDimension.objects.filter(schema=dataset.schema).select_related('dimension'):
        dimension = schema_dimension.dimension
        categories = {
            str(identifier): str(uuid)
            for identifier, uuid in DimensionCategory.objects.filter(dimension=dimension).values_list('identifier', 'uuid')
        }
        for role, wanted in (('sector', STATIONARY_SECTORS), ('carrier', CARRIERS)):
            if set(wanted) <= set(categories):
                if role in found:
                    raise ValueError(f'{ENERGY_DATASET}: more than one dimension has the {role} categories.')
                found[role] = (str(dimension.uuid), {identifier: categories[identifier] for identifier in wanted})
    missing = {'sector', 'carrier'} - set(found)
    if missing or len(found) != len(DatasetSchemaDimension.objects.filter(schema=dataset.schema)):
        raise ValueError(f'{ENERGY_DATASET}: expected exactly a sector and a carrier dimension; missing {sorted(missing)}')
    return (*found['sector'], *found['carrier'])


@transaction.atomic
def provision_bisko_plausibility_ranges(framework: Framework) -> int:
    """
    Create or update the framework's reference ranges from ``BANDS``.

    A range whose bounds or cells change gets a new revision. Ranges of this source
    that are no longer in ``BANDS`` are removed. Returns the number of ranges, or 0
    when the template has no final energy dataset to attach them to.
    """
    template = framework.template_instance
    assert template is not None
    dataset = Dataset.objects.for_instance_config(template).filter(identifier=ENERGY_DATASET).select_related('schema').first()
    if dataset is None or dataset.schema is None:
        return 0
    metric = dataset.schema.metrics.get(name=VALUE_METRIC)
    sector_dimension, sectors, carrier_dimension, carriers = _category_uuids(dataset)

    source, _ = PlausibilitySource.objects.update_or_create(
        identifier=SOURCE_IDENTIFIER,
        defaults={
            'name': 'Klimaschutz-Planer dashboards',
            'url': 'https://dashboard-be.klimaschutz-planer.de/v1/communes',
            'revision': SOURCE_REVISION,
            'method': SOURCE_METHOD,
            'is_example': False,
        },
    )
    source.full_clean()

    for band in BANDS:
        values = {
            'source': source,
            'selection': {
                sector_dimension: [sectors[identifier] for identifier in band.sectors],
                carrier_dimension: [carriers[identifier] for identifier in band.carriers],
            },
            'aggregation': PlausibilityAggregation.SUM,
            'denominator': band.denominator,
            'reference': band.reference,
            'max_gap_years': 1 if band.reference == PlausibilityReference.PREVIOUS_YEAR else None,
            'lower': band.lower,
            'upper': band.upper,
            'first_year': None,
            'last_year': None,
            'sample_size': band.sample_size,
            'enabled': True,
        }
        rule = DatasetMetricPlausibilityRange.objects.filter(
            framework=framework, metric=metric, identifier=band.identifier
        ).first()
        if rule is None:
            rule = DatasetMetricPlausibilityRange(framework=framework, metric=metric, identifier=band.identifier)
        elif all(getattr(rule, field) == value for field, value in values.items()):
            continue
        else:
            rule.revision += 1
        for field, value in values.items():
            setattr(rule, field, value)
        rule.full_clean()
        rule.save()

    DatasetMetricPlausibilityRange.objects.filter(framework=framework, source=source).exclude(
        identifier__in=[band.identifier for band in BANDS]
    ).delete()
    return len(BANDS)
