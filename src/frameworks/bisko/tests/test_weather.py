from datetime import date
from decimal import Decimal

from django.contrib.contenttypes.models import ContentType

import polars as pl
import pytest

from kausal_common.datasets.models import DimensionScope
from kausal_common.datasets.tests.factories import (
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)

from frameworks.bisko.weather import (
    HEATING_SECTORS,
    NEUTRAL_SECTORS,
    mark_seeded_weather_defaults,
    regional_weather_factors,
    seed_weather_defaults,
)
from frameworks.models import DataEvidenceKind, DataPointEvidence
from nodes.defs.instance_defs import YearsSpec
from nodes.models import DatasetMaterialization
from nodes.tests.factories import InstanceConfigFactory
from users.tests.factories import UserFactory

pytestmark = pytest.mark.django_db


def degree_days() -> pl.DataFrame:
    return pl.DataFrame({
        'nuts': ['DE405'] * 46,
        'Year': list(range(1980, 2026)),
        'hdd_eurostat': [500 if year == 2023 else 1000 for year in range(1980, 2026)],
    })


def test_regional_factors_use_fixed_reference_and_require_complete_history() -> None:
    factors = regional_weather_factors(degree_days(), 'DE405', first_year=2020, last_year=2025)
    assert factors[2020] == Decimal('1.0000')
    assert factors[2023] == Decimal('2.0000')

    incomplete = degree_days().filter(pl.col('Year') != 1980)
    with pytest.raises(ValueError, match='reference HDD years'):
        regional_weather_factors(incomplete, 'DE405', first_year=2020, last_year=2025)


def weather_dataset():
    instance = InstanceConfigFactory.create(name='Weather city', config_source='database')
    spec = instance.ensure_spec()
    spec.years = YearsSpec(reference=2020, min_historical=2020, max_historical=2023, target=2025)
    instance.spec = spec
    instance.save(update_fields=['spec'])
    schema = DatasetSchemaFactory.create()
    dimension = DimensionFactory.create(name='Sectors')
    DatasetSchemaDimensionFactory.create(schema=schema, dimension=dimension)
    DimensionScope.objects.create(
        dimension=dimension,
        scope_content_type=ContentType.objects.get_for_model(instance),
        scope_id=instance.pk,
        identifier='sector',
    )
    for identifier in (*HEATING_SECTORS, *NEUTRAL_SECTORS):
        DimensionCategoryFactory.create(dimension=dimension, identifier=identifier)
    DatasetMetricFactory.create(schema=schema, name='default', unit='')
    dataset = DatasetFactory.create(scope=instance, schema=schema, identifier='kommune/witterungsbereinigung')

    return instance, dataset


def test_seed_weather_defaults_preserves_municipal_edits() -> None:
    instance, dataset = weather_dataset()
    assert seed_weather_defaults(instance, dataset, degree_days(), nuts3='DE405', source_revision='test-commit')
    assert dataset.data_points.count() == 30
    household = dataset.data_points.get(date=date(2023, 1, 1), dimension_categories__identifier='private_households')
    industry = dataset.data_points.get(date=date(2023, 1, 1), dimension_categories__identifier='industry')
    assert household.value == 2
    assert industry.value == 1
    assert DatasetMaterialization.objects.filter(dataset=dataset).exists()
    dataset.refresh_from_db()
    assert dataset.spec['bisko_weather_default']['nuts3'] == 'DE405'
    assert dataset.spec['bisko_weather_default']['reference_years'] == [1980, 2014]

    household.value = Decimal('1.7')
    household.save(update_fields=['value'])
    assert not seed_weather_defaults(instance, dataset, degree_days(), nuts3='DE405', source_revision='test-commit')
    household.refresh_from_db()
    assert household.value == Decimal('1.7')
    assert dataset.data_points.count() == 30


def test_seeded_factors_are_provider_defaults_and_old_seeds_are_marked() -> None:
    instance, dataset = weather_dataset()
    assert seed_weather_defaults(instance, dataset, degree_days(), nuts3='DE405', source_revision='test-commit')
    evidence = DataPointEvidence.objects.filter(data_point__dataset=dataset)
    assert evidence.count() == 30
    assert set(evidence.values_list('kind', flat=True)) == {DataEvidenceKind.PROVIDER_DEFAULT}
    dataset.refresh_from_db()
    assert mark_seeded_weather_defaults(dataset) == 0

    # As seeded before evidence was recorded, with one factor since edited by a user.
    evidence.delete()
    edited = dataset.data_points.get(date=date(2023, 1, 1), dimension_categories__identifier='private_households')
    edited.value = Decimal('1.7')
    edited.last_modified_by = UserFactory.create()
    edited.save(update_fields=['value', 'last_modified_by'])

    assert mark_seeded_weather_defaults(dataset) == 29
    assert not DataPointEvidence.objects.filter(data_point=edited).exists()
    assert mark_seeded_weather_defaults(dataset) == 0
