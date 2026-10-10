from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError

import polars as pl
import pytest

from kausal_common.datasets.models import Dataset, DimensionScope
from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)

from frameworks.bisko.default_sources import (
    POPULATION_DATASET,
    SECTORS,
    DefaultProvenance,
    PopulationDefault,
    prepare_energy_defaults,
    prepare_population_defaults,
)
from frameworks.bisko.defaults import seed_energy_defaults, seed_population_defaults
from frameworks.models import DataEvidenceKind, DataQualityLevel, DataQualityScheme, OrganizationPopulation
from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.defs.instance_defs import DatasetRepoSpec
from nodes.tests.factories import InstanceConfigFactory
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


def history() -> pl.DataFrame:
    return pl.DataFrame({
        'ags': ['03001001', '03001002'],
        'year': [2025, 2025],
        'population': [10, 20],
        'source_url': ['history', 'history'],
    })


def geography() -> pl.DataFrame:
    return pl.DataFrame({
        'ars': ['030015001', '030015001001', '030015001002'],
        'parent_ars': [None, '030015001', '030015001'],
        'ags': [None, '03001001', '03001002'],
    })


def forecasts() -> pl.DataFrame:
    return pl.DataFrame({
        'state_code': ['03'] * 3,
        'variant': ['trend', 'trend', 'other'],
        'administrative_level': ['association'] * 3,
        'source_region_code': ['030015001'] * 3,
        'ars': ['030015001'] * 3,
        'ags': [None] * 3,
        'year': [2025, 2027, 2027],
        'population': [30, 35, 9000],
        'source_edition': ['edition'] * 3,
        'source_vintage': [date(2024, 1, 1)] * 3,
        'source_url': ['forecast'] * 3,
        'source_sha256': ['hash'] * 3,
    })


def test_population_interpolates_allocates_and_holds_last() -> None:
    values = prepare_population_defaults(history(), forecasts(), geography(), historical_sha256='history-hash', horizon=2029)
    lookup = {(value.ags, value.year): value for value in values}
    assert sum(lookup[ags, 2026].value for ags in ('03001001', '03001002')) == 32
    assert sum(lookup[ags, 2027].value for ags in ('03001001', '03001002')) == 35
    assert lookup['03001001', 2027].value == 12
    assert lookup['03001002', 2027].value == 23
    assert lookup['03001002', 2029].value == 23
    assert lookup['03001002', 2029].provenance.method == 'association_allocation:hold_last'
    assert lookup['03001002', 2029].provenance.allocation_year == 2025
    assert lookup['03001001', 2025].provenance.source_sha256 == 'history-hash'
    assert not lookup['03001001', 2025].is_forecast
    assert lookup['03001001', 2026].is_forecast


def test_combined_municipalities_resolve_through_geography() -> None:
    frame = forecasts().with_columns(pl.lit('combined_municipalities').alias('administrative_level'), pl.lit(None).alias('ars'))
    values = prepare_population_defaults(history(), frame, geography(), historical_sha256='hash', horizon=2027)
    assert sum(value.value for value in values if value.year == 2027) == 35


def test_population_without_forecast_holds_observations() -> None:
    values = prepare_population_defaults(history(), forecasts().head(0), geography(), historical_sha256='hash', horizon=2027)
    assert [value.value for value in values if value.ags == '03001001'] == [10, 10, 10]
    assert values[-1].provenance.method == 'hold_last'


def energy_estimates() -> pl.DataFrame:
    return pl.DataFrame([
        {
            'ags': ags,
            'year': 2023,
            'sector': sector,
            'quantity': quantity,
            'value_mwh_per_year': value,
            'needs_review': ags == '03001002' and sector == 'industry',
            'source_sha256': 'hash',
            'source_year': 2023,
            'method': 'calibrated_proportional_default',
            'empirical_lower_mwh': 1.0,
            'empirical_upper_mwh': 200.0,
        }
        for ags in ('03001001', '03001002')
        for sector in SECTORS
        for quantity, value in (('stationary_final_energy', 100.0), ('grid_electricity', 20.0))
    ])


def prior() -> pl.DataFrame:
    return pl.DataFrame([
        {'sector': sector, 'energy_carrier': carrier, 'value_mwh_per_year': value}
        for sector in SECTORS
        for carrier, value in (('electricity', 50.0), ('natural_gas', 30.0), ('biomass', 10.0))
    ])


def test_energy_skips_whole_municipality_and_conserves_sector_totals() -> None:
    values, excluded = prepare_energy_defaults(energy_estimates(), prior(), prior_sha256='prior-hash')
    assert excluded == {'03001002'}
    assert {value.ags for value in values} == {'03001001'}
    for sector in SECTORS:
        cells = {value.energy_carrier: value for value in values if value.sector == sector}
        assert sum(value.value for value in cells.values()) == pytest.approx(100)
        assert cells['electricity'].value == 20
        assert cells['natural_gas'].value == 60
        assert cells['biomass'].value == 20
        assert cells['biomass'].provenance.prior_sha256 == 'prior-hash'


def test_energy_rejects_unflagged_inconsistent_quantities() -> None:
    frame = energy_estimates().with_columns(
        pl
        .when(pl.col('quantity') == 'grid_electricity')
        .then(200.0)
        .otherwise(pl.col('value_mwh_per_year'))
        .alias('value_mwh_per_year'),
    )
    with pytest.raises(ValueError, match='Unflagged inconsistent'):
        prepare_energy_defaults(frame, prior(), prior_sha256='hash')


def population_dataset() -> tuple[InstanceConfig, Dataset]:
    instance = InstanceConfigFactory.create(name='Population city', config_source='database')
    schema = DatasetSchemaFactory.create()
    DatasetMetricFactory.create(schema=schema, name='population', unit='cap')
    return instance, DatasetFactory.create(scope=instance, schema=schema, identifier=POPULATION_DATASET)


def population_values(value: int) -> list[PopulationDefault]:
    return [
        PopulationDefault(
            ags='03001001',
            year=2023,
            value=value,
            is_forecast=False,
            provenance=DefaultProvenance(source_dataset='destatis', source_sha256='hash', method='observed'),
        )
    ]


@pytest.mark.django_db
def test_population_refresh_preserves_manual_override_and_records_provider_reference() -> None:
    instance, dataset = population_dataset()
    assert seed_population_defaults(instance, dataset, population_values(100), source_revision='first') == 1
    point = dataset.data_points.get()
    assert point.evidence.kind == DataEvidenceKind.PROVIDER_DEFAULT
    assert seed_population_defaults(instance, dataset, population_values(101), source_revision='second') == 1
    point.refresh_from_db()
    assert point.value == 101
    point.value = Decimal(110)
    point.last_modified_by = UserFactory.create()
    point.save(update_fields=['value', 'last_modified_by'])
    assert seed_population_defaults(instance, dataset, population_values(102), source_revision='third') == 0
    point.refresh_from_db()
    dataset.refresh_from_db()
    assert point.value == 110
    assert dataset.spec['provider_defaults']['cells']['2023']['value'] == '102'
    assert point.source_references.get().data_source.edition == 'second'


@pytest.mark.django_db
def test_population_refresh_preserves_changed_value_even_without_editor() -> None:
    instance, dataset = population_dataset()
    seed_population_defaults(instance, dataset, population_values(100), source_revision='first')
    dataset.data_points.update(value=120)
    assert seed_population_defaults(instance, dataset, population_values(101), source_revision='second') == 0
    assert dataset.data_points.get().value == 120
    # Once identified as a local override, coincidentally matching a later provider value
    # must not make a subsequent refresh reclaim the cell.
    seed_population_defaults(instance, dataset, population_values(120), source_revision='third')
    seed_population_defaults(instance, dataset, population_values(121), source_revision='fourth')
    assert dataset.data_points.get().value == 120


def test_population_reference_pin_validation_checks_all_years() -> None:
    framework = FrameworkFactory.create()
    instance, _dataset = population_dataset()
    for year, revision in ((2023, 'first'), (2024, 'second')):
        OrganizationPopulation.objects.create(
            framework=framework,
            organization=instance.organization,
            year=year,
            value=100,
            source_dataset='bisko/defaults/population',
            source_revision=revision,
        )
    pin = DatasetRepoSpec(url='https://example.com/data.git', commit='second')
    with pytest.raises(ValidationError, match='Refresh organization population'):
        framework.validate_population_revision(pin)
    framework.organization_populations.update(source_revision='second')
    framework.validate_population_revision(pin)


def test_population_citations_keep_each_years_source_url() -> None:
    instance, dataset = population_dataset()
    values = [
        PopulationDefault(
            ags='03001001',
            year=year,
            value=100,
            is_forecast=False,
            provenance=DefaultProvenance(
                source_dataset='destatis',
                source_sha256='hash',
                method='observed',
                source_url=f'https://example.com/population-{year}.xlsx',
            ),
        )
        for year in (2023, 2024)
    ]
    seed_population_defaults(instance, dataset, values, source_revision='first')
    assert dict(dataset.data_points.values_list('date__year', 'source_references__data_source__url')) == {
        year: f'https://example.com/population-{year}.xlsx' for year in (2023, 2024)
    }


@pytest.mark.django_db
def test_energy_defaults_have_grade_c_and_leave_existing_values_alone() -> None:
    framework = FrameworkFactory.create(identifier='bisko')
    scheme = DataQualityScheme.objects.create(framework=framework, identifier='quality', version='1', name='BISKO')
    DataQualityLevel.objects.create(scheme=scheme, identifier='C', name='Regional statistics', score=Decimal('0.25'))
    instance = FrameworkConfigFactory.create(
        framework=framework,
        instance_config__name='Energy city',
        instance_config__config_source='database',
    ).instance_config
    schema = DatasetSchemaFactory.create()
    metric = DatasetMetricFactory.create(schema=schema, name='Value', unit='MWh/a')
    for identifier, categories in (('sector', SECTORS), ('energy_carrier', ('electricity', 'natural_gas', 'biomass'))):
        dimension = DimensionFactory.create()
        DatasetSchemaDimensionFactory.create(schema=schema, dimension=dimension)
        DimensionScope.objects.create(
            dimension=dimension,
            scope_content_type=ContentType.objects.get_for_model(instance),
            scope_id=instance.pk,
            identifier=identifier,
        )
        for category in categories:
            DimensionCategoryFactory.create(dimension=dimension, identifier=category)
    dataset = DatasetFactory.create(scope=instance, schema=schema, identifier='kommune/endenergieverbrauch')
    values, _ = prepare_energy_defaults(energy_estimates(), prior(), prior_sha256='hash')
    assert seed_energy_defaults(instance, dataset, values, source_revision='first') == 12
    assert set(dataset.data_points.values_list('evidence__quality_level__identifier', flat=True)) == {'C'}
    assert seed_energy_defaults(instance, dataset, values, source_revision='second') == 0
    DataPointFactory.create(dataset=dataset, metric=metric, date=date(2024, 1, 1), value=123)
    assert seed_energy_defaults(instance, dataset, values, source_revision='third') == 0
