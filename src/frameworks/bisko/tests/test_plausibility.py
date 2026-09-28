import dataclasses
import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)

from datasets.models import DatasetMetricPlausibilityRange
from datasets.plausibility import evaluate_dataset_plausibility
from frameworks.bisko import plausibility
from frameworks.bisko.plausibility import BANDS, CARRIERS, STATIONARY_SECTORS, provision_bisko_plausibility_ranges
from frameworks.bisko.provisioning import setup_bisko
from frameworks.models import OrganizationPopulation
from frameworks.tests.factories import FrameworkConfigFactory
from nodes.tests.factories import InstanceConfigFactory

if TYPE_CHECKING:
    from kausal_common.datasets.models import DatasetSchema, DimensionCategory

    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


@pytest.fixture
def template() -> InstanceConfig:
    return InstanceConfigFactory.create(identifier='bisko', name='BISKO')


@pytest.fixture
def energy(template: InstanceConfig) -> tuple[DatasetSchema, dict[str, DimensionCategory]]:
    schema = DatasetSchemaFactory.create(name='Endenergie')
    DatasetMetricFactory.create(schema=schema, name='Value', unit='MWh/a')
    DatasetMetricFactory.create(schema=schema, name='quality', unit='')
    categories: dict[str, DimensionCategory] = {}
    for name, identifiers in (('Sektoren', (*STATIONARY_SECTORS, 'transport')), ('Energieträger', (*CARRIERS, 'petrol'))):
        dimension = DimensionFactory.create(name=name)
        DatasetSchemaDimensionFactory.create(schema=schema, dimension=dimension)
        for identifier in identifiers:
            categories[identifier] = DimensionCategoryFactory.create(dimension=dimension, identifier=identifier)
    DatasetFactory.create(schema=schema, identifier=plausibility.ENERGY_DATASET, scope=template)
    return schema, categories


def test_setup_provisions_every_band_as_a_framework_sum(template: InstanceConfig, energy) -> None:
    schema, categories = energy
    framework = setup_bisko()

    ranges = {rule.identifier: rule for rule in DatasetMetricPlausibilityRange.objects.filter(framework=framework)}
    assert set(ranges) == {band.identifier for band in BANDS}
    households = ranges['private_households']
    assert households.metric == schema.metrics.get(name='Value')
    assert households.aggregation == 'sum'
    assert households.denominator == 'population'
    assert sorted(households.selection[str(categories['private_households'].dimension.uuid)]) == [
        str(categories['private_households'].uuid)
    ]
    assert len(households.selection[str(categories['electricity'].dimension.uuid)]) == len(CARRIERS)
    ratio = ranges['electricity-yoy']
    assert (ratio.reference, ratio.denominator, ratio.max_gap_years) == ('previous_year', 'none', 1)
    assert not ranges['district_heating'].lower
    assert households.source.url
    assert not households.source.is_example


def test_reprovisioning_is_idempotent_and_revises_changed_bands(template: InstanceConfig, energy, monkeypatch) -> None:
    framework = setup_bisko()
    before = dict(DatasetMetricPlausibilityRange.objects.values_list('identifier', 'uuid'))

    assert provision_bisko_plausibility_ranges(framework) == len(BANDS)
    assert dict(DatasetMetricPlausibilityRange.objects.values_list('identifier', 'uuid')) == before
    assert set(DatasetMetricPlausibilityRange.objects.values_list('revision', flat=True)) == {1}

    changed = tuple(
        dataclasses.replace(band, upper=band.upper * 2) if band.identifier == 'natural_gas' else band for band in BANDS
    )
    monkeypatch.setattr(plausibility, 'BANDS', changed[1:])  # also retire the first band
    provision_bisko_plausibility_ranges(framework)

    revisions = dict(DatasetMetricPlausibilityRange.objects.values_list('identifier', 'revision'))
    assert BANDS[0].identifier not in revisions
    assert revisions['natural_gas'] == 2
    assert revisions['electricity'] == 1
    assert DatasetMetricPlausibilityRange.objects.get(identifier='natural_gas').uuid == before['natural_gas']


def test_template_without_energy_dataset_gets_no_ranges(template: InstanceConfig) -> None:
    framework = setup_bisko()
    assert provision_bisko_plausibility_ranges(framework) == 0
    assert not DatasetMetricPlausibilityRange.objects.exists()


def test_member_municipality_sees_household_sum_finding(template: InstanceConfig, energy) -> None:
    schema, categories = energy
    framework = setup_bisko()
    municipality = InstanceConfigFactory.create(identifier='bisko-test', name='Testgemeinde')
    FrameworkConfigFactory.create(instance_config=municipality, framework=framework)
    OrganizationPopulation.objects.create(
        framework=framework,
        organization=municipality.organization,
        year=2023,
        value=1000,
        source_dataset='test/population',
        source_revision='test-v1',
    )
    dataset = DatasetFactory.create(schema=schema, identifier=plausibility.ENERGY_DATASET, scope=municipality)
    value = schema.metrics.get(name='Value')
    # 7 MWh per inhabitant in households would be typical; a unit slip makes it 28.
    for carrier in CARRIERS:
        DataPointFactory.create(
            dataset=dataset,
            metric=value,
            date=datetime.date(2023, 1, 1),
            value=Decimal(4000),
            dimension_categories=[categories['private_households'], categories[carrier]],
        )

    findings = {finding.rule_uuid: finding for finding in evaluate_dataset_plausibility(dataset)}
    households = DatasetMetricPlausibilityRange.objects.get(framework=framework, identifier='private_households')
    finding = findings[households.uuid]
    assert (finding.normalized, finding.component_count, finding.complete) == (28, len(CARRIERS), True)
