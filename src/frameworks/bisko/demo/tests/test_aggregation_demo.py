"""Generation preserves coordinates and repeat runs protect existing inventories."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)

from frameworks import submissions
from frameworks.bisko.demo.aggregation import (
    DEMO_KEY,
    DemoRecipe,
    DemoRecord,
    _energy,
    _repeat,
    energy_factor,
    inventory_fingerprint,
)
from frameworks.bisko.demo.dashboards import CARRIERS, SECTORS
from frameworks.models import DataEvidenceKind, DataQualityLevel, DataQualityScheme
from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.tests.factories import InstanceConfigFactory

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset

    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


@pytest.fixture
def energy_instance() -> tuple[InstanceConfig, Dataset, DemoRecipe]:
    instance = InstanceConfigFactory.create(
        name='Aggregation demo',
        config_source='database',
        spec=InstanceModelSpec(years=YearsSpec(reference=2022, min_historical=2022, max_historical=2023, target=2030)),
    )
    framework = FrameworkFactory.create(identifier='bisko')
    FrameworkConfigFactory.create(framework=framework, instance_config=instance)
    scheme = DataQualityScheme.objects.create(framework=framework, identifier='quality', version='1', name='Quality')
    for identifier, score in [('A', 1), ('B', 0.5), ('C', 0.25), ('D', 0)]:
        DataQualityLevel.objects.create(scheme=scheme, identifier=identifier, name=identifier, score=Decimal(str(score)))
    schema = DatasetSchemaFactory.create()
    metric = DatasetMetricFactory.create(schema=schema, name='Value', unit='MWh/a')
    dataset = DatasetFactory.create(scope=instance, schema=schema, identifier='kommune/endenergieverbrauch')
    sector_dimension = DimensionFactory.create(name='Sektoren')
    carrier_dimension = DimensionFactory.create(name='Energieträger')
    sectors = {s: DimensionCategoryFactory.create(dimension=sector_dimension, identifier=s) for s in SECTORS}
    carriers = {c: DimensionCategoryFactory.create(dimension=carrier_dimension, identifier=c) for c in CARRIERS}
    for sector in SECTORS:
        for carrier in CARRIERS:
            DataPointFactory.create(
                dataset=dataset,
                metric=metric,
                date=date(2023, 1, 1),
                value=100,
                dimension_categories=[sectors[sector], carriers[carrier]],
            )
    recipe = DemoRecipe(seed=42, years=[2022, 2023], district='03451', template_revision=1)
    return instance, dataset, recipe


def test_two_year_generation_preserves_2023_and_assigns_grades(
    energy_instance: tuple[InstanceConfig, Dataset, DemoRecipe],
) -> None:
    instance, dataset, recipe = energy_instance
    _energy(instance, recipe, '03451001')
    assert dataset.data_points.count() == 120
    assert set(dataset.data_points.filter(date__year=2023).values_list('value', flat=True)) == {Decimal(100)}
    for point in dataset.data_points.select_related('evidence__quality_level').prefetch_related('dimension_categories'):
        assert point.evidence.kind == DataEvidenceKind.ESTIMATED
        assert point.evidence.quality_level is not None
        coordinates = {category.identifier for category in point.dimension_categories.all()}
        if 'electricity' in coordinates or 'natural_gas' in coordinates or 'municipal_facilities' in coordinates:
            assert point.evidence.quality_level.identifier == 'A'
    assert dataset.data_points.filter(date__year=2022, value__gt=100).exists()
    assert dataset.data_points.filter(date__year=2022, value__lt=100).exists()


def test_repeat_accepts_unchanged_inventories_and_rejects_edits(
    energy_instance: tuple[InstanceConfig, Dataset, DemoRecipe],
) -> None:
    instance, dataset, recipe = energy_instance
    _energy(instance, recipe, '03451001')
    for year in recipe.years:
        submission = submissions.create_submission(instance, period_start=year, user=None)
        submissions.finalise(submissions.request_review(submission, user=None), user=None)
    instance.refresh_from_db()
    dataset.refresh_from_db()
    record = DemoRecord(recipe=recipe, fingerprint=inventory_fingerprint(instance))
    dataset.spec = {**(dataset.spec or {}), DEMO_KEY: record.model_dump(mode='json')}
    dataset.save(update_fields=['spec'])
    assert _repeat(instance, recipe) is False
    with pytest.raises(ValueError, match='recipe changed'):
        _repeat(instance, recipe.model_copy(update={'seed': 43}))
    point = dataset.data_points.first()
    assert point is not None
    point.value = Decimal(999)
    point.save(update_fields=['value'])
    with pytest.raises(ValueError, match='changed since demo generation'):
        _repeat(instance, recipe)


def test_generation_is_stable_and_keeps_heat_pump_growth() -> None:
    factors = [energy_factor(42, ags, 'private_households', 'natural_gas', 2022) for ags in ['03451001', '12072002']]
    assert len(set(factors)) == 2
    assert factors == [energy_factor(42, ags, 'private_households', 'natural_gas', 2022) for ags in ['03451001', '12072002']]
    assert energy_factor(42, '03451001', 'private_households', 'natural_gas', 2023) == 1
    assert energy_factor(42, '03451001', 'private_households', 'environmental_heat', 2022) < 1
