"""Demo settings describe the synthetic source; its quality grades are invented for the demo."""

from datetime import date
from decimal import Decimal
from io import StringIO
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.core.management import call_command
from django.core.management.base import CommandError

import pytest

from kausal_common.datasets.models import DatasetSourceReference, DataSource
from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)

from frameworks.management.commands.load_bisko_demo import (
    CARRIERS,
    DEMO_GRADES,
    SECTORS,
    SOURCE_NAME,
    configure_demo_definition,
    existing_demo_years,
)
from frameworks.models import DataEvidenceKind, DataPointEvidence, DataQualityLevel, DataQualityScheme, FrameworkConfig
from frameworks.tests.factories import FrameworkFactory
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.instance_serialization import build_instance_snapshot
from nodes.scenario import Scenario, ScenarioKind
from nodes.template_graph import publish_template_instance
from nodes.tests.factories import InstanceConfigFactory
from params.base import ParameterOwner
from params.param import BoolParameter, NumberParameter

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset

    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


@pytest.fixture
def demo() -> tuple[InstanceConfig, InstanceConfig, Dataset]:
    template = InstanceConfigFactory.create(
        name='Demo template',
        owner='Test owner',
        config_source='database',
        spec=InstanceModelSpec(
            years=YearsSpec(reference=1990, min_historical=1990, max_historical=2023, target=2030),
            params=[
                BoolParameter(local_id='municipal_facilities_included_in_commerce', value=True),
                NumberParameter(local_id='untouched', value=2),
            ],
            scenarios=[Scenario(id='default', name='Default', kind=ScenarioKind.DEFAULT)],
        ),
    )
    framework = FrameworkFactory.create(identifier='bisko', template_instance=template)
    scheme = DataQualityScheme.objects.create(framework=framework, identifier='quality', version='1', name='Quality')
    for order, (identifier, score) in enumerate((('A', '1'), ('B', '0.5'), ('C', '0.25'), ('D', '0'))):
        DataQualityLevel.objects.create(scheme=scheme, identifier=identifier, name=identifier, order=order, score=Decimal(score))
    revision = publish_template_instance(template)
    instance = InstanceConfigFactory.create(
        name='Demo city',
        owner='Test owner',
        config_source='database',
        template_revision=revision,
        spec=InstanceModelSpec(years=template.ensure_spec().years),
    )
    FrameworkConfig.objects.create(framework=framework, instance_config=instance)
    schema = DatasetSchemaFactory.create()
    metric = DatasetMetricFactory.create(schema=schema, name='Value', unit='MWh/a')
    sector = DimensionFactory.create(name='Sektoren')
    carrier = DimensionFactory.create(name='Energieträger')
    DatasetSchemaDimensionFactory.create(schema=schema, dimension=sector)
    DatasetSchemaDimensionFactory.create(schema=schema, dimension=carrier)
    sectors = {identifier: DimensionCategoryFactory.create(dimension=sector, identifier=identifier) for identifier in SECTORS}
    carriers = {identifier: DimensionCategoryFactory.create(dimension=carrier, identifier=identifier) for identifier in CARRIERS}
    dataset = DatasetFactory.create(scope=instance, schema=schema, identifier='kommune/endenergieverbrauch')
    for year in (2000, 2005):
        for sector_id in SECTORS:
            for carrier_id in CARRIERS:
                point = DataPointFactory.create(
                    dataset=dataset,
                    metric=metric,
                    date=date(year, 1, 1),
                    dimension_categories=[sectors[sector_id], carriers[carrier_id]],
                )
                DataPointEvidence.objects.create(data_point=point, kind=DataEvidenceKind.ESTIMATED)
    source = DataSource.objects.create(
        scope_content_type=ContentType.objects.get_for_model(instance), scope_id=instance.pk, name=SOURCE_NAME
    )
    DatasetSourceReference.objects.create(dataset=dataset, data_source=source)
    return template, instance, dataset


def test_configuration_is_sparse_and_preserves_other_local_settings(demo: tuple[InstanceConfig, InstanceConfig, Dataset]) -> None:
    template, instance, dataset = demo
    local = instance.ensure_spec().local_scenario('default')
    local.param_values['untouched'] = 7
    instance.ensure_spec().features.show_accumulated_effects = False
    instance.save(update_fields=['spec'])
    configure_demo_definition(instance, existing_demo_years(dataset))
    instance.refresh_from_db()
    spec = instance.ensure_spec()
    assert spec.params == []
    assert spec.scenarios[0].param_values == {'untouched': 7, 'municipal_facilities_included_in_commerce': False}
    assert spec.years.reference == 2000
    assert spec.years.historical == [2000, 2005]
    assert spec.years.target == 2030
    assert spec.features.show_accumulated_effects is False
    effective = build_instance_snapshot(instance)
    assert next(p for p in effective.spec.params if p.local_id == 'municipal_facilities_included_in_commerce').value is False
    template.refresh_from_db()
    assert template.ensure_spec().params[0].value is True
    assert dataset.data_points.count() == 120
    assert not DataPointEvidence.objects.filter(data_point__dataset=dataset, quality_level__isnull=False).exists()


def test_reconcile_command_defaults_to_rollback(demo: tuple[InstanceConfig, InstanceConfig, Dataset]) -> None:
    _, instance, _ = demo
    output = StringIO()
    call_command('load_bisko_demo', reconcile_existing=True, instance=[instance.identifier], stdout=output)
    instance.refresh_from_db()
    assert instance.ensure_spec().scenarios == []
    assert instance.ensure_spec().years.min_historical == 1990
    assert 'definition changes rolled back' in output.getvalue()
    call_command('load_bisko_demo', reconcile_existing=True, instance=[instance.identifier], apply=True, stdout=output)
    instance.refresh_from_db()
    assert instance.ensure_spec().years.historical == [2000, 2005]
    grades = {
        (categories['Sektoren'], categories['Energieträger']): grade
        for categories, grade in (
            (
                {category.dimension.name: category.identifier for category in evidence.data_point.dimension_categories.all()},
                evidence.quality_level.identifier if evidence.quality_level else None,
            )
            for evidence in DataPointEvidence.objects
            .filter(data_point__dataset__scope_id=instance.pk)
            .prefetch_related('data_point__dimension_categories__dimension')
            .select_related('quality_level')
        )
    }
    assert grades[('private_households', 'electricity')] == 'A'
    assert grades[('private_households', 'heating_oil')] == 'C'
    assert grades[('municipal_facilities', 'heating_oil')] == 'A'
    assert set(grades.values()) == set(DEMO_GRADES.values())
    assert 'grades changed on 120 cells' in output.getvalue()
    previous_invalidation = instance.cache_invalidated_at
    call_command('load_bisko_demo', reconcile_existing=True, instance=[instance.identifier], apply=True, stdout=output)
    instance.refresh_from_db()
    assert instance.cache_invalidated_at == previous_invalidation


def test_reconciliation_refuses_manually_reported_values(demo: tuple[InstanceConfig, InstanceConfig, Dataset]) -> None:
    _, instance, dataset = demo
    point = dataset.data_points.first()
    assert point is not None
    DataPointEvidence.objects.filter(data_point=point).update(kind=DataEvidenceKind.OBSERVED)
    with pytest.raises(CommandError, match='without estimated provenance'):
        call_command('load_bisko_demo', reconcile_existing=True, instance=[instance.identifier], apply=True)
    instance.refresh_from_db()
    assert instance.ensure_spec().scenarios == []


def test_reconciliation_refuses_a_framework_owned_overlap_parameter(demo: tuple[InstanceConfig, InstanceConfig, Dataset]) -> None:
    template, instance, dataset = demo
    template.ensure_spec().params[0].owner = ParameterOwner.FRAMEWORK
    template.save(update_fields=['spec'])
    instance.template_revision = publish_template_instance(template)
    instance.save(update_fields=['template_revision'])
    with pytest.raises(ValueError, match='instance-owned'):
        configure_demo_definition(instance, existing_demo_years(dataset))
    instance.refresh_from_db()
    assert instance.ensure_spec().scenarios == []
