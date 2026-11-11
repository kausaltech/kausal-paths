"""Demo settings describe the synthetic source; its quality grades are invented for the demo."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.core.management import call_command
from django.core.management.base import CommandError

import polars as pl
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

from frameworks.bisko.default_sources import SECTORS
from frameworks.bisko.demo.aggregation import DEMO_KEY, DemoRecipe, _adopt_dashboard_demo, _repeat
from frameworks.bisko.demo.inventories import DEMO_GRADES, LEGACY_SOURCE_NAME, configure_demo_definition
from frameworks.models import DataEvidenceKind, DataPointEvidence, DataQualityLevel, DataQualityScheme, FrameworkConfig
from frameworks.tests.factories import FrameworkFactory
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.scenario import Scenario, ScenarioKind
from nodes.template_graph import publish_template_instance
from nodes.tests.factories import InstanceConfigFactory
from params.param import NumberParameter

if TYPE_CHECKING:
    from pathlib import Path

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
    template.refresh_from_db()
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
    carriers = {
        identifier: DimensionCategoryFactory.create(dimension=carrier, identifier=identifier) for identifier in DEMO_GRADES
    }
    dataset = DatasetFactory.create(scope=instance, schema=schema, identifier='kommune/endenergieverbrauch')
    for year in (2000, 2005):
        for sector_id in SECTORS:
            for carrier_id in DEMO_GRADES:
                point = DataPointFactory.create(
                    dataset=dataset,
                    metric=metric,
                    date=date(year, 1, 1),
                    value=100,
                    dimension_categories=[sectors[sector_id], carriers[carrier_id]],
                )
                DataPointEvidence.objects.create(data_point=point, kind=DataEvidenceKind.ESTIMATED)
    source = DataSource.objects.create(
        scope_content_type=ContentType.objects.get_for_model(instance), scope_id=instance.pk, name=LEGACY_SOURCE_NAME
    )
    DatasetSourceReference.objects.create(dataset=dataset, data_source=source)
    return template, instance, dataset


def test_configuration_is_sparse_and_preserves_other_local_settings(demo: tuple[InstanceConfig, InstanceConfig, Dataset]) -> None:
    template, instance, dataset = demo
    local = instance.ensure_spec().local_scenario('default')
    local.param_values['untouched'] = 7
    instance.ensure_spec().features.show_accumulated_effects = False
    instance.save(update_fields=['spec'])
    configure_demo_definition(instance, [2000, 2005])
    instance.refresh_from_db()
    spec = instance.ensure_spec()
    assert spec.params == []
    assert spec.scenarios[0].param_values == {'untouched': 7}
    assert spec.years.reference == 2000
    assert spec.years.historical == [2000, 2005]
    assert spec.years.target == 2030
    assert spec.features.show_accumulated_effects is False
    template.refresh_from_db()
    assert template.ensure_spec().params[0].value == 2
    assert dataset.data_points.count() == 120
    assert not DataPointEvidence.objects.filter(data_point__dataset=dataset, quality_level__isnull=False).exists()


def test_existing_dashboard_demo_is_adopted_without_replacing_values(
    demo: tuple[InstanceConfig, InstanceConfig, Dataset],
) -> None:
    template, instance, dataset = demo
    assert template.live_revision_id is not None
    recipe = DemoRecipe(seed=42, years=[2000, 2005], district='03404', template_revision=template.live_revision_id)
    before = list(dataset.data_points.order_by('uuid').values_list('uuid', 'value'))
    assert _adopt_dashboard_demo(instance, recipe)
    instance.refresh_from_db()
    dataset.refresh_from_db()
    assert list(dataset.data_points.order_by('uuid').values_list('uuid', 'value')) == before
    assert dataset.spec is not None
    assert DEMO_KEY in dataset.spec
    assert instance.submissions.filter(status='final').count() == 2
    assert not _repeat(instance, recipe)


def test_dashboard_adoption_refuses_manually_entered_values(demo: tuple[InstanceConfig, InstanceConfig, Dataset]) -> None:
    template, instance, dataset = demo
    assert template.live_revision_id is not None
    recipe = DemoRecipe(seed=42, years=[2000, 2005], district='03404', template_revision=template.live_revision_id)
    point = dataset.data_points.first()
    assert point is not None
    DataPointEvidence.objects.filter(data_point=point).update(kind=DataEvidenceKind.OBSERVED)
    with pytest.raises(ValueError, match='manually supplied'):
        _adopt_dashboard_demo(instance, recipe)
    assert not instance.submissions.exists()


def test_production_energy_prior_can_be_exported_without_the_dashboard_loader(
    demo: tuple[InstanceConfig, InstanceConfig, Dataset],
    tmp_path: Path,
) -> None:
    _template, instance, dataset = demo
    output = tmp_path / 'energy-prior.parquet'
    call_command('export_bisko_energy_prior', instance.identifier, dataset.identifier, year=2000, output=output)
    frame = pl.read_parquet(output)
    assert frame.height == 60
    assert set(frame['year']) == {2000}
    assert set(frame['unit']) == {'MWh/a'}
    assert set(frame.columns) == {'sector', 'energy_carrier', 'value_mwh_per_year', 'year', 'unit'}
    with pytest.raises(CommandError, match='new output file'):
        call_command('export_bisko_energy_prior', instance.identifier, dataset.identifier, year=2000, output=output)
    assert dataset.data_points.count() == 120
