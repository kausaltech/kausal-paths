"""Generation preserves coordinates and repeat runs protect existing inventories."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

from django.core.management import call_command
from django.db import transaction

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
    _refresh_submissions,
    _repeat,
    _save_record,
    energy_factor,
    inventory_fingerprint,
)
from frameworks.bisko.demo.dashboards import CARRIERS, SECTORS
from frameworks.conversion import share_template_catalogue
from frameworks.models import DataEvidenceKind, DataQualityLevel, DataQualityScheme, SubmissionStatus
from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.defs.indicator_defs import SubmissionIndicatorsSpec
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.models import NodeInputPortBinding
from nodes.template_graph import publish_template_instance
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from nodes.units import unit_registry

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset

    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


@pytest.fixture
def energy_instance() -> tuple[InstanceConfig, Dataset, DemoRecipe]:
    instance = InstanceConfigFactory.create(
        name='Aggregation demo',
        owner='Demo',
        config_source='database',
        spec=InstanceModelSpec(years=YearsSpec(reference=2022, min_historical=2022, max_historical=2023, target=2030)),
    )
    spec = instance.ensure_spec()
    spec.features.use_datasets_from_db = True
    instance.spec = spec
    instance.save(update_fields=['spec'])
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


@pytest.fixture
def refresh_demo(energy_instance: tuple[InstanceConfig, Dataset, DemoRecipe]) -> tuple[InstanceConfig, Dataset, DemoRecipe]:
    instance, dataset, recipe = energy_instance
    framework = instance.framework_config.framework
    template = InstanceConfigFactory.create(
        name='Demo template', owner='Demo', config_source='database', spec=instance.ensure_spec()
    )
    framework.template_instance = template
    framework.save(update_fields=['template_instance'])
    source = DatasetFactory.create(scope=template, identifier='de/demo_energy')
    metric = DatasetMetricFactory.create(schema=source.schema, name='Value', unit='MWh/a')
    for year, value in [(2022, 7), (2023, 8)]:
        DataPointFactory.create(dataset=source, metric=metric, date=date(year, 1, 1), value=value)
    port_id = uuid4()
    unit = unit_registry.parse_units('MWh/a')
    node = NodeConfigFactory.create(
        instance=template,
        identifier='energy',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='nodes.simple.AdditiveNode'),
            input_ports=[InputPortDef(id=port_id, unit=unit, quantity='energy')],
            output_ports=[OutputPortDef(id=uuid4(), unit=unit, quantity='energy', dimensions=[])],
        ),
    )
    NodeInputPortBinding.objects.create(instance=template, node=node, port_id=port_id, dataset=source, metric=metric)
    share_template_catalogue(framework)
    old_revision = publish_template_instance(template)
    instance.template_revision = old_revision
    instance.save(update_fields=['template_revision'])
    recipe = recipe.model_copy(update={'template_revision': old_revision.pk})
    _energy(instance, recipe, '03451001')
    for year in recipe.years:
        submission = submissions.create_submission(instance, period_start=year, user=None)
        submissions.finalise(submissions.request_review(submission, user=None), user=None)
    _save_record(instance, recipe)
    template.refresh_from_db()
    spec = template.ensure_spec()
    spec.submission_indicators = SubmissionIndicatorsSpec.model_validate({'indicators': [{'id': 'energy', 'node': 'energy'}]})
    template.spec = spec
    template.save(update_fields=['spec'])
    revision = publish_template_instance(template)
    return instance, dataset, recipe.model_copy(update={'template_revision': revision.pk})


def test_refresh_supersedes_finals_and_freezes_indicators_without_replacing_inputs(
    refresh_demo: tuple[InstanceConfig, Dataset, DemoRecipe],
) -> None:
    instance, dataset, recipe = refresh_demo
    previous = list(instance.submissions.all())
    points = list(dataset.data_points.order_by('uuid').values_list('uuid', 'value'))
    assert _refresh_submissions(instance, recipe)
    instance.refresh_from_db()
    assert instance.template_revision_id == recipe.template_revision
    assert list(dataset.data_points.order_by('uuid').values_list('uuid', 'value')) == points
    for year, value in [(2022, 7), (2023, 8)]:
        final = instance.submissions.get(status=SubmissionStatus.FINAL, period_start=year)
        assert final.indicators is not None
        assert final.indicators.values['energy'].total == value
        assert final.indicators.provenance.instance_revision == final.instance_revision_id
        assert final.template_revision_id == recipe.template_revision
        assert final.supersedes_id in {submission.pk for submission in previous}
    for submission in previous:
        submission.refresh_from_db()
        assert submission.status == SubmissionStatus.SUPERSEDED
        assert submission.indicators is None
    count = instance.submissions.count()
    assert not _refresh_submissions(instance, recipe)
    assert not _repeat(instance, recipe)
    assert instance.submissions.count() == count
    # Backfill must also find the declarations inherited by an authored snapshot.
    instance.submissions.filter(pk=final.pk).update(indicators=None)
    call_command('materialise_submission_indicators', instance.identifier, '--apply')
    final.refresh_from_db()
    assert final.indicators is not None
    assert final.indicators.values['energy'].total == 8
    assert final.indicators.provenance.instance_revision == final.instance_revision_id
    assert instance.submissions.count() == count


def test_refresh_dry_run_rolls_back_corrections(refresh_demo: tuple[InstanceConfig, Dataset, DemoRecipe]) -> None:
    instance, _dataset, recipe = refresh_demo
    previous = list(instance.submissions.values_list('uuid', 'status', 'instance_revision_id'))
    template_revision = instance.template_revision_id
    with transaction.atomic():
        _refresh_submissions(instance, recipe)
        transaction.set_rollback(True)
    instance.refresh_from_db()
    assert list(instance.submissions.values_list('uuid', 'status', 'instance_revision_id')) == previous
    assert instance.template_revision_id == template_revision


def test_refresh_refuses_open_work_and_changed_seed(refresh_demo: tuple[InstanceConfig, Dataset, DemoRecipe]) -> None:
    instance, _dataset, recipe = refresh_demo
    with pytest.raises(ValueError, match='only the template may change'):
        _refresh_submissions(instance, recipe.model_copy(update={'seed': 43}))
    submissions.create_submission(instance, period_start=2022, user=None)
    with pytest.raises(ValueError, match='open submissions exist'):
        _refresh_submissions(instance, recipe)
    assert instance.submissions.filter(status=SubmissionStatus.FINAL).count() == 2
