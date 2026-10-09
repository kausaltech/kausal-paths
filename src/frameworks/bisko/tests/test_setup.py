from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction

import pytest

from kausal_common.datasets.tests.factories import DataPointFactory, DatasetFactory, DatasetMetricFactory, DatasetSchemaFactory

from frameworks.bisko.provisioning import prepare_bisko_template, setup_bisko
from frameworks.evidence import QUALITY_OF_SPEC_KEY
from frameworks.models import DataPointEvidence, DataQualityLevel, DataQualityScheme, Framework, FrameworkConfig
from frameworks.tests.factories import FrameworkFactory
from nodes.defs.transform_def import FilterColumnOp
from nodes.models import NodeInputPortBinding
from nodes.scenario import Scenario
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from params.param import StringParameter

if TYPE_CHECKING:
    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


@pytest.fixture
def template() -> InstanceConfig:
    return InstanceConfigFactory.create(identifier='bisko', name='BISKO')


def test_setup_is_idempotent_and_keeps_models_and_settings(template: InstanceConfig) -> None:
    framework = setup_bisko()
    assert framework.template_instance == template
    assert framework.root_instance is None
    assert not framework.allow_user_registration
    assert not framework.allow_instance_creation
    assert not FrameworkConfig.objects.exists()
    assert not template.has_framework_config()
    framework.description = 'Locally maintained description'
    framework.save(update_fields=['description'])
    identities = list(DataQualityLevel.objects.values_list('uuid', flat=True))

    assert setup_bisko().pk == framework.pk
    framework.refresh_from_db()
    assert framework.description == 'Locally maintained description'
    assert list(DataQualityLevel.objects.values_list('uuid', flat=True)) == identities
    scheme = framework.quality_schemes.get()
    assert scheme.version == '1'
    assert list(scheme.levels.values_list('identifier', 'score')) == [
        ('A', Decimal(1)),
        ('B', Decimal('0.5')),
        ('C', Decimal('0.25')),
        ('D', Decimal(0)),
    ]


def test_setup_retires_unused_passenger_input_without_deleting_historical_data(template: InstanceConfig) -> None:
    node = NodeConfigFactory.create(instance=template, identifier='passenger_kilometers_own')
    dataset = DatasetFactory.create(scope=template, identifier='kommune/verkehrsleistung_personen')
    spec = template.ensure_spec()
    spec.params.append(StringParameter(local_id='municipality_name', value='Düsseldorf'))
    spec.scenarios.append(
        Scenario(id='default', name='Default', param_values={'passenger_kilometers_own.formula': 'extend_all(mileage)'})
    )
    template.spec = spec
    template.save(update_fields=['spec'])

    framework = setup_bisko()
    prepare_bisko_template(framework)
    prepare_bisko_template(framework)

    assert not template.nodes.filter(pk=node.pk).exists()
    assert type(dataset).objects.filter(pk=dataset.pk).exists()
    template.refresh_from_db()
    assert template.spec is not None
    assert all(param.local_id != 'municipality_name' for param in template.spec.params)
    assert 'passenger_kilometers_own.formula' not in template.spec.scenarios[0].param_values


def test_setup_removes_name_filter_from_ags_selected_template_input(template: InstanceConfig) -> None:
    node = NodeConfigFactory.create(instance=template, identifier='vehicle_kilometers_ifeu')
    dataset = DatasetFactory.create(scope=template, identifier='de/fahrleistung_strassenverkehr')
    assert dataset.schema is not None
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='mileage')
    binding = NodeInputPortBinding.objects.create(
        instance=template,
        node=node,
        port_id=uuid4(),
        dataset=dataset,
        metric=metric,
        transformations=[
            FilterColumnOp(column='ags', ref='ags_number'),
            FilterColumnOp(column='municipality', ref='municipality_name'),
        ],
    )

    framework = setup_bisko()
    prepare_bisko_template(framework)
    prepare_bisko_template(framework)

    binding.refresh_from_db()
    assert binding.transformations == [FilterColumnOp(column='ags', ref='ags_number'), FilterColumnOp(column='municipality')]


def test_setup_attaches_only_explicit_database_instances(template: InstanceConfig) -> None:
    municipality = InstanceConfigFactory.create(name='Municipality', config_source='database')
    untouched = InstanceConfigFactory.create(name='Unattached municipality', config_source='database')
    before = municipality.spec
    framework = setup_bisko(instance_identifiers=(municipality.identifier,))
    setup_bisko(instance_identifiers=(municipality.identifier,))
    assert framework.configs.get().instance_config == municipality
    assert not untouched.has_framework_config()
    municipality.refresh_from_db()
    assert municipality.spec == before
    assert municipality.root_page_id is None


@pytest.mark.parametrize('already_projected', [False, True])
def test_setup_projects_shared_quality_schema_and_preserves_legacy_values(
    template: InstanceConfig, already_projected: bool
) -> None:
    schema = DatasetSchemaFactory.create()
    value = DatasetMetricFactory.create(schema=schema, name='Value')
    quality = DatasetMetricFactory.create(schema=schema, name='quality')
    if already_projected:
        quality.spec = {QUALITY_OF_SPEC_KEY: str(value.uuid)}
        quality.save(update_fields=['spec'])
    template_dataset = DatasetFactory.create(scope=template, schema=schema)
    municipality = InstanceConfigFactory.create(name='Municipality', config_source='database')
    municipal_dataset = DatasetFactory.create(scope=municipality, schema=schema)
    outside = InstanceConfigFactory.create(name='Outside', config_source='database')
    outside_dataset = DatasetFactory.create(scope=outside, schema=schema)
    value_point = DataPointFactory.create(dataset=template_dataset, metric=value, date=date(2023, 1, 1), value=10)
    DataPointFactory.create(dataset=template_dataset, metric=quality, date=date(2023, 1, 1), value=Decimal('0.5'))
    outside_value = DataPointFactory.create(dataset=outside_dataset, metric=value, date=date(2023, 1, 1), value=20)
    DataPointFactory.create(dataset=outside_dataset, metric=quality, date=date(2023, 1, 1), value=Decimal('0.25'))

    setup_bisko(instance_identifiers=(municipality.identifier,))
    quality.refresh_from_db()
    assert quality.spec[QUALITY_OF_SPEC_KEY] == str(value.uuid)
    grades = dict(DataQualityLevel.objects.values_list('identifier', 'pk'))
    assert DataPointEvidence.objects.get(data_point=value_point).quality_level_id == grades['B']
    assert DataPointEvidence.objects.get(data_point=outside_value).quality_level_id == grades['C']
    assert not municipal_dataset.data_points.exists()

    setup_bisko(instance_identifiers=(municipality.identifier,))
    assert DataPointEvidence.objects.filter(data_point__in=[value_point, outside_value]).count() == 2


def test_setup_refuses_unmappable_shared_quality_values(template: InstanceConfig) -> None:
    schema = DatasetSchemaFactory.create()
    value = DatasetMetricFactory.create(schema=schema, name='Value')
    quality = DatasetMetricFactory.create(schema=schema, name='quality')
    dataset = DatasetFactory.create(scope=template, schema=schema)
    DataPointFactory.create(dataset=dataset, metric=value, date=date(2023, 1, 1), value=10)
    DataPointFactory.create(dataset=dataset, metric=quality, date=date(2023, 1, 1), value=Decimal('0.2'))

    with pytest.raises(ValueError, match='no unique BISKO grade'):
        setup_bisko()
    quality.refresh_from_db()
    assert QUALITY_OF_SPEC_KEY not in quality.spec
    assert not DataPointEvidence.objects.exists()


def test_setup_rejects_yaml_membership_without_changes(template: InstanceConfig) -> None:
    municipality = InstanceConfigFactory.create(name='Municipality', config_source='yaml')
    with pytest.raises(ValueError, match='YAML-backed'):
        setup_bisko(instance_identifiers=(municipality.identifier,))
    assert not Framework.objects.exists()
    assert not DataQualityScheme.objects.exists()


def test_setup_rejects_missing_instances(template: InstanceConfig) -> None:
    with pytest.raises(ValueError, match='Unknown instances'):
        setup_bisko(instance_identifiers=('missing',))
    assert not Framework.objects.exists()


def test_setup_rolls_back_on_existing_framework_membership(template: InstanceConfig) -> None:
    municipality = InstanceConfigFactory.create(name='Municipality', config_source='database')
    other = FrameworkFactory.create()
    FrameworkConfig.objects.create(framework=other, instance_config=municipality)
    with pytest.raises(ValueError, match='another framework'):
        setup_bisko(instance_identifiers=(municipality.identifier,))
    assert not Framework.objects.filter(identifier='bisko').exists()
    assert not DataQualityScheme.objects.exists()
    assert municipality.framework_config.framework == other


def test_setup_rejects_conflicting_template(template: InstanceConfig) -> None:
    setup_bisko()
    other = InstanceConfigFactory.create(name='Other template')
    with pytest.raises(ValueError, match='different template'):
        setup_bisko(template_identifier=other.identifier)


def test_setup_reports_conflicting_scores(template: InstanceConfig) -> None:
    framework = setup_bisko()
    # Simulate an existing catalogue imported outside the model save boundary.
    DataQualityLevel.objects.filter(identifier='B').update(score=Decimal('0.75'))
    with pytest.raises(ValueError, match='refusing to overwrite'):
        setup_bisko()
    assert framework.quality_schemes.get().levels.get(identifier='B').score == Decimal('0.75')


def test_scheme_version_and_grade_identity_are_preserved(template: InstanceConfig) -> None:
    scheme = setup_bisko().quality_schemes.get()
    level = scheme.levels.get(identifier='B')
    level.score = Decimal('0.75')
    with pytest.raises(ValidationError, match='reinterpreting'):
        level.save()
    scheme.version = '2'
    with pytest.raises(ValidationError, match='identity'):
        scheme.save()
    level.refresh_from_db()
    assert level.score == Decimal('0.5')


@pytest.mark.parametrize('score', [Decimal('-0.1'), Decimal('1.1')])
def test_quality_scores_have_database_constraints(template: InstanceConfig, score: Decimal) -> None:
    scheme = setup_bisko().quality_schemes.get()
    with pytest.raises(IntegrityError), transaction.atomic():
        DataQualityLevel.objects.filter(scheme=scheme, identifier='A').update(score=score)


def test_quality_versions_can_coexist(template: InstanceConfig) -> None:
    scheme = setup_bisko().quality_schemes.get()
    newer = DataQualityScheme.objects.create(framework=scheme.framework, identifier='quality', version='2', name='New scale')
    DataQualityLevel.objects.create(scheme=newer, identifier='B', name='B', score=Decimal('0.75'))
    assert scheme.levels.get(identifier='B').score == Decimal('0.5')
    assert newer.levels.get(identifier='B').score == Decimal('0.75')


def test_default_quality_release_is_idempotent_and_keeps_member_pins() -> None:  # noqa: PLR0915
    from frameworks.bisko.provisioning import reconcile_bisko_default_quality
    from nodes.defs.graph import QualityLevelKey
    from nodes.defs.port_def import InputPortDef
    from nodes.instance_serialization import InstanceSnapshot, build_instance_snapshot
    from nodes.template_graph import publish_template_instance
    from nodes.units import unit_registry

    template = InstanceConfigFactory.create(identifier='bisko', name='Method', config_source='database')
    node = NodeConfigFactory.create(instance=template, identifier='ifeu')
    assert node.spec is not None
    port = InputPortDef(id=uuid4(), identifier='input', unit=unit_registry.parse_units('kt/a'))
    node.spec.input_ports = [port]
    node.save(update_fields=['spec'])
    dataset = DatasetFactory.create(
        scope=template,
        identifier='de/fahrleistung_strassenverkehr',
        is_external_placeholder=True,
        external_ref={
            'repo_url': 'https://example.com/data.git',
            'commit': 'abc123',
            'dataset_id': 'de/fahrleistung_strassenverkehr',
        },
        spec={'preserved': True},
    )
    assert dataset.schema is not None
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='Value', unit='kt/a')
    NodeInputPortBinding.objects.create(instance=template, node=node, port_id=port.id, dataset=dataset, metric=metric)
    framework = FrameworkFactory.create(identifier='bisko', template_instance=template)
    old = publish_template_instance(template)
    members = [InstanceConfigFactory.create(name=f'Member {i}', config_source='database') for i in range(2)]
    for member in members:
        FrameworkConfig.objects.create(framework=framework, instance_config=member)
        member.template_revision = old
        member.save(update_fields=['template_revision'])
    unpinned = InstanceConfigFactory.create(name='Standalone member', config_source='database')
    FrameworkConfig.objects.create(framework=framework, instance_config=unpinned)
    assert dataset.identifier is not None
    unused = DatasetFactory.create(scope=template, identifier='unused/default')
    defaults = {
        dataset.identifier: QualityLevelKey(scheme='quality', level='B'),
        'unused/default': QualityLevelKey(scheme='quality', level='B'),
    }
    # Also repair releases where the row was already updated but the publication was missed.
    reconcile_bisko_default_quality(framework, defaults, publish=False)
    new = reconcile_bisko_default_quality(framework, defaults)
    assert new is not None
    assert new.pk != old.pk
    dataset.refresh_from_db()
    assert dataset.spec == {'preserved': True, 'default_quality': {'scheme': 'quality', 'level': 'B'}}
    unused.refresh_from_db()
    assert unused.spec['default_quality'] == {'scheme': 'quality', 'level': 'B'}
    assert InstanceSnapshot.from_serialized_data(old.content['model_snapshot']['structured']).datasets[0].default_quality is None
    for member in members:
        member.refresh_from_db()
        assert member.template_revision_id == old.pk
        assert build_instance_snapshot(member).datasets[0].default_quality is None
    unpinned.refresh_from_db()
    assert unpinned.template_revision_id is None
    repeated = reconcile_bisko_default_quality(framework, defaults)
    assert repeated is not None
    assert repeated.pk == new.pk
    # A member left on an older release does not trigger another template publication.
    members[0].template_revision = old
    members[0].save(update_fields=['template_revision'])
    advanced = reconcile_bisko_default_quality(framework, defaults)
    members[0].refresh_from_db()
    assert advanced is not None
    assert members[0].template_revision_id == old.pk
    assert advanced.pk == new.pk
    final = reconcile_bisko_default_quality(framework, defaults)
    assert final is not None
    assert final.pk == advanced.pk
