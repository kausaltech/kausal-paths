from decimal import Decimal
from typing import TYPE_CHECKING

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction

import pytest

from frameworks.models import DataQualityLevel, DataQualityScheme, Framework, FrameworkConfig
from frameworks.provisioning import setup_bisko
from frameworks.tests.factories import FrameworkFactory
from nodes.tests.factories import InstanceConfigFactory

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
    newer = DataQualityScheme.objects.create(framework=scheme.framework, identifier='bisko', version='2', name='New scale')
    DataQualityLevel.objects.create(scheme=newer, identifier='B', name='B', score=Decimal('0.75'))
    assert scheme.levels.get(identifier='B').score == Decimal('0.5')
    assert newer.levels.get(identifier='B').score == Decimal('0.75')
