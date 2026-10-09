"""Framework qualifier namespaces, grade identity, and dynamic payloads."""

from decimal import Decimal
from importlib import import_module
from types import SimpleNamespace

from django.apps import apps
from django.db import connection

import polars as pl
import pytest

from common import polars as ppl, qualifiers
from datasets.snapshot import CellLabels, QualityLevelRef
from frameworks.models import DataQualityLevel, DataQualityScheme
from frameworks.qualifiers import attach_evidence_qualifiers, qualifier_catalog_for_framework, qualifier_catalog_for_instance
from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def test_catalog_is_scoped_and_includes_template_and_member_instances() -> None:
    template = InstanceConfigFactory.create(name='Qualifier template')
    framework = FrameworkFactory.create(identifier='bisko', template_instance=template)
    scheme = DataQualityScheme.objects.create(framework=framework, identifier='quality', version='1', name='Quality')
    DataQualityLevel.objects.create(scheme=scheme, identifier='B', name='B', score=Decimal('0.5'))
    member = InstanceConfigFactory.create(name='Qualifier member')
    FrameworkConfigFactory.create(framework=framework, instance_config=member)
    catalog = qualifier_catalog_for_framework(framework)
    assert [d.identifier for d in catalog.definitions] == ['reported', 'bisko_quality']
    assert catalog.dtype == pl.Struct({'reported': qualifiers.REPORTING_DTYPE, 'bisko_quality': qualifiers.COVERED_SCORE_DTYPE})
    assert qualifier_catalog_for_instance(member) == catalog
    assert qualifier_catalog_for_instance(template) == catalog
    assert qualifier_catalog_for_instance(None) == qualifiers.BUILTIN_QUALIFIERS
    assert qualifier_catalog_for_instance(InstanceConfigFactory.create(name='Qualifier test')) == qualifiers.BUILTIN_QUALIFIERS


def test_versions_preserve_grade_identity_and_change_cache_identity() -> None:
    instance = InstanceFactory.create()
    config = InstanceConfigFactory.create(instance=instance)
    instance.config = config
    framework = FrameworkFactory.create(identifier='bisko')
    FrameworkConfigFactory.create(framework=framework, instance_config=config)
    first = DataQualityScheme.objects.create(framework=framework, identifier='quality', version='2', name='Quality')
    original = DataQualityLevel.objects.create(scheme=first, identifier='B', name='B', score=Decimal('0.5'))
    catalog = qualifier_catalog_for_framework(framework)
    old_hash = instance.context.instance_hash
    # Numeric or lexical ordering of version labels must not silently choose defaults.
    latest = DataQualityScheme.objects.create(framework=framework, identifier='quality', version='10', name='Quality')
    DataQualityLevel.objects.create(scheme=latest, identifier='B', name='B', score=Decimal('0.75'))
    second = DataQualityScheme.objects.create(framework=framework, identifier='completeness', version='1', name='Completeness')
    DataQualityLevel.objects.create(scheme=second, identifier='B', name='B', score=Decimal('0.1'))
    updated = qualifier_catalog_for_framework(framework)
    assert catalog.hash_data() != updated.hash_data()
    assert updated['bisko_quality'].schemes[-1].version == '10'
    assert updated['bisko_quality'].schemes[0].levels[0].uuid == str(original.uuid)
    instance.context.__dict__.pop('qualifiers', None)
    instance.context.__dict__.pop('instance_hash', None)
    assert old_hash != instance.context.instance_hash


def test_evidence_is_attached_to_its_scheme_without_reinterpreting_old_grades() -> None:
    framework = FrameworkFactory.create(identifier='bisko')
    for identifier, score in [('quality', '0.5'), ('completeness', '0.25')]:
        scheme = DataQualityScheme.objects.create(framework=framework, identifier=identifier, version='1', name=identifier)
        DataQualityLevel.objects.create(scheme=scheme, identifier='B', name='B', score=Decimal(score))
    catalog = qualifier_catalog_for_framework(framework)
    grade = catalog['bisko_completeness'].schemes[0].levels[0]
    frame = ppl.to_ppdf(
        pl.DataFrame({'Year': [2020, 2021], 'Value': [10.0, 20.0]}),
        meta=ppl.DataFrameMeta(units={'Value': unit_registry.parse_units('MWh/a')}, primary_keys=['Year']),
    )
    grades: dict[CellLabels, QualityLevelRef] = {
        (2020, 'Value', ()): QualityLevelRef(uuid=grade.uuid, scheme='completeness', scheme_version='1', level='B')
    }
    result = attach_evidence_qualifiers(frame, grades, catalog)
    rows = result['Value__qual'].to_list()
    assert rows[0]['bisko_completeness'] == {'score': 0.25, 'coverage': 1.0}
    assert rows[0]['bisko_quality'] is None
    assert rows[1]['bisko_completeness'] is None
    assert result.schema['Value__qual'] == catalog.dtype

    # An identically named scale from another framework is not portable evidence
    # in a live database read.
    other = FrameworkFactory.create(identifier='other')
    foreign = DataQualityScheme.objects.create(framework=other, identifier='completeness', version='1', name='Other')
    foreign_grade = DataQualityLevel.objects.create(scheme=foreign, identifier='B', name='B', score=Decimal('0.9'))
    grades = {
        (2020, 'Value', ()): QualityLevelRef(uuid=str(foreign_grade.uuid), scheme='completeness', scheme_version='1', level='B'),
    }
    ignored = attach_evidence_qualifiers(frame, grades, catalog)
    assert ignored['Value__qual'][0]['bisko_completeness'] is None


def test_portable_historical_scheme_name_is_resolved_in_the_current_framework() -> None:
    framework = FrameworkFactory.create(identifier='bisko')
    scheme = DataQualityScheme.objects.create(framework=framework, identifier='quality', version='1', name='Quality')
    DataQualityLevel.objects.create(scheme=scheme, identifier='B', name='B', score=Decimal('0.5'))
    frame = ppl.to_ppdf(
        pl.DataFrame({'Year': [2020], 'Value': [10.0]}),
        meta=ppl.DataFrameMeta(units={'Value': unit_registry.parse_units('MWh/a')}, primary_keys=['Year']),
    )
    portable: dict[CellLabels, QualityLevelRef] = {
        (2020, 'Value', ()): QualityLevelRef(uuid='different-deployment', scheme='bisko', scheme_version='1', level='B')
    }
    result = attach_evidence_qualifiers(frame, portable, qualifier_catalog_for_framework(framework), portable=True)
    assert result['Value__qual'][0]['bisko_quality'] == {'score': 0.5, 'coverage': 1.0}


def test_scheme_rename_preserves_scheme_and_level_uuids() -> None:
    framework = FrameworkFactory.create(identifier='bisko')
    scheme = DataQualityScheme.objects.create(framework=framework, identifier='bisko', version='1', name='BISKO')
    grade = DataQualityLevel.objects.create(scheme=scheme, identifier='B', name='B', score=Decimal('0.5'))
    identity = (scheme.uuid, grade.uuid)
    migration = import_module('frameworks.migrations.0034_bisko_quality_scheme_identifier')
    migration.forwards(apps, SimpleNamespace(connection=connection))
    scheme.refresh_from_db()
    grade.refresh_from_db()
    assert scheme.identifier == 'quality'
    assert (scheme.uuid, grade.uuid) == identity


def test_each_named_assessment_reduces_its_own_coverage() -> None:
    catalog = qualifiers.QualifierCatalog((
        *qualifiers.BUILTIN_QUALIFIERS.definitions,
        *(
            qualifiers.QualifierDefinition(name, qualifiers.Propagation.COVERED_SCORE)
            for name in ('bisko_quality', 'bisko_completeness')
        ),
    ))
    frame = ppl.to_ppdf(
        pl.DataFrame({'Year': [2020, 2020], 'carrier': ['gas', 'oil'], 'Value': [12.0, 4.0]}).with_columns(
            qualifiers.make(
                catalog=catalog,
                reported=pl.lit(value=True),
                assessments={
                    'bisko_quality': qualifiers.covered_score(pl.lit(0.5), (pl.col('carrier') == 'gas').cast(pl.Float64)),
                    'bisko_completeness': qualifiers.covered_score(pl.lit(0.25), (pl.col('carrier') == 'oil').cast(pl.Float64)),
                },
            ).alias('Value__qual')
        ),
        meta=ppl.DataFrameMeta(units={'Value': unit_registry.parse_units('MWh/a')}, primary_keys=['Year', 'carrier']),
    )
    result = frame.paths.sum_over_dims('carrier')['Value__qual'][0]
    assert result['bisko_quality'] == {'score': 0.5, 'coverage': 0.75}
    assert result['bisko_completeness'] == {'score': 0.25, 'coverage': 0.25}
    assert result['reported']['all'] is True
