import importlib
from types import SimpleNamespace
from uuid import uuid4

from django.apps import apps
from django.db import connection
from strawberry.types import get_object_definition

import polars as pl
import pytest

from kausal_common.datasets.models import DatasetMetricValidationRule
from kausal_common.datasets.tests.factories import DatasetMetricFactory

from datasets.validation import dump_violations, evaluate_rule, load_violations
from datasets.validation_rules import RequiredCombinationsRule, ValidationRuleSpecInput, rule_to_gql, validation_rule_adapter
from nodes.instance_serialization import upgrade_dataset_catalog_v15
from nodes.spec_sync import _apply_declared_metric_rules

pytestmark = pytest.mark.django_db


def test_deprecated_requirements_accept_alternatives_and_explicit_zero() -> None:
    a, b = uuid4(), uuid4()
    rule = validation_rule_adapter.validate_python({
        'kind': 'required_combinations',
        'enforcement': 'block_submission',
        'groups': [{'id': 'minimum', 'combinations': [str(a), str(b)]}],
    })
    frame = pl.DataFrame({'Year': [2020, 2021, 2022], 'region': ['a', 'b', 'a'], 'amount': [0.0, 2.0, None]})
    violations = evaluate_rule(rule, uuid4(), uuid4(), 'amount', frame, ['region'], {a: {'region': 'a'}, b: {'region': 'b'}})
    (violation,) = violations
    assert violation.kind == 'required_combinations'
    assert violation.enforcement == 'block_submission'
    assert violation.years == [2022]
    assert violation.requirement_group == 'minimum'
    assert violation.combination_ids == [a, b]
    assert load_violations(dump_violations(violations)) == violations


def test_deprecated_requirements_report_unresolved_combinations() -> None:
    combination = uuid4()
    rule = validation_rule_adapter.validate_python({
        'kind': 'required_combinations',
        'enforcement': 'block_edit',
        'groups': [{'id': 'minimum', 'combinations': [str(combination)]}],
    })
    (violation,) = evaluate_rule(rule, uuid4(), uuid4(), 'amount', pl.DataFrame({'Year': [2020], 'amount': [1.0]}), [])
    assert violation.kind == 'invalid_rule'
    assert violation.enforcement == 'block_publish'
    assert violation.combination_ids == [combination]


def test_deprecated_rule_has_graphql_output_but_no_editor_input() -> None:
    rule = validation_rule_adapter.validate_python({
        'kind': 'required_combinations',
        'enforcement': 'block_publish',
        'groups': [{'id': 'minimum', 'combinations': [str(uuid4())]}],
    })
    projected = rule_to_gql(rule)
    assert isinstance(projected, RequiredCombinationsRule.ObjectType)
    output = get_object_definition(type(projected), strict=True)
    assert next(field for field in output.fields if field.name == 'groups').deprecation_reason
    inputs = get_object_definition(ValidationRuleSpecInput, strict=True)
    assert 'required_combinations' not in {field.name for field in inputs.fields}


def test_instance_catalog_keeps_deprecated_rules_and_removes_retired_rules() -> None:
    required = {'kind': 'required_combinations', 'groups': [{'id': 'minimum', 'combinations': [str(uuid4())]}]}
    metric = {'validation_rules': [required, {'kind': 'allowed_combinations'}]}
    node_metric = {'validation_rules': [{'rule': required}]}
    snapshot = {'datasets': [{'metrics': [metric]}], 'nodes': [{'datasets': [{'metrics': [node_metric]}]}]}
    upgrade_dataset_catalog_v15(snapshot)
    assert metric['validation_rules'] == [{'rule': required}]
    assert node_metric['validation_rules'] == [{'rule': required}]


@pytest.mark.django_db
def test_migration_reports_legacy_rules_and_sync_can_remove_them(capsys: pytest.CaptureFixture[str]) -> None:
    metric = DatasetMetricFactory.create()
    row = DatasetMetricValidationRule.objects.create(
        metric=metric,
        rule={
            'kind': 'required_combinations',
            'enforcement': 'block_publish',
            'groups': [{'id': 'minimum', 'combinations': [str(uuid4())]}],
        },
    )
    migration = importlib.import_module('datasets.migrations.0003_retire_combination_rules')
    migration.refuse_retired_rules(apps, SimpleNamespace(connection=connection))
    assert 'Deprecated required_combinations' in capsys.readouterr().out
    assert DatasetMetricValidationRule.objects.filter(pk=row.pk).exists()
    assert _apply_declared_metric_rules(metric, [])
    assert not metric.validation_rules.exists()
    migration.refuse_retired_rules(apps, SimpleNamespace(connection=connection))
    assert not capsys.readouterr().out


@pytest.mark.django_db
def test_migration_still_refuses_retired_allowed_combinations() -> None:
    metric = DatasetMetricFactory.create()
    row = DatasetMetricValidationRule.objects.create(metric=metric, rule={'kind': 'allowed_combinations'})
    migration = importlib.import_module('datasets.migrations.0003_retire_combination_rules')
    with pytest.raises(RuntimeError, match='allowed_combinations'):
        migration.refuse_retired_rules(apps, SimpleNamespace(connection=connection))
    assert DatasetMetricValidationRule.objects.filter(pk=row.pk).exists()
