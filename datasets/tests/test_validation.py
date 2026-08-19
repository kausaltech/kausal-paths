"""Tests for dataset metric validation rules: evaluator, enforcement, publish gate, GraphQL."""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from pydantic import ValidationError as PydanticValidationError

import pytest

from kausal_common.datasets.category_domain import DatasetCategoryCombination, DatasetCategoryDomain
from kausal_common.datasets.models import DataPoint, DatasetMetricValidationRule, DimensionScope
from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)

from paths.tests.graphql import PathsTestClient

from datasets.validation import (
    DatasetValidationError,
    InstanceDatasetValidationError,
    evaluate_dataset_rules,
    load_violations,
)
from nodes.dataset_materialization import (
    materialize_dataset,
    refresh_dataset_materialization,
    require_valid_dataset_rules,
)
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.instance_serialization import DatasetSnapshot
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from users.tests.factories import UserFactory

pytestmark = pytest.mark.django_db


@pytest.fixture
def rig(db_instance_config):
    """One dataset: metric 'amount', dimension column 'region' with categories a/b."""
    schema = DatasetSchemaFactory.create(name='Validation schema')
    metric = DatasetMetricFactory.create(schema=schema, name='amount', label='Amount', unit='t/a')
    dimension = DimensionFactory.create(name='Region')
    DatasetSchemaDimensionFactory.create(schema=schema, dimension=dimension, column_name='region')
    cat_a = DimensionCategoryFactory.create(dimension=dimension, identifier='a', label='A')
    cat_b = DimensionCategoryFactory.create(dimension=dimension, identifier='b', label='B')
    DimensionScope.objects.create(
        dimension=dimension,
        scope_content_type=ContentType.objects.get_for_model(db_instance_config),
        scope_id=db_instance_config.pk,
        identifier='region',
    )
    dataset = DatasetFactory.create(schema=schema, identifier='validation-ds', scope=db_instance_config)
    return dataset, metric, cat_a, cat_b


def add_point(dataset, metric, year: int, value: float | None, category) -> DataPoint:
    return DataPointFactory.create(
        dataset=dataset,
        metric=metric,
        date=datetime.date(year, 1, 1),
        value=None if value is None else Decimal(str(value)),
        dimension_categories=[category],
    )


def set_rule(metric, blob: dict[str, Any]) -> DatasetMetricValidationRule:
    return DatasetMetricValidationRule.objects.create(metric=metric, rule=blob, order=0)


def test_value_range_rule_locates_offending_cells(rig):
    dataset, metric, cat_a, cat_b = rig
    set_rule(metric, {'kind': 'value_range', 'enforcement': 'block_edit', 'min': 0})
    add_point(dataset, metric, 2020, 10, cat_a)
    add_point(dataset, metric, 2020, -5, cat_b)

    violations = evaluate_dataset_rules(dataset)

    assert len(violations) == 1
    violation = violations[0]
    assert violation.kind == 'value_range'
    assert violation.enforcement == 'block_edit'
    assert violation.years == [2020]
    assert violation.categories == {'region': 'b'}
    assert violation.metric == 'amount'
    assert violation.dataset_uuid == dataset.uuid


def test_value_range_rule_supports_exclusive_bounds(rig):
    dataset, metric, cat_a, _ = rig
    set_rule(
        metric,
        {
            'kind': 'value_range',
            'enforcement': 'block_edit',
            'min': 10,
            'max': 20,
            'exclusive_min': True,
            'exclusive_max': True,
        },
    )
    add_point(dataset, metric, 2020, 10, cat_a)
    add_point(dataset, metric, 2021, 15, cat_a)
    add_point(dataset, metric, 2022, 20, cat_a)

    violations = evaluate_dataset_rules(dataset)

    assert [violation.years for violation in violations] == [[2020], [2022]]


@pytest.mark.parametrize(
    'rule',
    [
        {'max': 5, 'exclusive_min': True},
        {'min': 0, 'exclusive_max': True},
        {'min': 2, 'max': 1},
        {'min': 1, 'max': 1, 'exclusive_min': True},
    ],
)
def test_value_range_rule_rejects_empty_or_unbounded_exclusive_ranges(rule):
    from pydantic import ValidationError

    from datasets.validation_rules import ValueRangeRule

    with pytest.raises(ValidationError):
        ValueRangeRule.model_validate({'enforcement': 'block_edit', **rule})


def test_no_gaps_rule_observed_union(rig):
    dataset, metric, cat_a, cat_b = rig
    set_rule(metric, {'kind': 'no_gaps', 'enforcement': 'block_publish'})
    add_point(dataset, metric, 2020, 1, cat_a)
    # An explicit 0 counts as a value; an explicit null and a missing row do not.
    add_point(dataset, metric, 2020, 0, cat_b)
    add_point(dataset, metric, 2021, 2, cat_a)
    add_point(dataset, metric, 2021, None, cat_b)
    add_point(dataset, metric, 2022, 3, cat_a)

    violations = evaluate_dataset_rules(dataset)

    assert len(violations) == 1
    violation = violations[0]
    assert violation.kind == 'no_gaps'
    assert violation.categories == {'region': 'b'}
    assert violation.years == [2021, 2022]


def test_dimension_sum_rule(rig):
    dataset, metric, cat_a, cat_b = rig
    set_rule(metric, {'kind': 'dimension_sum', 'enforcement': 'block_publish', 'dimension': 'region', 'target': 1.0})
    add_point(dataset, metric, 2020, 0.4, cat_a)
    add_point(dataset, metric, 2020, 0.6, cat_b)
    add_point(dataset, metric, 2021, 0.4, cat_a)
    add_point(dataset, metric, 2021, 0.7, cat_b)

    violations = evaluate_dataset_rules(dataset)

    assert len(violations) == 1
    assert violations[0].years == [2021]
    assert violations[0].categories == {}


def test_required_combinations_rule_locates_missing_domain_cells(rig):
    dataset, metric, cat_a, cat_b = rig
    dimension = cat_a.dimension
    combo_a = DatasetCategoryCombination(
        id=uuid4(),
        identifier='region_a',
        categories={dimension.uuid: cat_a.uuid},
    )
    combo_b = DatasetCategoryCombination(
        id=uuid4(),
        identifier='region_b',
        categories={dimension.uuid: cat_b.uuid},
    )
    dataset.schema.category_domain = DatasetCategoryDomain(combinations=[combo_a, combo_b])
    dataset.schema.save(update_fields=['category_domain'])
    set_rule(
        metric,
        {
            'kind': 'required_combinations',
            'enforcement': 'block_publish',
            'groups': [{'id': 'region_b', 'combinations': [str(combo_b.id)]}],
        },
    )
    add_point(dataset, metric, 2020, 1, cat_a)
    add_point(dataset, metric, 2020, 0, cat_b)
    add_point(dataset, metric, 2021, 2, cat_a)

    (violation,) = evaluate_dataset_rules(dataset)

    assert violation.kind == 'required_combinations'
    assert violation.years == [2021]
    assert violation.categories == {'region': 'b'}
    assert violation.combination_ids == [combo_b.id]


def test_allowed_combinations_rule_rejects_rows_outside_closed_domain(rig):
    dataset, metric, cat_a, cat_b = rig
    dimension = cat_a.dimension
    combo_a = DatasetCategoryCombination(
        id=uuid4(),
        identifier='region_a',
        categories={dimension.uuid: cat_a.uuid},
    )
    dataset.schema.category_domain = DatasetCategoryDomain(mode='closed', combinations=[combo_a])
    dataset.schema.save(update_fields=['category_domain'])
    set_rule(metric, {'kind': 'allowed_combinations', 'enforcement': 'block_edit'})
    add_point(dataset, metric, 2020, 1, cat_a)
    add_point(dataset, metric, 2020, 2, cat_b)

    (violation,) = evaluate_dataset_rules(dataset)

    assert violation.kind == 'allowed_combinations'
    assert violation.years == [2020]
    assert violation.categories == {'region': 'b'}


def test_unparseable_rule_fails_loudly(rig):
    # Every write path validates rule blobs before persisting, so an
    # unparseable row is a bug: evaluation crashes instead of degrading.
    dataset, metric, _cat_a, _cat_b = rig
    set_rule(metric, {'kind': 'nonsense'})
    add_point(dataset, metric, 2020, 1, _cat_a)

    with pytest.raises(PydanticValidationError):
        evaluate_dataset_rules(dataset)


def test_block_edit_enforcement_uses_baseline_diff(rig):
    dataset, metric, cat_a, cat_b = rig
    add_point(dataset, metric, 2020, -5, cat_a)
    materialize_dataset(dataset)  # import-style refresh: no enforcement
    set_rule(metric, {'kind': 'value_range', 'enforcement': 'block_edit', 'min': 0})
    materialize_dataset(dataset)  # persist the pre-existing violation as baseline

    # An edit that does not introduce new violations is allowed even though
    # the pre-existing violation remains.
    add_point(dataset, metric, 2021, 10, cat_b)
    with transaction.atomic():
        materialization = refresh_dataset_materialization(dataset, touch=False, enforce_edit_rules=True)
    assert len(load_violations(materialization.validation_violations)) == 1

    # An edit that introduces a new violation is rejected.
    add_point(dataset, metric, 2022, -1, cat_b)
    with pytest.raises(DatasetValidationError) as excinfo, transaction.atomic():
        refresh_dataset_materialization(dataset, touch=False, enforce_edit_rules=True)
    assert len(excinfo.value.violations) == 1
    assert excinfo.value.violations[0].years == [2022]


def test_publish_gate_helper(rig):
    dataset, metric, cat_a, _cat_b = rig
    set_rule(metric, {'kind': 'value_range', 'enforcement': 'block_publish', 'max': 100})
    add_point(dataset, metric, 2020, 50, cat_a)
    materialization = materialize_dataset(dataset)
    require_valid_dataset_rules([materialization])  # no violations: passes

    add_point(dataset, metric, 2021, 200, cat_a)
    materialization = materialize_dataset(dataset)
    with pytest.raises(InstanceDatasetValidationError) as excinfo:
        require_valid_dataset_rules([materialization])
    assert len(excinfo.value.violations) == 1


def test_snapshot_carries_validation_rules(rig):
    dataset, metric, _cat_a, _cat_b = rig
    blob = {'kind': 'no_gaps', 'enforcement': 'block_publish'}
    rule = set_rule(metric, blob)

    snapshot = DatasetSnapshot.from_model(dataset)

    (metric_snapshot,) = snapshot.metrics
    (rule_snapshot,) = metric_snapshot.validation_rules
    assert rule_snapshot.uuid == rule.uuid
    assert rule_snapshot.rule.model_dump(mode='json') == blob


# --- GraphQL surface ---------------------------------------------------------


@pytest.fixture
def db_instance_config():
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(
        years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
    )
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        owner='Test Owner',
        spec=spec,
    )


@pytest.fixture
def gql_client(client, db_instance_config) -> PathsTestClient:
    user = UserFactory.create(is_superuser=True)
    client.force_login(user)
    tc = PathsTestClient(client)
    tc.set_instance(db_instance_config)
    return tc


SET_METRIC_VALIDATION_RULES = """
mutation SetMetricValidationRules($instanceId: ID!, $datasetId: ID!, $metricId: UUID!, $rules: [ValidationRuleInput!]!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            setMetricValidationRules(metricId: $metricId, rules: $rules) {
                __typename
                ... on MetricValidationRulesResult {
                    validationRules {
                        id
                        rule {
                            __typename
                            enforcement
                            ... on ValueRangeRule { min max }
                        }
                    }
                    violations {
                        code
                        message
                        severity
                        enforcement
                        metric
                        years
                        coordinates { dimension category }
                    }
                }
                ... on OperationInfo {
                    messages { kind message field code }
                }
            }
        }
    }
}
"""


CREATE_DATA_POINTS_WITH_VIOLATIONS = """
mutation CreateDataPoints($instanceId: ID!, $datasetId: ID!, $input: [CreateDataPointInput!]!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            createDataPoints(input: $input) {
                __typename
                ... on DataPointsMutationResult {
                    dataPoints { id }
                    violations { code enforcement years coordinates { dimension category } }
                }
                ... on OperationInfo {
                    messages { kind message field code }
                }
            }
        }
    }
}
"""


GET_DATASET_VALIDATION = """
query DatasetValidation($datasetId: ID!) {
    instance {
        editor {
            dataset(id: $datasetId) {
                categoryDomain {
                    mode
                    combinations { id identifier coordinates { dimensionId categoryId } }
                }
                validationViolations {
                    code
                    metric
                    years
                    requirementGroup
                    combinationIds
                    coordinates { dimension category }
                }
            }
        }
    }
}
"""


def test_set_metric_validation_rules_mutation(gql_client: PathsTestClient, db_instance_config, rig):
    dataset, metric, _cat_a, _cat_b = rig

    data = gql_client.query_data(
        SET_METRIC_VALIDATION_RULES,
        variables={
            'instanceId': str(db_instance_config.pk),
            'datasetId': str(dataset.uuid),
            'metricId': str(metric.uuid),
            'rules': [{'rule': {'valueRange': {'enforcement': 'BLOCK_PUBLISH', 'min': 0}}}],
        },
    )
    result = data['instanceEditor']['datasetEditor']['setMetricValidationRules']
    assert result['__typename'] == 'MetricValidationRulesResult'
    (rule_payload,) = result['validationRules']
    assert rule_payload['rule'] == {
        '__typename': 'ValueRangeRule',
        'enforcement': 'BLOCK_PUBLISH',
        'min': 0.0,
        'max': None,
    }
    assert result['violations'] == []
    rule_id = rule_payload['id']
    assert DatasetMetricValidationRule.objects.filter(metric=metric, uuid=rule_id).exists()

    # Re-setting with the uuid updates in place; omitting it would delete.
    data = gql_client.query_data(
        SET_METRIC_VALIDATION_RULES,
        variables={
            'instanceId': str(db_instance_config.pk),
            'datasetId': str(dataset.uuid),
            'metricId': str(metric.uuid),
            'rules': [{'uuid': rule_id, 'rule': {'valueRange': {'enforcement': 'BLOCK_EDIT', 'min': 0}}}],
        },
    )
    result = data['instanceEditor']['datasetEditor']['setMetricValidationRules']
    (rule_payload,) = result['validationRules']
    assert rule_payload['id'] == rule_id
    assert DatasetMetricValidationRule.objects.get(metric=metric).rule == {
        'kind': 'value_range',
        'enforcement': 'block_edit',
        'min': 0.0,
        'max': None,
        'exclusive_min': False,
        'exclusive_max': False,
    }


def test_dataset_query_exposes_category_domain_and_required_combination_violations(
    gql_client: PathsTestClient,
    rig,
):
    dataset, metric, cat_a, cat_b = rig
    dimension = cat_a.dimension
    combo_b = DatasetCategoryCombination(
        id=uuid4(),
        identifier='region_b',
        categories={dimension.uuid: cat_b.uuid},
    )
    dataset.schema.category_domain = DatasetCategoryDomain(combinations=[combo_b])
    dataset.schema.save(update_fields=['category_domain'])
    set_rule(
        metric,
        {
            'kind': 'required_combinations',
            'enforcement': 'block_publish',
            'groups': [{'id': 'region_b', 'combinations': [str(combo_b.id)]}],
        },
    )
    add_point(dataset, metric, 2021, 1, cat_a)
    materialize_dataset(dataset)

    data = gql_client.query_data(GET_DATASET_VALIDATION, variables={'datasetId': str(dataset.uuid)})
    payload = data['instance']['editor']['dataset']

    assert payload['categoryDomain'] == {
        'mode': 'open',
        'combinations': [
            {
                'id': str(combo_b.id),
                'identifier': 'region_b',
                'coordinates': [{'dimensionId': str(dimension.uuid), 'categoryId': str(cat_b.uuid)}],
            }
        ],
    }
    assert payload['validationViolations'] == [
        {
            'code': 'required_combinations',
            'metric': 'amount',
            'years': [2021],
            'requirementGroup': 'region_b',
            'combinationIds': [str(combo_b.id)],
            'coordinates': [{'dimension': 'region', 'category': 'b'}],
        }
    ]


def test_set_metric_validation_rules_rejects_invalid_rule(gql_client: PathsTestClient, db_instance_config, rig):
    dataset, metric, _cat_a, _cat_b = rig
    # A value_range rule without either bound parses at the GraphQL layer
    # but fails the model validator.
    data = gql_client.query_data(
        SET_METRIC_VALIDATION_RULES,
        variables={
            'instanceId': str(db_instance_config.pk),
            'datasetId': str(dataset.uuid),
            'metricId': str(metric.uuid),
            'rules': [{'rule': {'valueRange': {'enforcement': 'BLOCK_PUBLISH'}}}],
        },
    )
    result = data['instanceEditor']['datasetEditor']['setMetricValidationRules']
    assert result['__typename'] == 'OperationInfo'
    assert not DatasetMetricValidationRule.objects.filter(metric=metric).exists()


def test_set_metric_validation_rules_requires_exactly_one_variant(gql_client: PathsTestClient, db_instance_config, rig):
    dataset, metric, _cat_a, _cat_b = rig
    errors = gql_client.query_errors(
        SET_METRIC_VALIDATION_RULES,
        variables={
            'instanceId': str(db_instance_config.pk),
            'datasetId': str(dataset.uuid),
            'metricId': str(metric.uuid),
            'rules': [
                {
                    'rule': {
                        'valueRange': {'enforcement': 'BLOCK_PUBLISH', 'min': 0},
                        'noGaps': {'enforcement': 'BLOCK_PUBLISH'},
                    }
                }
            ],
        },
    )
    assert 'exactly one' in errors[0]['message'].lower()
    assert not DatasetMetricValidationRule.objects.filter(metric=metric).exists()


def test_create_data_points_returns_block_publish_violations(gql_client: PathsTestClient, db_instance_config, rig):
    dataset, metric, cat_a, _cat_b = rig
    set_rule(metric, {'kind': 'value_range', 'enforcement': 'block_publish', 'min': 0})

    data = gql_client.query_data(
        CREATE_DATA_POINTS_WITH_VIOLATIONS,
        variables={
            'instanceId': str(db_instance_config.pk),
            'datasetId': str(dataset.uuid),
            'input': [
                {
                    'date': '2020-01-01',
                    'value': -5.0,
                    'metricId': str(metric.uuid),
                    'dimensionCategoryIds': [str(cat_a.uuid)],
                }
            ],
        },
    )
    result = data['instanceEditor']['datasetEditor']['createDataPoints']
    assert result['__typename'] == 'DataPointsMutationResult'
    assert len(result['dataPoints']) == 1
    (violation,) = result['violations']
    assert violation['code'] == 'value_range'
    assert violation['enforcement'] == 'BLOCK_PUBLISH'
    assert violation['years'] == [2020]
    assert violation['coordinates'] == [{'dimension': 'region', 'category': 'a'}]


def test_create_data_points_blocked_by_block_edit_rule(gql_client: PathsTestClient, db_instance_config, rig):
    dataset, metric, cat_a, _cat_b = rig
    set_rule(metric, {'kind': 'value_range', 'enforcement': 'block_edit', 'min': 0})

    data = gql_client.query_data(
        CREATE_DATA_POINTS_WITH_VIOLATIONS,
        variables={
            'instanceId': str(db_instance_config.pk),
            'datasetId': str(dataset.uuid),
            'input': [
                {
                    'date': '2020-01-01',
                    'value': -5.0,
                    'metricId': str(metric.uuid),
                    'dimensionCategoryIds': [str(cat_a.uuid)],
                }
            ],
        },
    )
    result = data['instanceEditor']['datasetEditor']['createDataPoints']
    assert result['__typename'] == 'OperationInfo'
    assert not DataPoint.objects.filter(dataset=dataset).exists()
