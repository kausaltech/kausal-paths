"""Tests for dataset and dataset-metric GraphQL mutations in the model editor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from django.contrib.contenttypes.models import ContentType

import pytest

from kausal_common.datasets.models import (
    Dataset,
    DatasetMetric,
    DatasetSchema,
    DatasetSchemaScope,
    DimensionScope,
)
from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaFactory,
    DimensionFactory,
)

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

if TYPE_CHECKING:
    from paths.tests.graphql import PathsTestClient

    from nodes.models import InstanceConfig


gql = str

pytestmark = pytest.mark.django_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_instance_config() -> InstanceConfig:
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
def gql_client(client, db_instance_config: InstanceConfig) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    from users.tests.factories import UserFactory

    user = UserFactory.create(is_superuser=True)
    client.force_login(user)
    tc = PathsTestClient(client)
    tc.set_instance(db_instance_config)
    return tc


def _make_dataset(ic: InstanceConfig, name: str = 'Emissions', unit: str = 't/a') -> tuple[Dataset, DatasetMetric]:
    """Create a Dataset with a 1:1 schema and one metric, scoped to the instance."""
    schema = DatasetSchemaFactory.create(name=name)
    ct = ContentType.objects.get_for_model(ic)
    DatasetSchemaScope.objects.create(schema=schema, scope_content_type=ct, scope_id=ic.pk)
    metric = DatasetMetricFactory.create(schema=schema, name='emissions', label=name, unit=unit)
    dataset = DatasetFactory.create(schema=schema, scope=ic)
    return dataset, metric


# ---------------------------------------------------------------------------
# createDataset
# ---------------------------------------------------------------------------

CREATE_DATASET = gql("""
mutation CreateDataset($instanceId: ID!, $input: CreateDatasetInput!) {
    instanceEditor(instanceId: $instanceId) {
        createDataset(input: $input) {
            ... on Dataset {
                id
                identifier
                name
                metrics {
                    id
                    label
                    unit
                    quantity { id identifier label }
                }
                dimensions { id }
            }
        }
    }
}
""")


def test_create_dataset(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    ic = db_instance_config
    dim = DimensionFactory.create(name='Sector')
    ct = ContentType.objects.get_for_model(ic)
    DimensionScope.objects.create(dimension=dim, scope_content_type=ct, scope_id=ic.pk, identifier='sector')

    data = gql_client.query_data(
        CREATE_DATASET,
        variables={
            'instanceId': str(ic.pk),
            'input': {
                'name': 'Building emissions',
                'identifier': 'building_emissions',
                'metrics': [
                    {'label': 'Emissions', 'unit': 't/a', 'quantity': 'emissions'},
                ],
                'dimensions': [str(dim.uuid)],
            },
        },
    )
    result = data['instanceEditor']['createDataset']
    assert result['identifier'] == 'building_emissions'
    assert result['name'] == 'Building emissions'
    assert len(result['metrics']) == 1
    metric = result['metrics'][0]
    assert metric['unit'] == 't/a'
    assert metric['quantity']['id'] == 'emissions'
    assert metric['quantity']['identifier'] == 'emissions'
    assert [d['id'] for d in result['dimensions']] == [str(dim.uuid)]

    dataset = Dataset.objects.get(uuid=result['id'])
    assert dataset.scope_id == ic.pk
    assert dataset.schema is not None
    row = dataset.schema.metrics.get()
    assert row.spec == {'quantity': 'emissions'}
    schema_dims = list(dataset.schema.dimensions.all())
    assert [sd.dimension.pk for sd in schema_dims] == [dim.pk]


def test_create_dataset_unknown_quantity_fails(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    gql_client.query_errors(
        CREATE_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'name': 'Bad',
                'metrics': [{'label': 'X', 'unit': 't/a', 'quantity': 'no_such_quantity'}],
            },
        },
    )
    assert not Dataset.objects.filter(scope_id=db_instance_config.pk).exists()


def test_create_dataset_requires_metric(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    gql_client.query_errors(
        CREATE_DATASET,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {'name': 'Empty', 'metrics': []},
        },
    )


def test_create_dataset_duplicate_identifier_fails(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    ic = db_instance_config
    dataset, _ = _make_dataset(ic)
    dataset.identifier = 'taken'
    dataset.save(update_fields=['identifier'])
    gql_client.query_errors(
        CREATE_DATASET,
        variables={
            'instanceId': str(ic.pk),
            'input': {
                'name': 'Clash',
                'identifier': 'taken',
                'metrics': [{'label': 'X', 'unit': 't/a'}],
            },
        },
    )


# ---------------------------------------------------------------------------
# updateDataset
# ---------------------------------------------------------------------------

UPDATE_DATASET = gql("""
mutation UpdateDataset($instanceId: ID!, $input: UpdateDatasetInput!) {
    instanceEditor(instanceId: $instanceId) {
        updateDataset(input: $input) {
            ... on Dataset {
                id
                identifier
                name
            }
        }
    }
}
""")


def test_update_dataset(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    ic = db_instance_config
    dataset, _ = _make_dataset(ic, name='Old name')
    data = gql_client.query_data(
        UPDATE_DATASET,
        variables={
            'instanceId': str(ic.pk),
            'input': {
                'datasetId': str(dataset.uuid),
                'name': 'New name',
                'identifier': 'new_identifier',
            },
        },
    )
    result = data['instanceEditor']['updateDataset']
    assert result['name'] == 'New name'
    assert result['identifier'] == 'new_identifier'
    dataset.refresh_from_db()
    assert dataset.identifier == 'new_identifier'
    assert dataset.schema is not None
    assert dataset.schema.name == 'New name'


# ---------------------------------------------------------------------------
# deleteDataset
# ---------------------------------------------------------------------------

DELETE_DATASET = gql("""
mutation DeleteDataset($instanceId: ID!, $datasetId: UUID!, $force: Boolean!) {
    instanceEditor(instanceId: $instanceId) {
        deleteDataset(datasetId: $datasetId, force: $force) {
            ... on ModelDeletePayload {
                ok
            }
        }
    }
}
""")


def test_delete_dataset(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    ic = db_instance_config
    dataset, _ = _make_dataset(ic)
    assert dataset.schema is not None
    schema_pk = dataset.schema.pk
    data = gql_client.query_data(
        DELETE_DATASET,
        variables={'instanceId': str(ic.pk), 'datasetId': str(dataset.uuid), 'force': False},
    )
    assert data['instanceEditor']['deleteDataset']['ok'] is True
    assert not Dataset.objects.filter(pk=dataset.pk).exists()
    assert not DatasetSchema.objects.filter(pk=schema_pk).exists()


def test_delete_dataset_with_data_requires_force(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    ic = db_instance_config
    dataset, metric = _make_dataset(ic)
    DataPointFactory.create(dataset=dataset, metric=metric)

    errors = gql_client.query_errors(
        DELETE_DATASET,
        variables={'instanceId': str(ic.pk), 'datasetId': str(dataset.uuid), 'force': False},
    )
    assert errors[0].get('extensions', {}).get('code') == 'dataset_has_data_points'
    assert Dataset.objects.filter(pk=dataset.pk).exists()

    data = gql_client.query_data(
        DELETE_DATASET,
        variables={'instanceId': str(ic.pk), 'datasetId': str(dataset.uuid), 'force': True},
    )
    assert data['instanceEditor']['deleteDataset']['ok'] is True
    assert not Dataset.objects.filter(pk=dataset.pk).exists()


# ---------------------------------------------------------------------------
# Metric mutations (via datasetEditor)
# ---------------------------------------------------------------------------

CREATE_METRIC = gql("""
mutation CreateMetric($instanceId: ID!, $datasetId: ID!, $input: CreateDatasetMetricInput!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            createMetric(input: $input) {
                ... on DatasetMetric {
                    id
                    label
                    unit
                    quantity { id }
                }
                ... on OperationInfo {
                    messages { message code }
                }
            }
        }
    }
}
""")


def test_create_metric(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    ic = db_instance_config
    dataset, _ = _make_dataset(ic)
    data = gql_client.query_data(
        CREATE_METRIC,
        variables={
            'instanceId': str(ic.pk),
            'datasetId': str(dataset.uuid),
            'input': {'label': 'Energy', 'unit': 'GWh/a', 'quantity': 'energy'},
        },
    )
    result = data['instanceEditor']['datasetEditor']['createMetric']
    assert result['label'] == 'Energy'
    assert result['quantity']['id'] == 'energy'
    metric = DatasetMetric.objects.get(uuid=result['id'])
    assert dataset.schema is not None
    assert metric.schema.pk == dataset.schema.pk
    assert metric.spec == {'quantity': 'energy'}


UPDATE_METRIC = gql("""
mutation UpdateMetric($instanceId: ID!, $datasetId: ID!, $metricId: UUID!, $input: UpdateDatasetMetricInput!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            updateMetric(metricId: $metricId, input: $input) {
                ... on DatasetMetric {
                    id
                    label
                    unit
                    quantity { id }
                }
                ... on OperationInfo {
                    messages { message code }
                }
            }
        }
    }
}
""")


def test_update_metric_quantity(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    ic = db_instance_config
    dataset, metric = _make_dataset(ic)

    data = gql_client.query_data(
        UPDATE_METRIC,
        variables={
            'instanceId': str(ic.pk),
            'datasetId': str(dataset.uuid),
            'metricId': str(metric.uuid),
            'input': {'quantity': 'emissions', 'label': 'CO2e'},
        },
    )
    result = data['instanceEditor']['datasetEditor']['updateMetric']
    assert result['quantity']['id'] == 'emissions'
    assert result['label'] == 'CO2e'
    metric.refresh_from_db()
    assert metric.spec == {'quantity': 'emissions'}
    assert metric.label == 'CO2e'

    # Clearing the quantity with an explicit null.
    data = gql_client.query_data(
        UPDATE_METRIC,
        variables={
            'instanceId': str(ic.pk),
            'datasetId': str(dataset.uuid),
            'metricId': str(metric.uuid),
            'input': {'quantity': None},
        },
    )
    result = data['instanceEditor']['datasetEditor']['updateMetric']
    assert result['quantity'] is None
    metric.refresh_from_db()
    assert metric.spec == {}


def test_update_metric_invalid_quantity(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    ic = db_instance_config
    dataset, metric = _make_dataset(ic)
    result = gql_client.query_data(
        UPDATE_METRIC,
        variables={
            'instanceId': str(ic.pk),
            'datasetId': str(dataset.uuid),
            'metricId': str(metric.uuid),
            'input': {'quantity': 'bogus'},
        },
    )['instanceEditor']['datasetEditor']['updateMetric']
    # An unknown quantity surfaces as an OperationInfo payload, not a data write.
    assert 'messages' in result
    metric.refresh_from_db()
    assert metric.spec == {}


DELETE_METRIC = gql("""
mutation DeleteMetric($instanceId: ID!, $datasetId: ID!, $metricId: UUID!, $force: Boolean!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            deleteMetric(metricId: $metricId, force: $force) {
                ... on OperationInfo {
                    messages { message code }
                }
            }
        }
    }
}
""")


def test_delete_metric(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    ic = db_instance_config
    dataset, metric = _make_dataset(ic)
    assert dataset.schema is not None
    other = DatasetMetricFactory.create(schema=dataset.schema, name='other', label='Other', unit='GWh/a')

    data = gql_client.query_data(
        DELETE_METRIC,
        variables={
            'instanceId': str(ic.pk),
            'datasetId': str(dataset.uuid),
            'metricId': str(metric.uuid),
            'force': False,
        },
    )
    assert data['instanceEditor']['datasetEditor']['deleteMetric'] is None
    assert not DatasetMetric.objects.filter(pk=metric.pk).exists()
    assert DatasetMetric.objects.filter(pk=other.pk).exists()


def test_delete_last_metric_fails(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    ic = db_instance_config
    dataset, metric = _make_dataset(ic)
    result = gql_client.query_data(
        DELETE_METRIC,
        variables={
            'instanceId': str(ic.pk),
            'datasetId': str(dataset.uuid),
            'metricId': str(metric.uuid),
            'force': False,
        },
    )['instanceEditor']['datasetEditor']['deleteMetric']
    assert result is not None
    assert result['messages'][0]['code'] == 'last_metric'
    assert DatasetMetric.objects.filter(pk=metric.pk).exists()


def test_delete_metric_with_data_requires_force(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    ic = db_instance_config
    dataset, metric = _make_dataset(ic)
    assert dataset.schema is not None
    DatasetMetricFactory.create(schema=dataset.schema, name='other', label='Other', unit='GWh/a')
    DataPointFactory.create(dataset=dataset, metric=metric)

    result = gql_client.query_data(
        DELETE_METRIC,
        variables={
            'instanceId': str(ic.pk),
            'datasetId': str(dataset.uuid),
            'metricId': str(metric.uuid),
            'force': False,
        },
    )['instanceEditor']['datasetEditor']['deleteMetric']
    assert result is not None
    assert result['messages'][0]['code'] == 'metric_has_data_points'
    assert DatasetMetric.objects.filter(pk=metric.pk).exists()

    result = gql_client.query_data(
        DELETE_METRIC,
        variables={
            'instanceId': str(ic.pk),
            'datasetId': str(dataset.uuid),
            'metricId': str(metric.uuid),
            'force': True,
        },
    )['instanceEditor']['datasetEditor']['deleteMetric']
    assert result is None
    assert not DatasetMetric.objects.filter(pk=metric.pk).exists()
    assert not dataset.data_points.exists()


def test_metric_quantity_in_snapshot_and_meta(db_instance_config: InstanceConfig):
    """`quantity` survives into the revision snapshot and the graph catalog."""
    from nodes.instance_serialization import DatasetSnapshot, dataset_meta_from_model

    ic = db_instance_config
    dataset, metric = _make_dataset(ic)
    metric.spec = {'quantity': 'emissions'}
    metric.save(update_fields=['spec'])

    snap = DatasetSnapshot.from_model_for_instance(dataset, ic)
    assert snap.metrics[0].quantity == 'emissions'

    meta = dataset_meta_from_model(dataset, primary_language=ic.primary_language)
    assert meta.metrics[0].quantity == 'emissions'


def test_metric_mutation_refused_on_shared_schema(gql_client: PathsTestClient, db_instance_config: InstanceConfig):
    ic = db_instance_config
    dataset, _ = _make_dataset(ic)
    # A second dataset sharing the same schema (pre-Trailhead shape).
    other_instance = InstanceFactory.create(id=f'other-{uuid4().hex[:8]}')
    other_ic = InstanceConfigFactory.create(identifier=other_instance.id, instance=other_instance)
    DatasetFactory.create(schema=dataset.schema, scope=other_ic)

    result = gql_client.query_data(
        CREATE_METRIC,
        variables={
            'instanceId': str(ic.pk),
            'datasetId': str(dataset.uuid),
            'input': {'label': 'Nope', 'unit': 't/a'},
        },
    )['instanceEditor']['datasetEditor']['createMetric']
    assert result['messages'][0]['code'] == 'schema_shared'
