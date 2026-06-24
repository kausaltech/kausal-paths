from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
    DatasetSourceReference,
    DataSource,
)
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

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from users.tests.factories import UserFactory

pytestmark = pytest.mark.django_db


CREATE_DATA_POINT = """
mutation CreateDataPoint($instanceId: ID!, $datasetId: ID!, $input: CreateDataPointInput!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            createDataPoint(input: $input) {
                __typename
                ... on DataPoint {
                    id
                    date
                    value
                    metric { id name }
                    dimensionCategories { uuid label }
                }
                ... on OperationInfo {
                    messages { kind message field code }
                }
            }
        }
    }
}
"""


UPDATE_DATA_POINT = """
mutation UpdateDataPoint($instanceId: ID!, $datasetId: ID!, $dataPointId: ID!, $input: UpdateDataPointInput!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            updateDataPoint(dataPointId: $dataPointId, input: $input) {
                __typename
                ... on DataPoint {
                    id
                    date
                    value
                    metric { id name }
                    dimensionCategories { uuid label }
                }
                ... on OperationInfo {
                    messages { kind message field code }
                }
            }
        }
    }
}
"""


DELETE_DATA_POINT = """
mutation DeleteDataPoint($instanceId: ID!, $datasetId: ID!, $dataPointId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            deleteDataPoint(dataPointId: $dataPointId) {
                messages { kind message field code }
            }
        }
    }
}
"""


INSTANCE_DATASETS = """
query InstanceDatasets($instanceId: ID!) {
    modelInstance(instanceId: $instanceId) {
        editor {
            datasets {
                id
                forecastFrom
                metrics {
                    id
                    name
                    previousSibling
                    nextSibling
                }
                dataPoints {
                    id
                    date
                    value
                    metric { id name }
                    dimensionCategories { uuid label }
                }
            }
        }
    }
}
"""


DATASET_DATA = """
query DatasetData($instanceId: ID!) {
    modelInstance(instanceId: $instanceId) {
        editor {
            datasets {
                id
                data {
                    forecastFrom
                }
            }
        }
    }
}
"""


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
def dataset_setup(db_instance_config):
    schema = DatasetSchemaFactory.create(name='Emissions schema')
    dimension = DimensionFactory.create(name='Sector')
    category = DimensionCategoryFactory.create(dimension=dimension, label='Energy')
    DatasetSchemaDimensionFactory.create(schema=schema, dimension=dimension)
    metric = DatasetMetricFactory.create(schema=schema, name='emissions', label='Emissions', unit='t/a')
    dataset = DatasetFactory.create(schema=schema, scope=db_instance_config)
    return db_instance_config, dataset, metric, category


@pytest.fixture
def gql_client(client, db_instance_config) -> PathsTestClient:
    user = UserFactory.create(is_superuser=True)
    client.force_login(user)
    tc = PathsTestClient(client)
    tc.set_instance(db_instance_config)
    return tc


def test_create_data_point(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup

    data = gql_client.query_data(
        CREATE_DATA_POINT,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'input': {
                'date': '2024-01-01',
                'value': 150.0,
                'metricId': str(metric.uuid),
                'dimensionCategoryIds': [str(category.uuid)],
            },
        },
    )

    data_point = data['instanceEditor']['datasetEditor']['createDataPoint']
    assert data_point['__typename'] == 'DataPoint'
    assert data_point['date'] == '2024-01-01'
    assert data_point['value'] == 150.0
    assert data_point['metric']['id'] == str(metric.uuid)
    assert [cat['uuid'] for cat in data_point['dimensionCategories']] == [str(category.uuid)]
    assert DataPoint.objects.filter(uuid=data_point['id'], dataset=dataset).exists()


def test_create_data_point_rejects_duplicate_with_no_dimension_categories(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, _category = dataset_setup
    DataPointFactory.create(
        dataset=dataset,
        metric=metric,
        date=date(2024, 1, 1),
        value=Decimal('100.0'),
    )

    data = gql_client.query_data(
        CREATE_DATA_POINT,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'input': {
                'date': '2024-01-01',
                'value': 150.0,
                'metricId': str(metric.uuid),
                'dimensionCategoryIds': [],
            },
        },
    )

    result = data['instanceEditor']['datasetEditor']['createDataPoint']
    assert result['__typename'] == 'OperationInfo'
    assert result['messages'][0]['kind'] == 'VALIDATION'


def test_update_data_point(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    data_point = DataPointFactory.create(
        dataset=dataset,
        metric=metric,
        date=date(2024, 1, 1),
        value=Decimal('100.0'),
        dimension_categories=[category],
    )

    data = gql_client.query_data(
        UPDATE_DATA_POINT,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'dataPointId': str(data_point.uuid),
            'input': {'date': '2025-01-01', 'value': None},
        },
    )

    updated = data['instanceEditor']['datasetEditor']['updateDataPoint']
    assert updated['__typename'] == 'DataPoint'
    assert updated['id'] == str(data_point.uuid)
    assert updated['date'] == '2025-01-01'
    assert updated['value'] is None
    data_point.refresh_from_db()
    assert data_point.date == date(2025, 1, 1)
    assert data_point.value is None


def test_update_data_point_rejects_duplicate_coordinates(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    existing = DataPointFactory.create(
        dataset=dataset,
        metric=metric,
        date=date(2024, 1, 1),
        value=Decimal('100.0'),
        dimension_categories=[category],
    )
    data_point = DataPointFactory.create(
        dataset=dataset,
        metric=metric,
        date=date(2025, 1, 1),
        value=Decimal('200.0'),
    )

    data = gql_client.query_data(
        UPDATE_DATA_POINT,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'dataPointId': str(data_point.uuid),
            'input': {'date': '2024-01-01', 'dimensionCategoryIds': [str(category.uuid)]},
        },
    )

    result = data['instanceEditor']['datasetEditor']['updateDataPoint']
    assert result['__typename'] == 'OperationInfo'
    assert result['messages'][0]['kind'] == 'VALIDATION'
    data_point.refresh_from_db()
    assert data_point.date == date(2025, 1, 1)
    assert list(data_point.dimension_categories.all()) == []
    assert DataPoint.objects.filter(pk=existing.pk).exists()


def test_delete_data_point(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    data_point = DataPointFactory.create(dataset=dataset, metric=metric, dimension_categories=[category])

    data = gql_client.query_data(
        DELETE_DATA_POINT,
        variables={'instanceId': str(instance_config.pk), 'datasetId': str(dataset.uuid), 'dataPointId': str(data_point.uuid)},
    )

    assert data['instanceEditor']['datasetEditor']['deleteDataPoint'] is None
    assert not DataPoint.objects.filter(pk=data_point.pk).exists()


def test_create_data_point_validation_error_returns_operation_info(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, _metric, category = dataset_setup

    data = gql_client.query_data(
        CREATE_DATA_POINT,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'input': {
                'date': '2024-01-01',
                'value': 150.0,
                'metricId': str(uuid4()),
                'dimensionCategoryIds': [str(category.uuid)],
            },
        },
    )

    result = data['instanceEditor']['datasetEditor']['createDataPoint']
    assert result['__typename'] == 'OperationInfo'
    assert result['messages'][0]['kind'] == 'VALIDATION'
    assert result['messages'][0]['field'] == 'metric'


def test_dataset_metrics_include_sibling_ids(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, first_metric, _category = dataset_setup
    second_metric = DatasetMetricFactory.create(schema=dataset.schema, name='energy', label='Energy', unit='MWh')
    third_metric = DatasetMetricFactory.create(schema=dataset.schema, name='cost', label='Cost', unit='EUR')

    data = gql_client.query_data(
        INSTANCE_DATASETS,
        variables={'instanceId': str(instance_config.pk)},
    )

    datasets = data['modelInstance']['editor']['datasets']
    result = next(item for item in datasets if item['id'] == str(dataset.uuid))
    metrics = result['metrics']

    assert [metric['id'] for metric in metrics] == [str(first_metric.uuid), str(second_metric.uuid), str(third_metric.uuid)]
    assert metrics[0]['previousSibling'] is None
    assert metrics[0]['nextSibling'] == str(second_metric.uuid)
    assert metrics[1]['previousSibling'] == str(first_metric.uuid)
    assert metrics[1]['nextSibling'] == str(third_metric.uuid)
    assert metrics[2]['previousSibling'] == str(second_metric.uuid)
    assert metrics[2]['nextSibling'] is None


def test_dataset_forecast_from(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, _metric, _category = dataset_setup
    dataset.spec = {'forecast_from': 2025}
    dataset.save(update_fields=['spec'])

    data = gql_client.query_data(
        INSTANCE_DATASETS,
        variables={'instanceId': str(instance_config.pk)},
    )

    datasets = data['modelInstance']['editor']['datasets']
    result = next(item for item in datasets if item['id'] == str(dataset.uuid))
    assert result['forecastFrom'] == 2025


def test_dataset_data_uses_dataset_forecast_from(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    dataset.spec = {'forecast_from': 2025}
    dataset.save(update_fields=['spec'])
    DataPointFactory.create(
        dataset=dataset,
        metric=metric,
        date=date(2024, 1, 1),
        value=Decimal('42.5'),
        dimension_categories=[category],
    )

    data = gql_client.query_data(
        DATASET_DATA,
        variables={'instanceId': str(instance_config.pk)},
    )

    datasets = data['modelInstance']['editor']['datasets']
    result = next(item for item in datasets if item['id'] == str(dataset.uuid))
    assert result['data'][0]['forecastFrom'] == 2025


def test_dataset_data_points_include_dimension_categories(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    data_point = DataPointFactory.create(
        dataset=dataset,
        metric=metric,
        date=date(2024, 1, 1),
        value=Decimal('42.5'),
        dimension_categories=[category],
    )

    data = gql_client.query_data(
        INSTANCE_DATASETS,
        variables={'instanceId': str(instance_config.pk)},
    )

    datasets = data['modelInstance']['editor']['datasets']
    result = next(item for item in datasets if item['id'] == str(dataset.uuid))
    points = result['dataPoints']

    assert len(points) == 1
    assert points[0]['id'] == str(data_point.uuid)
    assert points[0]['date'] == '2024-01-01'
    assert points[0]['value'] == 42.5
    assert points[0]['metric']['id'] == str(metric.uuid)
    assert points[0]['dimensionCategories'] == [{'uuid': str(category.uuid), 'label': 'Energy'}]


def test_dataset_editor_rejects_dataset_outside_instance(gql_client: PathsTestClient, db_instance_config):
    other_instance = InstanceConfigFactory.create(
        identifier='other-instance',
        instance=InstanceFactory.create(),
        config_source='database',
        spec=db_instance_config.spec,
    )
    dataset = DatasetFactory.create(scope=other_instance)

    gql_client.query_errors(
        DELETE_DATA_POINT,
        variables={'instanceId': str(db_instance_config.pk), 'datasetId': str(dataset.uuid), 'dataPointId': str(dataset.uuid)},
        assert_error_message='not found',
    )


def test_create_data_point_emits_change_operation(gql_client: PathsTestClient, dataset_setup):
    """Creating a datapoint opens an InstanceChangeOperation for the instance."""
    from nodes.models import InstanceChangeOperation, InstanceModelLogEntry

    instance_config, dataset, metric, category = dataset_setup

    gql_client.query_data(
        CREATE_DATA_POINT,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'input': {
                'date': '2024-01-01',
                'value': 150.0,
                'metricId': str(metric.uuid),
                'dimensionCategoryIds': [str(category.uuid)],
            },
        },
    )

    op = InstanceChangeOperation.objects.filter(instance_config=instance_config, action='dataset.datapoint.create').first()
    assert op is not None
    entry = InstanceModelLogEntry.objects.filter(operation=op).first()
    assert entry is not None
    assert entry.action == 'dataset.datapoint.create'
    assert entry.data['before'] is None
    assert entry.data['after']['value'] == 150.0
    assert entry.data['after']['dataset_uuid'] == str(dataset.uuid)
    assert entry.data['after']['metric_uuid'] == str(metric.uuid)
    assert entry.data['after']['dimension_category_uuids'] == [str(category.uuid)]


# ----------------------------------------------------------------------
# DataPointComment
# ----------------------------------------------------------------------


CREATE_DATA_POINT_COMMENT = """
mutation CreateComment($instanceId: ID!, $datasetId: ID!, $dataPointId: ID!, $input: CreateDataPointCommentInput!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            createDataPointComment(dataPointId: $dataPointId, input: $input) {
                __typename
                ... on DataPointComment {
                    id
                    text
                    isSticky
                    isReview
                    reviewState
                    createdBy { email }
                    lastModifiedBy { email }
                }
                ... on OperationInfo {
                    messages { kind message field code }
                }
            }
        }
    }
}
"""


UPDATE_DATA_POINT_COMMENT = """
mutation UpdateComment($instanceId: ID!, $datasetId: ID!, $commentId: ID!, $input: UpdateDataPointCommentInput!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            updateDataPointComment(commentId: $commentId, input: $input) {
                __typename
                ... on DataPointComment {
                    id
                    text
                    isSticky
                    reviewState
                }
            }
        }
    }
}
"""


DELETE_DATA_POINT_COMMENT = """
mutation DeleteComment($instanceId: ID!, $datasetId: ID!, $commentId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            deleteDataPointComment(commentId: $commentId) {
                messages { kind message field code }
            }
        }
    }
}
"""


RESOLVE_DATA_POINT_COMMENT = """
mutation ResolveComment($instanceId: ID!, $datasetId: ID!, $commentId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            resolveDataPointComment(commentId: $commentId) {
                __typename
                ... on DataPointComment {
                    id
                    reviewState
                    resolvedAt
                    resolvedBy { email }
                }
            }
        }
    }
}
"""


UNRESOLVE_DATA_POINT_COMMENT = """
mutation UnresolveComment($instanceId: ID!, $datasetId: ID!, $commentId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            unresolveDataPointComment(commentId: $commentId) {
                __typename
                ... on DataPointComment {
                    id
                    reviewState
                    resolvedAt
                    resolvedBy { email }
                }
            }
        }
    }
}
"""


DATA_POINT_COMMENTS_QUERY = """
query DatasetWithComments($instanceId: ID!) {
    modelInstance(instanceId: $instanceId) {
        editor {
            datasets {
                id
                dataPointComments {
                    id
                    text
                    reviewState
                }
                dataPoints {
                    id
                    comments {
                        id
                        text
                    }
                }
            }
        }
    }
}
"""


def test_create_data_point_comment(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    data_point = DataPointFactory.create(dataset=dataset, metric=metric, dimension_categories=[category])

    data = gql_client.query_data(
        CREATE_DATA_POINT_COMMENT,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'dataPointId': str(data_point.uuid),
            'input': {'text': 'Looks suspicious', 'isReview': True, 'reviewState': 'UNRESOLVED'},
        },
    )
    result = data['instanceEditor']['datasetEditor']['createDataPointComment']
    assert result['__typename'] == 'DataPointComment'
    assert result['text'] == 'Looks suspicious'
    assert result['isReview'] is True
    assert result['reviewState'] == 'UNRESOLVED'
    assert result['createdBy']['email']
    assert DataPointComment.objects.filter(uuid=result['id']).exists()


def test_update_data_point_comment(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    data_point = DataPointFactory.create(dataset=dataset, metric=metric, dimension_categories=[category])
    comment = DataPointComment.objects.create(data_point=data_point, text='Original')

    data = gql_client.query_data(
        UPDATE_DATA_POINT_COMMENT,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'commentId': str(comment.uuid),
            'input': {'text': 'Edited', 'isSticky': True},
        },
    )
    result = data['instanceEditor']['datasetEditor']['updateDataPointComment']
    assert result['text'] == 'Edited'
    assert result['isSticky'] is True
    comment.refresh_from_db()
    assert comment.text == 'Edited'
    assert comment.is_sticky is True


def test_delete_data_point_comment_soft_deletes(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    data_point = DataPointFactory.create(dataset=dataset, metric=metric, dimension_categories=[category])
    comment = DataPointComment.objects.create(data_point=data_point, text='Bye')

    data = gql_client.query_data(
        DELETE_DATA_POINT_COMMENT,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'commentId': str(comment.uuid),
        },
    )
    assert data['instanceEditor']['datasetEditor']['deleteDataPointComment'] is None
    comment.refresh_from_db()
    assert comment.is_soft_deleted is True
    assert not DataPointComment.objects.filter(pk=comment.pk).exists()
    assert DataPointComment.objects_including_soft_deleted.filter(pk=comment.pk).exists()


def test_resolve_then_unresolve_data_point_comment(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    data_point = DataPointFactory.create(dataset=dataset, metric=metric, dimension_categories=[category])
    comment = DataPointComment.objects.create(
        data_point=data_point,
        text='Review please',
        is_review=True,
        review_state=DataPointComment.ReviewState.UNRESOLVED,
    )

    resolved = gql_client.query_data(
        RESOLVE_DATA_POINT_COMMENT,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'commentId': str(comment.uuid),
        },
    )['instanceEditor']['datasetEditor']['resolveDataPointComment']
    assert resolved['reviewState'] == 'RESOLVED'
    assert resolved['resolvedAt'] is not None
    assert resolved['resolvedBy']['email']

    unresolved = gql_client.query_data(
        UNRESOLVE_DATA_POINT_COMMENT,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'commentId': str(comment.uuid),
        },
    )['instanceEditor']['datasetEditor']['unresolveDataPointComment']
    assert unresolved['reviewState'] == 'UNRESOLVED'
    assert unresolved['resolvedAt'] is None
    assert unresolved['resolvedBy'] is None


def test_data_point_comments_query(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    data_point = DataPointFactory.create(dataset=dataset, metric=metric, dimension_categories=[category])
    comment = DataPointComment.objects.create(data_point=data_point, text='Hello')
    DataPointComment.objects.create(data_point=data_point, text='Deleted one', is_soft_deleted=True)

    data = gql_client.query_data(
        DATA_POINT_COMMENTS_QUERY,
        variables={'instanceId': str(instance_config.pk)},
    )
    ds = next(d for d in data['modelInstance']['editor']['datasets'] if d['id'] == str(dataset.uuid))
    assert [c['id'] for c in ds['dataPointComments']] == [str(comment.uuid)]
    dp = next(d for d in ds['dataPoints'] if d['id'] == str(data_point.uuid))
    assert [c['id'] for c in dp['comments']] == [str(comment.uuid)]


def test_comment_mutation_emits_change_operation(gql_client: PathsTestClient, dataset_setup):
    from nodes.models import InstanceChangeOperation, InstanceModelLogEntry

    instance_config, dataset, metric, category = dataset_setup
    data_point = DataPointFactory.create(dataset=dataset, metric=metric, dimension_categories=[category])

    gql_client.query_data(
        CREATE_DATA_POINT_COMMENT,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'dataPointId': str(data_point.uuid),
            'input': {'text': 'Tracked'},
        },
    )
    op = InstanceChangeOperation.objects.filter(
        instance_config=instance_config,
        action='dataset.datapoint.comment.create',
    ).first()
    assert op is not None
    entry = InstanceModelLogEntry.objects.filter(operation=op).first()
    assert entry is not None
    assert entry.data['before'] is None
    assert entry.data['after']['text'] == 'Tracked'


# ----------------------------------------------------------------------
# DataSource + DatasetSourceReference
# ----------------------------------------------------------------------


CREATE_DATA_SOURCE = """
mutation CreateDataSource($instanceId: ID!, $input: CreateDataSourceInput!) {
    instanceEditor(instanceId: $instanceId) {
        createDataSource(input: $input) {
            __typename
            ... on DataSource {
                id
                name
                authority
                edition
                description
                url
                label
            }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""


UPDATE_DATA_SOURCE = """
mutation UpdateDataSource($instanceId: ID!, $dataSourceId: ID!, $input: UpdateDataSourceInput!) {
    instanceEditor(instanceId: $instanceId) {
        updateDataSource(dataSourceId: $dataSourceId, input: $input) {
            __typename
            ... on DataSource { id name edition authority }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""


DELETE_DATA_SOURCE = """
mutation DeleteDataSource($instanceId: ID!, $dataSourceId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        deleteDataSource(dataSourceId: $dataSourceId) {
            __typename
            ... on ModelDeletePayload { ok }
            ... on OperationInfo { messages { kind message } }
        }
    }
}
"""


CREATE_SOURCE_REFERENCE = """
mutation CreateRef($instanceId: ID!, $datasetId: ID!, $input: CreateDatasetSourceReferenceInput!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            createSourceReference(input: $input) {
                __typename
                ... on DatasetSourceReference {
                    id
                    dataSource { id name }
                    dataPoint { id }
                    dataset { id }
                }
                ... on OperationInfo { messages { kind message } }
            }
        }
    }
}
"""


DELETE_SOURCE_REFERENCE = """
mutation DeleteRef($instanceId: ID!, $datasetId: ID!, $referenceId: ID!) {
    instanceEditor(instanceId: $instanceId) {
        datasetEditor(datasetId: $datasetId) {
            deleteSourceReference(referenceId: $referenceId) {
                messages { kind message }
            }
        }
    }
}
"""


INSTANCE_DATA_SOURCES = """
query InstanceDataSources($instanceId: ID!) {
    modelInstance(instanceId: $instanceId) {
        editor {
            dataSources { id name }
        }
    }
}
"""


DATASET_SOURCE_REFERENCES = """
query DatasetSources($instanceId: ID!, $target: DatasetSourceReferenceTarget!) {
    modelInstance(instanceId: $instanceId) {
        editor {
            datasets {
                id
                sourceReferences(target: $target) {
                    id
                    dataPoint { id }
                    dataset { id }
                    dataSource { id }
                }
                dataSources { id name }
                dataPoints { id sourceReferences { id dataSource { id } } }
            }
        }
    }
}
"""


def _make_data_source(instance_config, **kwargs) -> DataSource:
    from django.contrib.contenttypes.models import ContentType

    ct = ContentType.objects.get_for_model(type(instance_config))
    defaults = {
        'name': 'Test Source',
        'authority': 'Test Authority',
    }
    defaults.update(kwargs)
    return DataSource.objects.create(scope_content_type=ct, scope_id=instance_config.pk, **defaults)


def test_create_data_source(gql_client: PathsTestClient, db_instance_config):
    data = gql_client.query_data(
        CREATE_DATA_SOURCE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {
                'name': 'IPCC AR6',
                'authority': 'IPCC',
                'edition': '2021',
                'description': 'Sixth Assessment Report',
                'url': 'https://example.org/ar6',
            },
        },
    )
    result = data['instanceEditor']['createDataSource']
    assert result['__typename'] == 'DataSource'
    assert result['name'] == 'IPCC AR6'
    assert result['authority'] == 'IPCC'
    assert result['edition'] == '2021'
    assert result['label']
    assert DataSource.objects.filter(uuid=result['id']).exists()


def test_update_data_source(gql_client: PathsTestClient, db_instance_config):
    ds = _make_data_source(db_instance_config, name='Old', authority='Old Auth')

    data = gql_client.query_data(
        UPDATE_DATA_SOURCE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'dataSourceId': str(ds.uuid),
            'input': {'name': 'New', 'edition': 'rev2'},
        },
    )
    result = data['instanceEditor']['updateDataSource']
    assert result['__typename'] == 'DataSource'
    assert result['name'] == 'New'
    assert result['edition'] == 'rev2'
    assert result['authority'] == 'Old Auth'  # untouched
    ds.refresh_from_db()
    assert ds.name == 'New'
    assert ds.edition == 'rev2'


def test_delete_unreferenced_data_source(gql_client: PathsTestClient, db_instance_config):
    ds = _make_data_source(db_instance_config)

    data = gql_client.query_data(
        DELETE_DATA_SOURCE,
        variables={'instanceId': str(db_instance_config.pk), 'dataSourceId': str(ds.uuid)},
    )
    payload = data['instanceEditor']['deleteDataSource']
    assert payload['__typename'] == 'ModelDeletePayload'
    assert payload['ok'] is True
    assert not DataSource.objects.filter(pk=ds.pk).exists()


def test_delete_referenced_data_source_fails(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, _metric, _category = dataset_setup
    ds = _make_data_source(instance_config)
    DatasetSourceReference.objects.create(data_source=ds, dataset=dataset)

    data = gql_client.query_data(
        DELETE_DATA_SOURCE,
        variables={'instanceId': str(instance_config.pk), 'dataSourceId': str(ds.uuid)},
    )
    result = data['instanceEditor']['deleteDataSource']
    assert result['__typename'] == 'OperationInfo'
    assert 'still referenced' in result['messages'][0]['message']
    assert DataSource.objects.filter(pk=ds.pk).exists()


def test_create_source_reference_to_dataset(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, _metric, _category = dataset_setup
    ds = _make_data_source(instance_config)

    data = gql_client.query_data(
        CREATE_SOURCE_REFERENCE,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'input': {'dataSourceId': str(ds.uuid), 'toDataset': True},
        },
    )
    result = data['instanceEditor']['datasetEditor']['createSourceReference']
    assert result['__typename'] == 'DatasetSourceReference'
    assert result['dataSource']['id'] == str(ds.uuid)
    assert result['dataset']['id'] == str(dataset.uuid)
    assert result['dataPoint'] is None
    assert DatasetSourceReference.objects.filter(uuid=result['id'], dataset=dataset).exists()


def test_create_source_reference_to_data_point(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    data_point = DataPointFactory.create(dataset=dataset, metric=metric, dimension_categories=[category])
    ds = _make_data_source(instance_config)

    data = gql_client.query_data(
        CREATE_SOURCE_REFERENCE,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'input': {'dataSourceId': str(ds.uuid), 'dataPointId': str(data_point.uuid)},
        },
    )
    result = data['instanceEditor']['datasetEditor']['createSourceReference']
    assert result['__typename'] == 'DatasetSourceReference'
    assert result['dataPoint']['id'] == str(data_point.uuid)
    assert result['dataset'] is None
    assert DatasetSourceReference.objects.filter(uuid=result['id'], data_point=data_point).exists()


def test_create_source_reference_rejects_both_targets(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    data_point = DataPointFactory.create(dataset=dataset, metric=metric, dimension_categories=[category])
    ds = _make_data_source(instance_config)

    data = gql_client.query_data(
        CREATE_SOURCE_REFERENCE,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'input': {
                'dataSourceId': str(ds.uuid),
                'toDataset': True,
                'dataPointId': str(data_point.uuid),
            },
        },
    )
    result = data['instanceEditor']['datasetEditor']['createSourceReference']
    assert result['__typename'] == 'OperationInfo'
    assert any('Exactly one of' in m['message'] for m in result['messages'])


def test_delete_source_reference(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, _metric, _category = dataset_setup
    ds = _make_data_source(instance_config)
    ref = DatasetSourceReference.objects.create(data_source=ds, dataset=dataset)

    data = gql_client.query_data(
        DELETE_SOURCE_REFERENCE,
        variables={
            'instanceId': str(instance_config.pk),
            'datasetId': str(dataset.uuid),
            'referenceId': str(ref.uuid),
        },
    )
    assert data['instanceEditor']['datasetEditor']['deleteSourceReference'] is None
    assert not DatasetSourceReference.objects.filter(pk=ref.pk).exists()


def test_instance_data_sources_query(gql_client: PathsTestClient, db_instance_config):
    s1 = _make_data_source(db_instance_config, name='Alpha')
    s2 = _make_data_source(db_instance_config, name='Beta')

    data = gql_client.query_data(
        INSTANCE_DATA_SOURCES,
        variables={'instanceId': str(db_instance_config.pk)},
    )
    ids = {s['id'] for s in data['modelInstance']['editor']['dataSources']}
    assert {str(s1.uuid), str(s2.uuid)} <= ids


def test_dataset_source_references_filtering(gql_client: PathsTestClient, dataset_setup):
    instance_config, dataset, metric, category = dataset_setup
    data_point = DataPointFactory.create(dataset=dataset, metric=metric, dimension_categories=[category])
    ds = _make_data_source(instance_config)

    ref_on_dataset = DatasetSourceReference.objects.create(data_source=ds, dataset=dataset)
    ref_on_dp = DatasetSourceReference.objects.create(data_source=ds, data_point=data_point)

    def query(target: str) -> list[str]:
        data = gql_client.query_data(
            DATASET_SOURCE_REFERENCES,
            variables={'instanceId': str(instance_config.pk), 'target': target},
        )
        ds_node = next(d for d in data['modelInstance']['editor']['datasets'] if d['id'] == str(dataset.uuid))
        return [r['id'] for r in ds_node['sourceReferences']]

    assert query('DATASET') == [str(ref_on_dataset.uuid)]
    assert query('DATA_POINT') == [str(ref_on_dp.uuid)]
    assert set(query('ALL')) == {str(ref_on_dataset.uuid), str(ref_on_dp.uuid)}

    data = gql_client.query_data(
        DATASET_SOURCE_REFERENCES,
        variables={'instanceId': str(instance_config.pk), 'target': 'ALL'},
    )
    ds_node = next(d for d in data['modelInstance']['editor']['datasets'] if d['id'] == str(dataset.uuid))
    assert [s['id'] for s in ds_node['dataSources']] == [str(ds.uuid)]
    dp_node = next(p for p in ds_node['dataPoints'] if p['id'] == str(data_point.uuid))
    assert [r['id'] for r in dp_node['sourceReferences']] == [str(ref_on_dp.uuid)]


def test_data_source_mutation_emits_change_operation(gql_client: PathsTestClient, db_instance_config):
    from nodes.models import InstanceChangeOperation, InstanceModelLogEntry

    gql_client.query_data(
        CREATE_DATA_SOURCE,
        variables={
            'instanceId': str(db_instance_config.pk),
            'input': {'name': 'Tracked source'},
        },
    )
    op = InstanceChangeOperation.objects.filter(
        instance_config=db_instance_config,
        action='dataset.data_source.create',
    ).first()
    assert op is not None
    entry = InstanceModelLogEntry.objects.filter(operation=op).first()
    assert entry is not None
    assert entry.data['before'] is None
    assert entry.data['after']['name'] == 'Tracked source'
