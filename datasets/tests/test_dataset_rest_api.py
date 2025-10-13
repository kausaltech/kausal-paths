from __future__ import annotations

from rest_framework.test import APIClient

import pytest

from datasets.tests.fixtures import *

pytestmark = pytest.mark.django_db()

@pytest.fixture
def api_client():
    return APIClient()


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'has_access', 'expected_schemas'), [
    pytest.param('superuser', True, {'Schema 1', 'Schema 2', 'Unused schema'}, id='superuser'),
    pytest.param('super_admin_user', True, {'Schema 1', 'Unused schema'}, id='super_admin_user'),
    pytest.param('admin_user', True, {'Schema 1', 'Unused schema'}, id='admin_user'),
    pytest.param('reviewer_user', True, {'Schema 1', 'Unused schema'}, id='reviewer_user'),
    pytest.param('viewer_user', True, {'Schema 1', 'Unused schema'}, id='viewer_user'),
    pytest.param('regular_user', False, set(), id='regular_user'),
])
def test_dataset_schema_list(api_client, dataset_test_data, user_key, has_access, expected_schemas):
    user = dataset_test_data[user_key]
    api_client.force_authenticate(user=user)
    response = api_client.get('/v1/dataset_schemas/')

    if has_access is False:
        assert response.status_code == 403
        return
    assert response.status_code == 200

    data = response.json()
    data_schemas = set(schema['name'] for schema in data)
    assert data_schemas == expected_schemas


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'schema_key', 'access_allowed', 'should_be_found'), [
    ('superuser', 'schema1', True, True),
    ('superuser', 'schema2', True, True),
    ('admin_user', 'schema1', True, True),
    ('admin_user', 'schema2', True, False),
    ('super_admin_user', 'schema1', True, True),
    ('super_admin_user', 'schema2', True, False),
    ('reviewer_user', 'schema1', True, True),
    ('reviewer_user', 'schema2', True, False),
    ('viewer_user', 'schema1', True, True),
    ('viewer_user', 'schema2', True, False),
    ('regular_user', 'schema1', False, False),
    ('regular_user', 'schema2', False, False),
])
def test_dataset_schema_retrieve(api_client, dataset_test_data, user_key, schema_key, access_allowed, should_be_found):
    user = dataset_test_data[user_key]
    schema = dataset_test_data[schema_key]
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/dataset_schemas/{schema.uuid}/')

    if access_allowed and should_be_found:
        assert response.status_code == 200
        data = response.json()
        assert data['uuid'] == str(schema.uuid)
    elif access_allowed and should_be_found is False:
        assert response.status_code == 404
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'schema_key', 'access_allowed', 'should_be_found'), [
    ('superuser', 'schema1', True, True),
    ('superuser', 'schema2', True, True),
    ('admin_user', 'schema1', True, True),
    ('admin_user', 'schema2', True, False),
    ('super_admin_user', 'schema1', True, True),
    ('super_admin_user', 'schema2', True, False),
    ('reviewer_user', 'schema1', False, False),
    ('reviewer_user', 'schema2', False, False),
    ('viewer_user', 'schema1', False, False),
    ('viewer_user', 'schema2', False, False),
    ('regular_user', 'schema1', False, False),
    ('regular_user', 'schema2', False, False),
])
def test_dataset_schema_update(api_client, dataset_test_data, user_key, schema_key, access_allowed, should_be_found):
    user = dataset_test_data[user_key]
    schema = dataset_test_data[schema_key]
    api_client.force_authenticate(user=user)

    update_data = {'name_en': 'Updated Schema Name'}
    response = api_client.patch(f'/v1/dataset_schemas/{schema.uuid}/', update_data, format='json')

    if access_allowed and should_be_found:
        assert response.status_code == 200
        data = response.json()
        assert data['name'] == 'Updated Schema Name'
        # TODO test schema is there
    elif access_allowed and should_be_found is False:
        assert response.status_code == 404
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'schema_key', 'access_allowed', 'should_be_found', 'has_linked_objects'), [
    ('superuser', 'schema1', True, True, True),
    ('superuser', 'schema2', True, True, True),
    ('admin_user', 'schema1', True, True, True),
    ('admin_user', 'schema2', True, False, True),
    ('super_admin_user', 'schema1', True, True, True),
    ('super_admin_user', 'schema2', True, False, True),
    ('reviewer_user', 'schema1', False, False, True),
    ('reviewer_user', 'schema2', False, False, True),
    ('viewer_user', 'schema1', False, False, True),
    ('viewer_user', 'schema2', False, False, True),
    ('regular_user', 'schema1', False, False, True),
    ('regular_user', 'schema2', False, False, True),
    ('superuser', 'unused_schema', True, True, False),
    ('admin_user', 'unused_schema', True, True, False),
    ('super_admin_user', 'unused_schema', True, True, False),
    ('reviewer_user', 'unused_schema', False, False, False),
    ('viewer_user', 'unused_schema', False, False, False),
    ('regular_user', 'unused_schema', False, False, False),

])
def test_dataset_schema_delete(
    api_client,
    dataset_test_data,
    user_key,
    schema_key,
    access_allowed,
    should_be_found,
    has_linked_objects,
):
    user = dataset_test_data[user_key]
    schema = dataset_test_data[schema_key]

    api_client.force_authenticate(user=user)
    response = api_client.delete(f'/v1/dataset_schemas/{schema.uuid}/')

    if not access_allowed:
        assert response.status_code == 403
        return

    if should_be_found and has_linked_objects:
        assert response.status_code == 400
    elif should_be_found and not has_linked_objects:
        assert response.status_code == 204
    else:
        assert response.status_code == 404

     # TODO test deleted


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'access_allowed'), [
    ('superuser', True),
    ('admin_user', True),
    ('super_admin_user', True),
    ('reviewer_user', False),
    ('viewer_user', False),
    ('regular_user', False),
])
def test_dataset_schema_create(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    api_client.force_authenticate(user=user)

    create_data = {'name_en': 'New Schema'}
    response = api_client.post('/v1/dataset_schemas/', create_data, format='json')

    if access_allowed:
        assert response.status_code == 201
        data = response.json()
        assert data['name'] == 'New Schema'
        # TODO TEST SCHEMA IS THERE
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize('user_key', [
    'superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user', 'regular_user'
])
def test_dataset_list(api_client, dataset_test_data, user_key):
    user = dataset_test_data[user_key]
    api_client.force_authenticate(user=user)

    if user_key == 'superuser':
        expected_datasets = {'dataset1', 'dataset2'}
    elif user_key in ['admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user']:
        expected_datasets = {'dataset1'}
    else:
        expected_datasets = {}

    response = api_client.get('/v1/datasets/')

    if user_key == 'regular_user':
        assert response.status_code == 403
        return

    assert response.status_code == 200
    data = response.json()
    dataset_uuids = set(str(dataset_test_data[key].uuid) for key in expected_datasets)
    result_uuids = set(d['uuid'] for d in data)
    assert dataset_uuids == result_uuids


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'dataset_key', 'access_allowed', 'should_be_found'), [
    ('superuser', 'dataset1', True, True),
    ('superuser', 'dataset2', True, True),
    ('admin_user', 'dataset1', True, True),
    ('admin_user', 'dataset2', True, False),
    ('super_admin_user', 'dataset1', True, True),
    ('super_admin_user', 'dataset2', True, False),
    ('reviewer_user', 'dataset1', True, True),
    ('reviewer_user', 'dataset2', True, False),
    ('viewer_user', 'dataset1', True, True),
    ('viewer_user', 'dataset2', True, False),
    ('regular_user', 'dataset1', False, False),
    ('regular_user', 'dataset2', False, False),
])
def test_dataset_retrieve(api_client, dataset_test_data, user_key, dataset_key, access_allowed, should_be_found):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/')

    if access_allowed and should_be_found:
        assert response.status_code == 200
        data = response.json()
        assert data['uuid'] == str(dataset.uuid)
    elif access_allowed and should_be_found is False:
        assert response.status_code == 404
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'dataset_key', 'should_be_found', 'access_allowed'), [
    ('superuser', 'dataset1', True, True),
    ('superuser', 'dataset2', True, True),
    ('admin_user', 'dataset1', True, True),
    ('admin_user', 'dataset2', False, True),
    ('super_admin_user', 'dataset1', True, True),
    ('super_admin_user', 'dataset2', False, True),
    ('reviewer_user', 'dataset1', False, False),
    ('reviewer_user', 'dataset2', False, False),
    ('viewer_user', 'dataset1', False, False),
    ('viewer_user', 'dataset2', False, False),
    ('regular_user', 'dataset1', False, False),
    ('regular_user', 'dataset2', False, False),
])
def test_dataset_update(api_client, dataset_test_data, user_key, dataset_key, should_be_found, access_allowed):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    api_client.force_authenticate(user=user)

    update_data = {}
    response = api_client.patch(f'/v1/datasets/{dataset.uuid}/', update_data, format='json')

    if access_allowed and should_be_found:
        assert response.status_code == 200
    elif access_allowed and not should_be_found:
        assert response.status_code == 404
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'dataset_key', 'access_allowed', 'should_be_found'), [
    ('superuser', 'dataset1', True, True),
    ('superuser', 'dataset2', True, True),
    ('admin_user', 'dataset1', True, True),
    ('admin_user', 'dataset2', True, False),
    ('super_admin_user', 'dataset1', True, True),
    ('super_admin_user', 'dataset2', True, False),
    ('reviewer_user', 'dataset1', False, False),
    ('reviewer_user', 'dataset2', False, False),
    ('viewer_user', 'dataset1', False, False),
    ('viewer_user', 'dataset2', False, False),
    ('regular_user', 'dataset1', False, False),
    ('regular_user', 'dataset2', False, False),
])
def test_dataset_delete(api_client, dataset_test_data, user_key, dataset_key, access_allowed, should_be_found):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    api_client.force_authenticate(user=user)

    response = api_client.delete(f'/v1/datasets/{dataset.uuid}/')

    if access_allowed and should_be_found:
        assert response.status_code == 204
    elif access_allowed and not should_be_found:
        assert response.status_code == 404
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'access_allowed'), [
    ('superuser', True),
    ('admin_user', True),
    ('super_admin_user', True),
    ('reviewer_user', False),
    ('viewer_user', False),
    ('regular_user', False),
])
def test_dataset_create(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    api_client.force_authenticate(user=user)

    if user_key in ['admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user']:
        schema = dataset_test_data['schema1']
        instance = dataset_test_data['instance1']
    else:
        schema = dataset_test_data['schema2']
        instance = dataset_test_data['instance2']

    create_data = {
        'schema': str(schema.uuid),
        'scope_content_type': 'nodes.instanceconfig',
        'scope_id': instance.id,
    }
    response = api_client.post('/v1/datasets/', create_data, format='json')

    if access_allowed:
        assert response.status_code == 201
        data = response.json()
        assert data['schema'] == str(schema.uuid)
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'dataset_key', 'expected_status'), [
    # Access to everything
    ('superuser', 'dataset1', 200),
    ('superuser', 'dataset2', 200),

    # Access to dataset
    ('admin_user', 'dataset1', 200),
    ('super_admin_user', 'dataset1', 200),
    ('reviewer_user', 'dataset1', 200),
    ('viewer_user', 'dataset1', 200),

    # No access to dataset
    ('admin_user', 'dataset2', 404),
    ('super_admin_user', 'dataset2', 404),
    ('reviewer_user', 'dataset2', 404),
    ('viewer_user', 'dataset2', 404),

    # No access to endpoint
    ('regular_user', 'dataset1', 403),
    ('regular_user', 'dataset2', 403),
])
def test_datapoint_list(api_client, dataset_test_data, user_key, dataset_key, expected_status):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/data_points/')
    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert 'results' in data


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'datapoint_key', 'access_allowed', 'should_be_found'), [
    ('superuser', 'data_point1', True, True),
    ('superuser', 'data_point2', True, True),
    ('admin_user', 'data_point1', True, True),
    ('admin_user', 'data_point2', True, False),
    ('super_admin_user', 'data_point1', True, True),
    ('super_admin_user', 'data_point2', True, False),
    ('reviewer_user', 'data_point1', True, True),
    ('reviewer_user', 'data_point2', True, False),
    ('viewer_user', 'data_point1', True, True),
    ('viewer_user', 'data_point2', True, False),
    ('regular_user', 'data_point1', False, False),
    ('regular_user', 'data_point2', False, False),
])
def test_datapoint_retrieve(api_client, dataset_test_data, user_key, datapoint_key, access_allowed, should_be_found):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data[datapoint_key]
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/')

    if access_allowed and should_be_found:
        assert response.status_code == 200
        data = response.json()
        assert data['uuid'] == str(datapoint.uuid)
    elif access_allowed and should_be_found is False:
        assert response.status_code == 404
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'dataset_key', 'expected_status'), [
    ('superuser', 'dataset1', 201),
    ('admin_user', 'dataset1', 201),
    ('super_admin_user', 'dataset1', 201),
    ('reviewer_user', 'dataset1', 403),
    ('viewer_user', 'dataset1', 403),
    ('regular_user', 'dataset1', 403),
])
def test_datapoint_create(api_client, dataset_test_data, user_key, dataset_key, expected_status):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    metric = dataset_test_data['metric1']
    dimension_category = dataset_test_data['dimension_category1']
    api_client.force_authenticate(user=user)

    create_data = {
        'date': '2024-01-01',
        'value': 150.0,
        'metric': str(metric.uuid),
        'dimension_categories': [str(dimension_category.uuid)],
    }
    response = api_client.post(f'/v1/datasets/{dataset.uuid}/data_points/', create_data, format='json')

    assert response.status_code == expected_status
    if response.status_code == 201:
        data = response.json()
        assert data['value'] == 150.0  # TODO test object is there


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'datapoint_key', 'access_allowed', 'should_be_found'), [
    ('superuser', 'data_point1', True, True),
    ('superuser', 'data_point2', True, True),
    ('admin_user', 'data_point1', True, True),
    ('admin_user', 'data_point2', True, False),
    ('super_admin_user', 'data_point1', True, True),
    ('super_admin_user', 'data_point2', True, False),
    ('reviewer_user', 'data_point1', False, False),
    ('reviewer_user', 'data_point2', False, False),
    ('viewer_user', 'data_point1', False, False),
    ('viewer_user', 'data_point2', False, False),
    ('regular_user', 'data_point1', False, False),
    ('regular_user', 'data_point2', False, False),
])
def test_datapoint_delete(api_client, dataset_test_data, user_key, datapoint_key, access_allowed, should_be_found):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data[datapoint_key]
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.delete(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/')

    if access_allowed and should_be_found:
        assert response.status_code == 204
    elif access_allowed and not should_be_found:
        assert response.status_code == 404
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'data_point_key', 'expected_status'), [
    # Access to data_point1 (instance1)
    ('superuser', 'data_point1', 200),
    ('admin_user', 'data_point1', 200),
    ('super_admin_user', 'data_point1', 200),
    ('reviewer_user', 'data_point1', 200),
    ('viewer_user', 'data_point1', 200),

    # Access to data_point2 (instance2)
    ('superuser', 'data_point2', 200),

    # No access to data_point2 (parent not visible)
    ('admin_user', 'data_point2', 404),
    ('super_admin_user', 'data_point2', 404),
    ('reviewer_user', 'data_point2', 404),
    ('viewer_user', 'data_point2', 404),

    # No access to endpoint
    ('regular_user', 'data_point1', 403),
    ('regular_user', 'data_point2', 403),
])
def test_datapoint_comment_list(api_client, dataset_test_data, user_key, data_point_key, expected_status):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data[data_point_key]
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/comments/')
    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'comment_key', 'expected_status'), [
    # Access to comment1 (on data_point1, instance1)
    ('superuser', 'comment1', 200),
    ('admin_user', 'comment1', 200),
    ('super_admin_user', 'comment1', 200),
    ('reviewer_user', 'comment1', 200),
    ('viewer_user', 'comment1', 200),

    # Access to comment2 (on data_point2, instance2)
    ('superuser', 'comment2', 200),

    # No access to comment2 (parent not visible)
    ('admin_user', 'comment2', 404),
    ('super_admin_user', 'comment2', 404),
    ('reviewer_user', 'comment2', 404),
    ('viewer_user', 'comment2', 404),

    # No access to endpoint
    ('regular_user', 'comment1', 403),
    ('regular_user', 'comment2', 403),
])
def test_datapoint_comment_retrieve(api_client, dataset_test_data, user_key, comment_key, expected_status):
    user = dataset_test_data[user_key]
    comment = dataset_test_data[comment_key]
    datapoint = comment.data_point
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/comments/{comment.uuid}/')

    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert data['uuid'] == str(comment.uuid)


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'data_point_key', 'expected_status'), [
    # Access to data_point1 (instance1)
    ('superuser', 'data_point1', 201),
    ('admin_user', 'data_point1', 201),
    ('super_admin_user', 'data_point1', 201),
    ('reviewer_user', 'data_point1', 201),

    # Access to data_point2 (instance2)
    ('superuser', 'data_point2', 201),

    # No write access to data_point1
    ('viewer_user', 'data_point1', 403),

    # No access to data_point2 (parent not visible)
    ('admin_user', 'data_point2', 404),
    ('super_admin_user', 'data_point2', 404),
    ('reviewer_user', 'data_point2', 404),

    # No access to endpoint / method
    ('viewer_user', 'data_point2', 403),
    ('regular_user', 'data_point1', 403),
    ('regular_user', 'data_point2', 403),
])
def test_datapoint_comment_create(api_client, dataset_test_data, user_key, data_point_key, expected_status):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data[data_point_key]
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    create_data = {
        'text': 'New test comment',
        'type': 'plain',
    }
    response = api_client.post(
        f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/comments/',
        create_data,
        format='json'
    )
    assert response.status_code == expected_status
    if response.status_code == 201:
        data = response.json()
        assert data['text'] == 'New test comment'


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'comment_key', 'expected_status'), [
    # Access to comment1 (on data_point1, instance1)
    ('superuser', 'comment1', 204),
    ('admin_user', 'comment1', 204),
    ('super_admin_user', 'comment1', 204),

    # Access to comment2 (on data_point2, instance2)
    ('superuser', 'comment2', 204),

    # No delete access to comment1
    ('reviewer_user', 'comment1', 403),
    ('viewer_user', 'comment1', 403),

    # No access to comment2 (parent not visible)
    ('admin_user', 'comment2', 404),
    ('super_admin_user', 'comment2', 404),

    # No access to endpoint / method
    ('reviewer_user', 'comment2', 403),
    ('viewer_user', 'comment2', 403),
    ('regular_user', 'comment1', 403),
    ('regular_user', 'comment2', 403),
])
def test_datapoint_comment_delete(api_client, dataset_test_data, user_key, comment_key, expected_status):
    user = dataset_test_data[user_key]
    comment = dataset_test_data[comment_key]
    datapoint = comment.data_point
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.delete(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/comments/{comment.uuid}/')
    assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'dataset_key', 'expected_status'), [
    ('superuser', 'dataset1', 200),
    ('superuser', 'dataset2', 200),
    ('admin_user', 'dataset1', 200),
    ('admin_user', 'dataset2', 404),
    ('super_admin_user', 'dataset1', 200),
    ('super_admin_user', 'dataset2', 404),
    ('reviewer_user', 'dataset1', 200),
    ('reviewer_user', 'dataset2', 404),
    ('viewer_user', 'dataset1', 200),
    ('viewer_user', 'dataset2', 404),
    ('regular_user', 'dataset1', 403),
    ('regular_user', 'dataset2', 403),
])
def test_dataset_comment_list(api_client, dataset_test_data, user_key, dataset_key, expected_status):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/comments/')
    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.django_db
@pytest.mark.parametrize('user_key', [
    'superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user', 'regular_user'
])
def test_dataset_comment_create(api_client, dataset_test_data, user_key):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data['dataset1']
    api_client.force_authenticate(user=user)

    create_data = {
        'text': 'New test comment',
        'type': 'plain',
    }
    response = api_client.post(f'/v1/datasets/{dataset.uuid}/comments/', create_data, format='json')

    assert response.status_code == 405


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'dataset_key', 'expected_status'), [
    # Access to dataset1 (instance1)
    ('superuser', 'dataset1', 200),
    ('admin_user', 'dataset1', 200),
    ('super_admin_user', 'dataset1', 200),
    ('reviewer_user', 'dataset1', 200),
    ('viewer_user', 'dataset1', 200),

    # Access to dataset2 (instance2)
    ('superuser', 'dataset2', 200),

    # No access to dataset2 (parent not visible)
    ('admin_user', 'dataset2', 404),
    ('super_admin_user', 'dataset2', 404),
    ('reviewer_user', 'dataset2', 404),
    ('viewer_user', 'dataset2', 404),

    # No access to endpoint
    ('regular_user', 'dataset1', 403),
    ('regular_user', 'dataset2', 403),
])
def test_dataset_source_reference_list_via_dataset(api_client, dataset_test_data, user_key, dataset_key, expected_status):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/sources/')

    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'source_ref_key', 'expected_status'), [
    # Access to source_ref1 (on dataset1, instance1)
    ('superuser', 'source_ref1', 200),
    ('admin_user', 'source_ref1', 200),
    ('super_admin_user', 'source_ref1', 200),
    ('reviewer_user', 'source_ref1', 200),
    ('viewer_user', 'source_ref1', 200),

    # Access to source_ref2 (on dataset2, instance2)
    ('superuser', 'source_ref2', 200),

    # No access to source_ref2 (parent not visible)
    ('admin_user', 'source_ref2', 404),
    ('super_admin_user', 'source_ref2', 404),
    ('reviewer_user', 'source_ref2', 404),
    ('viewer_user', 'source_ref2', 404),

    # No access to endpoint
    ('regular_user', 'source_ref1', 403),
    ('regular_user', 'source_ref2', 403),
])
def test_dataset_source_reference_retrieve_via_dataset(api_client, dataset_test_data, user_key, source_ref_key, expected_status):
    user = dataset_test_data[user_key]
    source_ref = dataset_test_data[source_ref_key]
    dataset = source_ref.dataset
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/sources/{source_ref.uuid}/')

    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert data['uuid'] == str(source_ref.uuid)


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'dataset_key', 'expected_status'), [
    # Access to dataset1 (instance1)
    ('superuser', 'dataset1', 201),
    ('admin_user', 'dataset1', 201),
    ('super_admin_user', 'dataset1', 201),

    # Access to dataset2 (instance2)
    ('superuser', 'dataset2', 201),

    # No write access to dataset1
    ('reviewer_user', 'dataset1', 403),
    ('viewer_user', 'dataset1', 403),

    # No access to dataset2 (parent not visible)
    ('admin_user', 'dataset2', 404),
    ('super_admin_user', 'dataset2', 404),

    # No access to endpoint
    ('reviewer_user', 'dataset2', 403),
    ('viewer_user', 'dataset2', 403),
    ('regular_user', 'dataset1', 403),
    ('regular_user', 'dataset2', 403),
])
def test_dataset_source_reference_create_via_dataset(api_client, dataset_test_data, user_key, dataset_key, expected_status):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    data_source = dataset_test_data['data_source1'] if dataset_key == 'dataset1' else dataset_test_data['data_source2']
    api_client.force_authenticate(user=user)

    create_data = {
        'data_source': str(data_source.uuid),
    }
    response = api_client.post(f'/v1/datasets/{dataset.uuid}/sources/', create_data, format='json')

    assert response.status_code == expected_status
    if response.status_code == 201:
        data = response.json()
        assert data['data_source'] == str(data_source.uuid)


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'source_ref_key', 'expected_status'), [
    # Access to source_ref1 (on dataset1, instance1)
    ('superuser', 'source_ref1', 204),
    ('admin_user', 'source_ref1', 204),
    ('super_admin_user', 'source_ref1', 204),

    # Access to source_ref2 (on dataset2, instance2)
    ('superuser', 'source_ref2', 204),

    # No delete access to source_ref1
    ('reviewer_user', 'source_ref1', 403),
    ('viewer_user', 'source_ref1', 403),

    # No access to source_ref2 (parent not visible)
    ('admin_user', 'source_ref2', 404),
    ('super_admin_user', 'source_ref2', 404),

    # No access to endpoint
    ('reviewer_user', 'source_ref2', 403),
    ('viewer_user', 'source_ref2', 403),
    ('regular_user', 'source_ref1', 403),
    ('regular_user', 'source_ref2', 403),
])
def test_dataset_source_reference_delete_via_dataset(api_client, dataset_test_data, user_key, source_ref_key, expected_status):
    user = dataset_test_data[user_key]
    source_ref = dataset_test_data[source_ref_key]
    dataset = source_ref.dataset
    api_client.force_authenticate(user=user)

    response = api_client.delete(f'/v1/datasets/{dataset.uuid}/sources/{source_ref.uuid}/')
    assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'data_point_key', 'expected_status'), [
    # Access to data_point1 (instance1)
    ('superuser', 'data_point1', 200),
    ('admin_user', 'data_point1', 200),
    ('super_admin_user', 'data_point1', 200),
    ('reviewer_user', 'data_point1', 200),
    ('viewer_user', 'data_point1', 200),

    # Access to data_point2 (instance2)
    ('superuser', 'data_point2', 200),

    # No access to data_point2 (parent not visible)
    ('admin_user', 'data_point2', 404),
    ('super_admin_user', 'data_point2', 404),
    ('reviewer_user', 'data_point2', 404),
    ('viewer_user', 'data_point2', 404),

    # No access to endpoint
    ('regular_user', 'data_point1', 403),
    ('regular_user', 'data_point2', 403),
])
def test_dataset_source_reference_list_via_datapoint(api_client, dataset_test_data, user_key, data_point_key, expected_status):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data[data_point_key]
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/')

    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'source_ref_key', 'expected_status'), [
    # Access to source_ref_on_datapoint (on data_point1, instance1)
    ('superuser', 'source_ref_on_datapoint', 200),
    ('admin_user', 'source_ref_on_datapoint', 200),
    ('super_admin_user', 'source_ref_on_datapoint', 200),
    ('reviewer_user', 'source_ref_on_datapoint', 200),
    ('viewer_user', 'source_ref_on_datapoint', 200),

    # Access to source_ref_on_datapoint2 (on data_point2, instance2)
    ('superuser', 'source_ref_on_datapoint2', 200),

    # No access to source_ref_on_datapoint2 (parent not visible)
    ('admin_user', 'source_ref_on_datapoint2', 404),
    ('super_admin_user', 'source_ref_on_datapoint2', 404),
    ('reviewer_user', 'source_ref_on_datapoint2', 404),
    ('viewer_user', 'source_ref_on_datapoint2', 404),

    # No access to endpoint
    ('regular_user', 'source_ref_on_datapoint', 403),
    ('regular_user', 'source_ref_on_datapoint2', 403),
])
def test_dataset_source_reference_retrieve_via_datapoint(
    api_client,
    dataset_test_data,
    user_key,
    source_ref_key,
    expected_status
):
    user = dataset_test_data[user_key]
    source_ref = dataset_test_data[source_ref_key]
    datapoint = source_ref.data_point
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/{source_ref.uuid}/')

    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert data['uuid'] == str(source_ref.uuid)


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'data_point_key', 'expected_status'), [
    # Access to data_point1 (instance1)
    ('superuser', 'data_point1', 201),
    ('admin_user', 'data_point1', 201),
    ('super_admin_user', 'data_point1', 201),

    # Access to data_point2 (instance2)
    ('superuser', 'data_point2', 201),

    # No write access to data_point1
    ('reviewer_user', 'data_point1', 403),
    ('viewer_user', 'data_point1', 403),

    # No access to data_point2 (parent not visible)
    ('admin_user', 'data_point2', 404),
    ('super_admin_user', 'data_point2', 404),

    # No access to endpoint
    ('reviewer_user', 'data_point2', 403),
    ('viewer_user', 'data_point2', 403),
    ('regular_user', 'data_point1', 403),
    ('regular_user', 'data_point2', 403),
])
def test_dataset_source_reference_create_via_datapoint(api_client, dataset_test_data, user_key, data_point_key, expected_status):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data[data_point_key]
    dataset = datapoint.dataset
    data_source = dataset_test_data['data_source1'] if data_point_key == 'data_point1' else dataset_test_data['data_source2']
    api_client.force_authenticate(user=user)

    create_data = {
        'data_source': str(data_source.uuid),
    }
    response = api_client.post(
        f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/',
        create_data,
        format='json'
    )
    assert response.status_code == expected_status
    if response.status_code == 201:
        data = response.json()
        assert data['data_source'] == str(data_source.uuid)


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'expected_status'), [
    ('superuser', 204),
    ('admin_user', 204),
    ('super_admin_user', 204),
    ('reviewer_user', 403),
    ('viewer_user', 403),
    ('regular_user', 403),
])
def test_dataset_source_reference_delete_via_datapoint(api_client, dataset_test_data, user_key, expected_status):
    user = dataset_test_data[user_key]
    source_ref = dataset_test_data['source_ref_on_datapoint']
    datapoint = dataset_test_data['data_point1']
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.delete(
        f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/{source_ref.uuid}/'
    )
    assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'schema_key', 'access_allowed', 'schema_should_be_found'), [
    ('superuser', 'schema1', True, True),
    ('superuser', 'schema2', True, True),
    ('admin_user', 'schema1', True, True),
    ('admin_user', 'schema2', True, False),
    ('super_admin_user', 'schema1', True, True),
    ('super_admin_user', 'schema2', True, False),
    ('reviewer_user', 'schema1', True, True),
    ('reviewer_user', 'schema2', True, False),
    ('viewer_user', 'schema1', True, True),
    ('viewer_user', 'schema2', True, False),
    ('regular_user', 'schema1', False, False),
    ('regular_user', 'schema2', False, False),
])
def test_dataset_metric_list(api_client, dataset_test_data, user_key, schema_key, access_allowed, schema_should_be_found):
    user = dataset_test_data[user_key]
    schema = dataset_test_data[schema_key]
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/dataset_schemas/{schema.uuid}/metrics/')

    if access_allowed and schema_should_be_found:
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    elif access_allowed and not schema_should_be_found:
        assert response.status_code == 404
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'metric_key', 'access_allowed', 'should_be_found'), [
    ('superuser', 'metric1', True, True),
    ('superuser', 'metric2', True, True),
    ('admin_user', 'metric1', True, True),
    ('admin_user', 'metric2', True, False),
    ('super_admin_user', 'metric1', True, True),
    ('super_admin_user', 'metric2', True, False),
    ('reviewer_user', 'metric1', True, True),
    ('reviewer_user', 'metric2', True, False),
    ('viewer_user', 'metric1', True, True),
    ('viewer_user', 'metric2', True, False),
    ('regular_user', 'metric1', False, False),
    ('regular_user', 'metric2', False, False),
])
def test_dataset_metric_retrieve(api_client, dataset_test_data, user_key, metric_key, access_allowed, should_be_found):
    user = dataset_test_data[user_key]
    metric = dataset_test_data[metric_key]
    schema = metric.schema
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/dataset_schemas/{schema.uuid}/metrics/{metric.uuid}/')

    if access_allowed and should_be_found:
        assert response.status_code == 200
        data = response.json()
        assert data['uuid'] == str(metric.uuid)
    elif access_allowed and not should_be_found:
        assert response.status_code == 404
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize('user_key', [
    'superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user', 'regular_user'
])
def test_dataset_metric_create(api_client, dataset_test_data, user_key):
    user = dataset_test_data[user_key]
    schema = dataset_test_data['schema1']
    api_client.force_authenticate(user=user)

    create_data = {
        'label': 'New Metric',
        'unit': 'tons',
        'order': 1,
    }
    response = api_client.post(f'/v1/dataset_schemas/{schema.uuid}/metrics/', create_data, format='json')
    assert response.status_code == 405


@pytest.mark.django_db
@pytest.mark.parametrize('user_key', [
    'superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user', 'regular_user'
])
def test_dataset_metric_update(api_client, dataset_test_data, user_key):
    user = dataset_test_data[user_key]
    metric = dataset_test_data['metric1']
    schema = metric.schema
    api_client.force_authenticate(user=user)

    update_data = {'label': 'Updated Metric'}
    response = api_client.patch(f'/v1/dataset_schemas/{schema.uuid}/metrics/{metric.uuid}/', update_data, format='json')

    assert response.status_code == 405


@pytest.mark.django_db
@pytest.mark.parametrize('user_key', [
    'superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user', 'regular_user'
])
def test_dataset_metric_delete(api_client, dataset_test_data, user_key):
    user = dataset_test_data[user_key]
    metric = dataset_test_data['metric1']
    schema = metric.schema
    api_client.force_authenticate(user=user)

    response = api_client.delete(f'/v1/dataset_schemas/{schema.uuid}/metrics/{metric.uuid}/')

    assert response.status_code == 405
