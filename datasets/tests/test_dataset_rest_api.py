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
@pytest.mark.parametrize(('user_key', 'schema_key', 'access_allowed'), [
    ('superuser', 'schema1', True),
    ('superuser', 'schema2', True),
    ('admin_user', 'schema1', True),
    ('admin_user', 'schema2', False),
    ('super_admin_user', 'schema1', True),
    ('super_admin_user', 'schema2', False),
    ('reviewer_user', 'schema1', False),
    ('reviewer_user', 'schema2', False),
    ('viewer_user', 'schema1', False),
    ('viewer_user', 'schema2', False),
    ('regular_user', 'schema1', False),
    ('regular_user', 'schema2', False),
])
def test_dataset_schema_update(api_client, dataset_test_data, user_key, schema_key, access_allowed):
    user = dataset_test_data[user_key]
    schema = dataset_test_data[schema_key]
    api_client.force_authenticate(user=user)

    update_data = {'name_en': 'Updated Schema Name'}
    response = api_client.patch(f'/v1/dataset_schemas/{schema.uuid}/', update_data, format='json')

    if access_allowed:
        assert response.status_code == 200
        data = response.json()
        assert data['name'] == 'Updated Schema Name'
    else:
        assert response.status_code in [403, 404]


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'schema_key', 'access_allowed', 'has_linked_objects'), [
    ('superuser', 'schema1', True, True),
    ('superuser', 'schema2', True, True),
    ('admin_user', 'schema1', True, True),
    ('admin_user', 'schema2', False, True),
    ('super_admin_user', 'schema1', True, True),
    ('super_admin_user', 'schema2', False, True),
    ('reviewer_user', 'schema1', False, True),
    ('reviewer_user', 'schema2', False, True),
    ('viewer_user', 'schema1', False, True),
    ('viewer_user', 'schema2', False, True),
    ('regular_user', 'schema1', False, True),
    ('regular_user', 'schema2', False, True),
    ('superuser', 'unused_schema', True, False),
    ('admin_user', 'unused_schema', True, False),
    ('super_admin_user', 'unused_schema', True, False),
    ('reviewer_user', 'unused_schema', False, False),
    ('viewer_user', 'unused_schema', False, False),
    ('regular_user', 'unused_schema', False, False),

])
def test_dataset_schema_delete(
    api_client,
    dataset_test_data,
    user_key,
    schema_key,
    access_allowed,
    has_linked_objects,
):
    user = dataset_test_data[user_key]
    schema = dataset_test_data[schema_key]

    api_client.force_authenticate(user=user)
    response = api_client.delete(f'/v1/dataset_schemas/{schema.uuid}/')

    if not access_allowed:
        assert response.status_code in [403, 404]
    elif has_linked_objects:
        assert response.status_code == 400
    else:
        assert response.status_code == 204


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
@pytest.mark.parametrize(('user_key', 'dataset_key', 'access_allowed'), [
    ('superuser', 'dataset1', True),
    ('superuser', 'dataset2', True),
    ('admin_user', 'dataset1', True),
    ('admin_user', 'dataset2', False),
    ('super_admin_user', 'dataset1', True),
    ('super_admin_user', 'dataset2', False),
    ('reviewer_user', 'dataset1', False),
    ('reviewer_user', 'dataset2', False),
    ('viewer_user', 'dataset1', False),
    ('viewer_user', 'dataset2', False),
    ('regular_user', 'dataset1', False),
    ('regular_user', 'dataset2', False),
])
def test_dataset_update(api_client, dataset_test_data, user_key, dataset_key, access_allowed):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    api_client.force_authenticate(user=user)

    update_data = {}
    response = api_client.patch(f'/v1/datasets/{dataset.uuid}/', update_data, format='json')

    if access_allowed:
        assert response.status_code == 200
    else:
        assert response.status_code in [403, 404]


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'dataset_key', 'access_allowed'), [
    ('superuser', 'dataset1', True),
    ('superuser', 'dataset2', True),
    ('admin_user', 'dataset1', True),
    ('admin_user', 'dataset2', False),
    ('super_admin_user', 'dataset1', True),
    ('super_admin_user', 'dataset2', False),
    ('reviewer_user', 'dataset1', False),
    ('reviewer_user', 'dataset2', False),
    ('viewer_user', 'dataset1', False),
    ('viewer_user', 'dataset2', False),
    ('regular_user', 'dataset1', False),
    ('regular_user', 'dataset2', False),
])
def test_dataset_delete(api_client, dataset_test_data, user_key, dataset_key, access_allowed):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    api_client.force_authenticate(user=user)

    response = api_client.delete(f'/v1/datasets/{dataset.uuid}/')

    if access_allowed:
        assert response.status_code == 204
    else:
        assert response.status_code in [403, 404]


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
@pytest.mark.parametrize('user_key', [
    'superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user', 'regular_user'
])
def test_datapoint_list(api_client, dataset_test_data, user_key):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data['dataset1']
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/data_points/')

    if user_key in ['superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user']:
        assert response.status_code == 200
        data = response.json()
        assert 'results' in data  # TODO a bit lacking in actual testing
    else:
        assert response.status_code in [200, 403]
        if response.status_code == 200:
            data = response.json()
            assert len(data['results']) == 0


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
@pytest.mark.parametrize(('user_key', 'access_allowed'), [
    ('superuser', True),
    ('admin_user', True),
    ('super_admin_user', True),
    ('reviewer_user', False),
    ('viewer_user', False),
    ('regular_user', False),
])
def test_datapoint_create(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data['dataset1']
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

    if access_allowed:
        assert response.status_code == 201
        data = response.json()
        assert data['value'] == 150.0
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'datapoint_key', 'access_allowed'), [
    ('superuser', 'data_point1', True),
    ('superuser', 'data_point2', True),
    ('admin_user', 'data_point1', True),
    ('admin_user', 'data_point2', False),
    ('super_admin_user', 'data_point1', True),
    ('super_admin_user', 'data_point2', False),
    ('reviewer_user', 'data_point1', False),
    ('reviewer_user', 'data_point2', False),
    ('viewer_user', 'data_point1', False),
    ('viewer_user', 'data_point2', False),
    ('regular_user', 'data_point1', False),
    ('regular_user', 'data_point2', False),
])
def test_datapoint_delete(api_client, dataset_test_data, user_key, datapoint_key, access_allowed):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data[datapoint_key]
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.delete(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/')

    if access_allowed:
        assert response.status_code == 204
    else:
        assert response.status_code in [403, 404]


@pytest.mark.django_db
@pytest.mark.parametrize('user_key', [
    'superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user', 'regular_user'
])
def test_datapoint_comment_list(api_client, dataset_test_data, user_key):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data['data_point1']
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/comments/')

    if user_key in ['superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user']:
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    else:
        assert response.status_code in [200, 403]


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'access_allowed', 'should_be_found'), [
    ('superuser', True, True),
    ('admin_user', True, True),
    ('super_admin_user', True, True),
    ('reviewer_user', True, True),
    ('viewer_user', True, True),
    ('regular_user', False, False),
])
def test_datapoint_comment_retrieve(api_client, dataset_test_data, user_key, access_allowed, should_be_found):
    user = dataset_test_data[user_key]
    comment = dataset_test_data['comment1']
    datapoint = comment.data_point
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/comments/{comment.uuid}/')

    if access_allowed and should_be_found:
        assert response.status_code == 200
        data = response.json()
        assert data['uuid'] == str(comment.uuid)
    elif access_allowed and should_be_found is False:
        assert response.status_code == 404
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'access_allowed'), [
    ('superuser', True),
    ('admin_user', True),
    ('super_admin_user', True),
    ('reviewer_user', True),
    ('viewer_user', False),
    ('regular_user', False),
])
def test_datapoint_comment_create(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data['data_point1']
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

    if access_allowed:
        assert response.status_code == 201
        data = response.json()
        assert data['text'] == 'New test comment'
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
def test_datapoint_comment_delete(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    comment = dataset_test_data['comment1']
    datapoint = comment.data_point
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.delete(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/comments/{comment.uuid}/')

    if access_allowed:
        assert response.status_code == 204
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize('user_key', [
    'superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user', 'regular_user'
])
def test_dataset_source_reference_list_via_dataset(api_client, dataset_test_data, user_key):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data['dataset1']
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/sources/')

    if user_key in ['superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user']:
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    else:
        assert response.status_code in [200, 403]


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'access_allowed', 'should_be_found'), [
    ('superuser', True, True),
    ('admin_user', True, True),
    ('super_admin_user', True, True),
    ('reviewer_user', True, True),
    ('viewer_user', True, True),
    ('regular_user', False, False),
])
def test_dataset_source_reference_retrieve_via_dataset(api_client, dataset_test_data, user_key, access_allowed, should_be_found):
    user = dataset_test_data[user_key]
    source_ref = dataset_test_data['source_ref1']
    dataset = source_ref.dataset
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/sources/{source_ref.uuid}/')

    if access_allowed and should_be_found:
        assert response.status_code == 200
        data = response.json()
        assert data['uuid'] == str(source_ref.uuid)
    elif access_allowed and should_be_found is False:
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
def test_dataset_source_reference_create_via_dataset(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data['dataset1']
    data_source = dataset_test_data['data_source1']
    api_client.force_authenticate(user=user)

    create_data = {
        'data_source': str(data_source.uuid),
    }
    response = api_client.post(f'/v1/datasets/{dataset.uuid}/sources/', create_data, format='json')

    if access_allowed:
        assert response.status_code == 201, response.json()
        data = response.json()
        assert data['data_source'] == str(data_source.uuid)
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
def test_dataset_source_reference_delete_via_dataset(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    source_ref = dataset_test_data['source_ref1']
    dataset = source_ref.dataset
    api_client.force_authenticate(user=user)

    response = api_client.delete(f'/v1/datasets/{dataset.uuid}/sources/{source_ref.uuid}/')

    if access_allowed:
        assert response.status_code == 204
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize('user_key', [
    'superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user', 'regular_user'
])
def test_dataset_source_reference_list_via_datapoint(api_client, dataset_test_data, user_key):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data['data_point1']
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/')

    if user_key in ['superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user']:
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    else:
        assert response.status_code in [200, 403]


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'access_allowed', 'should_be_found'), [
    ('superuser', True, True),
    ('admin_user', True, True),
    ('super_admin_user', True, True),
    ('reviewer_user', True, True),
    ('viewer_user', True, True),
    ('regular_user', False, False),
])
def test_dataset_source_reference_retrieve_via_datapoint(
    api_client,
    dataset_test_data,
    user_key,
    access_allowed,
    should_be_found
):
    user = dataset_test_data[user_key]
    source_ref = dataset_test_data['source_ref_on_datapoint']
    datapoint = dataset_test_data['data_point1']
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/{source_ref.uuid}/')

    if access_allowed and should_be_found:
        assert response.status_code == 200, response.json()
        data = response.json()
        assert data['uuid'] == str(source_ref.uuid)
    elif access_allowed and should_be_found is False:
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
def test_dataset_source_reference_create_via_datapoint(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data['data_point1']
    dataset = datapoint.dataset
    data_source = dataset_test_data['data_source1']
    api_client.force_authenticate(user=user)

    create_data = {
        'data_source': str(data_source.uuid),
    }
    response = api_client.post(
        f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/',
        create_data,
        format='json'
    )

    if access_allowed:
        assert response.status_code == 201, response.json()
        data = response.json()
        assert data['data_source'] == str(data_source.uuid)
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
def test_dataset_source_reference_delete_via_datapoint(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    source_ref = dataset_test_data['source_ref_on_datapoint']
    datapoint = dataset_test_data['data_point1']
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    response = api_client.delete(
        f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/{source_ref.uuid}/'
    )

    if access_allowed:
        assert response.status_code == 204, response.json()
    else:
        assert response.status_code == 403
