from __future__ import annotations

from rest_framework.test import APIClient

import pytest

from datasets.tests.fixtures import *


@pytest.fixture
def api_client():
    return APIClient()


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'expected_schemas'), [
    pytest.param('superuser', ['Schema 1', 'Schema 2', 'Unused schema'], id='superuser'),
    pytest.param('admin_user', ['Schema 1', 'Unused schema'], id='admin_user'),
    pytest.param('super_admin_user', ['Schema 1', 'Unused schema'], id='super_admin_user'),
    pytest.param('reviewer_user', ['Schema 1', 'Unused schema'], id='reviewer_user'),
    pytest.param('regular_user', [], id='regular_user'),
])
def test_dataset_schema_list(api_client, dataset_test_data, user_key, expected_schemas):
    user = dataset_test_data[user_key]
    api_client.force_authenticate(user=user)

    response = api_client.get('/v1/dataset_schemas/')

    all_schema_names = ['Schema 1', 'Schema 2', 'Unused schema']

    if expected_schemas:
        assert response.status_code == 200
        data = response.json()
        schema_names = [schema['name'] for schema in data]
        for schema_name in expected_schemas:
            assert schema_name in schema_names
        for schema_name in all_schema_names:
            if schema_name not in expected_schemas:
                assert schema_name not in schema_names
    else:
        assert response.status_code in [200, 403]
        if response.status_code == 200:
            data = response.json()
            assert len(data) == 0


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'schema_key', 'access_allowed'), [
    ('superuser', 'schema1', True),
    ('superuser', 'schema2', True),
    ('admin_user', 'schema1', True),
    ('admin_user', 'schema2', False),
    ('super_admin_user', 'schema1', True),
    ('super_admin_user', 'schema2', False),
    ('reviewer_user', 'schema1', True),
    ('reviewer_user', 'schema2', False),
    ('regular_user', 'schema1', False),
    ('regular_user', 'schema2', False),
])
def test_dataset_schema_retrieve(api_client, dataset_test_data, user_key, schema_key, access_allowed):
    user = dataset_test_data[user_key]
    schema = dataset_test_data[schema_key]
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/dataset_schemas/{schema.uuid}/')

    if access_allowed:
        assert response.status_code == 200
        data = response.json()
        assert data['uuid'] == str(schema.uuid)
    else:
        assert response.status_code in [403, 404]


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
    ('regular_user', 'schema1', False, True),
    ('regular_user', 'schema2', False, True),
    ('superuser', 'unused_schema', True, False),
    ('admin_user', 'unused_schema', True, False),
    ('super_admin_user', 'unused_schema', True, False),
    ('reviewer_user', 'unused_schema', False, False),
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
@pytest.mark.parametrize('user_key', ['superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'regular_user'])
def test_dataset_list(api_client, dataset_test_data, user_key):
    user = dataset_test_data[user_key]
    api_client.force_authenticate(user=user)

    if user_key == 'superuser':
        expected_datasets = ['dataset1', 'dataset2']
    elif user_key in ['admin_user', 'super_admin_user', 'reviewer_user']:
        expected_datasets = ['dataset1']
    else:
        expected_datasets = []

    response = api_client.get('/v1/datasets/')

    if expected_datasets:
        assert response.status_code == 200
        data = response.json()
        dataset_uuids = [str(dataset_test_data[key].uuid) for key in expected_datasets]
        result_uuids = [d['uuid'] for d in data]
        for uuid in dataset_uuids:
            assert uuid in result_uuids
    else:
        assert response.status_code in [200, 403]
        if response.status_code == 200:
            data = response.json()
            assert len(data) == 0


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'dataset_key', 'access_allowed'), [
    ('superuser', 'dataset1', True),
    ('superuser', 'dataset2', True),
    ('admin_user', 'dataset1', True),
    ('admin_user', 'dataset2', False),
    ('super_admin_user', 'dataset1', True),
    ('super_admin_user', 'dataset2', False),
    ('reviewer_user', 'dataset1', True),
    ('reviewer_user', 'dataset2', False),
    ('regular_user', 'dataset1', False),
    ('regular_user', 'dataset2', False),
])
def test_dataset_retrieve(api_client, dataset_test_data, user_key, dataset_key, access_allowed):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/datasets/{dataset.uuid}/')

    if access_allowed:
        assert response.status_code == 200
        data = response.json()
        assert data['uuid'] == str(dataset.uuid)
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
    ('regular_user', False),
])
def test_dataset_create(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    api_client.force_authenticate(user=user)

    if user_key in ['admin_user', 'super_admin_user', 'reviewer_user']:
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
