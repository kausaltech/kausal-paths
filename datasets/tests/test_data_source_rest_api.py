from __future__ import annotations

from rest_framework.test import APIClient

import pytest

from kausal_common.datasets.models import DataSource
from kausal_common.testing.utils import parse_table

from datasets.tests.fixtures import *
from datasets.tests.utils import AssertIdenticalUUIDs, AssertNewUUID, AssertRemovedUUID

pytestmark = pytest.mark.django_db()

@pytest.fixture
def api_client():
    return APIClient()


@pytest.mark.django_db
@pytest.mark.parametrize('user_key', [
    'superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user',
    'schema1_viewer', 'schema1_editor', 'schema1_admin',
    'schema1_viewer_group_user', 'schema1_editor_group_user', 'schema1_admin_group_user',
    'regular_user'
])
def test_data_source_list(api_client, dataset_test_data, user_key):
    user = dataset_test_data[user_key]
    api_client.force_authenticate(user=user)


    if user_key == 'superuser':
        expected_data_sources = {
            'data_source1',
            'data_source1_alternative',
            'data_source2'
        }
    elif user_key in [
        'admin_user',
        'super_admin_user',
        'reviewer_user',
        'viewer_user',
        'schema1_viewer',
        'schema1_editor',
        'schema1_admin',
        'schema1_viewer_group_user',
        'schema1_editor_group_user',
        'schema1_admin_group_user'
    ]:
        expected_data_sources = {
            'data_source1',
            'data_source1_alternative'
        }
    else:
        expected_data_sources = {}

    response = api_client.get('/v1/data_sources/')
    if user_key == 'regular_user':
        assert response.status_code == 403
        return

    assert response.status_code == 200
    data = response.json()
    data_source_uuids = set(str(dataset_test_data[key].uuid) for key in expected_data_sources)
    result_uuids = set(d['uuid'] for d in data['results'])
    assert data_source_uuids == result_uuids


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   data_source_key           expected_status
#-------------------------------------------------------------------
superuser                  data_source1              200
superuser                  data_source2              200

admin_user                 data_source1              200
super_admin_user           data_source1              200
reviewer_user              data_source1              200
viewer_user                data_source1              200

# Note: subsector admin persons with rights through
# DatasetSchemaPersonPermission objects or
# DatasetSchemaGroupPermission objects do not have
# direct access to individual data sources. If the
# need arises, access should be granted in the
# relevant permission policy
schema1_viewer             data_source1              404
schema1_editor             data_source1              404
schema1_admin              data_source1              404
schema1_viewer_group_user  data_source1              404
schema1_editor_group_user  data_source1              404
schema1_admin_group_user   data_source1              404

admin_user                 data_source2              404
super_admin_user           data_source2              404
reviewer_user              data_source2              404
viewer_user                data_source2              404
schema1_viewer             data_source2              404
schema1_editor             data_source2              404
schema1_admin              data_source2              404
schema1_viewer_group_user  data_source2              404
schema1_editor_group_user  data_source2              404
schema1_admin_group_user   data_source2              404

regular_user               data_source1              403
regular_user               data_source2              403
"""))
def test_data_source_retrieve(api_client, dataset_test_data, user_key, data_source_key, expected_status):
    user = dataset_test_data[user_key]
    data_source = dataset_test_data[data_source_key]
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/data_sources/{data_source.uuid}/')
    assert response.status_code == expected_status

    if response.status_code == 200:
        data = response.json()
        assert data['uuid'] == str(data_source.uuid)


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   data_source_key           expected_status

superuser                  data_source1              200
superuser                  data_source2              200
admin_user                 data_source1              200
super_admin_user           data_source1              200

viewer_user                data_source1              403
reviewer_user              data_source1              403
schema1_viewer             data_source1              403
schema1_editor             data_source1              403
schema1_admin              data_source1              403
schema1_viewer_group_user  data_source1              403
schema1_editor_group_user  data_source1              403
schema1_admin_group_user   data_source1              403
regular_user               data_source1              403

# TODO: the 403s below should be 404s to be consistent.
# Leaving them be for now.
viewer_user                data_source2              403
reviewer_user              data_source2              403
schema1_viewer             data_source2              403
schema1_editor             data_source2              403
schema1_admin              data_source2              403
schema1_viewer_group_user  data_source2              403
schema1_editor_group_user  data_source2              403
schema1_admin_group_user   data_source2              403
regular_user               data_source2              403

admin_user                 data_source2              404
super_admin_user           data_source2              404
"""))
def test_data_source_update(api_client, dataset_test_data, user_key, data_source_key, expected_status):
    user = dataset_test_data[user_key]
    data_source = dataset_test_data[data_source_key]
    api_client.force_authenticate(user=user)

    update_data = {'name': 'Updated Data Source Name'}

    with AssertIdenticalUUIDs(DataSource.objects):
        response = api_client.patch(f'/v1/data_sources/{data_source.uuid}/', update_data, format='json')

    assert response.status_code == expected_status

    if response.status_code == 200:
        data = response.json()
        assert data['name'] == 'Updated Data Source Name'


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   data_source_key                expected_status

superuser                  data_source1_alternative       204
admin_user                 data_source1_alternative       204
super_admin_user           data_source1_alternative       204

reviewer_user              data_source1_alternative       403
viewer_user                data_source1_alternative       403
schema1_viewer             data_source1_alternative       403
schema1_editor             data_source1_alternative       403
schema1_admin              data_source1_alternative       403
schema1_viewer_group_user  data_source1_alternative       403
schema1_editor_group_user  data_source1_alternative       403
schema1_admin_group_user   data_source1_alternative       403
regular_user               data_source1_alternative       403

reviewer_user              data_source2                   403
viewer_user                data_source2                   403
schema1_viewer             data_source2                   403
schema1_editor             data_source2                   403
schema1_admin              data_source2                   403
schema1_viewer_group_user  data_source2                   403
schema1_editor_group_user  data_source2                   403
schema1_admin_group_user   data_source2                   403
regular_user               data_source2                   403
"""))
def test_data_source_delete(api_client, dataset_test_data, user_key, data_source_key, expected_status):
    user = dataset_test_data[user_key]
    data_source = dataset_test_data[data_source_key]
    api_client.force_authenticate(user=user)

    if expected_status == 204:
        with AssertRemovedUUID(DataSource.objects, data_source.uuid):
            response = api_client.delete(f'/v1/data_sources/{data_source.uuid}/')
            assert response.status_code == expected_status
            return

    with AssertIdenticalUUIDs(DataSource.objects):
        response = api_client.delete(f'/v1/data_sources/{data_source.uuid}/')
        assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key          access_allowed
superuser         +
admin_user        +
super_admin_user  +
reviewer_user     -
viewer_user       -
regular_user      -
"""))
def test_data_source_create(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    api_client.force_authenticate(user=user)

    if user_key in ['admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user']:
        instance = dataset_test_data['instance1']
    else:
        instance = dataset_test_data['instance2']

    create_data = {
        'name': 'New Data Source',
        'authority': 'Test Authority',
        'content_type_app': 'nodes',
        'content_type_model': 'instanceconfig',
        'object_id': instance.id,
    }

    if access_allowed:
        with AssertNewUUID(DataSource.objects) as uuid_tracker:
            response = api_client.post('/v1/data_sources/', create_data, format='json')
        assert response.status_code == 201
        data = response.json()
        assert data['name'] == 'New Data Source'
        uuid_tracker.assert_created(data)
    else:
        with AssertIdenticalUUIDs(DataSource.objects):
            response = api_client.post('/v1/data_sources/', create_data, format='json')
        assert response.status_code == 403
