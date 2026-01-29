from __future__ import annotations

from typing import Any

from django.contrib.contenttypes.models import ContentType
from rest_framework.test import APIClient

import pytest

from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
    Dataset,
    DatasetSchema,
    DatasetSourceReference,
)
from kausal_common.testing.utils import parse_table

from datasets.tests.fixtures import *
from datasets.tests.utils import AssertIdenticalUUIDs, AssertNewUUID, AssertRemovedUUID

pytestmark = pytest.mark.django_db()

@pytest.fixture
def api_client():
    return APIClient()


@pytest.mark.django_db
@pytest.mark.parametrize(('user_key', 'has_access', 'expected_schemas'), [
    pytest.param('superuser', True, {'Schema 1', 'Schema 2', 'Schema 3', 'Unused schema', 'Unused schema 2'}, id='superuser'),
    pytest.param('super_admin_user', True, {'Schema 1', 'Unused schema'}, id='super_admin_user'),
    pytest.param('admin_user', True, {'Schema 1', 'Unused schema'}, id='admin_user'),
    pytest.param('reviewer_user', True, {'Schema 1', 'Unused schema'}, id='reviewer_user'),
    pytest.param('viewer_user', True, {'Schema 1', 'Unused schema'}, id='viewer_user'),
    pytest.param('schema1_viewer', True, {'Schema 1'}, id='schema1_viewer'),
    pytest.param('schema1_editor', True, {'Schema 1'}, id='schema1_editor'),
    pytest.param('schema1_admin', True, {'Schema 1'}, id='schema1_admin'),
    pytest.param('schema1_viewer_group_user', True, {'Schema 1'}, id='schema1_viewer_group_user'),
    pytest.param('schema1_editor_group_user', True, {'Schema 1'}, id='schema1_editor_group_user'),
    pytest.param('schema1_admin_group_user', True, {'Schema 1'}, id='schema1_admin_group_user'),
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
    data_schemas = set(schema['name'] for schema in data['results'])
    assert data_schemas == expected_schemas


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   schema_key  access_allowed  should_be_found
superuser                  schema1     +               +
superuser                  schema2     +               +
admin_user                 schema1     +               +
admin_user                 schema2     +               -
super_admin_user           schema1     +               +
super_admin_user           schema2     +               -
reviewer_user              schema1     +               +
reviewer_user              schema2     +               -
viewer_user                schema1     +               +
viewer_user                schema2     +               -
schema1_viewer             schema1     +               +
schema1_viewer             schema2     +               -
schema1_editor             schema1     +               +
schema1_editor             schema2     +               -
schema1_admin              schema1     +               +
schema1_admin              schema2     +               -
schema1_viewer_group_user  schema1     +               +
schema1_viewer_group_user  schema2     +               -
schema1_editor_group_user  schema1     +               +
schema1_editor_group_user  schema2     +               -
schema1_admin_group_user   schema1     +               +
schema1_admin_group_user   schema2     +               -
regular_user               schema1     -               -
regular_user               schema2     -               -
"""))
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
@pytest.mark.parametrize(*parse_table("""
user_key                   schema_key  expected_status

superuser                  schema1     200
superuser                  schema2     200
admin_user                 schema1     200
super_admin_user           schema1     200
schema1_editor             schema1     200
schema1_admin              schema1     200
schema1_editor_group_user  schema1     200
schema1_admin_group_user   schema1     200

reviewer_user              schema1     403
reviewer_user              schema2     403
viewer_user                schema1     403
viewer_user                schema2     403
schema1_viewer             schema1     403
schema1_viewer_group_user  schema1     403
regular_user               schema1     403
regular_user               schema2     403

admin_user                 schema2     404
super_admin_user           schema2     404
schema1_viewer             schema2     404
schema1_editor             schema2     404
schema1_admin              schema2     404
schema1_viewer_group_user  schema2     404
schema1_editor_group_user  schema2     404
schema1_admin_group_user   schema2     404
"""))
def test_dataset_schema_update(api_client, dataset_test_data, user_key, schema_key, expected_status):
    user = dataset_test_data[user_key]
    schema = dataset_test_data[schema_key]
    api_client.force_authenticate(user=user)

    update_data = {'name_en': 'Updated Schema Name'}

    with AssertIdenticalUUIDs(DatasetSchema.objects):
        response = api_client.patch(f'/v1/dataset_schemas/{schema.uuid}/', update_data, format='json')

    assert response.status_code == expected_status

    if response.status_code == 200:
        data = response.json()
        assert data['name'] == 'Updated Schema Name'


# TODO add unused_schema2 (to instance2)

@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   schema_key     delete_allowed  should_be_found  has_linked_objects

# The schemas below have datasets linked to them so deletion won't work (HTTP 400)
admin_user                 schema1        +               +                +
admin_user                 schema2        +               -                +
regular_user               schema1        -               -                +
regular_user               schema2        -               -                +
reviewer_user              schema1        -               -                +
reviewer_user              schema2        -               -                +
schema1_admin              schema1        +               +                +
schema1_admin              schema2        +               -                +
schema1_admin_group_user   schema1        +               +                +
schema1_admin_group_user   schema2        +               -                +
schema1_editor             schema1        -               -                +
schema1_editor             schema2        +               -                +
schema1_editor_group_user  schema1        -               -                +
schema1_editor_group_user  schema2        +               -                +
schema1_viewer             schema1        -               -                +
schema1_viewer             schema2        +               -                +
schema1_viewer_group_user  schema1        -               -                +
schema1_viewer_group_user  schema2        +               -                +
super_admin_user           schema1        +               +                +
super_admin_user           schema2        +               -                +
superuser                  schema1        +               +                +
superuser                  schema2        +               +                +
viewer_user                schema1        -               -                +
viewer_user                schema2        -               -                +


# These are the only ones that should succeed (204)
superuser                  unused_schema  +               +                -
super_admin_user           unused_schema  +               +                -
admin_user                 unused_schema  +               +                -

# 403 for these since the roles of the users simply disallow the delete action
regular_user               unused_schema  -               -                -
reviewer_user              unused_schema  -               -                -
viewer_user                unused_schema  -               -                -

# 404: subsector admins should only know about schemas they are directly linked to
schema1_admin              unused_schema  +               -                -
schema1_admin_group_user   unused_schema  +               -                -
schema1_editor             unused_schema  +               -                -
schema1_editor_group_user  unused_schema  +               -                -
schema1_viewer             unused_schema  +               -                -
schema1_viewer_group_user  unused_schema  +               -                -
"""))
def test_dataset_schema_delete(
    api_client,
    dataset_test_data,
    user_key,
    schema_key,
    delete_allowed,
    should_be_found,
    has_linked_objects,
):
    user = dataset_test_data[user_key]
    schema = dataset_test_data[schema_key]

    assert DatasetSchema.objects.filter(uuid=schema.uuid).exists()
    api_client.force_authenticate(user=user)

    if not delete_allowed:
        with AssertIdenticalUUIDs(DatasetSchema.objects):
            response = api_client.delete(f'/v1/dataset_schemas/{schema.uuid}/')
        assert response.status_code == 403
        return

    if not should_be_found:
        with AssertIdenticalUUIDs(DatasetSchema.objects):
            response = api_client.delete(f'/v1/dataset_schemas/{schema.uuid}/')
        assert response.status_code == 404
        return

    if has_linked_objects:
        with AssertIdenticalUUIDs(DatasetSchema.objects):
            response = api_client.delete(f'/v1/dataset_schemas/{schema.uuid}/')
        assert response.status_code == 400
        return

    with AssertRemovedUUID(DatasetSchema.objects, schema.uuid):
        response = api_client.delete(f'/v1/dataset_schemas/{schema.uuid}/')
    assert response.status_code == 204
    assert not DatasetSchema.objects.filter(uuid=schema.uuid).exists()


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
def test_dataset_schema_create(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    api_client.force_authenticate(user=user)

    create_data = {'name_en': 'New Schema'}

    if access_allowed:
        with AssertNewUUID(DatasetSchema.objects) as uuid_tracker:
            response = api_client.post('/v1/dataset_schemas/', create_data, format='json')
        assert response.status_code == 201
        data = response.json()
        assert data['name'] == 'New Schema'
        uuid_tracker.assert_created(data)
    else:
        with AssertIdenticalUUIDs(DatasetSchema.objects):
            response = api_client.post('/v1/dataset_schemas/', create_data, format='json')
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize('user_key', [
    'superuser', 'admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user',
    'schema1_viewer', 'schema1_editor', 'schema1_admin',
    'schema1_viewer_group_user', 'schema1_editor_group_user', 'schema1_admin_group_user',
    'regular_user'
])
def test_dataset_list(api_client, dataset_test_data, user_key):
    user = dataset_test_data[user_key]
    api_client.force_authenticate(user=user)

    if user_key == 'superuser':
        expected_datasets = {'dataset1', 'dataset2', 'dataset3'}
    elif user_key in ['admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user',
                      'schema1_viewer', 'schema1_editor', 'schema1_admin',
                      'schema1_viewer_group_user', 'schema1_editor_group_user', 'schema1_admin_group_user']:
        expected_datasets = {'dataset1'}
    else:
        expected_datasets = set()

    response = api_client.get('/v1/datasets/')

    if user_key == 'regular_user':
        assert response.status_code == 403
        return

    assert response.status_code == 200
    data = response.json()
    dataset_uuids = set(str(dataset_test_data[key].uuid) for key in expected_datasets)
    result_uuids = set(d['uuid'] for d in data['results'])
    assert dataset_uuids == result_uuids


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   dataset_key  access_allowed  should_be_found
superuser                  dataset1     +               +
superuser                  dataset2     +               +
admin_user                 dataset1     +               +
admin_user                 dataset2     +               -
super_admin_user           dataset1     +               +
super_admin_user           dataset2     +               -
reviewer_user              dataset1     +               +
reviewer_user              dataset2     +               -
viewer_user                dataset1     +               +
viewer_user                dataset2     +               -
schema1_viewer             dataset1     +               +
schema1_viewer             dataset2     +               -
schema1_editor             dataset1     +               +
schema1_editor             dataset2     +               -
schema1_admin              dataset1     +               +
schema1_admin              dataset2     +               -
schema1_viewer_group_user  dataset1     +               +
schema1_viewer_group_user  dataset2     +               -
schema1_editor_group_user  dataset1     +               +
schema1_editor_group_user  dataset2     +               -
schema1_admin_group_user   dataset1     +               +
schema1_admin_group_user   dataset2     +               -
regular_user               dataset1     -               -
regular_user               dataset2     -               -
"""))
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
@pytest.mark.parametrize(*parse_table("""
user_key                   dataset_key  expected_status

superuser                  dataset1     200
superuser                  dataset2     200

admin_user                 dataset1     200
super_admin_user           dataset1     200
schema1_editor             dataset1     200
schema1_admin              dataset1     200
schema1_editor_group_user  dataset1     200
schema1_admin_group_user   dataset1     200

viewer_user                dataset1     403
viewer_user                dataset2     403
reviewer_user              dataset1     403
reviewer_user              dataset2     403
schema1_viewer             dataset1     403
schema1_viewer_group_user  dataset1     403
regular_user               dataset1     403
regular_user               dataset2     403

schema1_editor             dataset2     404
schema1_admin              dataset2     404
schema1_editor_group_user  dataset2     404
schema1_admin_group_user   dataset2     404
schema1_viewer             dataset2     404
schema1_viewer_group_user  dataset2     404
admin_user                 dataset2     404
super_admin_user           dataset2     404
"""))
def test_dataset_update(api_client, dataset_test_data, user_key, dataset_key, expected_status):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    api_client.force_authenticate(user=user)

    update_data: dict[str, Any] = {}

    with AssertIdenticalUUIDs(Dataset.objects):
        response = api_client.patch(f'/v1/datasets/{dataset.uuid}/', update_data, format='json')

    assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   dataset_key  expected_status

superuser                  dataset1     204
superuser                  dataset2     204
admin_user                 dataset1     204
super_admin_user           dataset1     204
schema1_admin              dataset1     204
schema1_admin_group_user   dataset1     204

reviewer_user              dataset1     403
reviewer_user              dataset2     403
viewer_user                dataset1     403
viewer_user                dataset2     403
regular_user               dataset1     403
regular_user               dataset2     403
schema1_viewer             dataset1     403
schema1_editor             dataset1     403
schema1_viewer_group_user  dataset1     403
schema1_editor_group_user  dataset1     403

admin_user                 dataset2     404
super_admin_user           dataset2     404
schema1_viewer             dataset2     404
schema1_editor             dataset2     404
schema1_admin              dataset2     404
schema1_viewer_group_user  dataset2     404
schema1_editor_group_user  dataset2     404
schema1_admin_group_user   dataset2     404
"""))
def test_dataset_delete(api_client, dataset_test_data, user_key, dataset_key, expected_status):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    api_client.force_authenticate(user=user)

    if expected_status == 204:
        with AssertRemovedUUID(Dataset.objects, dataset.uuid):
            response = api_client.delete(f'/v1/datasets/{dataset.uuid}/')
            assert response.status_code == expected_status
            return

    with AssertIdenticalUUIDs(Dataset.objects):
        response = api_client.delete(f'/v1/datasets/{dataset.uuid}/')
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
def test_dataset_create(api_client, dataset_test_data, user_key, access_allowed):
    user = dataset_test_data[user_key]
    api_client.force_authenticate(user=user)

    # Use unused schemas which have scopes but no existing datasets
    if user_key in ['admin_user', 'super_admin_user', 'reviewer_user', 'viewer_user']:
        schema = dataset_test_data['unused_schema']
        instance = dataset_test_data['instance1']
    else:
        schema = dataset_test_data['unused_schema2']
        instance = dataset_test_data['instance2']

    content_type = ContentType.objects.get(app_label='nodes', model='instanceconfig')
    create_data = {
        'schema': str(schema.uuid),
        'scope_content_type_id': content_type.pk,
        'scope_id': instance.id,
    }

    if access_allowed:
        with AssertNewUUID(Dataset.objects) as uuid_tracker:
            response = api_client.post('/v1/datasets/', create_data, format='json')
        assert response.status_code == 201
        data = response.json()
        assert data['schema'] == str(schema.uuid)
        uuid_tracker.assert_created(data)
    else:
        with AssertIdenticalUUIDs(Dataset.objects):
            response = api_client.post('/v1/datasets/', create_data, format='json')
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   dataset_key  expected_status

# Access to everything
superuser                  dataset1     200
superuser                  dataset2     200

# Access to dataset
admin_user                 dataset1     200
super_admin_user           dataset1     200
reviewer_user              dataset1     200
viewer_user                dataset1     200
schema1_viewer             dataset1     200
schema1_editor             dataset1     200
schema1_admin              dataset1     200
schema1_viewer_group_user  dataset1     200
schema1_editor_group_user  dataset1     200
schema1_admin_group_user   dataset1     200

# No access to dataset
admin_user                 dataset2     404
super_admin_user           dataset2     404
reviewer_user              dataset2     404
viewer_user                dataset2     404
schema1_viewer             dataset2     404
schema1_editor             dataset2     404
schema1_admin              dataset2     404
schema1_viewer_group_user  dataset2     404
schema1_editor_group_user  dataset2     404
schema1_admin_group_user   dataset2     404

# No access to endpoint
regular_user               dataset1     403
regular_user               dataset2     403
"""))
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
@pytest.mark.parametrize(*parse_table("""
user_key                   datapoint_key  access_allowed  should_be_found
superuser                  data_point1    +               +
superuser                  data_point2    +               +
admin_user                 data_point1    +               +
admin_user                 data_point2    +               -
super_admin_user           data_point1    +               +
super_admin_user           data_point2    +               -
reviewer_user              data_point1    +               +
reviewer_user              data_point2    +               -
viewer_user                data_point1    +               +
viewer_user                data_point2    +               -
schema1_viewer             data_point1    +               +
schema1_viewer             data_point2    +               -
schema1_editor             data_point1    +               +
schema1_editor             data_point2    +               -
schema1_admin              data_point1    +               +
schema1_admin              data_point2    +               -
schema1_viewer_group_user  data_point1    +               +
schema1_viewer_group_user  data_point2    +               -
schema1_editor_group_user  data_point1    +               +
schema1_editor_group_user  data_point2    +               -
schema1_admin_group_user   data_point1    +               +
schema1_admin_group_user   data_point2    +               -
regular_user               data_point1    -               -
regular_user               data_point2    -               -
"""))
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
@pytest.mark.parametrize(*parse_table("""
user_key                   dataset_key  expected_status
superuser                  dataset1     201
admin_user                 dataset1     201
super_admin_user           dataset1     201
schema1_editor             dataset1     201
schema1_admin              dataset1     201
schema1_editor_group_user  dataset1     201
schema1_admin_group_user   dataset1     201
reviewer_user              dataset1     403
viewer_user                dataset1     403
schema1_viewer             dataset1     403
schema1_viewer_group_user  dataset1     403
regular_user               dataset1     403
"""))
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

    if expected_status == 201:
        with AssertNewUUID(DataPoint.objects) as uuid_tracker:
            response = api_client.post(f'/v1/datasets/{dataset.uuid}/data_points/', create_data, format='json')
        assert response.status_code == 201
        data = response.json()
        assert data['value'] == 150.0
        uuid_tracker.assert_created(data)
    else:
        with AssertIdenticalUUIDs(DataPoint.objects):
            response = api_client.post(f'/v1/datasets/{dataset.uuid}/data_points/', create_data, format='json')
        assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   datapoint_key  expected_status

# Access to data_point2 ( instance2 )
superuser                  data_point2    200

#  Access to data_point1 instance1
superuser                  data_point1    200
admin_user                 data_point1    200
super_admin_user           data_point1    200
schema1_editor             data_point1    200
schema1_admin              data_point1    200
schema1_editor_group_user  data_point1    200
schema1_admin_group_user   data_point1    200

# No write access to data_point1
reviewer_user              data_point1    403
viewer_user                data_point1    403
schema1_viewer             data_point1    403
schema1_viewer_group_user  data_point1    403

# No access to data_point2 ( parent not visible )
admin_user                 data_point2    404
super_admin_user           data_point2    404
schema1_editor             data_point2    404
schema1_admin              data_point2    404
schema1_editor_group_user  data_point2    404
schema1_admin_group_user   data_point2    404
schema1_viewer             data_point2    404
schema1_viewer_group_user  data_point2    404

# No access to endpoint
reviewer_user              data_point2    403
viewer_user                data_point2    403
regular_user               data_point1    403
regular_user               data_point2    403
"""))
def test_datapoint_update(api_client, dataset_test_data, user_key, datapoint_key, expected_status):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data[datapoint_key]
    dataset = datapoint.dataset
    metric = datapoint.metric
    dimension_categories = list(datapoint.dimension_categories.all())
    api_client.force_authenticate(user=user)

    # Test PATCH (partial update)
    patch_data = {'value': 999.99}
    with AssertIdenticalUUIDs(DataPoint.objects):
        response = api_client.patch(
            f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/',
            patch_data,
            format='json'
        )
    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert data['value'] == 999.99

    # Test PUT (full update)
    put_data = {
        'date': '2025-01-01',
        'value': 888.88,
        'metric': str(metric.uuid),
        'dimension_categories': [str(dc.uuid) for dc in dimension_categories],
    }
    with AssertIdenticalUUIDs(DataPoint.objects):
        response = api_client.put(
            f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/',
            put_data,
            format='json'
        )
    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert data['value'] == 888.88
        assert data['date'] == '2025-01-01'


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   datapoint_key  access_allowed  should_be_found
superuser                  data_point1    +               +
superuser                  data_point2    +               +
admin_user                 data_point1    +               +
super_admin_user           data_point1    +               +
schema1_admin              data_point1    +               +
schema1_admin_group_user   data_point1    +               +

admin_user                 data_point2    +               -
super_admin_user           data_point2    +               -
schema1_admin              data_point2    +               -
schema1_admin_group_user   data_point2    +               -

reviewer_user              data_point1    -               -
reviewer_user              data_point2    -               -
viewer_user                data_point1    -               -
viewer_user                data_point2    -               -
schema1_viewer             data_point1    -               -
schema1_viewer             data_point2    +               -
schema1_editor             data_point1    -               -
schema1_editor             data_point2    +               -
schema1_viewer_group_user  data_point1    -               -
schema1_viewer_group_user  data_point2    +               -
schema1_editor_group_user  data_point1    -               -
schema1_editor_group_user  data_point2    +               -
regular_user               data_point1    -               -
regular_user               data_point2    -               -
"""))
def test_datapoint_delete(api_client, dataset_test_data, user_key, datapoint_key, access_allowed, should_be_found):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data[datapoint_key]
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    if access_allowed and should_be_found:
        with AssertRemovedUUID(DataPoint.objects, datapoint.uuid):
            response = api_client.delete(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/')
        assert response.status_code == 204
    else:
        with AssertIdenticalUUIDs(DataPoint.objects):
            response = api_client.delete(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/')
        if access_allowed and not should_be_found:
            assert response.status_code == 404
        else:
            assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   data_point_key  expected_status

# Access to data_point1 (instance1)
superuser                  data_point1     200
admin_user                 data_point1     200
super_admin_user           data_point1     200
reviewer_user              data_point1     200
viewer_user                data_point1     200
schema1_viewer             data_point1     200
schema1_editor             data_point1     200
schema1_admin              data_point1     200
schema1_viewer_group_user  data_point1     200
schema1_editor_group_user  data_point1     200
schema1_admin_group_user   data_point1     200

# Access to data_point2 (instance2)
superuser                  data_point2     200

# No access to data_point2 (parent not visible)
admin_user                 data_point2     404
super_admin_user           data_point2     404
reviewer_user              data_point2     404
viewer_user                data_point2     404
schema1_viewer             data_point2     404
schema1_editor             data_point2     404
schema1_admin              data_point2     404
schema1_viewer_group_user  data_point2     404
schema1_editor_group_user  data_point2     404
schema1_admin_group_user   data_point2     404

# No access to endpoint
regular_user               data_point1     403
regular_user               data_point2     403
"""))
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
@pytest.mark.parametrize(*parse_table("""
user_key                   comment_key  expected_status

# Access to comment1 (on data_point1, instance1)
superuser                  comment1     200
admin_user                 comment1     200
super_admin_user           comment1     200
reviewer_user              comment1     200
viewer_user                comment1     200
schema1_viewer             comment1     200
schema1_editor             comment1     200
schema1_admin              comment1     200
schema1_viewer_group_user  comment1     200
schema1_editor_group_user  comment1     200
schema1_admin_group_user   comment1     200

# Access to comment2 (on data_point2, instance2)
superuser                  comment2     200

# No access to comment2 (parent not visible)
admin_user                 comment2     404
super_admin_user           comment2     404
reviewer_user              comment2     404
viewer_user                comment2     404
schema1_viewer             comment2     404
schema1_editor             comment2     404
schema1_admin              comment2     404
schema1_viewer_group_user  comment2     404
schema1_editor_group_user  comment2     404
schema1_admin_group_user   comment2     404

# No access to endpoint
regular_user               comment1     403
regular_user               comment2     403
"""))
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
@pytest.mark.parametrize(*parse_table("""
user_key                   data_point_key  expected_status

# Access to data_point1 instance1
superuser                  data_point1     201
admin_user                 data_point1     201
super_admin_user           data_point1     201
reviewer_user              data_point1     201
schema1_editor             data_point1     201
schema1_admin              data_point1     201
schema1_editor_group_user  data_point1     201
schema1_admin_group_user   data_point1     201

# Access to data_point2 instance2
superuser                  data_point2     201

# No write access to data_point1
viewer_user                data_point1     403
schema1_viewer             data_point1     403
schema1_viewer_group_user  data_point1     403

# No access to data_point2: parent not visible
admin_user                 data_point2     404
super_admin_user           data_point2     404
reviewer_user              data_point2     404
schema1_editor             data_point2     404
schema1_admin              data_point2     404
schema1_editor_group_user  data_point2     404
schema1_admin_group_user   data_point2     404
schema1_viewer             data_point2     404
schema1_viewer_group_user  data_point2     404

# No access to endpoint / method
viewer_user                data_point2     403
regular_user               data_point1     403
regular_user               data_point2     403
"""))
def test_datapoint_comment_create(api_client, dataset_test_data, user_key, data_point_key, expected_status):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data[data_point_key]
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    create_data = {
        'text': 'New test comment',
        'type': 'plain',
    }

    if expected_status == 201:
        with AssertNewUUID(DataPointComment.objects) as uuid_tracker:
            response = api_client.post(
                f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/comments/',
                create_data,
                format='json'
            )
        assert response.status_code == 201
        data = response.json()
        assert data['text'] == 'New test comment'
        uuid_tracker.assert_created(data)
    else:
        with AssertIdenticalUUIDs(DataPointComment.objects):
            response = api_client.post(
                f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/comments/',
                create_data,
                format='json'
            )
        assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   comment_key  expected_status

# Access to comment1 on data_point1 instance1
superuser                  comment1     204
admin_user                 comment1     204
super_admin_user           comment1     204
schema1_admin              comment1     204
schema1_admin_group_user   comment1     204

# Access to comment2 on data_point2 instance2
superuser                  comment2     204

# No delete access to comment1
reviewer_user              comment1     403
viewer_user                comment1     403
schema1_viewer             comment1     403
schema1_editor             comment1     403
schema1_viewer_group_user  comment1     403
schema1_editor_group_user  comment1     403

# No access to comment2 parent not visible
admin_user                 comment2     404
super_admin_user           comment2     404
schema1_admin              comment2     404
schema1_admin_group_user   comment2     404
schema1_viewer             comment2     404
schema1_editor             comment2     404
schema1_viewer_group_user  comment2     404
schema1_editor_group_user  comment2     404

# No access to endpoint / method
reviewer_user              comment2     403
viewer_user                comment2     403
regular_user               comment1     403
regular_user               comment2     403
"""))
def test_datapoint_comment_delete(api_client, dataset_test_data, user_key, comment_key, expected_status):
    user = dataset_test_data[user_key]
    comment = dataset_test_data[comment_key]
    datapoint = comment.data_point
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    if expected_status == 204:
        with AssertRemovedUUID(DataPointComment.objects, comment.uuid):
            response = api_client.delete(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/comments/{comment.uuid}/')
        assert response.status_code == 204
    else:
        with AssertIdenticalUUIDs(DataPointComment.objects):
            response = api_client.delete(f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/comments/{comment.uuid}/')
        assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   dataset_key  expected_status
superuser                  dataset1     200
superuser                  dataset2     200
admin_user                 dataset1     200
admin_user                 dataset2     404
super_admin_user           dataset1     200
super_admin_user           dataset2     404
reviewer_user              dataset1     200
reviewer_user              dataset2     404
viewer_user                dataset1     200
viewer_user                dataset2     404
schema1_viewer             dataset1     200
schema1_viewer             dataset2     404
schema1_editor             dataset1     200
schema1_editor             dataset2     404
schema1_admin              dataset1     200
schema1_admin              dataset2     404
schema1_viewer_group_user  dataset1     200
schema1_viewer_group_user  dataset2     404
schema1_editor_group_user  dataset1     200
schema1_editor_group_user  dataset2     404
schema1_admin_group_user   dataset1     200
schema1_admin_group_user   dataset2     404
regular_user               dataset1     403
regular_user               dataset2     403
"""))
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
@pytest.mark.parametrize(*parse_table("""
user_key                   dataset_key  expected_status

# Access to dataset1 (instance1)
superuser                  dataset1     200
admin_user                 dataset1     200
super_admin_user           dataset1     200
reviewer_user              dataset1     200
viewer_user                dataset1     200
schema1_viewer             dataset1     200
schema1_editor             dataset1     200
schema1_admin              dataset1     200
schema1_viewer_group_user  dataset1     200
schema1_editor_group_user  dataset1     200
schema1_admin_group_user   dataset1     200

# Access to dataset2 (instance2)
superuser                  dataset2     200

# No access to dataset2 (parent not visible)
admin_user                 dataset2     404
super_admin_user           dataset2     404
reviewer_user              dataset2     404
viewer_user                dataset2     404
schema1_viewer             dataset2     404
schema1_editor             dataset2     404
schema1_admin              dataset2     404
schema1_viewer_group_user  dataset2     404
schema1_editor_group_user  dataset2     404
schema1_admin_group_user   dataset2     404

# No access to endpoint
regular_user               dataset1     403
regular_user               dataset2     403
"""))
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
@pytest.mark.parametrize(*parse_table("""
user_key                   source_ref_key  expected_status

# Access to source_ref1 (on dataset1, instance1)
superuser                  source_ref1     200
admin_user                 source_ref1     200
super_admin_user           source_ref1     200
reviewer_user              source_ref1     200
viewer_user                source_ref1     200
schema1_viewer             source_ref1     200
schema1_editor             source_ref1     200
schema1_admin              source_ref1     200
schema1_viewer_group_user  source_ref1     200
schema1_editor_group_user  source_ref1     200
schema1_admin_group_user   source_ref1     200

# Access to source_ref2 (on dataset2, instance2)
superuser                  source_ref2     200

# No access to source_ref2 (parent not visible)
admin_user                 source_ref2     404
super_admin_user           source_ref2     404
reviewer_user              source_ref2     404
viewer_user                source_ref2     404
schema1_viewer             source_ref2     404
schema1_editor             source_ref2     404
schema1_admin              source_ref2     404
schema1_viewer_group_user  source_ref2     404
schema1_editor_group_user  source_ref2     404
schema1_admin_group_user   source_ref2     404

# No access to endpoint
regular_user               source_ref1     403
regular_user               source_ref2     403
"""))
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
@pytest.mark.parametrize(*parse_table("""
user_key                   source_ref_key  expected_status

# Access to source_ref1 (on dataset1, instance1)
superuser                  source_ref1     200
admin_user                 source_ref1     200
super_admin_user           source_ref1     200
schema1_editor             source_ref1     200
schema1_admin              source_ref1     200
schema1_editor_group_user  source_ref1     200
schema1_admin_group_user   source_ref1     200

# Access to source_ref2 (on dataset2, instance2)
superuser                  source_ref2     200

# No write access to source_ref1
reviewer_user              source_ref1     403
viewer_user                source_ref1     403
schema1_viewer             source_ref1     403
schema1_viewer_group_user  source_ref1     403

# No access to source_ref2 (parent not visible)
admin_user                 source_ref2     404
super_admin_user           source_ref2     404
schema1_editor             source_ref2     404
schema1_admin              source_ref2     404
schema1_editor_group_user  source_ref2     404
schema1_admin_group_user   source_ref2     404
schema1_viewer             source_ref2     404
schema1_viewer_group_user  source_ref2     404

# No access to endpoint
reviewer_user              source_ref2     403
viewer_user                source_ref2     403
regular_user               source_ref1     403
regular_user               source_ref2     403
"""))
def test_dataset_source_reference_update_via_dataset(api_client, dataset_test_data, user_key, source_ref_key, expected_status):
    user = dataset_test_data[user_key]
    source_ref = dataset_test_data[source_ref_key]
    dataset = source_ref.dataset
    original_data_source = source_ref.data_source
    alternative_data_source = dataset_test_data['data_source1_alternative']
    api_client.force_authenticate(user=user)

    # Test PATCH (partial update)
    patch_data = {'data_source': str(alternative_data_source.uuid)}
    with AssertIdenticalUUIDs(DatasetSourceReference.objects):
        response = api_client.patch(
            f'/v1/datasets/{dataset.uuid}/sources/{source_ref.uuid}/',
            patch_data,
            format='json'
        )
    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert data.get('data_source') == str(alternative_data_source.uuid)

    # Test PUT (full update)
    put_data = {
        'data_source': str(original_data_source.uuid),
    }
    with AssertIdenticalUUIDs(DatasetSourceReference.objects):
        response = api_client.put(
            f'/v1/datasets/{dataset.uuid}/sources/{source_ref.uuid}/',
            put_data,
            format='json'
        )
    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert data.get('data_source') == str(original_data_source.uuid)


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   dataset_key  expected_status

# Access to dataset1 (instance1)
superuser                  dataset1     201
admin_user                 dataset1     201
super_admin_user           dataset1     201
schema1_editor             dataset1     201
schema1_admin              dataset1     201
schema1_editor_group_user  dataset1     201
schema1_admin_group_user   dataset1     201

# Access to dataset2 (instance2)
superuser                  dataset2     201

# No write access to dataset1
reviewer_user              dataset1     403
viewer_user                dataset1     403
schema1_viewer             dataset1     403
schema1_viewer_group_user  dataset1     403

# No access to dataset2 (parent not visible)
admin_user                 dataset2     404
super_admin_user           dataset2     404
schema1_editor             dataset2     404
schema1_admin              dataset2     404
schema1_editor_group_user  dataset2     404
schema1_admin_group_user   dataset2     404
schema1_viewer             dataset2     404
schema1_viewer_group_user  dataset2     404

# No access to endpoint
reviewer_user              dataset2     403
viewer_user                dataset2     403
regular_user               dataset1     403
regular_user               dataset2     403
"""))
def test_dataset_source_reference_create_via_dataset(api_client, dataset_test_data, user_key, dataset_key, expected_status):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    data_source = dataset_test_data['data_source1'] if dataset_key == 'dataset1' else dataset_test_data['data_source2']
    api_client.force_authenticate(user=user)

    create_data = {
        'data_source': str(data_source.uuid),
    }

    if expected_status == 201:
        with AssertNewUUID(DatasetSourceReference.objects) as uuid_tracker:
            response = api_client.post(f'/v1/datasets/{dataset.uuid}/sources/', create_data, format='json')
        assert response.status_code == 201
        data = response.json()
        assert data['data_source'] == str(data_source.uuid)
        uuid_tracker.assert_created(data)
    else:
        with AssertIdenticalUUIDs(DatasetSourceReference.objects):
            response = api_client.post(f'/v1/datasets/{dataset.uuid}/sources/', create_data, format='json')
        assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   source_ref_key  expected_status

# Access to source_ref1 on dataset1  instance1
superuser                  source_ref1     204
admin_user                 source_ref1     204
super_admin_user           source_ref1     204
schema1_admin              source_ref1     204
schema1_admin_group_user   source_ref1     204

# Access to source_ref2 on dataset2 instance2
superuser                  source_ref2     204

# No delete access to source_ref1
reviewer_user              source_ref1     403
viewer_user                source_ref1     403
schema1_viewer             source_ref1     403
schema1_editor             source_ref1     403
schema1_viewer_group_user  source_ref1     403
schema1_editor_group_user  source_ref1     403

# No access to source_ref2 parent not visible
admin_user                 source_ref2     404
super_admin_user           source_ref2     404
schema1_admin              source_ref2     404
schema1_admin_group_user   source_ref2     404
schema1_viewer             source_ref2     404
schema1_editor             source_ref2     404
schema1_viewer_group_user  source_ref2     404
schema1_editor_group_user  source_ref2     404

# No access to endpoint / method
reviewer_user              source_ref2     403
viewer_user                source_ref2     403
regular_user               source_ref1     403
regular_user               source_ref2     403
"""))
def test_dataset_source_reference_delete_via_dataset(api_client, dataset_test_data, user_key, source_ref_key, expected_status):
    user = dataset_test_data[user_key]
    source_ref = dataset_test_data[source_ref_key]
    dataset = source_ref.dataset
    api_client.force_authenticate(user=user)

    if expected_status == 204:
        with AssertRemovedUUID(DatasetSourceReference.objects, source_ref.uuid):
            response = api_client.delete(f'/v1/datasets/{dataset.uuid}/sources/{source_ref.uuid}/')
        assert response.status_code == 204
    else:
        with AssertIdenticalUUIDs(DatasetSourceReference.objects):
            response = api_client.delete(f'/v1/datasets/{dataset.uuid}/sources/{source_ref.uuid}/')
        assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   data_point_key  expected_status

# Access to data_point1 (instance1)
superuser                  data_point1     200
admin_user                 data_point1     200
super_admin_user           data_point1     200
reviewer_user              data_point1     200
viewer_user                data_point1     200
schema1_viewer             data_point1     200
schema1_editor             data_point1     200
schema1_admin              data_point1     200
schema1_viewer_group_user  data_point1     200
schema1_editor_group_user  data_point1     200
schema1_admin_group_user   data_point1     200

# Access to data_point2 (instance2)
superuser                  data_point2     200

# No access to data_point2 (parent not visible)
admin_user                 data_point2     404
super_admin_user           data_point2     404
reviewer_user              data_point2     404
viewer_user                data_point2     404
schema1_viewer             data_point2     404
schema1_editor             data_point2     404
schema1_admin              data_point2     404
schema1_viewer_group_user  data_point2     404
schema1_editor_group_user  data_point2     404
schema1_admin_group_user   data_point2     404

# No access to endpoint
regular_user               data_point1     403
regular_user               data_point2     403
"""))
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
@pytest.mark.parametrize(*parse_table("""
user_key                   source_ref_key              expected_status

# Access to source_ref_on_datapoint (on data_point1, instance1)
superuser                  source_ref_on_datapoint     200
admin_user                 source_ref_on_datapoint     200
super_admin_user           source_ref_on_datapoint     200
reviewer_user              source_ref_on_datapoint     200
viewer_user                source_ref_on_datapoint     200
schema1_viewer             source_ref_on_datapoint     200
schema1_editor             source_ref_on_datapoint     200
schema1_admin              source_ref_on_datapoint     200
schema1_viewer_group_user  source_ref_on_datapoint     200
schema1_editor_group_user  source_ref_on_datapoint     200
schema1_admin_group_user   source_ref_on_datapoint     200

# Access to source_ref_on_datapoint2 (on data_point2, instance2)
superuser                  source_ref_on_datapoint2    200

# No access to source_ref_on_datapoint2 (parent not visible)
admin_user                 source_ref_on_datapoint2    404
super_admin_user           source_ref_on_datapoint2    404
reviewer_user              source_ref_on_datapoint2    404
viewer_user                source_ref_on_datapoint2    404
schema1_viewer             source_ref_on_datapoint2    404
schema1_editor             source_ref_on_datapoint2    404
schema1_admin              source_ref_on_datapoint2    404
schema1_viewer_group_user  source_ref_on_datapoint2    404
schema1_editor_group_user  source_ref_on_datapoint2    404
schema1_admin_group_user   source_ref_on_datapoint2    404

# No access to endpoint
regular_user               source_ref_on_datapoint     403
regular_user               source_ref_on_datapoint2    403
"""))
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
@pytest.mark.parametrize(*parse_table("""
user_key                   source_ref_key              expected_status

# Access to source_ref_on_datapoint (on data_point1, instance1)
superuser                  source_ref_on_datapoint     200
admin_user                 source_ref_on_datapoint     200
super_admin_user           source_ref_on_datapoint     200
schema1_editor             source_ref_on_datapoint     200
schema1_admin              source_ref_on_datapoint     200
schema1_editor_group_user  source_ref_on_datapoint     200
schema1_admin_group_user   source_ref_on_datapoint     200

# Access to source_ref_on_datapoint2 (on data_point2, instance2)
superuser                  source_ref_on_datapoint2    200

# No write access to source_ref_on_datapoint
reviewer_user              source_ref_on_datapoint     403
viewer_user                source_ref_on_datapoint     403
schema1_viewer             source_ref_on_datapoint     403
schema1_viewer_group_user  source_ref_on_datapoint     403

# No access to source_ref_on_datapoint2 (parent not visible)
admin_user                 source_ref_on_datapoint2    404
super_admin_user           source_ref_on_datapoint2    404
schema1_editor             source_ref_on_datapoint2    404
schema1_admin              source_ref_on_datapoint2    404
schema1_editor_group_user  source_ref_on_datapoint2    404
schema1_admin_group_user   source_ref_on_datapoint2    404
schema1_viewer             source_ref_on_datapoint2    404
schema1_viewer_group_user  source_ref_on_datapoint2    404

# No access to endpoint
reviewer_user              source_ref_on_datapoint2    403
viewer_user                source_ref_on_datapoint2    403
regular_user               source_ref_on_datapoint     403
regular_user               source_ref_on_datapoint2    403
"""))
def test_dataset_source_reference_update_via_datapoint(api_client, dataset_test_data, user_key, source_ref_key, expected_status):
    user = dataset_test_data[user_key]
    source_ref = dataset_test_data[source_ref_key]
    datapoint = source_ref.data_point
    dataset = datapoint.dataset
    original_data_source = source_ref.data_source
    alternative_data_source = dataset_test_data['data_source1_alternative']
    api_client.force_authenticate(user=user)

    # Test PATCH (partial update)
    patch_data = {'data_source': str(alternative_data_source.uuid)}
    with AssertIdenticalUUIDs(DatasetSourceReference.objects):
        response = api_client.patch(
            f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/{source_ref.uuid}/',
            patch_data,
            format='json'
        )
    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert data.get('data_source') == str(alternative_data_source.uuid)

    # Test PUT (full update)
    put_data = {
        'data_source': str(original_data_source.uuid),
    }
    with AssertIdenticalUUIDs(DatasetSourceReference.objects):
        response = api_client.put(
            f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/{source_ref.uuid}/',
            put_data,
            format='json'
        )
    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert data.get('data_source') == str(original_data_source.uuid)


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   data_point_key  expected_status

# Access to data_point1 (instance1)
superuser                  data_point1     201
admin_user                 data_point1     201
super_admin_user           data_point1     201
schema1_editor             data_point1     201
schema1_admin              data_point1     201
schema1_editor_group_user  data_point1     201
schema1_admin_group_user   data_point1     201

# Access to data_point2 (instance2)
superuser                  data_point2     201

# No write access to data_point1
reviewer_user              data_point1     403
viewer_user                data_point1     403
schema1_viewer             data_point1     403
schema1_viewer_group_user  data_point1     403

# No access to data_point2 (parent not visible)
admin_user                 data_point2     404
super_admin_user           data_point2     404
schema1_editor             data_point2     404
schema1_admin              data_point2     404
schema1_editor_group_user  data_point2     404
schema1_admin_group_user   data_point2     404
schema1_viewer             data_point2     404
schema1_viewer_group_user  data_point2     404

# No access to endpoint
reviewer_user              data_point2     403
viewer_user                data_point2     403
regular_user               data_point1     403
regular_user               data_point2     403
"""))
def test_dataset_source_reference_create_via_datapoint(api_client, dataset_test_data, user_key, data_point_key, expected_status):
    user = dataset_test_data[user_key]
    datapoint = dataset_test_data[data_point_key]
    dataset = datapoint.dataset
    data_source = dataset_test_data['data_source1'] if data_point_key == 'data_point1' else dataset_test_data['data_source2']
    api_client.force_authenticate(user=user)

    create_data = {
        'data_source': str(data_source.uuid),
    }

    if expected_status == 201:
        with AssertNewUUID(DatasetSourceReference.objects) as uuid_tracker:
            response = api_client.post(
                f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/',
                create_data,
                format='json'
            )
        assert response.status_code == 201
        data = response.json()
        assert data['data_source'] == str(data_source.uuid)
        uuid_tracker.assert_created(data)
    else:
        with AssertIdenticalUUIDs(DatasetSourceReference.objects):
            response = api_client.post(
                f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/',
                create_data,
                format='json'
            )
        assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   expected_status
superuser                  204
admin_user                 204
super_admin_user           204
schema1_admin              204
schema1_admin_group_user   204
reviewer_user              403
viewer_user                403
schema1_viewer             403
schema1_editor             403
schema1_viewer_group_user  403
schema1_editor_group_user  403
regular_user               403
"""))
def test_dataset_source_reference_delete_via_datapoint(api_client, dataset_test_data, user_key, expected_status):
    user = dataset_test_data[user_key]
    source_ref = dataset_test_data['source_ref_on_datapoint']
    datapoint = dataset_test_data['data_point1']
    dataset = datapoint.dataset
    api_client.force_authenticate(user=user)

    if expected_status == 204:
        with AssertRemovedUUID(DatasetSourceReference.objects, source_ref.uuid):
            response = api_client.delete(
                f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/{source_ref.uuid}/'
            )
        assert response.status_code == 204
    else:
        with AssertIdenticalUUIDs(DatasetSourceReference.objects):
            response = api_client.delete(
                f'/v1/datasets/{dataset.uuid}/data_points/{datapoint.uuid}/sources/{source_ref.uuid}/'
            )
        assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   schema_key  access_allowed  schema_should_be_found
superuser                  schema1     +               +
superuser                  schema2     +               +
admin_user                 schema1     +               +
admin_user                 schema2     +               -
super_admin_user           schema1     +               +
super_admin_user           schema2     +               -
reviewer_user              schema1     +               +
reviewer_user              schema2     +               -
viewer_user                schema1     +               +
viewer_user                schema2     +               -
schema1_viewer             schema1     +               +
schema1_viewer             schema2     +               -
schema1_editor             schema1     +               +
schema1_editor             schema2     +               -
schema1_admin              schema1     +               +
schema1_admin              schema2     +               -
schema1_viewer_group_user  schema1     +               +
schema1_viewer_group_user  schema2     +               -
schema1_editor_group_user  schema1     +               +
schema1_editor_group_user  schema2     +               -
schema1_admin_group_user   schema1     +               +
schema1_admin_group_user   schema2     +               -
regular_user               schema1     -               -
regular_user               schema2     -               -
"""))
def test_dataset_metric_list(api_client, dataset_test_data, user_key, schema_key, access_allowed, schema_should_be_found):
    user = dataset_test_data[user_key]
    schema = dataset_test_data[schema_key]
    api_client.force_authenticate(user=user)

    response = api_client.get(f'/v1/dataset_schemas/{schema.uuid}/metrics/')

    if access_allowed and schema_should_be_found:
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data['results'], list)
    elif access_allowed and not schema_should_be_found:
        assert response.status_code == 404
    else:
        assert response.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key                   metric_key  access_allowed  should_be_found
superuser                  metric1     +               +
superuser                  metric2     +               +
admin_user                 metric1     +               +
admin_user                 metric2     +               -
super_admin_user           metric1     +               +
super_admin_user           metric2     +               -
reviewer_user              metric1     +               +
reviewer_user              metric2     +               -
viewer_user                metric1     +               +
viewer_user                metric2     +               -
schema1_viewer             metric1     +               +
schema1_viewer             metric2     +               -
schema1_editor             metric1     +               +
schema1_editor             metric2     +               -
schema1_admin              metric1     +               +
schema1_admin              metric2     +               -
schema1_viewer_group_user  metric1     +               +
schema1_viewer_group_user  metric2     +               -
schema1_editor_group_user  metric1     +               +
schema1_editor_group_user  metric2     +               -
schema1_admin_group_user   metric1     +               +
schema1_admin_group_user   metric2     +               -
regular_user               metric1     -               -
regular_user               metric2     -               -
"""))
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


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key         dataset_key expected_status
superuser        dataset1    201
admin_user       dataset1    201
super_admin_user dataset1    201
reviewer_user    dataset1    403
viewer_user      dataset1    403
regular_user     dataset1    403
"""))
def test_data_point_bulk_create(api_client, dataset_test_data, user_key, dataset_key, expected_status):
    user = dataset_test_data[user_key]
    dataset = dataset_test_data[dataset_key]
    metric = dataset_test_data['metric1']
    dimension_category = dataset_test_data['dimension_category1']
    api_client.force_authenticate(user=user)

    create_data = [
        {
            'date': '2024-01-01',
            'value': 150.0,
            'metric': str(metric.uuid),
            'dimension_categories': [str(dimension_category.uuid)],
        },
        {
            'date': '2025-01-01',
            'value': 10.0,
            'metric': str(metric.uuid),
            'dimension_categories': [str(dimension_category.uuid)],
        }
    ]

    if expected_status == 201:
        with AssertNewUUID(DataPoint.objects, bulk=True) as uuid_tracker:
            response = api_client.post(f'/v1/datasets/{dataset.uuid}/data_points/', create_data, format='json')
        assert response.status_code == 201
        data = response.json()
        # Only compare fields present in expected data, ignoring new fields (e.g., UUID of created object)
        data_to_compare = [
            {k: actual[k] for k in expected}
            for actual, expected in zip(data, create_data, strict=True)
        ]
        assert data_to_compare == create_data
        uuid_tracker.assert_created(data, 2)
    else:
        with AssertIdenticalUUIDs(DataPoint.objects):
            response = api_client.post(f'/v1/datasets/{dataset.uuid}/data_points/', create_data, format='json')
        assert response.status_code == expected_status


@pytest.mark.django_db
@pytest.mark.parametrize(*parse_table("""
user_key         data_point_keys         expected_status

# Actual bulk editing (see individual cases below for explanations about the status).
superuser        data_point2,data_point3 200
admin_user       data_point2,data_point3 404
super_admin_user data_point2,data_point3 404
reviewer_user    data_point2,data_point3 403
viewer_user      data_point2,data_point3 403
regular_user     data_point2,data_point3 403

# Test object-level permissions
# Data points of dataset2 (schema2, instance2)
schema1_admin    data_point3             404  # instance2 datasets not visible
schema1_viewer   data_point3             404  # instance2 datasets not visible
schema2_admin    data_point3             200
schema2_viewer   data_point3             403  # dataset visible, but no write access
schema3_admin    data_point3             404  # schema2 datasets not visible
schema3_viewer   data_point3             404  # schema2 datasets not visible
# Data points of dataset3 (schema3, instance2)
schema1_admin    data_point4             404  # instance2 datasets not visible
schema1_viewer   data_point4             404  # instance2 datasets not visible
schema2_admin    data_point4             404  # schema3 datasets not visible
schema2_viewer   data_point4             404  # schema3 datasets not visible
schema3_admin    data_point4             200
schema3_viewer   data_point4             403  # dataset visible, but no write access

# The following are the same as in test_datapoint_update
# Access to data_point1 (instance1)
superuser        data_point1             200
admin_user       data_point1             200
super_admin_user data_point1             200

# Access to data_point2 (instance2)
superuser        data_point2             200

# No write access to data_point1
reviewer_user    data_point1             403
viewer_user      data_point1             403

# No access to data_point2 (parent not visible)
admin_user       data_point2             404
super_admin_user data_point2             404

# No access to endpoint
reviewer_user    data_point2             403
viewer_user      data_point2             403
regular_user     data_point1             403
regular_user     data_point2             403
"""))
def test_data_point_bulk_update(api_client, dataset_test_data, user_key, data_point_keys, expected_status):
    data_point_keys = data_point_keys.split(',')
    user = dataset_test_data[user_key]
    data_points = [dataset_test_data[key] for key in data_point_keys]
    # For this test, we only work with data points from the same dataset
    assert {dp.dataset for dp in data_points} == {data_points[0].dataset}
    dataset = data_points[0].dataset
    # TODO: What about the metric and dimension categories? Think about whether this makes any sense.
    metric = data_points[0].metric
    dimension_categories = list(data_points[0].dimension_categories.all())
    api_client.force_authenticate(user=user)

    # Test PATCH (partial update)
    patch_data = [{
        'uuid': str(dp.uuid),
        'value': 999.99,
    } for dp in data_points]
    with AssertIdenticalUUIDs(DataPoint.objects):
        response = api_client.patch(f'/v1/datasets/{dataset.uuid}/data_points/', patch_data, format='json')
    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert len(data) == len(data_points)
        assert all(d['value'] == 999.99 for d in data)

    # Test PUT (full update)
    put_data = [{
        'uuid': str(dp.uuid),
        'date': '2025-01-01',
        'value': 888.88,
        'metric': str(metric.uuid),
        'dimension_categories': [str(dc.uuid) for dc in dimension_categories],
    } for dp in data_points]
    with AssertIdenticalUUIDs(DataPoint.objects):
        response = api_client.put(f'/v1/datasets/{dataset.uuid}/data_points/', put_data, format='json')
    assert response.status_code == expected_status
    if response.status_code == 200:
        data = response.json()
        assert len(data) == len(data_points)
        assert all(d['value'] == 888.88 for d in data)
        assert all(d['date'] == '2025-01-01' for d in data)
