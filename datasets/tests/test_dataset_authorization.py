from django.core.exceptions import PermissionDenied
from django.urls import reverse

import pytest

from paths.context import RealmContext, realm_context

from admin_site.dataset_admin import DatasetSchemaViewSet

from .fixtures import *


@pytest.mark.django_db
class TestDatasetAdminAuthorization:
    """Test authorization for dataset and dataset schema admin views."""

    @pytest.mark.parametrize(('user_key', 'expected_schemas'), [
        ('superuser', ['Schema 1', 'Schema 2']),
        ('admin_user', ['Schema 1']),
        ('regular_user', []),
        ('schema1_viewer', ['Schema 1']),
        ('schema1_editor', ['Schema 1']),
        ('schema1_admin', ['Schema 1']),
        ('schema1_viewer_group_user', ['Schema 1']),
        ('schema1_editor_group_user', ['Schema 1']),
        ('schema1_admin_group_user', ['Schema 1']),
    ])
    def test_dataset_schema_index_view(self, client, dataset_test_data, user_key, expected_schemas, get_in_admin_context):
        """Test access to dataset schema index view."""
        data = dataset_test_data
        user = data[user_key]

        view = DatasetSchemaViewSet().index_view
        url = reverse('datasets_datasetschema:list')
        if expected_schemas:
            response = get_in_admin_context(user, view, url, data['instance1'])
        else:
            with pytest.raises(PermissionDenied):
                response = get_in_admin_context(user, view, url, data['instance1'])
            return

        ctx = RealmContext(realm = data['instance1'], user=user)
        with realm_context.activate(ctx):
            response.render()

        assert response.status_code == 200
        content = response.content.decode('utf-8')
        for schema_name in expected_schemas:
            schema_in_content = schema_name in content
            assert schema_in_content
        for schema_name in ['Schema 1', 'Schema 2']:
            schema_in_content = schema_name in content
            if schema_name not in expected_schemas:
                assert not schema_in_content

    @pytest.mark.parametrize(('user_key', 'schema_key', 'access_allowed'), [
        ('superuser', 'schema1', True),
        ('superuser', 'schema2', True),
        ('admin_user', 'schema1', True),
        ('admin_user', 'schema2', False),
        ('regular_user', 'schema1', False),
        ('regular_user', 'schema2', False),
        ('schema1_viewer', 'schema1', False),
        ('schema1_viewer', 'schema2', False),
        ('schema1_editor', 'schema1', True),
        ('schema1_editor', 'schema2', False),
        ('schema1_admin', 'schema1', True),
        ('schema1_admin', 'schema2', False),
        ('schema1_viewer_group_user', 'schema1', False),
        ('schema1_viewer_group_user', 'schema2', False),
        ('schema1_editor_group_user', 'schema1', True),
        ('schema1_editor_group_user', 'schema2', False),
        ('schema1_admin_group_user', 'schema1', True),
        ('schema1_admin_group_user', 'schema2', False),
    ])
    def test_dataset_schema_edit_view(
        self,
        client,
        dataset_test_data,
        user_key,
        schema_key,
        access_allowed,
        get_in_admin_context
    ):
        """Test access to dataset schema edit view."""
        data = dataset_test_data
        user = data[user_key]
        schema = data[schema_key]

        view = DatasetSchemaViewSet().edit_view
        url = reverse('datasets_datasetschema:edit', args=[schema.id])
        if access_allowed:
            response = get_in_admin_context(user, view, url, data['instance1'], {'pk': schema.id})
            assert response.status_code == 200
            return
        with pytest.raises(PermissionDenied):
            response = get_in_admin_context(user, view, url, data['instance1'], {'pk': schema.id})

    @pytest.mark.parametrize(('user_key', 'schema_key', 'access_allowed'), [
        ('superuser', 'schema1', True),
        ('superuser', 'schema2', True),
        ('admin_user', 'schema1', True),
        ('admin_user', 'schema2', False),
        ('regular_user', 'schema1', False),
        ('regular_user', 'schema2', False),
        ('schema1_viewer', 'schema1', False),
        ('schema1_viewer', 'schema2', False),
        ('schema1_editor', 'schema1', False),
        ('schema1_editor', 'schema2', False),
        ('schema1_admin', 'schema1', True),
        ('schema1_admin', 'schema2', False),
        ('schema1_viewer_group_user', 'schema1', False),
        ('schema1_viewer_group_user', 'schema2', False),
        ('schema1_editor_group_user', 'schema1', False),
        ('schema1_editor_group_user', 'schema2', False),
        ('schema1_admin_group_user', 'schema1', True),
        ('schema1_admin_group_user', 'schema2', False),
    ])
    def test_dataset_schema_delete_view(
        self, client, dataset_test_data, user_key, schema_key, access_allowed, get_in_admin_context
    ):
        """Test access to dataset schema delete view."""
        data = dataset_test_data
        user = data[user_key]
        schema = data[schema_key]

        view = DatasetSchemaViewSet().delete_view
        url = reverse('datasets_datasetschema:delete', args=[schema.id])
        if access_allowed:
            response = get_in_admin_context(user, view, url, data['instance1'], {'pk': schema.id})
            assert response.status_code == 200
            return

        with pytest.raises(PermissionDenied):
            response = get_in_admin_context(user, view, url, data['instance1'], {'pk': schema.id})

    @pytest.mark.parametrize(('user_key', 'expected_datasets'), [
        ('superuser', [('instance1', 'dataset1'), ('instance2', 'dataset2')]),
        ('admin_user', [('instance1', 'dataset1'), ('instance2', None)]),
        ('regular_user', []),  # No datasets
        ('schema1_viewer', [('instance1', 'dataset1')]),
        ('schema1_editor', [('instance1', 'dataset1')]),
        ('schema1_admin', [('instance1', 'dataset1')]),
        ('schema1_viewer_group_user', [('instance1', 'dataset1')]),
        ('schema1_editor_group_user', [('instance1', 'dataset1')]),
        ('schema1_admin_group_user', [('instance1', 'dataset1')]),
    ])
    def test_dataset_index_view(self, client, dataset_test_data, user_key, expected_datasets, get_in_admin_context):
        """Test access to dataset index view."""
        data = dataset_test_data
        user = data[user_key]

        from kausal_paths_extensions.dataset_editor import DatasetViewSet
        view = DatasetViewSet().index_view
        url = reverse('datasets_dataset:list')

        if not expected_datasets:
            with pytest.raises(PermissionDenied):
                response = get_in_admin_context(user, view, url, data['instance1'])
            return

        for instance_key, dataset_key in expected_datasets:
            try:
                response = get_in_admin_context(user, view, url, data[instance_key])
            except PermissionDenied:
                assert dataset_key is None
                continue
            else:
                assert dataset_key is not None
            ctx = RealmContext(realm = data[instance_key], user=user)
            with realm_context.activate(ctx):
                response.render()

            assert response.status_code == 200
            content = response.content.decode('utf-8')

            schema1_in_content = str(data['dataset1'].schema.name) in content
            if dataset_key == 'dataset1':
                assert schema1_in_content
            else:
                assert not schema1_in_content
            schema2_in_content = str(data['dataset2'].schema.name) in content
            if dataset_key == 'dataset2':
                assert schema2_in_content
            else:
                assert not schema2_in_content

    @pytest.mark.parametrize(('user_key', 'dataset_key', 'access_allowed'), [
        ('superuser', 'dataset1', True),
        ('superuser', 'dataset2', True),
        ('admin_user', 'dataset1', True),
        ('admin_user', 'dataset2', False),
        ('regular_user', 'dataset1', False),
        ('regular_user', 'dataset2', False),
        ('schema1_viewer', 'dataset1', False),
        ('schema1_viewer', 'dataset2', False),
        ('schema1_editor', 'dataset1', True),
        ('schema1_editor', 'dataset2', False),
        ('schema1_admin', 'dataset1', True),
        ('schema1_admin', 'dataset2', False),
        ('schema1_viewer_group_user', 'dataset1', False),
        ('schema1_viewer_group_user', 'dataset2', False),
        ('schema1_editor_group_user', 'dataset1', True),
        ('schema1_editor_group_user', 'dataset2', False),
        ('schema1_admin_group_user', 'dataset1', True),
        ('schema1_admin_group_user', 'dataset2', False),

    ])
    def test_dataset_edit_view(self, client, dataset_test_data, user_key, dataset_key, access_allowed, get_in_admin_context):
        """Test access to dataset edit view."""
        data = dataset_test_data
        user = data[user_key]
        dataset = data[dataset_key]

        from kausal_paths_extensions.dataset_editor import DatasetViewSet
        view = DatasetViewSet().edit_view
        url = reverse('datasets_dataset:edit', args=[dataset.id])

        if access_allowed:
            response = get_in_admin_context(user, view, url, data['instance1'], {'pk': dataset.id})
            assert response.status_code == 200
            return

        with pytest.raises(PermissionDenied):
            response = get_in_admin_context(user, view, url, data['instance1'], {'pk': dataset.id})

    @pytest.mark.parametrize(('user_key', 'dataset_key', 'access_allowed'), [
        ('superuser', 'dataset1', True),
        ('superuser', 'dataset2', True),
        ('admin_user', 'dataset1', True),
        ('admin_user', 'dataset2', False),
        ('regular_user', 'dataset1', False),
        ('regular_user', 'dataset2', False),
        ('schema1_viewer', 'dataset1', False),
        ('schema1_viewer', 'dataset2', False),
        ('schema1_editor', 'dataset1', False),
        ('schema1_editor', 'dataset2', False),
        ('schema1_admin', 'dataset1', True),
        ('schema1_admin', 'dataset2', False),
        ('schema1_viewer_group_user', 'dataset1', False),
        ('schema1_viewer_group_user', 'dataset2', False),
        ('schema1_editor_group_user', 'dataset1', False),
        ('schema1_editor_group_user', 'dataset2', False),
        ('schema1_admin_group_user', 'dataset1', True),
        ('schema1_admin_group_user', 'dataset2', False),
    ])
    def test_dataset_delete_view(self, client, dataset_test_data, user_key, dataset_key, access_allowed, get_in_admin_context):
        """Test access to dataset delete view."""
        data = dataset_test_data
        user = data[user_key]
        dataset = data[dataset_key]

        from kausal_paths_extensions.dataset_editor import DatasetViewSet
        view = DatasetViewSet().delete_view
        url = reverse('datasets_dataset:delete', args=[dataset.id])

        if access_allowed:
            response = get_in_admin_context(user, view, url, data['instance1'], {'pk': dataset.id})
            assert response.status_code == 200
            return

        with pytest.raises(PermissionDenied):
            response = get_in_admin_context(user, view, url, data['instance1'], {'pk': dataset.id})

    @pytest.mark.parametrize(('user_key', 'access_allowed'), [
        ('superuser', True),
        ('admin_user', True),
        ('regular_user', False),
    ])
    def test_dataset_create_view(self, client, dataset_test_data, user_key, access_allowed, get_in_admin_context):
        """Test access to dataset create view."""
        data = dataset_test_data
        user = data[user_key]

        # Set the selected instance for the admin user
        instance = None
        schema = None
        if user_key == 'admin_user':
            instance = data['instance1']
            schema = data['schema1']
        else:
            instance = data['instance2']
            schema = data['schema2']

        from kausal_paths_extensions.dataset_editor import DatasetViewSet
        view = DatasetViewSet().add_view
        url = reverse('datasets_dataset:add')
        url += f'?model=nodes.InstanceConfig&object_id={instance.pk}&dataset_schema_uuid={schema.uuid!s}'

        if access_allowed:
            response = get_in_admin_context(user, view, url, data['instance1'])
            assert response.status_code == 200
            return

        with pytest.raises(PermissionDenied):
            response = get_in_admin_context(user, view, url, data['instance1'])

    @pytest.mark.parametrize(('user_key', 'access_allowed'), [
        ('superuser', True),
        ('admin_user', True),
        ('regular_user', False),
    ])
    def test_dataset_schema_create_view(self, client, dataset_test_data, user_key, access_allowed, get_in_admin_context):
        """Test access to dataset schema create view."""
        data = dataset_test_data
        user = data[user_key]

        view = DatasetSchemaViewSet().add_view
        url = reverse('datasets_datasetschema:add')

        if access_allowed:
            response = get_in_admin_context(user, view, url, data['instance1'])
            assert response.status_code == 200
            return

        with pytest.raises(PermissionDenied):
            response = get_in_admin_context(user, view, url, data['instance1'])
