from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import PermissionDenied
from django.urls import reverse

import pytest

from kausal_common.datasets.models import Dataset, DatasetSchema

from paths.context import RealmContext, realm_context

from admin_site.dataset_admin import DatasetSchemaViewSet
from nodes.models import InstanceConfig
from nodes.roles import instance_admin_role
from nodes.tests.factories import InstanceConfigFactory
from users.models import User


@pytest.fixture
def get_in_admin_context(rf):
    def get(user: User, view, url: str, instance_config: InstanceConfig, kwargs: dict | None = None):
        if kwargs is None:
            kwargs = {}
        request = rf.get(url)
        request.user = user
        ctx = RealmContext(realm = instance_config, user=request.user)
        with realm_context.activate(ctx):
            return view(request, **kwargs)
    return get


@pytest.mark.django_db
class TestDatasetAdminAuthorization:
    """Test authorization for dataset and dataset schema admin views."""

    @pytest.fixture
    def setup_test_data(self):
        # Create two instance configs with unique identifiers
        import uuid
        instance1 = InstanceConfigFactory(
            identifier=f'instance1-{uuid.uuid4().hex[:8]}',
            name='Instance 1',
        )
        instance2 = InstanceConfigFactory(
            identifier=f'instance2-{uuid.uuid4().hex[:8]}',
            name='Instance 2',
        )

        # Create a superuser
        superuser = User.objects.create_superuser(
            email='super@example.com',
            password='password',
        )

        # Create a regular user with admin access to instance1
        admin_user = User.objects.create_user(
            username='admin',
            email='admin@example.com',
            password='password',
        )
        # Assign the instance admin role to the user for instance1
        instance_admin_role.assign_user(instance1, admin_user)

        # Create a regular user with no special permissions
        regular_user = User.objects.create_user(
            username='regular',
            email='regular@example.com',
            password='password',
        )

        # Create dataset schemas for both instances
        schema1 = DatasetSchema.objects.create(
            name='Schema 1',
        )
        schema2 = DatasetSchema.objects.create(
            name='Schema 2',
        )

        # Create schema scopes to link schemas to instances
        content_type = ContentType.objects.get_for_model(InstanceConfig)
        from kausal_common.datasets.models import DatasetSchemaScope
        DatasetSchemaScope.objects.create(
            schema=schema1,
            scope_content_type=content_type,
            scope_id=instance1.id,
        )
        DatasetSchemaScope.objects.create(
            schema=schema2,
            scope_content_type=content_type,
            scope_id=instance2.id,
        )

        # Create datasets for both instances
        content_type = ContentType.objects.get_for_model(InstanceConfig)
        dataset1 = Dataset.objects.create(
            schema=schema1,
            scope_content_type=content_type,
            scope_id=instance1.id,
        )
        dataset2 = Dataset.objects.create(
            schema=schema2,
            scope_content_type=content_type,
            scope_id=instance2.id,
        )

        return {
            'instance1': instance1,
            'instance2': instance2,
            'superuser': superuser,
            'admin_user': admin_user,
            'regular_user': regular_user,
            'schema1': schema1,
            'schema2': schema2,
            'dataset1': dataset1,
            'dataset2': dataset2,
        }

    @pytest.mark.parametrize(('user_key', 'expected_schemas'), [
        ('superuser', ['Schema 1', 'Schema 2']),
        ('admin_user', ['Schema 1']),
        ('regular_user', []),
    ])
    def test_dataset_schema_index_view(self, client, setup_test_data, user_key, expected_schemas, get_in_admin_context):
        """Test access to dataset schema index view."""
        data = setup_test_data
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
            assert schema_name in content
        for schema_name in ['Schema 1', 'Schema 2']:
            if schema_name not in expected_schemas:
                assert schema_name not in content

    @pytest.mark.parametrize(('user_key', 'schema_key', 'access_allowed'), [
        ('superuser', 'schema1', True),
        ('superuser', 'schema2', True),
        ('admin_user', 'schema1', True),
        ('admin_user', 'schema2', False),
        ('regular_user', 'schema1', False),
        ('regular_user', 'schema2', False),
    ])
    def test_dataset_schema_edit_view(self, client, setup_test_data, user_key, schema_key, access_allowed, get_in_admin_context):
        """Test access to dataset schema edit view."""
        data = setup_test_data
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
    ])
    def test_dataset_schema_delete_view(
        self, client, setup_test_data, user_key, schema_key, access_allowed, get_in_admin_context
    ):
        """Test access to dataset schema delete view."""
        data = setup_test_data
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
    ])
    def test_dataset_index_view(self, client, setup_test_data, user_key, expected_datasets, get_in_admin_context):
        """Test access to dataset index view."""
        data = setup_test_data
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

            if dataset_key == 'dataset1':
                assert str(data['dataset1'].schema.name) in content
            else:
                assert str(data['dataset1'].schema.name) not in content
            if dataset_key == 'dataset2':
                assert str(data['dataset2'].schema.name) in content
            else:
                assert str(data['dataset2'].schema.name) not in content

    @pytest.mark.parametrize(('user_key', 'dataset_key', 'access_allowed'), [
        ('superuser', 'dataset1', True),
        ('superuser', 'dataset2', True),
        ('admin_user', 'dataset1', True),
        ('admin_user', 'dataset2', False),
        ('regular_user', 'dataset1', False),
        ('regular_user', 'dataset2', False),
    ])
    def test_dataset_edit_view(self, client, setup_test_data, user_key, dataset_key, access_allowed, get_in_admin_context):
        """Test access to dataset edit view."""
        data = setup_test_data
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
    ])
    def test_dataset_delete_view(self, client, setup_test_data, user_key, dataset_key, access_allowed, get_in_admin_context):
        """Test access to dataset delete view."""
        data = setup_test_data
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
    def test_dataset_create_view(self, client, setup_test_data, user_key, access_allowed, get_in_admin_context):
        """Test access to dataset create view."""
        data = setup_test_data
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
    def test_dataset_schema_create_view(self, client, setup_test_data, user_key, access_allowed, get_in_admin_context):
        """Test access to dataset schema create view."""
        data = setup_test_data
        user = data[user_key]

        view = DatasetSchemaViewSet().add_view
        url = reverse('datasets_datasetschema:add')

        if access_allowed:
            response = get_in_admin_context(user, view, url, data['instance1'])
            assert response.status_code == 200
            return

        with pytest.raises(PermissionDenied):
            response = get_in_admin_context(user, view, url, data['instance1'])
