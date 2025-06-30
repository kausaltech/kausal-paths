from django.core.exceptions import PermissionDenied
from django.urls import reverse

import pytest

from paths.admin_context import set_admin_instance

from nodes.instance_role_group_admin import InstanceRoleGroupSnippetViewSet
from nodes.models import InstanceConfig, InstanceRoleGroup
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
        if not user.selected_instance:
            user.selected_instance = instance_config
            user.save()
        from paths.context import RealmContext, realm_context
        set_admin_instance(instance_config, request=request)
        ctx = RealmContext(realm = instance_config, user=request.user)
        with realm_context.activate(ctx):
            return view(request, **kwargs)
    return get


@pytest.mark.django_db
class TestInstanceRoleGroupAdminAuthorization:
    """Test authorization for instance role group admin views."""

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

        # Create one instance role group for each instance
        instance1_group = InstanceRoleGroup.objects.create(
            instance=instance1,
            name="Instance role group for Instance 1",
        )
        instance2_group = InstanceRoleGroup.objects.create(
            instance=instance2,
            name="Instance role group for Instance 2",
        )

        return {
            'instance1': instance1,
            'instance2': instance2,
            'superuser': superuser,
            'admin_user': admin_user,
            'regular_user': regular_user,
            'instance1_group': instance1_group,
            'instance2_group': instance2_group,
        }

    @pytest.mark.parametrize(('user_key', 'expect_permission_denied', 'expected_groups'), [
        ('superuser', False, ['instance1_group', 'instance2_group']),
        ('admin_user', False, ['instance1_group']),
        ('regular_user', True, []),  # No groups
    ])
    def test_instance_role_group_index_view(
        self, client, setup_test_data, user_key, expect_permission_denied, expected_groups, get_in_admin_context
    ):
        """Test access to instance role group index view."""
        data = setup_test_data
        user = data[user_key]

        # Set the selected instance for the admin user
        if user_key == 'admin_user':
            user.selected_instance = data['instance1']
            user.save()

        view = InstanceRoleGroupSnippetViewSet().index_view
        url = reverse('nodes_instancerolegroup:list')

        for instance_key in ('instance1', 'instance2'):
            if expect_permission_denied:
                with pytest.raises(PermissionDenied):
                    get_in_admin_context(user, view, url, data[instance_key])
                continue

            response = get_in_admin_context(user, view, url, data[instance_key])
            response.render()
            assert response.status_code == 200
            content = response.content.decode('utf-8')
            expect_response_contains_group = {
                'instance1_group': 'instance1_group' in expected_groups and instance_key == 'instance1',
                'instance2_group': 'instance2_group' in expected_groups and instance_key == 'instance2',
            }
            for group_key in ('instance1_group', 'instance2_group'):
                response_contains_group = str(data[group_key].name) in content
                if expect_response_contains_group[group_key]:
                    assert response_contains_group
                else:
                    assert not response_contains_group
