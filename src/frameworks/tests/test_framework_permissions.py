from typing import TYPE_CHECKING, Any

import pytest

from kausal_common.models.permissions import PermissionedModel

from paths.tests.graphql import PathsTestClient

from frameworks.models import FrameworkConfig
from frameworks.roles import framework_admin_role, framework_viewer_role
from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.models import InstanceConfig
from nodes.roles import instance_editor_role
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from django.test import Client

    from kausal_common.models.permission_policy import ModelPermissionPolicy

    from users.models import User


@pytest.mark.django_db
def test_superuser_can_query_framework_config_population(client: Client) -> None:
    config = FrameworkConfigFactory.create()
    user = UserFactory.create(is_superuser=True)
    client.force_login(user)

    query = """
        query MunicipalityPopulation($framework: ID!, $config: ID!, $year: Int!) {
            framework(identifier: $framework) {
                config(id: $config) {
                    id
                    population(year: $year) { year value }
                }
            }
        }
    """
    result = PathsTestClient(client).query_data(
        query,
        variables={'framework': config.framework.identifier, 'config': str(config.uuid), 'year': 2023},
    )

    assert result['framework']['config']['population'] == {'year': 2023, 'value': None}


@pytest.mark.django_db
def test_nzc_framework_and_instance_roles() -> None:
    nzc = FrameworkFactory.create(identifier='nzc')
    own = FrameworkConfigFactory.create(framework=nzc)
    other = FrameworkConfigFactory.create(framework=nzc)
    outside = FrameworkConfigFactory.create()

    admin = UserFactory.create(is_superuser=False)
    viewer = UserFactory.create(is_superuser=False)
    editor = UserFactory.create(is_superuser=False)
    framework_admin_role.assign_user(nzc, admin)
    framework_viewer_role.assign_user(nzc, viewer)
    instance_editor_role.assign_user(own.instance_config, editor)

    fc_policy = FrameworkConfig.permission_policy()
    assert fc_policy.user_can_create(admin, nzc)
    assert not fc_policy.user_can_create(viewer, nzc)

    _assert_nzc_role_permissions(
        InstanceConfig.permission_policy(),
        (own.instance_config, other.instance_config, outside.instance_config),
        admin,
        viewer,
        editor,
    )
    _assert_nzc_role_permissions(fc_policy, (own, other, outside), admin, viewer, editor)

    other.instance_config.is_locked = True
    other.instance_config.save(update_fields=['is_locked'])
    _assert_locked_for_admin(InstanceConfig.permission_policy(), other.instance_config, admin)
    _assert_locked_for_admin(fc_policy, other, admin)


def _assert_nzc_role_permissions[M: PermissionedModel[Any]](
    policy: ModelPermissionPolicy[M, Any, Any],
    configs: tuple[M, M, M],
    admin: User,
    viewer: User,
    editor: User,
) -> None:
    own, other, outside = configs
    for action in ('view', 'change', 'delete'):
        admin_visible = policy.instances_user_has_permission_for(admin, action)
        assert policy.user_has_permission_for_instance(admin, action, own)
        assert policy.user_has_permission_for_instance(admin, action, other)
        assert admin_visible.filter(pk__in=[own.pk, other.pk]).count() == 2
        assert not policy.user_has_permission_for_instance(admin, action, outside)
        assert not admin_visible.filter(pk=outside.pk).exists()

    viewer_visible = policy.instances_user_has_permission_for(viewer, 'view')
    assert policy.user_has_permission_for_instance(viewer, 'view', own)
    assert policy.user_has_permission_for_instance(viewer, 'view', other)
    assert viewer_visible.filter(pk__in=[own.pk, other.pk]).count() == 2
    assert not policy.user_has_permission_for_instance(viewer, 'view', outside)
    assert not viewer_visible.filter(pk=outside.pk).exists()
    for action in ('change', 'delete'):
        assert all(not policy.user_has_permission_for_instance(viewer, action, obj) for obj in configs)
        assert not policy.instances_user_has_permission_for(viewer, action).filter(pk__in=[obj.pk for obj in configs]).exists()

    for action in ('view', 'change'):
        editor_visible = policy.instances_user_has_permission_for(editor, action)
        assert policy.user_has_permission_for_instance(editor, action, own)
        assert editor_visible.filter(pk=own.pk).exists()
        assert not policy.user_has_permission_for_instance(editor, action, other)
        assert not editor_visible.filter(pk=other.pk).exists()
        assert not policy.user_has_permission_for_instance(editor, action, outside)
    assert not policy.user_has_permission_for_instance(editor, 'delete', own)


def _assert_locked_for_admin[M: PermissionedModel[Any]](policy: ModelPermissionPolicy[M, Any, Any], obj: M, admin: User) -> None:
    assert policy.user_has_permission_for_instance(admin, 'view', obj)
    for action in ('change', 'delete'):
        assert not policy.user_has_permission_for_instance(admin, action, obj)
        assert not policy.instances_user_has_permission_for(admin, action).filter(pk=obj.pk).exists()
