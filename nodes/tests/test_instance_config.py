from __future__ import annotations

import pytest
from social_core.backends.base import BaseAuth

from paths.const import INSTANCE_VIEWER_ROLE

from admin_site.auth_pipeline import assign_roles
from frameworks.models import Framework, FrameworkConfig
from frameworks.roles import FrameworkRoleDef
from nodes.models import InstanceConfig, _pytest_instances
from nodes.roles import instance_admin_role, instance_super_admin_role
from nodes.tests.factories import InstanceFactory, NodeConfigFactory, SimpleNodeFactory
from orgs.tests.factories import OrganizationFactory
from users.tests.factories import UserFactory

pytestmark = pytest.mark.django_db


class DummyAuthBackend(BaseAuth):
    name = 'test'


def test_framework_backed_yaml_instance_resolves_outcome_node_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = InstanceFactory.create(id='framework-city', name='Framework City')
    SimpleNodeFactory.create(context=instance.context, id='net_emissions', is_outcome=True)
    _pytest_instances.pop(instance.id, None)

    ic = InstanceConfig.objects.create(
        identifier=instance.id,
        name='Framework City',
        primary_language='en',
        organization=OrganizationFactory.create(),
    )
    FrameworkConfig.objects.create(
        framework=Framework.objects.create(identifier='framework-test', name='Framework Test'),
        instance_config=ic,
        organization_name='Framework City',
        baseline_year=2020,
    )
    node_config = NodeConfigFactory.create(instance=ic, identifier='net_emissions')

    def create_model_instance(self: FrameworkConfig, _ic: InstanceConfig):
        return instance

    monkeypatch.setattr(FrameworkConfig, 'create_model_instance', create_model_instance)

    assert [node.identifier for node in ic.get_outcome_nodes()] == ['net_emissions']
    assert instance.context.nodes['net_emissions'].database_id == node_config.pk


def test_locked_instance_removes_mutating_permissions_for_superuser_and_children() -> None:
    user = UserFactory.create(is_superuser=True)
    ic = InstanceConfig.objects.create(
        identifier='locked-city',
        name='Locked City',
        primary_language='en',
        organization=OrganizationFactory.create(),
        is_locked=True,
    )
    node_config = NodeConfigFactory.create(instance=ic)

    ic_policy = InstanceConfig.permission_policy()
    assert ic_policy.user_has_permission_for_instance(user, 'view', ic)
    assert not ic_policy.user_has_permission_for_instance(user, 'change', ic)
    assert not ic_policy.user_has_permission_for_instance(user, 'delete', ic)
    assert ic_policy.user_can_preview_draft(user, ic)

    node_policy = node_config.permission_policy()
    assert not node_policy.user_has_permission_for_instance(user, 'change', node_config)
    assert not node_policy.user_has_permission_for_instance(user, 'delete', node_config)
    assert not node_policy.user_can_create(user, ic)


def test_instance_lock_permission_is_limited_to_super_admins() -> None:
    ic = InstanceConfig.objects.create(
        identifier='lockable-city',
        name='Lockable City',
        primary_language='en',
        organization=OrganizationFactory.create(),
    )
    admin_user = UserFactory.create()
    super_admin_user = UserFactory.create()
    instance_admin_role.assign_user(ic, admin_user)
    instance_super_admin_role.assign_user(ic, super_admin_user)

    pp = InstanceConfig.permission_policy()
    assert not pp.user_can_set_lock(admin_user, ic)
    assert pp.user_can_set_lock(super_admin_user, ic)


def test_nzc_instance_creates_super_admin_group_but_not_reviewer_group() -> None:
    ic = InstanceConfig.objects.create(
        identifier='nzc-city',
        name='NZC City',
        primary_language='en',
        organization=OrganizationFactory.create(),
    )
    FrameworkConfig.objects.create(
        framework=Framework.objects.create(identifier='nzc', name='NetZeroCities'),
        instance_config=ic,
        organization_name='NZC City',
        baseline_year=2020,
    )

    ic.create_or_update_instance_groups()
    ic.refresh_from_db()

    assert ic.admin_group is not None
    assert ic.viewer_group is not None
    assert ic.super_admin_group is not None
    assert ic.reviewer_group is None


def test_login_pipeline_refreshes_instance_groups_for_framework_instances() -> None:
    user = UserFactory.create()
    framework = Framework.objects.create(identifier='nzc', name='NetZeroCities')
    ic = InstanceConfig.objects.create(
        identifier='pipeline-city',
        name='Pipeline City',
        primary_language='en',
        organization=OrganizationFactory.create(),
    )
    FrameworkConfig.objects.create(
        framework=framework,
        instance_config=ic,
        organization_name='Pipeline City',
        organization_identifier='city-uid',
        baseline_year=2020,
    )

    assign_roles(
        backend=DummyAuthBackend,
        user=user,
        details={
            'framework_roles': [
                FrameworkRoleDef(
                    framework_id='nzc',
                    role_id=INSTANCE_VIEWER_ROLE,
                    org_slug='pipeline-city',
                    org_id='city-uid',
                ),
            ],
        },
    )
    ic.refresh_from_db()

    assert ic.viewer_group is not None
    assert ic.super_admin_group is not None
    assert ic.reviewer_group is None
    assert ic.viewer_group in user.groups.all()
