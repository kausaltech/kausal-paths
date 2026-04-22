from __future__ import annotations

import pytest
from social_core.backends.base import BaseAuth

from paths.const import INSTANCE_SUPER_ADMIN_ROLE

from admin_site.auth_backends import NZCPortalOAuth2
from admin_site.auth_pipeline import assign_roles
from frameworks.models import FrameworkConfig
from frameworks.tests.factories import FrameworkFactory
from nodes.tests.factories import InstanceConfigFactory
from users.tests.factories import UserFactory

pytestmark = pytest.mark.django_db


class DummyAuthBackend(BaseAuth):
    name = 'test'


def test_nzcportal_city_admin_maps_to_instance_super_admin() -> None:
    details = NZCPortalOAuth2(strategy=None)._get_user_details(
        {
            'Mail': 'CITY.ADMIN@example.com ',
            'FirstName': 'City',
            'LastName': 'Admin',
            'userType': 'cityAdmin',
            'userCity': 'test-city',
            'cityUID': 'city-uid',
        },
    )

    role = details['framework_roles'][0]
    assert details['email'] == 'city.admin@example.com'
    assert role.framework_id == 'nzc'
    assert role.role_id == INSTANCE_SUPER_ADMIN_ROLE
    assert role.org_slug == 'test-city'
    assert role.org_id == 'city-uid'


def test_nzcportal_city_admin_assignment_creates_super_admin_membership() -> None:
    user = UserFactory.create()
    framework = FrameworkFactory.create(identifier='nzc', name='NetZeroCities', public_base_fqdn='nzc.example.com')
    ic = InstanceConfigFactory.create(identifier='test-city', name='Test City')
    FrameworkConfig.objects.create(
        framework=framework,
        instance_config=ic,
        organization_name='Test City',
        organization_identifier='city-uid',
        baseline_year=2020,
    )

    details = NZCPortalOAuth2(strategy=None)._get_user_details(
        {
            'Mail': 'city.admin@example.com',
            'FirstName': 'City',
            'LastName': 'Admin',
            'userType': 'cityAdmin',
            'userCity': 'test-city',
            'cityUID': 'city-uid',
        },
    )
    assign_roles(backend=DummyAuthBackend, user=user, details=details)

    ic.refresh_from_db()
    assert ic.super_admin_group is not None
    assert ic.super_admin_group in user.groups.all()
    assert ic.admin_group is not None
    assert ic.admin_group not in user.groups.all()
