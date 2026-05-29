from __future__ import annotations

from typing import Never

from django.urls import reverse

import pytest
from social_core.backends.base import BaseAuth

from paths.const import INSTANCE_SUPER_ADMIN_ROLE

from admin_site.api import check_user_in_other_clusters
from admin_site.auth_backends import NZCPortalOAuth2
from admin_site.auth_pipeline import assign_roles
from frameworks.models import FrameworkConfig
from frameworks.tests.factories import FrameworkFactory
from nodes.tests.factories import InstanceConfigFactory
from users.tests.factories import UserFactory

pytestmark = pytest.mark.django_db


class DummyAuthBackend(BaseAuth):
    name = 'test'


class DummyClusterResponse:
    status_code = 200

    def json(self) -> dict[str, str]:
        return {'method': 'azure_ad'}


def test_check_login_method_redirects_to_user_cluster(client, monkeypatch, settings) -> None:
    settings.PATHS_BACKEND_REGION_URLS = ['https://eu.paths.example']
    url = reverse('admin_check_login_method')

    def post(url: str, json: dict[str, str], timeout: int, headers: dict[str, str]) -> DummyClusterResponse:
        assert url == 'https://eu.paths.example/admin/login/check/'
        assert json == {'email': 'user@example.com'}
        assert timeout == 5
        assert headers == {'Content-Type': 'application/json'}
        return DummyClusterResponse()

    monkeypatch.setattr('admin_site.api.requests.post', post)

    response = client.post(url, {'email': ' USER@example.com '}, content_type='application/json')

    assert response.status_code == 200
    assert response.json() == {
        'method': 'azure_ad',
        'cluster_redirect': True,
        'cluster_url': 'https://eu.paths.example',
    }


def test_check_login_method_ignores_inactive_local_user_when_checking_clusters(client, monkeypatch, settings) -> None:
    settings.PATHS_BACKEND_REGION_URLS = ['https://eu.paths.example']
    UserFactory.create(email='user@example.com', is_staff=True, is_superuser=True, is_active=False)
    url = reverse('admin_check_login_method')

    def post(*_args: object, **_kwargs: object) -> DummyClusterResponse:
        return DummyClusterResponse()

    monkeypatch.setattr('admin_site.api.requests.post', post)

    response = client.post(url, {'email': 'user@example.com'}, content_type='application/json')

    assert response.status_code == 200
    assert response.json()['cluster_redirect'] is True


def test_check_login_method_prefers_local_user(client, monkeypatch, settings, instance_config) -> None:
    settings.PATHS_BACKEND_REGION_URLS = ['https://eu.paths.example']
    user = UserFactory.create(email='user@example.com', is_staff=True, is_superuser=True)
    user.set_password('password')
    user.save()
    url = reverse('admin_check_login_method')

    def post(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError('local users should not be checked from other clusters')

    monkeypatch.setattr('admin_site.api.requests.post', post)

    response = client.post(url, {'email': 'user@example.com'}, content_type='application/json')

    assert response.status_code == 200
    assert response.json() == {'method': 'password'}


def test_check_user_in_other_clusters_skips_regional_host(rf, monkeypatch, settings) -> None:
    settings.PATHS_BACKEND_REGION_URLS = ['https://regional.paths.example']
    request = rf.post('/admin/login/check/', HTTP_HOST='regional.paths.example')

    def post(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError('regional hosts should not check peer clusters')

    monkeypatch.setattr('admin_site.api.requests.post', post)

    assert check_user_in_other_clusters('user@example.com', request) is None


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
