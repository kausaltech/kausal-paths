from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Never
from urllib.parse import parse_qs, urlparse

from django.test import override_settings
from django.urls import reverse

import pytest
from pytest_django.asserts import assertContains
from social_core.backends.base import BaseAuth

from paths.const import INSTANCE_SUPER_ADMIN_ROLE
from paths.context import RealmContext, realm_context

from admin_site.api import check_user_in_other_clusters
from admin_site.auth_backends import NZCPortalOAuth2
from admin_site.auth_pipeline import assign_roles
from admin_site.wagtail_hooks import instance_chooser
from frameworks.models import FrameworkConfig
from frameworks.tests.factories import FrameworkFactory
from nodes.tests.factories import InstanceConfigFactory
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.django_db
requires_kpe = pytest.mark.skipif(
    find_spec('kausal_paths_extensions') is None,
    reason='requires the optional kausal_paths_extensions package',
)


@pytest.fixture
def _reset_social_backends_cache() -> Iterator[None]:
    """
    Reset social_core's global load_backends() cache.

    load_backends() memoizes its result on a module-level BACKENDSCACHE, so
    tests that use @override_settings(AUTHENTICATION_BACKENDS=…) can pollute
    subsequent tests. Reset before and after to guarantee isolation.
    """
    import social_core.backends.utils as sc_utils

    sc_utils.BACKENDSCACHE = {}
    yield
    sc_utils.BACKENDSCACHE = {}


class DummyAuthBackend(BaseAuth):
    name = 'test'


class DummyClusterResponse:
    status_code = 200

    def json(self) -> dict[str, str]:
        return {'method': 'azure_ad'}


def test_admin_login_uses_post_form_for_social_auth(client):
    response = client.get(reverse('wagtailadmin_login'))

    assert response.status_code == 200
    assertContains(response, '<form id="social-login-form" method="post" hidden>')
    assertContains(response, '<input type="hidden" name="csrfmiddlewaretoken"', count=2)
    assertContains(response, '<input type="hidden" name="next" />')
    assertContains(response, '<input type="hidden" name="email" />')
    assertContains(response, "$('.login-form').attr('action', '/auth/complete/password/');")


@override_settings(SOCIAL_AUTH_AZURE_AD_KEY='client-id', SOCIAL_AUTH_AZURE_AD_SECRET='client-secret')  # noqa: S106
def test_azure_ad_auth_entry_requires_post_and_forwards_email(client):
    url = reverse('social:begin', args=['azure_ad'])

    assert client.get(url).status_code == 405

    response = client.post(url, {'next': '/admin/', 'email': 'user@example.com'})

    assert response.status_code == 302
    assert response['Location'].startswith('https://login.microsoftonline.com/organizations/oauth2/authorize?')
    query = parse_qs(urlparse(response['Location']).query)
    assert query['login_hint'] == ['user@example.com']
    assert client.session['next'] == '/admin/'


@pytest.mark.usefixtures('_reset_social_backends_cache')
@requires_kpe
@override_settings(
    SOCIAL_AUTH_NZCPORTAL_CLIENT_ID='client-id',
    SOCIAL_AUTH_NZCPORTAL_CLIENT_SECRET='client-secret',  # noqa: S106
    AUTHENTICATION_BACKENDS=[
        'admin_site.auth_backends.NZCPortalOAuth2',
        'django.contrib.auth.backends.ModelBackend',
    ],
)
def test_nzc_oauth_authorize_login_redirect_reaches_provider(client):
    """
    External OAuth-provider redirects into /o/authorize/ must not dead-end on 405.

    NZC Planner UI (netzero.kausal.tech) drives users through Paths as an OAuth
    provider. When unauthenticated, browsers land at /o/authorize/ via GET, and
    AuthorizationView.get_login_url() sends them onward via a 302 (GET) redirect.
    Since social_django's /auth/login/<backend>/ is hardcoded POST-only in 6.0.x,
    that hop is a dead end. This test guards the end-to-end invariant.
    """
    from kausal_paths_extensions.auth.models import AuthApplication

    AuthApplication.objects.create(
        client_id='nzc-test-client',
        client_secret='nzc-test-secret',  # noqa: S106
        client_type=AuthApplication.CLIENT_CONFIDENTIAL,
        authorization_grant_type=AuthApplication.GRANT_AUTHORIZATION_CODE,
        redirect_uris='https://netzero.example/callback',
        social_auth_backend='nzcportal',
    )

    authorize_url = (
        '/o/authorize/'
        '?response_type=code&client_id=nzc-test-client'
        '&redirect_uri=https://netzero.example/callback'
        '&scope=openid'
        '&code_challenge=nmTAQE68zHR1tT6_Nqc26hpDPpI1tCtIVA0PWjgypT4'
        '&code_challenge_method=S256'
    )
    r = client.get(authorize_url)
    assert r.status_code == 302, f'expected 302 from /o/authorize/, got {r.status_code}'
    # Verify the redirect fully — scheme+host+path, plus the OAuth params that
    # NZC needs to accept the request. A prefix check would silently pass if a
    # regression dropped or corrupted client_id, state, response_type, scope,
    # or the callback redirect_uri.
    parsed = urlparse(r['Location'])
    assert (parsed.scheme, parsed.netloc, parsed.path) == (
        'https',
        'netzerocities.app',
        '/sso/authorize',
    ), f'expected redirect to https://netzerocities.app/sso/authorize, got {r["Location"]!r}'
    qs = parse_qs(parsed.query)
    assert qs.get('client_id') == ['client-id'], (
        f'client_id must be forwarded from SOCIAL_AUTH_NZCPORTAL_CLIENT_ID, got {qs.get("client_id")!r}'
    )
    assert qs.get('response_type') == ['code'], f'response_type must be "code", got {qs.get("response_type")!r}'
    assert qs.get('redirect_uri') == ['http://testserver/auth/complete/nzcportal/'], (
        f'redirect_uri must point at social_django complete endpoint, got {qs.get("redirect_uri")!r}'
    )
    assert qs.get('scope') == ['basic'], f'scope must match NZCPortalOAuth2.DEFAULT_SCOPE, got {qs.get("scope")!r}'
    state = qs.get('state', [''])[0]
    assert state, f'state must be present (anti-CSRF token), got {qs.get("state")!r}'
    # The OAuth-provider flow resumes after NZC callback via the "next" value
    # stashed in the session. Verify it preserves the full original /o/authorize/
    # URL so client_id, redirect_uri, and PKCE params survive the round trip.
    assert client.session.get('next') == authorize_url, (
        f'session["next"] must preserve the original OAuth authorize URL for '
        f'resume after NZC login; got {client.session.get("next")!r}, '
        f'expected {authorize_url!r}.'
    )


@pytest.mark.usefixtures('_reset_social_backends_cache')
@requires_kpe
@override_settings(
    SOCIAL_AUTH_AZURE_AD_KEY='client-id',
    SOCIAL_AUTH_AZURE_AD_SECRET='client-secret',  # noqa: S106
)
def test_azure_ad_oauth_authorize_does_not_take_nzc_shortcut(client):
    """
    Non-nzcportal social-backed OAuth apps must skip the nzcportal shortcut.

    The AuthorizationView override in handle_no_permission() is deliberately
    narrowed to `nzcportal` so that Azure-AD-backed OAuth clients (should any
    exist) fall through to the default LoginRequiredMixin behavior — the same
    code path they took before the nzcportal fix. This test bounds the blast
    radius of the override.
    """
    from kausal_paths_extensions.auth.models import AuthApplication

    AuthApplication.objects.create(
        client_id='azure-oauth-test-client',
        client_secret='azure-oauth-test-secret',  # noqa: S106
        client_type=AuthApplication.CLIENT_CONFIDENTIAL,
        authorization_grant_type=AuthApplication.GRANT_AUTHORIZATION_CODE,
        redirect_uris='https://azure-client.example/callback',
        social_auth_backend='azure_ad',
    )

    authorize_url = (
        '/o/authorize/'
        '?response_type=code&client_id=azure-oauth-test-client'
        '&redirect_uri=https://azure-client.example/callback'
        '&scope=openid'
        '&code_challenge=nmTAQE68zHR1tT6_Nqc26hpDPpI1tCtIVA0PWjgypT4'
        '&code_challenge_method=S256'
    )
    r = client.get(authorize_url)
    assert r.status_code == 302
    # Positively assert the pre-existing fallback: redirect to the internal
    # social:begin URL for azure_ad, with the full original /o/authorize/ URL
    # preserved in ?next= so login can resume with client_id, redirect_uri and
    # PKCE params intact. A prefix check would silently accept a truncated
    # next= that drops these params.
    parsed = urlparse(r['Location'])
    assert parsed.path == '/auth/login/azure_ad/', (
        f'azure_ad OAuth app should fall through to the internal social:begin '
        f'redirect (not the nzcportal shortcut, admin login, or a loop); '
        f'got path {parsed.path!r}. If handle_no_permission() is widened '
        f'beyond nzcportal, this breaks.'
    )
    next_value = parse_qs(parsed.query).get('next', [''])[0]
    assert next_value == authorize_url, (
        f'next= must preserve the full original OAuth authorize URL so login '
        f'can resume with all OAuth params intact; got {next_value!r}, '
        f'expected {authorize_url!r}.'
    )


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


def _chooser_labels(user, realm, rf) -> set[str]:
    request = rf.get('/admin/')
    request.user = user
    ctx = RealmContext(realm=realm, user=user)
    with realm_context.activate(ctx):
        items = instance_chooser.menu_items_for_request(request)
    return {item.label for item in items}


def test_instance_chooser_omits_hidden_instances(rf) -> None:
    admin = UserFactory.create(is_staff=True, is_superuser=True)
    visible_a = InstanceConfigFactory.create(identifier='visible-a', name='Visible A')
    InstanceConfigFactory.create(identifier='visible-b', name='Visible B')
    InstanceConfigFactory.create(identifier='hidden-one', name='Hidden One', is_hidden=True)

    labels = _chooser_labels(admin, visible_a, rf)

    assert 'Visible A' in labels
    assert 'Visible B' in labels
    assert 'Hidden One' not in labels


def test_instance_chooser_keeps_active_hidden_instance(rf) -> None:
    # A user currently on a hidden instance must still see it (and be able to
    # switch away), so the active realm is exempt from the filter.
    admin = UserFactory.create(is_staff=True, is_superuser=True)
    InstanceConfigFactory.create(identifier='visible', name='Visible')
    hidden = InstanceConfigFactory.create(identifier='hidden', name='Hidden', is_hidden=True)

    labels = _chooser_labels(admin, hidden, rf)

    assert 'Hidden' in labels
    assert 'Visible' in labels


def test_hidden_instance_still_reachable() -> None:
    # The hiding is listing-only: it does not touch get_adminable_instances(),
    # which is the authorization gate for directly switching to an instance.
    admin = UserFactory.create(is_staff=True, is_superuser=True)
    hidden = InstanceConfigFactory.create(identifier='hidden', name='Hidden', is_hidden=True)

    assert admin.user_is_admin_for_instance(hidden)
    assert hidden in admin.get_adminable_instances()


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
