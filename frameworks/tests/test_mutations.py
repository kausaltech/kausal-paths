"""Tests for the CADS self-service GraphQL mutations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from frameworks.tests.factories import FrameworkFactory
from nodes.models import InstanceConfig
from users.models import User

if TYPE_CHECKING:
    from django.test import Client

    from paths.tests.graphql import PathsTestClient

    from frameworks.models import Framework


gql = str

pytestmark = pytest.mark.django_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def framework() -> Framework:
    return FrameworkFactory.create(
        identifier='testfw',
        name='Test Framework',
        public_base_fqdn='testfw.example.com',
        allow_user_registration=True,
        allow_instance_creation=True,
    )


@pytest.fixture
def closed_framework() -> Framework:
    return FrameworkFactory.create(
        identifier='closedfw',
        name='Closed Framework',
        public_base_fqdn='closedfw.example.com',
        allow_user_registration=False,
        allow_instance_creation=False,
    )


@pytest.fixture
def gql_client(client: Client) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    return PathsTestClient(client)


@pytest.fixture
def authenticated_gql_client(client: Client, framework: Framework) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    user = User.objects.create_user(username='existing', email='admin@test.com', password='testpass123!', is_staff=False)
    client.force_login(user)
    return PathsTestClient(client)


# ---------------------------------------------------------------------------
# registerUser
# ---------------------------------------------------------------------------

REGISTER_USER = gql("""
mutation RegisterUser($input: RegisterUserInput!) {
    registerUser(input: $input) {
        ... on RegisterUserResult {
            userId
            email
        }
        ... on OperationInfo {
            messages { message }
        }
    }
}
""")


def test_register_user_success(gql_client: PathsTestClient, framework: Framework) -> None:
    data = gql_client.query_data(
        REGISTER_USER,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'email': 'newuser@example.com',
                'password': 'SecurePass123!',
                'firstName': 'Test',
                'lastName': 'User',
            },
        },
    )
    result = data['registerUser']
    assert result['email'] == 'newuser@example.com'
    assert result['userId'] is not None

    user = User.objects.get(email='newuser@example.com')
    assert user.first_name == 'Test'
    assert user.last_name == 'User'
    assert user.is_staff is False
    assert user.check_password('SecurePass123!')


def test_register_user_duplicate_email(gql_client: PathsTestClient, framework: Framework) -> None:
    User.objects.create_user(username='existing', email='existing@example.com', password='pass123!')

    gql_client.query_errors(
        REGISTER_USER,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'email': 'existing@example.com',
                'password': 'SecurePass123!',
            },
        },
        assert_error_message='already exists',
    )


def test_register_user_weak_password(gql_client: PathsTestClient, framework: Framework) -> None:
    gql_client.query_errors(
        REGISTER_USER,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'email': 'newuser@example.com',
                'password': '123',
            },
        },
        assert_error_message='Invalid password',
    )


def test_register_user_framework_disallows(gql_client: PathsTestClient, closed_framework: Framework) -> None:
    gql_client.query_errors(
        REGISTER_USER,
        variables={
            'input': {
                'frameworkId': closed_framework.identifier,
                'email': 'newuser@example.com',
                'password': 'SecurePass123!',
            },
        },
        assert_error_message='not allowed',
    )


def test_register_user_unknown_framework(gql_client: PathsTestClient) -> None:
    gql_client.query_errors(
        REGISTER_USER,
        variables={
            'input': {
                'frameworkId': 'nonexistent',
                'email': 'newuser@example.com',
                'password': 'SecurePass123!',
            },
        },
        assert_error_message='not found',
    )


# ---------------------------------------------------------------------------
# createInstance
# ---------------------------------------------------------------------------

CREATE_INSTANCE = gql("""
mutation CreateInstance($input: CreateInstanceInput!) {
    createInstance(input: $input) {
        ... on CreateInstanceResult {
            instanceId
            instanceName
        }
        ... on OperationInfo {
            messages { message }
        }
    }
}
""")


def test_create_instance_success(authenticated_gql_client: PathsTestClient, framework: Framework) -> None:
    data = authenticated_gql_client.query_data(
        CREATE_INSTANCE,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'name': 'My City Model',
                'identifier': 'my-city',
                'organizationName': 'My City',
            },
        },
    )
    result = data['createInstance']
    assert result['instanceId'] == 'my-city'
    assert result['instanceName'] == 'My City Model'

    ic = InstanceConfig.objects.get(identifier='my-city')
    assert ic.config_source == 'database'
    assert ic.has_framework_config()
    assert ic.framework_config.framework == framework
    assert ic.admin_group is not None


def test_create_instance_unauthenticated(gql_client: PathsTestClient, framework: Framework) -> None:
    gql_client.query_errors(
        CREATE_INSTANCE,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'name': 'Anon City',
                'identifier': 'anon-city',
                'organizationName': 'Anon',
            },
        },
        assert_error_message='Authentication required',
    )


def test_create_instance_framework_disallows(authenticated_gql_client: PathsTestClient, closed_framework: Framework) -> None:
    authenticated_gql_client.query_errors(
        CREATE_INSTANCE,
        variables={
            'input': {
                'frameworkId': closed_framework.identifier,
                'name': 'Blocked City',
                'identifier': 'blocked-city',
                'organizationName': 'Blocked',
            },
        },
        assert_error_message='not allowed',
    )


# ---------------------------------------------------------------------------
# FrameworkLandingBlock query
# ---------------------------------------------------------------------------


PAGES_QUERY = gql("""
query Pages {
    pages {
        ... on InstanceRootPage {
            body {
                ... on FrameworkLandingBlock {
                    heading
                    body
                    ctaLabel
                    ctaUrl
                    framework {
                        identifier
                        allowUserRegistration
                        allowInstanceCreation
                    }
                }
            }
        }
    }
}
""")


def test_landing_block_exposes_framework(client: Client, framework: Framework) -> None:
    import json

    from wagtail.models import Locale, Page, Site

    from paths.tests.graphql import PathsTestClient

    from nodes.defs.instance_defs import InstanceSpec, YearsSpec
    from nodes.models import InstanceConfig
    from orgs.tests.factories import OrganizationFactory
    from pages.models import InstanceRootPage

    org = OrganizationFactory.create()
    spec = InstanceSpec(
        primary_language='en',
        owner='Test',
        years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
    )
    ic = InstanceConfig.objects.create(
        name='Landing Test',
        identifier='landing-test',
        primary_language='en',
        other_languages=[],
        organization=org,
        config_source='database',
        spec=spec,
    )
    locale, _ = Locale.objects.get_or_create(language_code='en')
    root = Page.get_first_root_node()
    body = json.dumps([
        {
            'type': 'framework_landing',
            'value': {
                'heading': 'Welcome',
                'body': '<p>Hello</p>',
                'cta_label': 'Go',
                'cta_url': '/register',
                'framework_identifier': framework.identifier,
            },
        }
    ])
    page = root.add_child(
        instance=InstanceRootPage(
            locale=locale,
            title='Landing Test',
            slug='landing-test',
            url_path='',
            body=body,
        )
    )
    site = Site.objects.create(site_name='Landing Test', hostname='landing-test.localhost', root_page=page)
    ic.site = site
    ic.save(update_fields=['site'])

    gql_client = PathsTestClient(client)
    gql_client.set_instance(ic)
    data = gql_client.query_data(PAGES_QUERY)

    pages = data['pages']
    assert len(pages) >= 1
    root_page_data = pages[0]
    assert 'body' in root_page_data
    blocks = root_page_data['body']
    assert len(blocks) == 1
    block = blocks[0]
    assert block['heading'] == 'Welcome'
    assert block['ctaLabel'] == 'Go'
    fw_data = block['framework']
    assert fw_data is not None
    assert fw_data['identifier'] == framework.identifier
    assert fw_data['allowUserRegistration'] is True
    assert fw_data['allowInstanceCreation'] is True


def test_create_instance_duplicate_identifier(authenticated_gql_client: PathsTestClient, framework: Framework) -> None:
    # Create first
    authenticated_gql_client.query_data(
        CREATE_INSTANCE,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'name': 'First City',
                'identifier': 'dupe-city',
                'organizationName': 'First',
            },
        },
    )
    # Try duplicate
    authenticated_gql_client.query_errors(
        CREATE_INSTANCE,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'name': 'Second City',
                'identifier': 'dupe-city',
                'organizationName': 'Second',
            },
        },
        assert_error_message='already exists',
    )
