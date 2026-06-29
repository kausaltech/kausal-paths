"""Tests for the instance user-management mutations and related schema bits."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from frameworks.tests.factories import FrameworkConfigFactory
from nodes.models import InstanceConfig, InstanceInvitation
from users.models import User

if TYPE_CHECKING:
    from django.test import Client

    from paths.tests.graphql import PathsTestClient

    from frameworks.models import Framework


pytestmark = pytest.mark.django_db


@pytest.fixture
def gql_client(client: Client) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    return PathsTestClient(client)


@pytest.fixture
def framework_with_invite_url() -> Framework:
    from frameworks.tests.factories import FrameworkFactory

    return FrameworkFactory.create(
        identifier='testfw',
        name='Test Framework',
        accept_invitation_url='https://app.example.com/register?token={code}',
        enable_user_management=True,
    )


@pytest.fixture
def managed_instance(instance_config: InstanceConfig, framework_with_invite_url: Framework) -> InstanceConfig:
    """Build an instance with user management enabled, hooked up to a framework + roles."""
    instance_config.create_or_update_instance_groups()
    spec = instance_config.spec
    assert spec is not None
    spec.features.enable_user_management = True
    instance_config.spec = spec
    instance_config.save(update_fields=['spec'])
    FrameworkConfigFactory.create(
        framework=framework_with_invite_url,
        instance_config=instance_config,
        baseline_year=2020,
    )
    instance_config.refresh_from_db()
    return instance_config


@pytest.fixture
def owner_user(managed_instance: InstanceConfig) -> User:
    user = User.objects.create_user(username='owner', email='owner@example.com', password='Sufficient1!', is_staff=False)
    managed_instance.permission_policy().admin_role.assign_user(managed_instance, user)
    managed_instance.owned_by = user
    managed_instance.save(update_fields=['owned_by'])
    return user


@pytest.fixture
def admin_user(managed_instance: InstanceConfig) -> User:
    user = User.objects.create_user(username='admin', email='admin@example.com', password='Sufficient1!', is_staff=False)
    managed_instance.permission_policy().admin_role.assign_user(managed_instance, user)
    return user


@pytest.fixture
def other_user() -> User:
    return User.objects.create_user(username='other', email='other@example.com', password='Sufficient1!', is_staff=False)


# ----------------------------------------------------------------------
# addUserToInstance
# ----------------------------------------------------------------------


ADD_USER_QUERY = """
mutation Add($instanceId: ID!, $email: String!) {
  instanceAdmin(instanceId: $instanceId) {
    addUserToInstance(email: $email) {
      __typename
      ... on User { id email }
      ... on UserNotFoundError { email }
    }
  }
}
"""


def test_add_user_returns_user_when_exists(
    gql_client: PathsTestClient,
    client: Client,
    managed_instance: InstanceConfig,
    owner_user: User,
    other_user: User,
):
    client.force_login(owner_user)
    data = gql_client.query_data(
        ADD_USER_QUERY,
        variables={'instanceId': managed_instance.identifier, 'email': other_user.email},
    )
    payload = data['instanceAdmin']['addUserToInstance']
    assert payload['__typename'] == 'User'
    assert payload['email'] == other_user.email
    pp = managed_instance.permission_policy()
    assert pp.is_admin(other_user, managed_instance)


def test_add_user_returns_not_found_when_no_such_email(
    gql_client: PathsTestClient,
    client: Client,
    managed_instance: InstanceConfig,
    owner_user: User,
):
    client.force_login(owner_user)
    data = gql_client.query_data(
        ADD_USER_QUERY,
        variables={'instanceId': managed_instance.identifier, 'email': 'ghost@nowhere.test'},
    )
    payload = data['instanceAdmin']['addUserToInstance']
    assert payload['__typename'] == 'UserNotFoundError'
    assert payload['email'] == 'ghost@nowhere.test'


def test_add_user_requires_admin(
    gql_client: PathsTestClient,
    client: Client,
    managed_instance: InstanceConfig,
    other_user: User,
):
    client.force_login(other_user)
    errors = gql_client.query_errors(
        ADD_USER_QUERY,
        variables={'instanceId': managed_instance.identifier, 'email': 'whoever@example.com'},
    )
    assert any('Permission denied' in e['message'] for e in errors)


def test_add_user_blocked_when_feature_disabled(
    gql_client: PathsTestClient,
    client: Client,
    managed_instance: InstanceConfig,
    owner_user: User,
    other_user: User,
):
    spec = managed_instance.spec
    assert spec is not None
    spec.features.enable_user_management = False
    managed_instance.spec = spec
    managed_instance.save(update_fields=['spec'])
    client.force_login(owner_user)
    errors = gql_client.query_errors(
        ADD_USER_QUERY,
        variables={'instanceId': managed_instance.identifier, 'email': other_user.email},
    )
    assert any('user management' in e['message'].lower() for e in errors)


# ----------------------------------------------------------------------
# inviteUserToInstance / registerUser via invitation
# ----------------------------------------------------------------------


INVITE_QUERY = """
mutation Invite($instanceId: ID!, $email: String!) {
  instanceAdmin(instanceId: $instanceId) {
    inviteUserToInstance(email: $email) {
      __typename
      ... on InstanceInvitation { id email expiresAt }
    }
  }
}
"""


def test_invite_creates_invitation_and_triggers_email(
    gql_client: PathsTestClient,
    client: Client,
    managed_instance: InstanceConfig,
    owner_user: User,
):
    client.force_login(owner_user)
    with patch('users.graphql.mutations.send_instance_invitation') as send_mock:
        data = gql_client.query_data(
            INVITE_QUERY,
            variables={'instanceId': managed_instance.identifier, 'email': 'guest@example.com'},
        )
    payload = data['instanceAdmin']['inviteUserToInstance']
    assert payload['__typename'] == 'InstanceInvitation'
    assert payload['email'] == 'guest@example.com'
    send_mock.assert_called_once()
    inv = InstanceInvitation.objects.get(instance_config=managed_instance, email='guest@example.com')
    assert inv.is_valid()


def test_invite_rejects_existing_user_email(
    gql_client: PathsTestClient,
    client: Client,
    managed_instance: InstanceConfig,
    owner_user: User,
    other_user: User,
):
    client.force_login(owner_user)
    errors = gql_client.query_errors(
        INVITE_QUERY,
        variables={'instanceId': managed_instance.identifier, 'email': other_user.email},
    )
    assert any('already exists' in e['message'] for e in errors)


REGISTER_QUERY = """
mutation Register($input: RegisterUserInput!) {
  registerUser(input: $input) {
    __typename
    ... on RegisterUserResult { userId email }
  }
}
"""


def test_register_with_invitation_token_grants_admin(
    gql_client: PathsTestClient,
    managed_instance: InstanceConfig,
    owner_user: User,
):
    inv = InstanceInvitation.objects.create(
        instance_config=managed_instance,
        email='claim@example.com',
        created_by=owner_user,
        last_modified_by=owner_user,
    )
    data = gql_client.query_data(
        REGISTER_QUERY,
        variables={
            'input': {
                'email': 'claim@example.com',
                'password': 'Sufficient1!',
                'invitationToken': inv.token,
            },
        },
    )
    new_user = User.objects.get(email='claim@example.com')
    assert data['registerUser']['userId'] == str(new_user.uuid)
    inv.refresh_from_db()
    assert inv.accepted_at is not None
    assert inv.accepted_by == new_user
    assert managed_instance.permission_policy().is_admin(new_user, managed_instance)


def test_register_with_wrong_email_fails(
    gql_client: PathsTestClient,
    managed_instance: InstanceConfig,
    owner_user: User,
):
    inv = InstanceInvitation.objects.create(
        instance_config=managed_instance,
        email='claim@example.com',
        created_by=owner_user,
        last_modified_by=owner_user,
    )
    errors = gql_client.query_errors(
        REGISTER_QUERY,
        variables={
            'input': {
                'email': 'mismatch@example.com',
                'password': 'Sufficient1!',
                'invitationToken': inv.token,
            },
        },
    )
    assert any('does not match' in e['message'] for e in errors)


def test_register_double_use_token_fails(
    gql_client: PathsTestClient,
    managed_instance: InstanceConfig,
    owner_user: User,
):
    inv = InstanceInvitation.objects.create(
        instance_config=managed_instance,
        email='claim@example.com',
        created_by=owner_user,
        last_modified_by=owner_user,
    )
    gql_client.query_data(
        REGISTER_QUERY,
        variables={
            'input': {
                'email': 'claim@example.com',
                'password': 'Sufficient1!',
                'invitationToken': inv.token,
            },
        },
    )
    # Trying to register a second time with the same token (and a different email)
    errors = gql_client.query_errors(
        REGISTER_QUERY,
        variables={
            'input': {
                'email': 'other.claim@example.com',
                'password': 'Sufficient1!',
                'invitationToken': inv.token,
            },
        },
    )
    assert any('invalid' in e['message'].lower() or 'expired' in e['message'].lower() for e in errors)


# ----------------------------------------------------------------------
# removeUserFromInstance
# ----------------------------------------------------------------------


REMOVE_USER_QUERY = """
mutation Remove($instanceId: ID!, $userId: ID!) {
  instanceAdmin(instanceId: $instanceId) {
    removeUserFromInstance(userId: $userId) { __typename }
  }
}
"""


def test_remove_user_clears_role_and_selected_instance(
    gql_client: PathsTestClient,
    client: Client,
    managed_instance: InstanceConfig,
    owner_user: User,
    admin_user: User,
):
    admin_user.selected_instance = managed_instance
    admin_user.save(update_fields=['selected_instance'])
    client.force_login(owner_user)
    gql_client.query_data(
        REMOVE_USER_QUERY,
        variables={'instanceId': managed_instance.identifier, 'userId': str(admin_user.uuid)},
    )
    assert not managed_instance.permission_policy().is_admin(admin_user, managed_instance)
    admin_user.refresh_from_db()
    assert admin_user.selected_instance_id is None


def test_remove_user_cannot_remove_owner(
    gql_client: PathsTestClient,
    client: Client,
    managed_instance: InstanceConfig,
    owner_user: User,
):
    client.force_login(owner_user)
    errors = gql_client.query_errors(
        REMOVE_USER_QUERY,
        variables={'instanceId': managed_instance.identifier, 'userId': str(owner_user.uuid)},
    )
    assert any('owner' in e['message'].lower() for e in errors)


def test_remove_user_requires_owner_not_just_admin(
    gql_client: PathsTestClient,
    client: Client,
    managed_instance: InstanceConfig,
    admin_user: User,
    other_user: User,
):
    managed_instance.permission_policy().admin_role.assign_user(managed_instance, other_user)
    client.force_login(admin_user)
    errors = gql_client.query_errors(
        REMOVE_USER_QUERY,
        variables={'instanceId': managed_instance.identifier, 'userId': str(other_user.uuid)},
    )
    assert any('owner' in e['message'].lower() or 'superuser' in e['message'].lower() for e in errors)


# ----------------------------------------------------------------------
# removeInvitation
# ----------------------------------------------------------------------


REMOVE_INV_QUERY = """
mutation RemoveInv($instanceId: ID!, $invitationId: ID!) {
  instanceAdmin(instanceId: $instanceId) {
    removeInvitation(invitationId: $invitationId) { __typename }
  }
}
"""


def test_remove_invitation_soft_deletes(
    gql_client: PathsTestClient,
    client: Client,
    managed_instance: InstanceConfig,
    owner_user: User,
):
    inv = InstanceInvitation.objects.create(
        instance_config=managed_instance,
        email='to-revoke@example.com',
        created_by=owner_user,
        last_modified_by=owner_user,
    )
    client.force_login(owner_user)
    gql_client.query_data(
        REMOVE_INV_QUERY,
        variables={'instanceId': managed_instance.identifier, 'invitationId': str(inv.uuid)},
    )
    inv = InstanceInvitation.objects_including_soft_deleted.get(pk=inv.pk)
    assert inv.is_soft_deleted
    assert inv.soft_deleted_by == owner_user


# ----------------------------------------------------------------------
# UserType.editable_instances
# ----------------------------------------------------------------------


EDITABLE_QUERY = """
query Me($frameworkId: ID) {
  me { editableInstances(frameworkId: $frameworkId) { id identifier editor { configSource } } }
}
"""


def test_editable_instances_includes_owned_and_admin_instances(
    gql_client: PathsTestClient,
    client: Client,
    managed_instance: InstanceConfig,
    owner_user: User,
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_get_instance(*args: object, **kwargs: object) -> None:
        raise AssertionError('editableInstances must not hydrate runtime instances for config-only fields')

    monkeypatch.setattr(InstanceConfig, '_get_instance', fail_get_instance)
    monkeypatch.setattr(InstanceConfig, 'get_instance', fail_get_instance)

    client.force_login(owner_user)
    data = gql_client.query_data(EDITABLE_QUERY, variables={'frameworkId': None})
    identifiers = [i['identifier'] for i in data['me']['editableInstances']]
    assert managed_instance.identifier in identifiers
    editable_instance = next(i for i in data['me']['editableInstances'] if i['identifier'] == managed_instance.identifier)
    assert editable_instance['editor']['configSource'] == managed_instance.config_source


def test_editable_instances_filters_by_framework(
    gql_client: PathsTestClient,
    client: Client,
    managed_instance: InstanceConfig,
    owner_user: User,
    framework_with_invite_url: Framework,
):
    client.force_login(owner_user)
    data = gql_client.query_data(
        EDITABLE_QUERY,
        variables={'frameworkId': framework_with_invite_url.identifier},
    )
    identifiers = [i['identifier'] for i in data['me']['editableInstances']]
    assert managed_instance.identifier in identifiers

    data_other = gql_client.query_data(EDITABLE_QUERY, variables={'frameworkId': 'nonexistent'})
    assert data_other['me']['editableInstances'] == []
