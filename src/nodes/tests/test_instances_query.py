"""The root ``instances`` query: a signed-in user's viewable instances."""

from typing import TYPE_CHECKING

import pytest

from paths.tests.graphql import PathsTestClient

from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from django.test import Client

    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db

QUERY = '{ instances { identifier isLocked editor { configSource } } }'


def test_anonymous_request_is_refused(client: Client, instance_config: InstanceConfig) -> None:
    errors = PathsTestClient(client).query_errors(QUERY, assert_error_message='signed in')

    assert errors[0].get('path') == ['instances']


def test_signed_in_user_sees_viewable_instances_sorted(client: Client, instance_config: InstanceConfig) -> None:
    client.force_login(UserFactory.create())

    data = PathsTestClient(client).query_data(QUERY)

    identifiers = [row['identifier'] for row in data['instances']]
    assert instance_config.identifier in identifiers, 'a public instance is viewable by any signed-in user'
    assert identifiers == sorted(identifiers)
    row = next(r for r in data['instances'] if r['identifier'] == instance_config.identifier)
    assert row['editor'] is None, 'view permission alone does not open the editor payload'


def test_superuser_sees_editor_payload(client: Client, instance_config: InstanceConfig) -> None:
    client.force_login(UserFactory.create(is_superuser=True))

    data = PathsTestClient(client).query_data(QUERY)

    row = next(r for r in data['instances'] if r['identifier'] == instance_config.identifier)
    assert row['editor'] == {'configSource': instance_config.config_source}
