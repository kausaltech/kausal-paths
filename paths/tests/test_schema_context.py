from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest

from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

if TYPE_CHECKING:
    from collections.abc import Generator

    from paths.tests.graphql import PathsTestClient

    from nodes.models import InstanceConfig


pytestmark = pytest.mark.django_db


@pytest.fixture
def instance_gql_client(client) -> tuple[PathsTestClient, InstanceConfig]:
    from paths.tests.graphql import PathsTestClient

    instance = InstanceFactory.create()
    config = InstanceConfigFactory.create(identifier=instance.id, instance=instance)
    gql_client = PathsTestClient(client)
    gql_client.set_instance(config)
    return gql_client, config


def test_instance_scoped_query_does_not_eagerly_create_runtime(
    instance_gql_client: tuple[PathsTestClient, InstanceConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nodes.models import InstanceConfig

    gql_client, _config = instance_gql_client

    def fail_enter_instance_context(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError('config-only instance scope must not create a runtime Instance')

    monkeypatch.setattr(InstanceConfig, 'enter_instance_context', fail_enter_instance_context)

    data = gql_client.query_data('{ unit(value: "kg") { short } }')

    assert data['unit']['short']


def test_runtime_instance_is_created_once_per_request(
    instance_gql_client: tuple[PathsTestClient, InstanceConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nodes.models import InstanceConfig

    gql_client, _config = instance_gql_client
    original = InstanceConfig.enter_instance_context
    enter_count = 0

    @contextmanager
    def counted_enter_instance_context(self: InstanceConfig, *args: Any, **kwargs: Any) -> Generator[Any]:
        nonlocal enter_count
        enter_count += 1
        with original(self, *args, **kwargs) as instance:
            yield instance

    monkeypatch.setattr(InstanceConfig, 'enter_instance_context', counted_enter_instance_context)

    data = gql_client.query_data('{ instance { id } nodes { id } }')

    assert data['instance']['id']
    assert data['nodes'] == []
    assert enter_count == 1
