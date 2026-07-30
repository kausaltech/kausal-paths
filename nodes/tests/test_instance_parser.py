from uuid import UUID, uuid3, uuid4

import pytest

from nodes.instance_parser import InstanceConfigParser

pytestmark = pytest.mark.django_db


def _parser(*, instance_uuid: UUID, node_uuids: dict[str, UUID] | None = None) -> InstanceConfigParser:
    return InstanceConfigParser(
        {'default_language': 'en'},
        instance_uuid=instance_uuid,
        node_uuids=node_uuids,
    )


def test_node_uuid_is_deterministic():
    instance_uuid = uuid4()

    first = _parser(instance_uuid=instance_uuid)._node_uuid('node')
    second = _parser(instance_uuid=instance_uuid)._node_uuid('node')

    assert first == second == uuid3(instance_uuid, 'node')


def test_authored_and_existing_node_uuids_take_precedence():
    instance_uuid = uuid4()
    existing_uuid = uuid4()
    authored_uuid = uuid4()

    assert _parser(instance_uuid=instance_uuid, node_uuids={'node': existing_uuid})._node_uuid('node') == existing_uuid
    assert (
        _parser(instance_uuid=instance_uuid, node_uuids={'node': existing_uuid})._node_uuid('node', authored_uuid)
        == authored_uuid
    )
