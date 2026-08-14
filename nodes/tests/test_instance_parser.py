from uuid import UUID, uuid3, uuid4

import pytest

from nodes.instance_export_sync import compile_instance_export_from_yaml
from nodes.instance_parser import InstanceConfigParser, parse_instance_snapshot
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory

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


def test_yaml_short_description_is_rendered_at_snapshot_boundary():
    snapshot = parse_instance_snapshot(
        {
            'id': 'test',
            'default_language': 'en',
            'name': 'Test',
            'owner': 'Owner',
            'target_year': 2030,
            'reference_year': 2020,
            'minimum_historical_year': 2010,
            'nodes': [
                {
                    'id': 'node',
                    'type': 'generic.GenericNode',
                    'name': 'Node',
                    'description': '**Rich**',
                    'unit': 'kg/a',
                    'quantity': 'mass',
                }
            ],
        },
        instance_uuid=uuid4(),
    )

    short_description = snapshot.nodes[0].short_description
    assert short_description is not None
    assert short_description.i18n == {'en': '<p><strong>Rich</strong></p>\n'}


def test_compile_instance_export_preserves_identity_without_db_metadata(tmp_path):
    instance_config = InstanceConfigFactory.create(identifier='test', name='Test', config_source='yaml')
    existing = NodeConfigFactory.create(instance=instance_config, identifier='node', name='Database name')
    yaml_path = tmp_path / 'test.yaml'
    yaml_path.write_text(
        """
id: test
default_language: en
name: Test
owner: Owner
target_year: 2030
reference_year: 2020
minimum_historical_year: 2010
nodes:
- id: node
  type: generic.GenericNode
  name: YAML name
  unit: kg/a
  quantity: mass
""".lstrip()
    )

    export = compile_instance_export_from_yaml(instance_config, yaml_path)

    assert export.datasets == []
    assert len(export.instance.nodes) == 1
    exported = export.instance.nodes[0]
    assert exported.uuid == existing.uuid
    assert str(exported.name) == 'YAML name'
