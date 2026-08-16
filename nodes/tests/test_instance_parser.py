from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid3, uuid4

import pytest

from nodes.instance_export_sync import compile_instance_export_from_yaml
from nodes.instance_loader import InstanceYAMLConfig
from nodes.instance_parser import InstanceConfigParser, parse_instance_snapshot
from nodes.spec_export import _export_node_params
from nodes.tests.factories import AdditiveActionFactory, InstanceConfigFactory, NodeConfigFactory

if TYPE_CHECKING:
    from nodes.instance_serialization import InstanceSnapshot

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


def test_datasets_key_parses_into_typed_catalog_entries():
    from datasets.validation_rules import NoGapsRule, ValueRangeRule
    from nodes.instance_parser import InstanceParseError

    config = {
        'id': 'test',
        'default_language': 'en',
        'name': 'Test',
        'owner': 'Owner',
        'target_year': 2030,
        'reference_year': 2020,
        'minimum_historical_year': 2010,
        'datasets': [
            {
                'id': 'test/energy',
                'metrics': [
                    {
                        'id': 'amount',
                        'validation_rules': [
                            {'kind': 'no_gaps', 'enforcement': 'block_publish'},
                            {'kind': 'value_range', 'enforcement': 'block_edit', 'min': 0},
                        ],
                    },
                ],
            },
        ],
    }
    instance_uuid = uuid4()

    snapshot = parse_instance_snapshot(config, instance_uuid=instance_uuid)

    (ds_meta,) = snapshot.datasets
    assert ds_meta.identifier == 'test/energy'
    (metric_meta,) = ds_meta.metrics
    assert metric_meta.identifier == 'amount'
    no_gaps, value_range = metric_meta.validation_rules
    assert isinstance(no_gaps, NoGapsRule)
    assert isinstance(value_range, ValueRangeRule)
    assert value_range.min == 0.0

    # Catalog UUIDs are parse-invented but deterministic per instance.
    again = parse_instance_snapshot(config, instance_uuid=instance_uuid)
    assert again.datasets[0].id == ds_meta.id
    assert again.datasets[0].metrics[0].id == metric_meta.id

    bad = dict(config)
    bad['datasets'] = [{'id': 'test/energy', 'metrics': [{'id': 'amount', 'validation_rules': [{'kind': 'nope'}]}]}]
    with pytest.raises(InstanceParseError, match='Invalid validation rule'):
        parse_instance_snapshot(bad, instance_uuid=instance_uuid)


def _action_snapshot(*, params: list[dict[str, Any]] | None = None) -> InstanceSnapshot:
    action: dict[str, Any] = {
        'id': 'action',
        'type': 'simple.AdditiveAction',
        'name': 'Action',
        'unit': 'kg/a',
        'quantity': 'mass',
    }
    if params is not None:
        action['params'] = params
    return parse_instance_snapshot(
        {
            'id': 'test',
            'default_language': 'en',
            'name': 'Test',
            'owner': 'Owner',
            'target_year': 2030,
            'reference_year': 2020,
            'minimum_historical_year': 2010,
            'actions': [action],
        },
        instance_uuid=uuid4(),
    )


def test_implicit_action_enabled_parameter_is_not_persisted():
    snapshot = _action_snapshot()

    assert snapshot.nodes[0].spec is not None
    assert snapshot.nodes[0].spec.params == []
    assert snapshot.spec.scenarios[0].param_values == {'action.enabled': False}


def test_authored_action_enabled_parameter_is_persisted():
    snapshot = _action_snapshot(params=[{'id': 'enabled', 'value': True}])

    assert snapshot.nodes[0].spec is not None
    assert [param.local_id for param in snapshot.nodes[0].spec.params] == ['enabled']


def test_runtime_export_omits_implicit_action_enabled_parameter():
    action = AdditiveActionFactory.create()

    assert _export_node_params(action) == []


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


def test_include_nodes_editable_applies_to_nodes_and_actions(tmp_path):
    module_path = tmp_path / 'module.yaml'
    module_path.write_text(
        """
nodes:
- id: included_node
  type: generic.GenericNode
  name: Included node
  unit: kg/a
  quantity: mass
actions:
- id: included_action
  type: simple.AdditiveAction
  name: Included action
  unit: kg/a
  quantity: mass
""".lstrip()
    )
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
include:
- file: module.yaml
  nodes_editable: false
""".lstrip()
    )

    yaml_config = InstanceYAMLConfig.load_for_entrypoint(yaml_path)
    assert yaml_config.data is not None
    assert yaml_config.data['nodes'][0]['is_editable'] is False
    assert yaml_config.data['actions'][0]['is_editable'] is False

    snapshot = parse_instance_snapshot(yaml_config.data, instance_uuid=uuid4())
    assert {node.identifier: node.is_editable for node in snapshot.nodes} == {
        'included_node': False,
        'included_action': False,
    }


def test_include_nodes_editable_must_be_boolean(tmp_path):
    (tmp_path / 'module.yaml').write_text('nodes: []\n')
    yaml_path = tmp_path / 'test.yaml'
    yaml_path.write_text(
        """
id: test
include:
- file: module.yaml
  nodes_editable: "false"
""".lstrip()
    )

    with pytest.raises(TypeError, match='nodes_editable must be a boolean'):
        InstanceYAMLConfig.load_for_entrypoint(yaml_path)
