from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid3, uuid4

import pytest

from nodes.instance_export_sync import compile_instance_export_from_yaml
from nodes.instance_loader import InstanceYAMLConfig
from nodes.instance_parser import InstanceConfigParser, parse_instance_snapshot
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory

if TYPE_CHECKING:
    from nodes.defs.port_def import InputPortDef
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
    from datasets.validation_rules import NoGapsRule, RequiredCombinationsRule, ValueRangeRule
    from nodes.instance_parser import InstanceParseError

    config = {
        'id': 'test',
        'default_language': 'en',
        'name': 'Test',
        'owner': 'Owner',
        'target_year': 2030,
        'reference_year': 2020,
        'minimum_historical_year': 2010,
        'dimensions': [
            {
                'id': 'region',
                'label_en': 'Region',
                'categories': [
                    {'id': 'a', 'label_en': 'A'},
                    {'id': 'b', 'label_en': 'B'},
                ],
            },
        ],
        'shapes': [
            {
                'id': 'test/regions',
                'dimensions': ['region'],
                'combinations': [
                    {'id': 'region_a', 'categories': {'region': 'a'}},
                    {'id': 'region_b', 'categories': {'region': 'b'}},
                ],
            },
        ],
        'datasets': [
            {
                'id': 'test/energy',
                'is_editable': False,
                'shape': 'test/regions',
                'metrics': [
                    {
                        'id': 'amount',
                        'validation_rules': [
                            {'kind': 'no_gaps', 'enforcement': 'block_publish'},
                            {'kind': 'value_range', 'enforcement': 'block_edit', 'min': 0},
                            {
                                'kind': 'required_combinations',
                                'enforcement': 'block_publish',
                                'groups': [{'id': 'region', 'combinations': ['region_a', 'region_b']}],
                            },
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
    assert ds_meta.is_editable is False
    (metric_meta,) = ds_meta.metrics
    assert metric_meta.identifier == 'amount'
    no_gaps, value_range, required = metric_meta.validation_rules
    assert isinstance(no_gaps, NoGapsRule)
    assert isinstance(value_range, ValueRangeRule)
    assert value_range.min == 0.0
    assert isinstance(required, RequiredCombinationsRule)
    # Rule groups name the combinations of the dataset's shape.
    (shape,) = snapshot.spec.shapes
    assert ds_meta.shape_id == shape.uuid
    assert required.groups[0].combinations == [combination.uuid for combination in shape.combinations]
    assert ds_meta.category_domain_spec is None

    # Catalog UUIDs are parse-invented but deterministic per instance.
    again = parse_instance_snapshot(config, instance_uuid=instance_uuid)
    assert again.datasets[0].id == ds_meta.id
    assert again.datasets[0].metrics[0].id == metric_meta.id

    bad = dict(config)
    bad['datasets'] = [{'id': 'test/energy', 'metrics': [{'id': 'amount', 'validation_rules': [{'kind': 'nope'}]}]}]
    with pytest.raises(InstanceParseError, match='Invalid validation rule'):
        parse_instance_snapshot(bad, instance_uuid=instance_uuid)

    bad = dict(config)
    bad['datasets'] = [{'id': 'test/energy', 'category_domain': {'combinations': []}}]
    with pytest.raises(InstanceParseError, match='replaced by shapes'):
        parse_instance_snapshot(bad, instance_uuid=instance_uuid)

    bad = dict(config)
    bad['datasets'] = [{'id': 'test/energy', 'shape': 'missing'}]
    with pytest.raises(InstanceParseError, match='unknown shape missing'):
        parse_instance_snapshot(bad, instance_uuid=instance_uuid)

    bad = dict(config)
    bad['datasets'] = [{'id': 'test/energy', 'is_editable': 'false'}]
    with pytest.raises(InstanceParseError, match=r'is_editable.*must be a boolean'):
        parse_instance_snapshot(bad, instance_uuid=instance_uuid)

    assert ds_meta.default_quality is None
    graded = dict(config)
    graded['datasets'] = [{'id': 'test/energy', 'default_quality': {'scheme': 'bisko', 'level': 'B'}}]
    (graded_meta,) = parse_instance_snapshot(graded, instance_uuid=instance_uuid).datasets
    assert graded_meta.default_quality is not None
    assert (graded_meta.default_quality.scheme, graded_meta.default_quality.level) == ('bisko', 'B')

    bad = dict(config)
    bad['datasets'] = [{'id': 'test/energy', 'default_quality': 'B'}]
    with pytest.raises(InstanceParseError, match='must name a scheme and a level'):
        parse_instance_snapshot(bad, instance_uuid=instance_uuid)


def _action_snapshot(*, params: list[dict[str, Any]] | None = None, default_actions_enabled: bool = True) -> InstanceSnapshot:
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
            'scenarios': [
                {'id': 'default', 'name': 'Default', 'default': True, 'all_actions_enabled': default_actions_enabled},
                {'id': 'baseline', 'name': 'Baseline'},
            ],
        },
        instance_uuid=uuid4(),
    )


def test_implicit_action_enabled_parameter_is_not_persisted():
    snapshot = _action_snapshot()

    assert snapshot.nodes[0].spec is not None
    assert snapshot.nodes[0].spec.params == []
    assert snapshot.spec.scenarios[0].param_values == {}
    assert snapshot.spec.scenarios[1].param_values == {'action.enabled': False}


def test_default_scenario_disabling_actions_declares_enabled_parameter():
    snapshot = _action_snapshot(default_actions_enabled=False)

    assert snapshot.nodes[0].spec is not None
    assert [(param.local_id, param.value) for param in snapshot.nodes[0].spec.params] == [('enabled', False)]
    assert snapshot.spec.scenarios[0].param_values == {}


def test_authored_action_enabled_parameter_is_persisted():
    snapshot = _action_snapshot(params=[{'id': 'enabled', 'value': True}])

    assert snapshot.nodes[0].spec is not None
    assert [param.local_id for param in snapshot.nodes[0].spec.params] == ['enabled']


def _target_input_ports(target: dict[str, Any], sources: list[dict[str, Any]]) -> list[InputPortDef]:
    snapshot = parse_instance_snapshot(
        {
            'id': 'test',
            'default_language': 'en',
            'name': 'Test',
            'owner': 'Owner',
            'target_year': 2030,
            'reference_year': 2020,
            'minimum_historical_year': 2010,
            'nodes': [*sources, target],
        },
        instance_uuid=uuid4(),
    )
    node = next(n for n in snapshot.nodes if n.identifier == 'target')
    assert node.spec is not None
    return node.spec.input_ports


def _source(identifier: str, *, unit: str = 'kt/a', quantity: str = 'emissions') -> dict[str, Any]:
    return {'id': identifier, 'type': 'simple.AdditiveNode', 'name': identifier, 'unit': unit, 'quantity': quantity}


def _additive_target(input_nodes: list[Any]) -> dict[str, Any]:
    return {
        'id': 'target',
        'type': 'simple.AdditiveNode',
        'name': 'Target',
        'unit': 'kt/a',
        'quantity': 'emissions',
        'input_nodes': input_nodes,
    }


def test_parser_collapses_plain_additive_inputs_onto_one_multi_port():
    ports = _target_input_ports(_additive_target(['source_a', 'source_b']), [_source('source_a'), _source('source_b')])

    assert len(ports) == 1
    assert ports[0].multi is True
    assert ports[0].role == 'additive'


def test_parser_keeps_a_non_additive_input_on_its_own_port():
    ports = _target_input_ports(
        _additive_target(['additive_source', {'id': 'non_additive_source', 'tags': ['non_additive']}]),
        [_source('additive_source'), _source('non_additive_source')],
    )

    assert [port.multi for port in ports] == [True, False]


def test_parser_keeps_additive_inputs_single_when_units_are_incompatible():
    ports = _target_input_ports(
        _additive_target(['emissions_source', 'energy_source']),
        [_source('emissions_source'), _source('energy_source', unit='MWh/a', quantity='energy')],
    )

    assert [port.multi for port in ports] == [False, False]


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
datasets:
- id: module/reference
  is_editable: false
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
  dataset_replacements:
  - from: module/reference
    to: test/reference
""".lstrip()
    )

    yaml_config = InstanceYAMLConfig.load_for_entrypoint(yaml_path)
    assert yaml_config.data is not None
    assert yaml_config.data['nodes'][0]['is_editable'] is False
    assert yaml_config.data['actions'][0]['is_editable'] is False
    assert yaml_config.data['datasets'] == [{'id': 'test/reference', 'is_editable': False}]

    snapshot = parse_instance_snapshot(yaml_config.data, instance_uuid=uuid4())
    assert {node.identifier: node.is_editable for node in snapshot.nodes} == {
        'included_node': False,
        'included_action': False,
    }
    assert snapshot.datasets[0].identifier == 'test/reference'
    assert snapshot.datasets[0].is_editable is False


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


_CATEGORY_MODULE = """
dimensions:
- id: fuel
  label: Fuel
  categories:
  - id: petrol
    label_en: Petrol
    label_de: Benzin
    aliases: [benzin]
    color: '#111111'
    order: 1
  - id: diesel
    label_en: Diesel
    color: '#222222'
""".lstrip()


def _write_category_override_instance(tmp_path, overrides: str) -> Any:
    (tmp_path / 'module.yaml').write_text(_CATEGORY_MODULE)
    yaml_path = tmp_path / 'test.yaml'
    yaml_path.write_text(
        f"""
id: test
default_language: en
supported_languages: [de]
name: Test
owner: Owner
target_year: 2030
reference_year: 2020
minimum_historical_year: 2010
_palette:
  petrol: &petrol '#abcdef'
include:
- file: module.yaml
{overrides}""".lstrip()
    )
    return yaml_path


def test_category_overrides_restyle_included_dimension(tmp_path):
    yaml_path = _write_category_override_instance(
        tmp_path,
        """
category_overrides:
  fuel:
    petrol:
      color: *petrol
      label_de: Ottokraftstoff
      order: 5
    diesel:
      color: null
""",
    )

    yaml_config = InstanceYAMLConfig.load_for_entrypoint(yaml_path)
    assert yaml_config.data is not None
    assert 'category_overrides' not in yaml_config.data
    petrol, diesel = yaml_config.data['dimensions'][0]['categories']
    # Overridden fields change; ids, aliases and untouched labels stay the module's.
    assert petrol == {
        'id': 'petrol',
        'label_en': 'Petrol',
        'label_de': 'Ottokraftstoff',
        'aliases': ['benzin'],
        'color': '#abcdef',
        'order': 5,
    }
    assert diesel == {'id': 'diesel', 'label_en': 'Diesel'}

    parser = InstanceConfigParser(yaml_config.data, instance_uuid=uuid4())
    parser._parse_dimensions()
    assert parser.dimensions['fuel'].get('petrol').color == '#abcdef'


@pytest.mark.parametrize(
    ('overrides', 'error', 'match'),
    [
        ('  road:\n    petrol:\n      color: red\n', KeyError, "dimension 'road' is not defined"),
        ('  fuel:\n    lpg:\n      color: red\n', KeyError, "has no category 'lpg'"),
        ('  fuel:\n    petrol:\n      aliases: [x]\n', ValueError, "field 'aliases' cannot be overridden"),
        ('  fuel:\n    petrol:\n      id: gasoline\n', ValueError, "field 'id' cannot be overridden"),
        ('  fuel:\n    petrol:\n      order: first\n', TypeError, 'order must be int or null'),
        ('  fuel:\n    petrol:\n      color: 3\n', TypeError, 'color must be str or null'),
    ],
)
def test_category_overrides_refuse_what_they_cannot_apply(tmp_path, overrides: str, error: type[Exception], match: str):
    yaml_path = _write_category_override_instance(tmp_path, 'category_overrides:\n' + overrides)

    with pytest.raises(error, match=match):
        InstanceYAMLConfig.load_for_entrypoint(yaml_path)


def test_yaml_dataset_binding_ownership_and_value_contract_are_persisted_on_the_port() -> None:
    snapshot = parse_instance_snapshot(
        {
            'id': 'owned-data',
            'name': 'Owned data',
            'owner': 'Test',
            'default_language': 'en',
            'reference_year': 2020,
            'minimum_historical_year': 2020,
            'target_year': 2030,
            'nodes': [
                {
                    'id': 'consumer',
                    'name': 'Consumer',
                    'type': 'simple.AdditiveNode',
                    'quantity': 'energy',
                    'unit': 'kWh',
                    'input_datasets': [{'id': 'test/activity', 'column': 'Value', 'tags': ['data'], 'binding_owner': 'instance'}],
                    'input_validation': {'data': {'combinations': [{'categories': {}}]}},
                }
            ],
        },
        instance_uuid=uuid4(),
    )
    spec = snapshot.nodes[0].spec
    assert spec is not None
    (port,) = spec.input_ports
    assert port.binding_owner == 'instance'
    assert port.validation is not None
    assert port.validation.combinations[0].categories == {}
    assert spec.model_dump(mode='json')['input_ports'][0]['validation']['combinations'] == [{'categories': {}, 'qualifiers': {}}]


def test_default_scenario_does_not_rewrite_parameter_declarations() -> None:
    snapshot = parse_instance_snapshot(
        {
            'id': 'sparse-defaults',
            'name': 'Sparse defaults',
            'owner': 'Owner',
            'default_language': 'en',
            'reference_year': 2020,
            'minimum_historical_year': 2020,
            'target_year': 2030,
            'params': [{'id': 'local_factor', 'type': 'number', 'value': 2}],
            'scenarios': [
                {'id': 'default', 'name': 'Default', 'default': True, 'params': [{'id': 'local_factor', 'value': 4}]},
                {'id': 'other', 'name': 'Other'},
            ],
        },
        instance_uuid=uuid4(),
    )
    assert snapshot.spec.params[0].value == 2
    assert snapshot.spec.scenarios[0].param_values == {'local_factor': 4}
    assert snapshot.spec.scenarios[1].param_values == {}
