"""Focused, non-ORM tests for the runtime node input contract."""

from pathlib import Path
from uuid import uuid4

import pytest

from nodes.constants import VALUE_COLUMN
from nodes.datasets import FixedDataset
from nodes.defs.port_def import InputPort, InputPortDeclaration
from nodes.exceptions import NodeError
from nodes.instance_loader import InstanceLoader
from nodes.instance_parser import parse_instance_snapshot
from nodes.tests.node_input_harness import bind, binding, frame, node_case
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def test_single_input_is_resolved_without_source_identity() -> None:
    port = InputPortDeclaration(role='energy')
    node = bind(node_case(port), [binding('energy', frame([1.0, 2.0]), source_id='dataset')])
    result = node.get_input(port)
    assert result is not None
    assert result[VALUE_COLUMN].to_list() == [1.0, 2.0]


def test_optional_missing_input_returns_none() -> None:
    port = InputPortDeclaration(role='optional', required=False)
    node = bind(node_case(port), [])
    assert node.get_input(port) is None


def test_required_and_conditionally_required_inputs_fail_at_accessor_boundary() -> None:
    required = InputPort.one('energy')
    optional = InputPort.optional('gas_mix')
    node = bind(node_case(required, optional), [])

    with pytest.raises(NodeError, match='Required input role'):
        node.get_input(required)
    with pytest.raises(NodeError, match='required in this configuration'):
        node.require_input(optional)


def test_duplicate_single_input_is_rejected() -> None:
    port = InputPortDeclaration(role='energy')
    node = bind(
        node_case(port),
        [binding('energy', frame([1.0])), binding('energy', frame([2.0]), position=1)],
    )
    with pytest.raises(NodeError):
        node.get_input(port)


def test_multi_input_iterator_preserves_position() -> None:
    port = InputPortDeclaration(role='values', multi=True)
    node = bind(
        node_case(port),
        [binding('values', frame([1.0]), position=1), binding('values', frame([2.0]), position=0)],
    )
    assert [df[VALUE_COLUMN][0] for df in node.iter_inputs(port)] == [2.0, 1.0]
    with pytest.raises(NodeError, match='use iter_inputs'):
        node.get_input(port)


def test_sum_aggregation_combines_multi_input() -> None:
    port = InputPortDeclaration(role='values', multi=True, aggregation='sum')
    node = bind(
        node_case(port),
        [
            binding('values', frame([1.0, 3.0]), position=0),
            binding('values', frame([2.0, 4.0]), position=1),
        ],
    )
    result = node.get_input(port)
    assert result is not None
    assert result[VALUE_COLUMN].to_list() == [3.0, 7.0]


def test_port_constructors_keep_cardinality_and_aggregation_separate() -> None:
    gas_mix = InputPort.optional('gas_mix')
    additive = InputPort.multi('additive', required=False, aggregation='sum')
    factors = InputPort.repeatable('factors', min_count=2)

    assert gas_mix.required is False
    assert gas_mix.min_count == 1
    assert additive.multi is True
    assert additive.required is False
    assert additive.aggregation == 'sum'
    assert factors.repeatable is True
    assert factors.min_count == 2


def test_instance_loader_attaches_graph_defined_runtime_bindings() -> None:
    config = {
        'id': 'runtime_inputs',
        'default_language': 'en',
        'supported_languages': [],
        'name': 'Runtime inputs',
        'owner': 'Owner',
        'target_year': 2030,
        'minimum_historical_year': 2010,
        'maximum_historical_year': 2020,
        'reference_year': 1990,
        'nodes': [
            {
                'id': 'source',
                'type': 'nodes.simple.AdditiveNode',
                'name': 'Source',
                'unit': 'kWh',
                'quantity': 'energy',
                'output_nodes': ['target'],
            },
            {
                'id': 'target',
                'type': 'nodes.simple.AdditiveNode',
                'name': 'Target',
                'unit': 'kWh',
                'quantity': 'energy',
            },
        ],
    }
    snapshot = parse_instance_snapshot(config, instance_uuid=uuid4())
    target = InstanceLoader(snapshot=snapshot).context.nodes['target']

    assert len(target.runtime_input_bindings) == 1
    (runtime_binding,) = target.runtime_input_bindings
    assert runtime_binding.port_role == 'additive'
    assert runtime_binding.source_kind == 'node'
    assert runtime_binding.source_id == 'source'
    assert runtime_binding.definition is not None


def test_instance_loader_does_not_resolve_stale_dataset_port_ids_for_unmigrated_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        'id': 'legacy_runtime_inputs',
        'default_language': 'en',
        'supported_languages': [],
        'name': 'Legacy runtime inputs',
        'owner': 'Owner',
        'target_year': 2030,
        'minimum_historical_year': 2010,
        'maximum_historical_year': 2020,
        'reference_year': 1990,
        'nodes': [
            {
                'id': 'target',
                'type': 'nodes.generic.GenericNode',
                'name': 'Target',
                'unit': 'dimensionless',
                'quantity': 'number',
                'input_datasets': ['some/data'],
            },
        ],
    }
    snapshot = parse_instance_snapshot(config, instance_uuid=uuid4())
    assert len(snapshot.bindings) == 1
    catalog_loader = object.__new__(InstanceLoader)
    catalog_loader.instance_config = None
    catalog_loader._stash_snapshot_bindings(snapshot)
    stale_binding = snapshot.bindings[0].model_copy(update={'port_id': uuid4()})
    snapshot = snapshot.model_copy(
        update={
            'bindings': [stale_binding],
            'datasets': list(catalog_loader._instance_graph.datasets),
        }
    )

    def make_fixed_dataset(loader: InstanceLoader, *_args, **_kwargs) -> list[FixedDataset]:
        return [
            FixedDataset(
                id='some/data',
                context=loader.context,
                unit=unit_registry.dimensionless,
                historical=[(2020, 1.0)],
                forecast=None,
                tags=[],
            )
        ]

    monkeypatch.setattr(InstanceLoader, '_make_node_datasets', make_fixed_dataset)

    target = InstanceLoader(snapshot=snapshot).context.nodes['target']

    assert target.input_port_declarations == ()
    assert target.runtime_input_bindings == ()


def test_yaml_graph_supplements_a_partial_embedded_dataset_catalog() -> None:
    config = {
        'id': 'partial_catalog',
        'default_language': 'en',
        'supported_languages': [],
        'name': 'Partial catalog',
        'owner': 'Owner',
        'target_year': 2030,
        'minimum_historical_year': 2010,
        'maximum_historical_year': 2020,
        'reference_year': 1990,
        'nodes': [
            {
                'id': 'target',
                'type': 'nodes.generic.GenericNode',
                'name': 'Target',
                'unit': 'dimensionless',
                'quantity': 'number',
                'input_datasets': ['some/data'],
            }
        ],
    }
    snapshot = parse_instance_snapshot(config, instance_uuid=uuid4())
    catalog_loader = object.__new__(InstanceLoader)
    catalog_loader.instance_config = None
    catalog_loader._stash_snapshot_bindings(snapshot)
    dataset = catalog_loader._instance_graph.datasets[0].model_copy(update={'metrics': ()})
    snapshot = snapshot.model_copy(update={'datasets': [dataset]})

    loader = object.__new__(InstanceLoader)
    loader.instance_config = None
    loader.yaml_file_path = Path('config.yaml')
    loader._stash_snapshot_bindings(snapshot)

    assert [metric.identifier for metric in loader._instance_graph.datasets[0].metrics] == ['Value']
