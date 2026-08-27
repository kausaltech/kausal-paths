"""Focused, non-ORM tests for the runtime node input contract."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common.polars import DataFrameMeta, to_ppdf
from nodes.actions.simple import AdditiveAction
from nodes.actions.values import BudgetingAction
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.datasets import FixedDataset
from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec
from nodes.defs.node_defs import ActionConfig, NodeSpec
from nodes.defs.port_def import InputPort, InputPortDeclaration, InputPortDef, OutputPortDef, pair_input_ports_to_outputs
from nodes.defs.transform_def import FilterDimensionOp
from nodes.exceptions import NodeError
from nodes.instance_graph import NodeMeta, build_instance_graph
from nodes.instance_loader import InstanceLoader
from nodes.instance_parser import parse_instance_snapshot
from nodes.instance_serialization import EdgeSnapshot, InstanceSnapshot, NodeSnapshot
from nodes.node import Node, NodeMetric
from nodes.runtime_input import RuntimeInputBinding
from nodes.simple import MultiplicativeNode
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.tests.node_input_harness import bind, binding, frame, node_case
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


class _EnabledAdditiveAction(AdditiveAction):
    def is_enabled(self) -> bool:
        return True


class _EnabledBudgetingAction(BudgetingAction):
    def is_enabled(self) -> bool:
        return True


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


def test_repeatable_input_ports_preserve_target_port_identity() -> None:
    port = InputPort.repeatable('metrics')
    energy_port_id = uuid4()
    cost_port_id = uuid4()
    node = bind(
        node_case(port),
        [
            binding('metrics', frame([1.0, 2.0], unit='kWh'), target_port_id=energy_port_id),
            binding(
                'metrics',
                frame([3.0, 4.0], unit='EUR'),
                position=1,
                target_port_id=cost_port_id,
            ),
        ],
    )

    runtime_ports = list(node.iter_input_ports(port))

    assert [runtime_port.id for runtime_port in runtime_ports] == [energy_port_id, cost_port_id]
    assert node.require_input_port(runtime_ports[0])[VALUE_COLUMN].to_list() == [1.0, 2.0]
    assert node.require_input_port(runtime_ports[1])[VALUE_COLUMN].to_list() == [3.0, 4.0]


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


def test_additive_action_receives_inline_values_through_its_input_port() -> None:
    config = {
        'id': 'runtime_action_input',
        'default_language': 'en',
        'supported_languages': [],
        'name': 'Runtime action input',
        'owner': 'Owner',
        'target_year': 2030,
        'minimum_historical_year': 2020,
        'maximum_historical_year': 2020,
        'reference_year': 2020,
        'nodes': [],
        'actions': [
            {
                'id': 'action',
                'type': 'nodes.actions.simple.AdditiveAction',
                'name': 'Action',
                'unit': 'kWh',
                'quantity': 'energy',
                'historical_values': [[2020, 1.0]],
                'forecast_values': [[2030, 2.0]],
            }
        ],
    }
    snapshot = parse_instance_snapshot(config, instance_uuid=uuid4())
    action = InstanceLoader(snapshot=snapshot).context.nodes['action']
    assert isinstance(action, AdditiveAction)

    (runtime_port,) = action.iter_input_ports(action.input_port)
    input_value = action.require_input_port(runtime_port)
    result = action.compute()

    assert input_value is not None
    assert input_value[VALUE_COLUMN].to_list() == [1.0, 2.0]
    assert result[VALUE_COLUMN].to_list() == [0.0, 0.0]
    assert [(binding.port_role, binding.source_kind) for binding in action.runtime_input_bindings] == [('input', 'dataset')]
    assert action.runtime_input_bindings[0].definition is None


@pytest.mark.parametrize('action_class', [_EnabledAdditiveAction, _EnabledBudgetingAction])
def test_action_reassembles_sparse_metrics_with_null_index_categories(
    action_class: type[AdditiveAction],
) -> None:
    instance = InstanceFactory.create(id='sparse_action_metrics', name='Sparse action metrics')
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name='Sparse action metrics')
    context = instance.context
    energy_unit = unit_registry.parse_units('kWh')
    fraction_unit = unit_registry.parse_units('%')
    outputs = [
        OutputPortDef(id=uuid4(), identifier='energy', column_id='energy', quantity='energy', unit=energy_unit),
        OutputPortDef(id=uuid4(), identifier='reduction', column_id='reduction', quantity='fraction', unit=fraction_unit),
    ]
    inputs = pair_input_ports_to_outputs([], outputs, role='input', keep_unpaired=False)
    spec = NodeSpec(
        type_config=ActionConfig(node_class='nodes.actions.simple.AdditiveAction'),
        input_ports=inputs,
        output_ports=outputs,
    )
    action = action_class(
        id='action',
        context=context,
        name=TranslatedString('Action', default_language='en'),
        output_metrics={
            'energy': NodeMetric(id='energy', column_id='energy', quantity='energy', unit=energy_unit),
            'reduction': NodeMetric(id='reduction', column_id='reduction', quantity='fraction', unit=fraction_unit),
        },
        spec=spec,
    )

    def sparse_frame(values: list[float | None], unit: str):
        raw = pl.DataFrame({
            YEAR_COLUMN: [2020, 2021],
            'asset': [None, 'ev_fleet'],
            VALUE_COLUMN: values,
            FORECAST_COLUMN: [False, True],
        })
        return to_ppdf(
            raw,
            DataFrameMeta(
                units={VALUE_COLUMN: unit_registry.parse_units(unit)},
                primary_keys=[YEAR_COLUMN, 'asset'],
            ),
        )

    action.bind_runtime_inputs(
        [
            binding('input', sparse_frame([1.0, None], 'kWh'), target_port_id=inputs[0].id),
            binding('input', sparse_frame([0.5, None], '%'), position=1, target_port_id=inputs[1].id),
        ],
        node_meta=NodeMeta(
            id=uuid4(),
            identifier='action',
            node_class_path='nodes.actions.simple.AdditiveAction',
            spec=spec,
        ),
    )

    result = action.compute_effect().sort([YEAR_COLUMN, 'asset'])

    assert len(result) == 2
    assert result.paths.index_has_duplicates() is False
    assert result['energy'].to_list() == [1.0, None]
    assert result['reduction'].to_list() == [0.5, None]


def test_assigned_dimension_is_part_of_runtime_binding_output_shape() -> None:
    config = {
        'id': 'runtime_assigned_dimension',
        'default_language': 'en',
        'supported_languages': [],
        'name': 'Runtime assigned dimension',
        'owner': 'Owner',
        'target_year': 2030,
        'minimum_historical_year': 2020,
        'maximum_historical_year': 2020,
        'reference_year': 2020,
        'dimensions': [
            {
                'id': 'scope',
                'label': 'Scope',
                'categories': [{'id': 'scope1', 'label': 'Scope 1'}],
            }
        ],
        'nodes': [
            {
                'id': 'power',
                'type': 'nodes.simple.AdditiveNode',
                'name': 'Power',
                'unit': 'kW',
                'quantity': 'energy',
                'historical_values': [[2020, 2.0]],
                'output_nodes': [
                    {
                        'id': 'target',
                        'tags': ['non_additive'],
                        'to_dimensions': [{'id': 'scope', 'categories': ['scope1']}],
                    }
                ],
            },
            {
                'id': 'time',
                'type': 'nodes.simple.AdditiveNode',
                'name': 'Time',
                'unit': 'h',
                'quantity': 'duration',
                'historical_values': [[2020, 3.0]],
                'output_nodes': [{'id': 'target', 'tags': ['non_additive']}],
            },
            {
                'id': 'target',
                'type': 'nodes.simple.MultiplicativeNode',
                'name': 'Target',
                'unit': 'kWh',
                'quantity': 'energy',
                'input_dimensions': ['scope'],
                'output_dimensions': ['scope'],
            },
        ],
    }
    snapshot = parse_instance_snapshot(config, instance_uuid=uuid4())
    target = InstanceLoader(snapshot=snapshot).context.nodes['target']
    assert isinstance(target, MultiplicativeNode)

    result = target.compute()

    assert result['scope'].unique().to_list() == ['scope1']
    assert result.filter(result[YEAR_COLUMN] == 2020)[VALUE_COLUMN].to_list() == [6.0]


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


@pytest.mark.parametrize(('has_null_row', 'expected_values'), [(False, []), (True, [0.0])])
def test_edge_value_preserves_shape_for_empty_and_null_only_inputs(
    has_null_row: bool,
    expected_values: list[float],
) -> None:
    source_id = uuid4()
    target_id = uuid4()
    source_port_id = uuid4()
    target_port_id = uuid4()
    unit = unit_registry.parse_units('EUR/a')
    graph = build_instance_graph(
        InstanceSnapshot(
            metadata=InstanceMetadata(uuid=uuid4(), identifier='empty-edge-shape', name='Empty edge shape'),
            spec=InstanceModelSpec(),
            nodes=[
                NodeSnapshot(
                    uuid=source_id,
                    identifier='source',
                    spec=NodeSpec(
                        output_ports=[OutputPortDef(id=source_port_id, unit=unit)],
                    ),
                ),
                NodeSnapshot(
                    uuid=target_id,
                    identifier='target',
                    spec=NodeSpec(
                        input_ports=[
                            InputPortDef(
                                id=target_port_id,
                                unit=unit,
                                required_dimensions=['cost_type', 'owner'],
                            ),
                        ],
                    ),
                ),
            ],
            bindings=[
                EdgeSnapshot(
                    uuid=uuid4(),
                    from_node=source_id,
                    from_port=source_port_id,
                    to_node=target_id,
                    to_port=target_port_id,
                    transformations=[
                        FilterDimensionOp(dimension='action', exclude=True, flatten=True),
                        FilterDimensionOp(dimension='heating', exclude=True, flatten=True),
                    ],
                ),
            ],
        ),
    )
    count = int(has_null_row)
    raw = pl.DataFrame(
        {
            YEAR_COLUMN: [2024] * count,
            'cost_type': ['investment'] * count,
            'action': ['action'] * count,
            'heating': ['geothermal'] * count,
            'owner': ['private'] * count,
            VALUE_COLUMN: [None] * count,
            FORECAST_COLUMN: [True] * count,
        },
        schema={
            YEAR_COLUMN: pl.Int64,
            'cost_type': pl.String,
            'action': pl.String,
            'heating': pl.String,
            'owner': pl.String,
            VALUE_COLUMN: pl.Float64,
            FORECAST_COLUMN: pl.Boolean,
        },
    )
    source_df = to_ppdf(
        raw,
        DataFrameMeta(
            units={VALUE_COLUMN: unit},
            primary_keys=[YEAR_COLUMN, 'cost_type', 'action', 'heating', 'owner'],
        ),
    )
    source: Any = object.__new__(Node)
    source.id = 'source'
    source.context = object()
    source.get_output_pl = lambda: source_df
    target = object.__new__(Node)
    target.id = 'target'
    runtime_binding = RuntimeInputBinding.from_graph_binding(
        graph.bindings[0],
        port_role='additive',
        source=source,
        target=target,
    )

    result = runtime_binding.get_value()

    assert result[VALUE_COLUMN].to_list() == expected_values
    assert set(result.dim_ids) == {'cost_type', 'owner'}


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
