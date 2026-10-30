"""Output dimensions are asserted on ports, or inferred where a port asserts nothing (snapshot v18)."""

from typing import Any
from uuid import uuid4

import pytest

from nodes.constraints.assertions import classify_output_assertions
from nodes.constraints.values import PortValue
from nodes.defs.graph import DimensionCategoryMeta, DimensionMeta
from nodes.defs.port_def import InputPortDef
from nodes.exceptions import NodeError
from nodes.instance_loader import InstanceLoader
from nodes.instance_parser import parse_instance_snapshot
from nodes.instance_serialization import InstanceSnapshot, move_output_dimensions_to_ports_v18
from nodes.tests.test_constraint_solver import _additive_target, _build, _codes, _dimension, _edge, _source_node
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def test_v18_upgrade_moves_the_node_list_onto_every_output_port() -> None:
    spec: dict[str, Any] = {
        'output_dimensions': ['sector'],
        'output_ports': [{'id': str(uuid4())}, {'id': str(uuid4()), 'dimensions': []}],
    }
    assert move_output_dimensions_to_ports_v18(spec)
    assert 'output_dimensions' not in spec
    assert [port['dimensions'] for port in spec['output_ports']] == [['sector'], ['sector']]
    assert not move_output_dimensions_to_ports_v18(spec)


def test_v18_upgrade_keeps_an_empty_list_a_scalar() -> None:
    spec: dict[str, Any] = {'output_dimensions': [], 'output_ports': [{'id': str(uuid4())}]}
    move_output_dimensions_to_ports_v18(spec)
    # Before v18 an empty list meant a scalar output, never an inferred one.
    assert spec['output_ports'][0]['dimensions'] == []


def _pass_through(source_dims: list[str] | None):
    sector = _dimension('sector', 'industry')
    port = InputPortDef(id=uuid4(), unit=unit_registry.parse_units('t/a'))
    target, output_id = _additive_target(port)
    source, source_port = _source_node('source', 't/a', dimensions=source_dims)
    graph = _build([source, target], [_edge(source, source_port, target, port.id)], dimensions=(sector,))
    return graph, sector, target, output_id


def test_an_unasserted_additive_output_takes_its_inputs_shape() -> None:
    graph, sector, target, output_id = _pass_through(['sector'])
    result = graph.solve_constraints()
    assert result.shapes[PortValue(target.uuid, output_id, 'output')].dimensions == {sector.id}
    assert not result.undetermined


def test_an_output_nothing_determines_is_undetermined_but_not_a_conflict() -> None:
    graph, _sector, target, output_id = _pass_through(None)
    result = graph.solve_constraints()
    # The source has no rule to infer from either, so it is undetermined too.
    assert ('undetermined_output_dimensions', PortValue(target.uuid, output_id, 'output')) in {
        (conflict.code, conflict.value) for conflict in result.undetermined
    }
    # Edits are validated against conflicts only; publication against both.
    assert not result.conflicts
    assert result.blocking == result.undetermined


def test_an_asserted_scalar_is_exact() -> None:
    graph, *_ = _pass_through(['sector'])
    data = graph.model_dump(mode='json')
    target = next(node for node in data['nodes'] if node['identifier'] == 'target')
    target['spec']['output_ports'][0]['dimensions'] = []
    result = type(graph).model_validate(data).solve_constraints()
    assert 'dimension_mismatch' in _codes(result)


def test_classifying_assertions_separates_redundant_from_needed() -> None:
    graph, *_ = _pass_through(['sector'])
    data = graph.model_dump(mode='json')
    target = next(node for node in data['nodes'] if node['identifier'] == 'target')
    target['spec']['output_ports'][0]['dimensions'] = ['sector']
    assertions = {
        graph.describe_uuid(item.node_id): item for item in classify_output_assertions(type(graph).model_validate(data))
    }

    # The additive node's rule reproduces its assertion; the source has no rule to infer from.
    assert assertions['target'].redundant
    assert not assertions['source'].redundant
    assert assertions['source'].inferred is None

    target['spec']['is_outcome'] = True
    assertions = {
        graph.describe_uuid(item.node_id): item for item in classify_output_assertions(type(graph).model_validate(data))
    }
    assert not assertions['target'].redundant  # an outcome's requested shape always stays


def _yaml_pass_through() -> dict[str, Any]:
    return {
        'id': 'inferred_dims',
        'default_language': 'en',
        'supported_languages': [],
        'name': 'Inferred dimensions',
        'owner': 'Owner',
        'theme_identifier': 'default',
        'target_year': 2030,
        'model_end_year': 2030,
        'reference_year': 2020,
        'minimum_historical_year': 2020,
        'maximum_historical_year': 2020,
        'emission_unit': 'kt/a',
        'dimensions': [{'id': 'sector', 'label': 'Sector', 'categories': [{'id': 'industry', 'label': 'Industry'}]}],
        'nodes': [
            {
                'id': 'source',
                'type': 'nodes.simple.AdditiveNode',
                'name': 'Source',
                'unit': 'kt/a',
                'quantity': 'emissions',
                'output_dimensions': ['sector'],
                'output_nodes': ['target'],
            },
            {
                'id': 'target',
                'type': 'nodes.simple.AdditiveNode',
                'name': 'Target',
                'unit': 'kt/a',
                'quantity': 'emissions',
                'output_dimensions': ['sector'],
            },
        ],
        'scenarios': [{'id': 'default', 'name': 'Default', 'default': True}],
    }


def _withdraw_assertion(snapshot: InstanceSnapshot, identifier: str) -> InstanceSnapshot:
    """
    Withdraw one node's assertion from a parsed snapshot.

    A parsed YAML snapshot carries no dimension catalog, and YAML always asserts, so its graph
    never has to infer; give it the catalog a database-built snapshot would have.
    """
    catalog = [
        DimensionMeta(
            id=uuid4(),
            identifier=dim['id'],
            categories=tuple(DimensionCategoryMeta(id=uuid4(), identifier=cat['id']) for cat in dim.get('categories', [])),
        )
        for dim in snapshot.spec.dimensions
    ]
    nodes = []
    for node in snapshot.nodes:
        if node.identifier == identifier:
            assert node.spec is not None
            ports = [port.model_copy(update={'dimensions': None}) for port in node.spec.output_ports]
            node = node.model_copy(update={'spec': node.spec.model_copy(update={'output_ports': ports})})  # noqa: PLW2901
        nodes.append(node)
    return snapshot.model_copy(update={'nodes': nodes, 'dimensions': catalog})


def test_the_runtime_takes_an_unasserted_output_from_the_solver() -> None:
    snapshot = _withdraw_assertion(parse_instance_snapshot(_yaml_pass_through(), instance_uuid=uuid4()), 'target')
    context = InstanceLoader.from_snapshot(snapshot).context
    assert list(context.get_node('target').output_dimensions) == ['sector']


def test_the_runtime_infers_a_source_from_its_asserting_consumer() -> None:
    snapshot = _withdraw_assertion(parse_instance_snapshot(_yaml_pass_through(), instance_uuid=uuid4()), 'source')
    context = InstanceLoader.from_snapshot(snapshot).context
    assert list(context.get_node('source').output_dimensions) == ['sector']


def test_the_runtime_refuses_an_undetermined_output() -> None:
    snapshot = parse_instance_snapshot(_yaml_pass_through(), instance_uuid=uuid4())
    snapshot = _withdraw_assertion(_withdraw_assertion(snapshot, 'source'), 'target')
    with pytest.raises(NodeError, match='cannot be inferred'):
        InstanceLoader.from_snapshot(snapshot)
