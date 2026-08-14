"""Step-8 gate tests: the overlay-validation application service."""

from uuid import uuid4

import pytest

from nodes.constraints.validation import BindingChange, graph_with_additions, validate_binding_change
from nodes.defs.binding_def import DatasetBindingDef, EdgeBindingDef, NodePortRef
from nodes.defs.graph import DatasetMeta, DatasetMetricMeta
from nodes.defs.port_def import InputPortDef
from nodes.instance_graph_cache import ResolvedInstanceSource
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.tests.test_constraint_solver import (
    _build,
    _dimension,
    _edge,
    _multiplicative_target,
    _source_node,
    _unit,
)

pytestmark = pytest.mark.django_db


@pytest.fixture
def config():
    instance = InstanceFactory.create()
    return InstanceConfigFactory.create(identifier=instance.id, instance=instance, config_source='database')


def _draft_source(config) -> ResolvedInstanceSource:
    return ResolvedInstanceSource(str(config.uuid), 'database-draft', 'test-version')


def _edge_candidate(graph, source, source_port, target, target_port) -> EdgeBindingDef:
    return EdgeBindingDef(
        id=uuid4(),
        port_ref=NodePortRef(node_uuid=target.uuid, node_id=target.identifier or str(target.uuid), port_id=target_port),
        from_ref=NodePortRef(node_uuid=source.uuid, node_id=source.identifier or str(source.uuid), port_id=source_port),
        position=len(graph.bindings_by_input.get((target.uuid, target_port), ())),
    )


def _pinning_parts():
    """Build an additive multi port fed by one sector-shaped source, with a fuel-shaped source standing by."""
    from nodes.tests.test_constraint_solver import _additive_target

    sector = _dimension('sector', 'industry')
    fuel = _dimension('fuel', 'oil')
    multi_port = InputPortDef(id=uuid4(), identifier='additive', role='additive', multi=True, unit=_unit('t/a'))
    target, _output_id = _additive_target(multi_port)
    source_a, port_a = _source_node('a', 't/a', dimensions=['sector'])
    source_b, port_b = _source_node('b', 't/a', dimensions=['fuel'])
    return sector, fuel, multi_port, target, source_a, port_a, source_b, port_b


def test_conflicting_candidate_is_rejected_with_only_its_conflicts(config):
    sector, fuel, multi_port, target, source_a, port_a, source_b, port_b = _pinning_parts()
    graph = _build([source_a, source_b, target], [_edge(source_a, port_a, target, multi_port.id)], dimensions=(sector, fuel))

    candidate = _edge_candidate(graph, source_b, port_b, target, multi_port.id)
    validation = validate_binding_change(config, graph, _draft_source(config), BindingChange(add_bindings=(candidate,)))

    assert not validation.ok
    assert {conflict.code for conflict in validation.new_conflicts} == {'dimension_mismatch'}


def test_baseline_debt_never_blocks_an_unrelated_change(config):
    sector, fuel, multi_port, target, source_a, port_a, source_b, port_b = _pinning_parts()
    source_c, port_c = _source_node('c', 't/a', dimensions=['sector'])
    # a and b already conflict (sector vs fuel); c matches a's shape.
    graph = _build(
        [source_a, source_b, source_c, target],
        [_edge(source_a, port_a, target, multi_port.id), _edge(source_b, port_b, target, multi_port.id)],
        dimensions=(sector, fuel),
    )
    assert graph.solve_constraints().conflicts, 'fixture must carry baseline debt'

    candidate = _edge_candidate(graph, source_c, port_c, target, multi_port.id)
    validation = validate_binding_change(config, graph, _draft_source(config), BindingChange(add_bindings=(candidate,)))

    assert validation.ok
    # The full result still shows the pre-existing debt for inspection.
    assert validation.result.conflicts


def test_removal_of_the_offending_binding_resolves_baseline_conflicts(config):
    sector, fuel, multi_port, target, source_a, port_a, source_b, port_b = _pinning_parts()
    edge_b = _edge(source_b, port_b, target, multi_port.id)
    graph = _build(
        [source_a, source_b, target],
        [_edge(source_a, port_a, target, multi_port.id), edge_b],
        dimensions=(sector, fuel),
    )

    assert edge_b.uuid is not None
    change = BindingChange(remove_binding_ids=frozenset({edge_b.uuid}))
    validation = validate_binding_change(config, graph, _draft_source(config), change)

    assert validation.ok
    assert not validation.result.conflicts


def test_added_port_participates_through_the_hypothetical_graph(config):
    factor_1 = InputPortDef(id=uuid4(), identifier='factor', role='factors', unit=_unit('kg/vkm'))
    target, _output_id = _multiplicative_target([factor_1], output_unit='kg/a')
    source_1, port_1 = _source_node('ef', 'kg/vkm')
    source_2, port_2 = _source_node('mileage', 'vkm/a')
    bad_source, bad_port = _source_node('price', 'EUR')
    graph = _build([source_1, source_2, bad_source, target], [_edge(source_1, port_1, target, factor_1.id)])

    new_port = InputPortDef(id=uuid4(), identifier='factor2', role='factors')

    good = BindingChange(
        add_bindings=(_edge_candidate(graph, source_2, port_2, target, new_port.id),),
        add_input_ports=((target.uuid, new_port),),
    )
    assert validate_binding_change(config, graph, _draft_source(config), good).ok

    bad = BindingChange(
        add_bindings=(_edge_candidate(graph, bad_source, bad_port, target, new_port.id),),
        add_input_ports=((target.uuid, new_port),),
    )
    validation = validate_binding_change(config, graph, _draft_source(config), bad)
    assert not validation.ok
    assert 'unit_incompatible' in {conflict.code for conflict in validation.new_conflicts}

    # The cached graph itself was never touched by the hypothetical copies.
    node = graph.node_by_id[target.uuid]
    assert [port.id for port in node.spec.input_ports] == [factor_1.id]


def test_graph_with_additions_is_independent_and_rebound():
    factor_1 = InputPortDef(id=uuid4(), identifier='factor', role='factors', unit=_unit('kg/vkm'))
    target, _output_id = _multiplicative_target([factor_1], output_unit='kg/a')
    graph = _build([target], [])

    new_port = InputPortDef(id=uuid4(), identifier='factor2', role='factors')
    dataset = DatasetMeta(
        id=uuid4(),
        schema_id=uuid4(),
        identifier='external',
        metrics=(DatasetMetricMeta(id=uuid4(), identifier='value', unit='kg/a'),),
        is_external_placeholder=True,
    )
    change = BindingChange(add_input_ports=((target.uuid, new_port),), add_datasets=(dataset,))
    hypothetical = graph_with_additions(graph, change)

    assert hypothetical is not graph
    assert (target.uuid, new_port.id) in hypothetical.input_port_by_id
    assert dataset.id in hypothetical.dataset_by_id
    assert (target.uuid, new_port.id) not in graph.input_port_by_id
    assert dataset.id not in graph.dataset_by_id
    # The copy's children are bound to the copy, not to the original.
    assert hypothetical.node_by_id[target.uuid].graph is hypothetical


def test_external_placeholder_dataset_candidate_validates_without_orm_rows(config):
    factor_1 = InputPortDef(id=uuid4(), identifier='factor', role='factors', unit=_unit('kg/a'))
    target, _output_id = _multiplicative_target([factor_1], output_unit='kg/a')
    graph = _build([target], [])

    metric_id = uuid4()
    dataset = DatasetMeta(
        id=uuid4(),
        schema_id=uuid4(),
        identifier='external',
        metrics=(DatasetMetricMeta(id=metric_id, identifier='value', unit='kg/a'),),
        is_external_placeholder=True,
    )
    candidate = DatasetBindingDef(
        id=uuid4(),
        port_ref=NodePortRef(node_uuid=target.uuid, node_id=target.identifier or str(target.uuid), port_id=factor_1.id),
        dataset_uuid=dataset.id,
        metric_uuid=metric_id,
        dataset_is_external_placeholder=True,
    )
    change = BindingChange(add_bindings=(candidate,), add_datasets=(dataset,))
    validation = validate_binding_change(config, graph, _draft_source(config), change)
    assert validation.ok
