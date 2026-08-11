"""Step-7 gate tests: the fixpoint constraint solver over InstanceGraph."""

from uuid import UUID, uuid4

import pytest

from nodes.constraints.rules import ProductShapeRule
from nodes.constraints.solver import GraphOverlay
from nodes.constraints.values import BindingValue, PortValue
from nodes.dataset_shape import DatasetShapeProfile
from nodes.defs.binding_def import EdgeBindingDef, NodePortRef
from nodes.defs.graph import DatasetMeta, DatasetMetricMeta, DimensionCategoryMeta, DimensionMeta
from nodes.defs.node_defs import DatasetPortSpec, NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.defs.transform_def import AssignDimensionOp, FilterDimensionOp
from nodes.instance_graph import InstanceGraph, build_instance_graph
from nodes.instance_serialization import DatasetPortSnapshot, EdgeSnapshot, InstanceSnapshot, NodeSnapshot
from nodes.node import Node
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def _unit(text: str):
    return unit_registry.parse_units(text)


def _dimension(identifier: str, *categories: str) -> DimensionMeta:
    return DimensionMeta(
        id=uuid4(),
        identifier=identifier,
        categories=tuple(DimensionCategoryMeta(id=uuid4(), identifier=category) for category in categories),
    )


def _source_node(
    identifier: str,
    unit: str,
    dimensions: list[str] | None = None,
    quantity: str | None = None,
) -> tuple[NodeSnapshot, UUID]:
    output_id = uuid4()
    snapshot = NodeSnapshot(
        uuid=uuid4(),
        identifier=identifier,
        spec=NodeSpec(
            output_ports=[
                OutputPortDef(
                    id=output_id,
                    identifier='default',
                    unit=_unit(unit),
                    dimensions=dimensions or [],
                    quantity=quantity,
                ),
            ],
        ),
    )
    return snapshot, output_id


def _edge(source: NodeSnapshot, source_port: UUID, target: NodeSnapshot, target_port: UUID, transformations=None):
    return EdgeSnapshot(
        uuid=uuid4(),
        from_node=source.uuid,
        from_port=source_port,
        to_node=target.uuid,
        to_port=target_port,
        transformations=transformations or [],
    )


def _build(
    nodes: list[NodeSnapshot],
    edges: list[EdgeSnapshot],
    dimensions: tuple[DimensionMeta, ...] = (),
    datasets: tuple[DatasetMeta, ...] = (),
    dataset_ports: list[DatasetPortSnapshot] | None = None,
) -> InstanceGraph:
    from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec

    return build_instance_graph(
        InstanceSnapshot(
            metadata=InstanceMetadata(uuid=uuid4(), identifier='solver-test', name='Solver test'),
            spec=InstanceModelSpec(),
            nodes=nodes,
            edges=edges,
            dimensions=list(dimensions),
            datasets=list(datasets),
            dataset_ports=dataset_ports or [],
        )
    )


def _additive_target(
    port: InputPortDef,
    output_unit: str = 't/a',
    output_quantity: str | None = None,
) -> tuple[NodeSnapshot, UUID]:
    output_id = uuid4()
    snapshot = NodeSnapshot(
        uuid=uuid4(),
        identifier='target',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='simple.AdditiveNode'),
            input_ports=[port],
            output_ports=[OutputPortDef(id=output_id, identifier='default', unit=_unit(output_unit), quantity=output_quantity)],
        ),
    )
    return snapshot, output_id


def _multiplicative_target(
    input_ports: list[InputPortDef],
    output_unit: str = 'kg/a',
    output_quantity: str | None = None,
) -> tuple[NodeSnapshot, UUID]:
    output_id = uuid4()
    snapshot = NodeSnapshot(
        uuid=uuid4(),
        identifier='target',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='simple.MultiplicativeNode'),
            input_ports=input_ports,
            output_ports=[OutputPortDef(id=output_id, identifier='default', unit=_unit(output_unit), quantity=output_quantity)],
        ),
    )
    return snapshot, output_id


def _codes(result) -> set[str]:
    return {conflict.code for conflict in result.conflicts}


# --- Additive pinning / unpinning ------------------------------------------------


def _pinning_fixture():
    sector = _dimension('sector', 'industry', 'transport')
    fuel = _dimension('fuel', 'oil', 'gas')
    multi_port = InputPortDef(id=uuid4(), identifier='additive', multi=True, unit=_unit('t/a'))
    target, output_id = _additive_target(multi_port)
    source_a, port_a = _source_node('a', 't/a', dimensions=['sector'])
    source_b, port_b = _source_node('b', 't/a', dimensions=['fuel'])
    edge_a = _edge(source_a, port_a, target, multi_port.id)
    edge_b = _edge(source_b, port_b, target, multi_port.id)
    graph = _build([source_a, source_b, target], [edge_a, edge_b], dimensions=(sector, fuel))
    return graph, sector, fuel, target, output_id, source_a, port_a, source_b, port_b, edge_b


def test_additive_pinning_conflicts_across_declared_source_shapes() -> None:
    graph, _sector, _fuel, _target, _output_id, _source_a, port_a, _source_b, port_b, _edge_b = _pinning_fixture()

    result = graph.solve_constraints()
    assert result.converged
    assert 'dimension_mismatch' in _codes(result)

    conflict = next(c for c in result.conflicts if c.code == 'dimension_mismatch')
    origin_ports = {origin.port_id for origin in conflict.origins}
    assert origin_ports == {port_a, port_b}
    assert all(origin.kind == 'declaration' for origin in conflict.origins)


def test_additive_unpinning_via_overlay_removes_the_conflict() -> None:
    graph, sector, _fuel, target, output_id, _source_a, _port_a, _source_b, _port_b, edge_b = _pinning_fixture()

    overlay = GraphOverlay(remove_binding_ids=frozenset({edge_b.uuid}))
    result = graph.solve_constraints(overlay=overlay)
    assert result.converged
    assert _codes(result) == set()
    output_shape = result.shapes[PortValue(target.uuid, output_id, 'output')]
    assert output_shape.dimensions == {sector.id}

    # The overlay did not touch the graph: the base solve still conflicts, and is cached.
    base = graph.solve_constraints()
    assert 'dimension_mismatch' in _codes(base)
    assert graph.solve_constraints() is base


def test_overlay_adding_a_candidate_binding_reports_its_conflict() -> None:
    graph, _sector, _fuel, target, _output_id, _source_a, _port_a, source_b, port_b, edge_b = _pinning_fixture()

    # Base graph without the bad edge: clean.
    clean = graph.solve_constraints(overlay=GraphOverlay(remove_binding_ids=frozenset({edge_b.uuid})))
    assert _codes(clean) == set()

    # Candidate re-adding an equivalent bad binding: the conflict comes back.
    multi_port_id = graph.node_by_identifier['target'].spec.input_ports[0].id
    candidate = EdgeBindingDef(
        id=uuid4(),
        port_ref=NodePortRef(node_uuid=target.uuid, node_id='target', port_id=multi_port_id),
        from_ref=NodePortRef(node_uuid=source_b.uuid, node_id='b', port_id=port_b),
        position=1,
    )
    overlay = GraphOverlay(
        add_bindings=(candidate,),
        remove_binding_ids=frozenset({edge_b.uuid}),
    )
    result = graph.solve_constraints(overlay=overlay)
    assert 'dimension_mismatch' in _codes(result)


# --- Multiplicative union, unit product, quantities --------------------------------


def test_multiplicative_dimension_union_and_unit_product() -> None:
    sector = _dimension('sector', 'industry')
    fuel = _dimension('fuel', 'oil')
    factor_1 = InputPortDef(id=uuid4(), role='factors', unit=_unit('kg/vkm'), quantity='emission_factor')
    factor_2 = InputPortDef(id=uuid4(), role='factors', unit=_unit('vkm/a'), quantity='vehicle_mileage')
    target, output_id = _multiplicative_target([factor_1, factor_2], output_unit='kg/a')
    source_1, port_1 = _source_node('ef', 'kg/vkm', dimensions=['sector'])
    source_2, port_2 = _source_node('mileage', 'vkm/a', dimensions=['fuel'])
    graph = _build(
        [source_1, source_2, target],
        [_edge(source_1, port_1, target, factor_1.id), _edge(source_2, port_2, target, factor_2.id)],
        dimensions=(sector, fuel),
    )

    result = graph.solve_constraints()
    assert result.converged
    assert _codes(result) == set()
    output_shape = result.shapes[PortValue(target.uuid, output_id, 'output')]
    assert output_shape.dimensions == {sector.id, fuel.id}
    # Factor cancellation: emission_factor times vehicle_mileage yields emissions.
    assert output_shape.quantity == 'emissions'


def test_multiplicative_wrong_output_unit_conflicts() -> None:
    factor_1 = InputPortDef(id=uuid4(), role='factors', unit=_unit('kg/vkm'))
    factor_2 = InputPortDef(id=uuid4(), role='factors', unit=_unit('vkm/a'))
    target, _output_id = _multiplicative_target([factor_1, factor_2], output_unit='m/a')
    source_1, port_1 = _source_node('ef', 'kg/vkm')
    source_2, port_2 = _source_node('mileage', 'vkm/a')
    graph = _build(
        [source_1, source_2, target],
        [_edge(source_1, port_1, target, factor_1.id), _edge(source_2, port_2, target, factor_2.id)],
    )

    result = graph.solve_constraints()
    assert 'unit_incompatible' in _codes(result)


def test_scalar_identity_preserves_quantity() -> None:
    factor_1 = InputPortDef(id=uuid4(), role='factors', unit=_unit('dimensionless'), quantity='fraction')
    factor_2 = InputPortDef(id=uuid4(), role='factors', unit=_unit('GWh/a'), quantity='energy')
    target, output_id = _multiplicative_target([factor_1, factor_2], output_unit='GWh/a')
    source_1, port_1 = _source_node('share', 'dimensionless')
    source_2, port_2 = _source_node('consumption', 'GWh/a')
    graph = _build(
        [source_1, source_2, target],
        [_edge(source_1, port_1, target, factor_1.id), _edge(source_2, port_2, target, factor_2.id)],
    )

    result = graph.solve_constraints()
    assert _codes(result) == set()
    assert result.shapes[PortValue(target.uuid, output_id, 'output')].quantity == 'energy'


def test_quantity_mismatch_between_derived_and_authored() -> None:
    factor_1 = InputPortDef(id=uuid4(), role='factors', unit=_unit('dimensionless'), quantity='fraction')
    factor_2 = InputPortDef(id=uuid4(), role='factors', unit=_unit('GWh/a'), quantity='energy')
    target, _output_id = _multiplicative_target([factor_1, factor_2], output_unit='GWh/a', output_quantity='emissions')
    source_1, port_1 = _source_node('share', 'dimensionless')
    source_2, port_2 = _source_node('consumption', 'GWh/a')
    graph = _build(
        [source_1, source_2, target],
        [_edge(source_1, port_1, target, factor_1.id), _edge(source_2, port_2, target, factor_2.id)],
    )

    result = graph.solve_constraints()
    conflict = next(c for c in result.conflicts if c.code == 'quantity_mismatch')
    kinds = {origin.kind for origin in conflict.origins}
    assert kinds == {'declaration', 'node_rule'}


class PerCapitaTestNode(Node):
    """Test-only exemplar for the inverse operand of a product rule."""

    @classmethod
    def shape_rules(cls, meta):  # noqa: ANN206
        return (
            ProductShapeRule(
                inputs=(meta.spec.input_ports[0].id,),
                inverse_inputs=(meta.spec.input_ports[1].id,),
                output=meta.spec.output_ports[0].id,
            ),
        )


def _per_capita_graph(output_unit: str) -> tuple[InstanceGraph, NodeSnapshot, UUID]:
    total_port = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    population_port = InputPortDef(id=uuid4(), unit=_unit('cap'))
    output_id = uuid4()
    target = NodeSnapshot(
        uuid=uuid4(),
        identifier='per_capita',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='nodes.tests.test_constraint_solver.PerCapitaTestNode'),
            input_ports=[total_port, population_port],
            output_ports=[OutputPortDef(id=output_id, identifier='default', unit=_unit(output_unit))],
        ),
    )
    source_1, port_1 = _source_node('total', 't/a')
    source_2, port_2 = _source_node('population', 'cap')
    graph = _build(
        [source_1, source_2, target],
        [_edge(source_1, port_1, target, total_port.id), _edge(source_2, port_2, target, population_port.id)],
    )
    return graph, target, output_id


def test_inverse_input_divides_the_unit() -> None:
    graph, _target, _output_id = _per_capita_graph('t/a/cap')
    assert _codes(graph.solve_constraints()) == set()

    wrong, _target, _output_id = _per_capita_graph('t/a')
    assert 'unit_incompatible' in _codes(wrong.solve_constraints())


# --- Consumes/produces -----------------------------------------------------------


def _gwp_graph(source_dimensions: list[str]) -> tuple[InstanceGraph, NodeSnapshot, UUID, DimensionMeta, DimensionMeta]:
    from nodes.tests.test_shape_rules import GWPLikeTestNode  # noqa: F401  (class path referenced below)

    ghg = _dimension('greenhouse_gases', 'co2', 'ch4')
    sector = _dimension('sector', 'industry')
    input_port = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    output_id = uuid4()
    target = NodeSnapshot(
        uuid=uuid4(),
        identifier='gwp',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='nodes.tests.test_shape_rules.GWPLikeTestNode'),
            input_ports=[input_port],
            output_ports=[OutputPortDef(id=output_id, identifier='default', unit=_unit('t/a'))],
        ),
    )
    source, source_port = _source_node('source', 't/a', dimensions=source_dimensions)
    graph = _build([source, target], [_edge(source, source_port, target, input_port.id)], dimensions=(ghg, sector))
    return graph, target, output_id, ghg, sector


def test_consumes_produces_removes_the_dimension() -> None:
    graph, target, output_id, _ghg, sector = _gwp_graph(['greenhouse_gases', 'sector'])
    result = graph.solve_constraints()
    assert _codes(result) == set()
    assert result.shapes[PortValue(target.uuid, output_id, 'output')].dimensions == {sector.id}


def test_consumes_produces_missing_required_dimension_conflicts() -> None:
    graph, _target, _output_id, _ghg, _sector = _gwp_graph(['sector'])
    result = graph.solve_constraints()
    assert 'missing_required_dimension' in _codes(result)


# --- Binding transformations -------------------------------------------------------


def test_filter_flatten_and_assign_reshape_the_binding_value() -> None:
    sector = _dimension('sector', 'industry', 'transport')
    scope = _dimension('scope', 'scope1')
    port = InputPortDef(id=uuid4(), unit=_unit('t/a'), required_dimensions=['scope'])
    target, _output_id = _additive_target(port)
    source, source_port = _source_node('source', 't/a', dimensions=['sector'])
    edge = _edge(
        source,
        source_port,
        target,
        port.id,
        transformations=[
            FilterDimensionOp(dimension='sector', categories=['industry'], flatten=True),
            AssignDimensionOp(dimension='scope', category='scope1'),
        ],
    )
    graph = _build([source, target], [edge], dimensions=(sector, scope))

    result = graph.solve_constraints()
    assert result.converged
    assert _codes(result) == set()
    binding_shape = result.shapes[BindingValue(edge.uuid)]
    assert binding_shape.dimensions == {scope.id}
    scope1 = scope.categories[0]
    assert binding_shape.categories == {scope.id: frozenset({scope1.id})}


def test_assigning_an_existing_dimension_conflicts() -> None:
    sector = _dimension('sector', 'industry')
    port = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    target, _output_id = _additive_target(port)
    source, source_port = _source_node('source', 't/a', dimensions=['sector'])
    edge = _edge(
        source,
        source_port,
        target,
        port.id,
        transformations=[AssignDimensionOp(dimension='sector', category='industry')],
    )
    graph = _build([source, target], [edge], dimensions=(sector,))

    result = graph.solve_constraints()
    conflict = next(c for c in result.conflicts if c.code == 'assign_existing_dimension')
    assert any(origin.kind == 'transformation' and origin.binding_id == edge.uuid for origin in conflict.origins)


# --- Dataset profiles and category disjointness ---------------------------------------


def _dataset_graph(filter_categories: list[str]):
    sector = _dimension('sector', 'industry', 'transport')
    metric = DatasetMetricMeta(id=uuid4(), identifier='total', unit='t/a')
    dataset = DatasetMeta(
        id=uuid4(),
        identifier='test/emissions',
        schema_id=uuid4(),
        metrics=(metric,),
        declared_dimension_ids=(sector.id,),
    )
    port = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    target, _output_id = _additive_target(port)
    dataset_port = DatasetPortSnapshot(
        uuid=uuid4(),
        node=target.uuid,
        dataset='test/emissions',
        dataset_uuid=dataset.id,
        port_id=port.id,
        metric='total',
        metric_uuid=metric.id,
        spec=DatasetPortSpec(
            transformations=[FilterDimensionOp(dimension='sector', categories=filter_categories)],
        ),
    )
    graph = _build([target], [], dimensions=(sector,), datasets=(dataset,), dataset_ports=[dataset_port])
    return graph, sector, dataset, metric, dataset_port


def test_disjoint_category_filter_against_observed_profile() -> None:
    graph, sector, dataset, metric, _dataset_port = _dataset_graph(filter_categories=['transport'])
    industry = sector.categories[0]
    profile = DatasetShapeProfile(
        dataset_id=dataset.id,
        metric_id=metric.id,
        categories_by_dimension={sector.id: frozenset({industry.id})},
        has_datapoints=True,
        source_version='test-1',
    )

    result = graph.solve_constraints(profiles={(dataset.id, metric.id): profile})
    conflict = next(c for c in result.conflicts if c.code == 'disjoint_category_filter')
    kinds = {origin.kind for origin in conflict.origins}
    assert kinds == {'transformation', 'dataset_profile'}

    # Without a profile the observed categories are unknown, and unknown never conflicts.
    assert 'disjoint_category_filter' not in _codes(graph.solve_constraints())


def test_overlapping_category_filter_is_clean_and_narrows_the_value() -> None:
    graph, sector, dataset, metric, dataset_port = _dataset_graph(filter_categories=['industry'])
    industry = sector.categories[0]
    transport = sector.categories[1]
    profile = DatasetShapeProfile(
        dataset_id=dataset.id,
        metric_id=metric.id,
        categories_by_dimension={sector.id: frozenset({industry.id, transport.id})},
        has_datapoints=True,
        source_version='test-1',
    )

    result = graph.solve_constraints(profiles={(dataset.id, metric.id): profile})
    assert 'disjoint_category_filter' not in _codes(result)
    binding_shape = result.shapes[BindingValue(dataset_port.uuid)]
    assert binding_shape.categories[sector.id] == frozenset({industry.id})


def test_legacy_flatten_declarations_are_recovered_onto_the_binding() -> None:
    from nodes.defs.transform_def import FlattenTransformation
    from nodes.instance_graph_cache import _dump_graph, _load_graph

    sector = _dimension('sector', 'industry')
    port = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    target, _output_id = _additive_target(port)
    source, source_port = _source_node('source', 't/a')
    edge = _edge(source, source_port, target, port.id, transformations=[FlattenTransformation(dimension='sector')])
    graph = _build([source, target], [edge], dimensions=(sector,))

    assert edge.uuid is not None
    binding = graph.binding_by_id[edge.uuid]
    assert isinstance(binding, EdgeBindingDef)
    assert binding.declared_dimensions == ['sector']
    assert not binding.transformations  # modernization dropped the non-executable op

    result = graph.solve_constraints()
    assert _codes(result) == set()
    assert result.shapes[BindingValue(edge.uuid)].dimensions == {sector.id}
    # The declaration constrains backward through the (empty) chain onto the source.
    assert result.shapes[PortValue(source.uuid, source_port, 'output')].dimensions == {sector.id}

    reloaded = _load_graph(_dump_graph(graph))
    reloaded_binding = reloaded.binding_by_id[edge.uuid]
    assert isinstance(reloaded_binding, EdgeBindingDef)
    assert reloaded_binding.declared_dimensions == ['sector']


def test_filter_column_drops_a_declared_dimension_but_not_a_raw_column() -> None:
    from nodes.defs.transform_def import FilterColumnOp

    sector = _dimension('sector', 'industry')
    metric = DatasetMetricMeta(id=uuid4(), identifier='total', unit='t/a')
    dataset = DatasetMeta(
        id=uuid4(),
        identifier='test/wide',
        schema_id=uuid4(),
        metrics=(metric,),
        declared_dimension_ids=(sector.id,),
    )
    port = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    target, _output_id = _additive_target(port)

    def build(column: str) -> tuple[InstanceGraph, DatasetPortSnapshot]:
        dataset_port = DatasetPortSnapshot(
            uuid=uuid4(),
            node=target.uuid,
            dataset='test/wide',
            dataset_uuid=dataset.id,
            port_id=port.id,
            metric='total',
            metric_uuid=metric.id,
            spec=DatasetPortSpec(transformations=[FilterColumnOp(column=column, value='industry')]),
        )
        graph = _build([target], [], dimensions=(sector,), datasets=(dataset,), dataset_ports=[dataset_port])
        return graph, dataset_port

    # Filtering a declared dimension column drops the dimension (drop_col defaults to True).
    graph, dataset_port = build('sector')
    assert dataset_port.uuid is not None
    result = graph.solve_constraints()
    assert result.shapes[BindingValue(dataset_port.uuid)].dimensions == frozenset()

    # A raw column outside the declared schema is shape-neutral, even if an
    # instance dimension happens to share its name.
    action = _dimension('action', 'zone')
    graph2, dataset_port2 = build('action')
    graph2 = _build(
        [target], [], dimensions=(sector, action), datasets=(dataset,), dataset_ports=[dataset_port2.model_copy(deep=True)]
    )
    assert dataset_port2.uuid is not None
    result2 = graph2.solve_constraints()
    assert result2.shapes[BindingValue(dataset_port2.uuid)].dimensions == {sector.id}


# --- Whole-result properties -----------------------------------------------------------


def test_multiple_independent_conflicts_are_all_reported() -> None:
    sector = _dimension('sector', 'industry')
    fuel = _dimension('fuel', 'oil')
    multi_port = InputPortDef(id=uuid4(), identifier='additive', multi=True, unit=_unit('t/a'))
    target, _output_id = _additive_target(multi_port)
    source_a, port_a = _source_node('a', 't/a', dimensions=['sector'])
    source_b, port_b = _source_node('b', 't/a', dimensions=['fuel'])
    source_c, port_c = _source_node('c', 'GWh/a')
    graph = _build(
        [source_a, source_b, source_c, target],
        [
            _edge(source_a, port_a, target, multi_port.id),
            _edge(source_b, port_b, target, multi_port.id),
            _edge(source_c, port_c, target, multi_port.id),
        ],
        dimensions=(sector, fuel),
    )

    result = graph.solve_constraints()
    assert {'dimension_mismatch', 'unit_incompatible'} <= _codes(result)


def test_quantity_mismatch_on_same_shape_rule() -> None:
    port = InputPortDef(id=uuid4(), identifier='additive', multi=True, unit=_unit('t/a'), quantity='emissions')
    target, _output_id = _additive_target(port, output_quantity='energy')
    source, source_port = _source_node('a', 't/a')
    graph = _build([source, target], [_edge(source, source_port, target, port.id)])

    result = graph.solve_constraints()
    conflict = next(c for c in result.conflicts if c.code == 'quantity_mismatch')
    assert all(origin.kind == 'declaration' for origin in conflict.origins)
