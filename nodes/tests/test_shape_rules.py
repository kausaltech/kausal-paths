"""Step-6 gate tests: shape-rule declaration, compilation, and legacy role classification."""

from uuid import UUID, uuid4, uuid5

import pytest

from nodes.constraints.compile import _validate_node_rules
from nodes.constraints.pipeline_compile import compile_pipeline_operations
from nodes.constraints.rules import (
    DimensionTransformRule,
    ProductShapeRule,
    SameShapeRule,
    ShapeRuleError,
)
from nodes.defs.graph import DimensionMeta
from nodes.defs.node_defs import NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.instance_graph import InstanceGraph, NodeMeta, build_instance_graph
from nodes.instance_graph_cache import _dump_graph, _load_graph
from nodes.instance_serialization import EdgeSnapshot, InstanceSnapshot, NodeSnapshot
from nodes.node import Node
from nodes.pipeline.ops.arithmetic import AddOperationSpec, AnyOperationSpec, MultiplyOperationSpec
from nodes.pipeline.ops.base import DatasetInputRef, IntermediateInputRef, PortInputRef, ScalarValue
from nodes.simple import AdditiveNode, MultiplicativeNode
from nodes.units import unit_registry
from params.param import StringParameter

pytestmark = pytest.mark.django_db


def _unit(text: str):
    return unit_registry.parse_units(text)


def _source_node(identifier: str, unit: str) -> tuple[NodeSnapshot, UUID]:
    output_id = uuid4()
    snapshot = NodeSnapshot(
        uuid=uuid4(),
        identifier=identifier,
        spec=NodeSpec(output_ports=[OutputPortDef(id=output_id, identifier='default', unit=_unit(unit))]),
    )
    return snapshot, output_id


def _edge(source: NodeSnapshot, source_port: UUID, target: NodeSnapshot, target_port: UUID, tags: list[str] | None = None):
    return EdgeSnapshot(
        uuid=uuid4(),
        from_node=source.uuid,
        from_port=source_port,
        to_node=target.uuid,
        to_port=target_port,
        tags=tags or [],
    )


def _build(nodes: list[NodeSnapshot], edges: list[EdgeSnapshot], dimensions: tuple[DimensionMeta, ...] = ()) -> InstanceGraph:
    from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec

    return build_instance_graph(
        InstanceSnapshot(
            metadata=InstanceMetadata(uuid=uuid4(), identifier='shape-test', name='Shape test'),
            spec=InstanceModelSpec(),
            nodes=nodes,
            bindings=list(edges),
            dimensions=list(dimensions),
        )
    )


def _multiplicative_target(input_ports: list[InputPortDef], output_unit: str = 'kt/a') -> tuple[NodeSnapshot, UUID]:
    output_id = uuid4()
    snapshot = NodeSnapshot(
        uuid=uuid4(),
        identifier='target',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='simple.MultiplicativeNode'),
            input_ports=input_ports,
            output_ports=[OutputPortDef(id=output_id, identifier='default', unit=_unit(output_unit))],
        ),
    )
    return snapshot, output_id


def test_legacy_multiplicative_ports_classify_and_compile_to_uuid_rules() -> None:
    factor_port = InputPortDef(id=uuid4(), unit=_unit('kg/vkm'))
    additive_port = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    impute_port = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    tagged_factor_port = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    target, output_id = _multiplicative_target([factor_port, additive_port, impute_port, tagged_factor_port])

    sources = [_source_node(f'source_{index}', unit) for index, unit in enumerate(('kg/vkm', 't/a', 't/a', 't/a'))]
    edges = [
        _edge(sources[0][0], sources[0][1], target, factor_port.id),
        _edge(sources[1][0], sources[1][1], target, additive_port.id),
        _edge(sources[2][0], sources[2][1], target, impute_port.id, tags=['impute']),
        _edge(sources[3][0], sources[3][1], target, tagged_factor_port.id, tags=['non_additive']),
    ]
    graph = _build([snapshot for snapshot, _ in sources] + [target], edges)

    meta = graph.node_by_id[target.uuid]
    assert meta.inferred_port_roles == {
        factor_port.id: 'factors',
        additive_port.id: 'additive',
        impute_port.id: 'impute',
        tagged_factor_port.id: 'factors',
    }
    codes = {diagnostic.code for diagnostic in graph.diagnostics}
    assert codes == {'inferred_port_role'}

    rules = graph.shape_rule_compilation.rules_by_node[target.uuid]
    assert rules == (
        ProductShapeRule(inputs=(factor_port.id, tagged_factor_port.id), output=output_id),
        SameShapeRule(inputs=(additive_port.id, impute_port.id), output=output_id),
    )
    for rule in rules:
        for value in (*rule.inputs, rule.output):
            assert isinstance(value, UUID)


def test_explicit_role_wins_and_unit_change_does_not_reclassify() -> None:
    # Unit-compatible with the output, but explicitly a factor: the heuristic must not touch it.
    explicit_factor = InputPortDef(id=uuid4(), role='factors', unit=_unit('t/a'))
    target, output_id = _multiplicative_target([explicit_factor])
    source, source_port = _source_node('source', 't/a')
    graph = _build([source, target], [_edge(source, source_port, target, explicit_factor.id)])

    meta = graph.node_by_id[target.uuid]
    assert meta.inferred_port_roles == {}
    assert meta.port_role_diagnostics == ()
    rules = graph.shape_rule_compilation.rules_by_node[target.uuid]
    assert rules == (ProductShapeRule(inputs=(explicit_factor.id,), output=output_id),)


def test_additive_grouped_port_resolves_role_from_identifier_fallback() -> None:
    multi_port = InputPortDef(id=uuid4(), identifier='additive', multi=True, unit=_unit('t/a'))
    output_id = uuid4()
    target = NodeSnapshot(
        uuid=uuid4(),
        identifier='target',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='simple.AdditiveNode'),
            input_ports=[multi_port],
            output_ports=[OutputPortDef(id=output_id, identifier='default', unit=_unit('t/a'))],
        ),
    )
    source_a, port_a = _source_node('a', 't/a')
    source_b, port_b = _source_node('b', 't/a')
    graph = _build(
        [source_a, source_b, target],
        [_edge(source_a, port_a, target, multi_port.id), _edge(source_b, port_b, target, multi_port.id)],
    )

    meta = graph.node_by_id[target.uuid]
    assert meta.inferred_port_roles == {}
    assert meta.require_input_port('additive').id == multi_port.id
    rules = graph.shape_rule_compilation.rules_by_node[target.uuid]
    assert rules == (SameShapeRule(inputs=(multi_port.id,), output=output_id),)


def test_legacy_additive_metric_parameter_selects_one_multi_output_edge_port() -> None:
    metric = StringParameter(local_id='metric', is_customizable=False)
    metric.set('Electricity')
    source_id = uuid4()
    source_ports = [
        OutputPortDef(id=uuid4(), identifier='default', column_id='Value', unit=_unit('%')),
        OutputPortDef(id=uuid4(), identifier='Heat', column_id='Heat', unit=_unit('kWh/m²/a')),
        OutputPortDef(id=uuid4(), identifier='Electricity', column_id='Electricity', unit=_unit('kWh/m²/a')),
    ]
    source = NodeSnapshot(
        uuid=source_id,
        identifier='building_action',
        spec=NodeSpec(output_ports=source_ports),
    )
    input_ports = [InputPortDef(id=uuid4(), unit=source_port.unit) for source_port in source_ports]
    output_id = uuid4()
    target = NodeSnapshot(
        uuid=uuid4(),
        identifier='electricity_per_area',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='simple.AdditiveNode'),
            params=[metric],
            input_ports=input_ports,
            output_ports=[OutputPortDef(id=output_id, identifier='default', unit=_unit('kWh/m²/a'))],
        ),
    )
    graph = _build(
        [source, target],
        [
            _edge(source, source_port.id, target, input_port.id)
            for source_port, input_port in zip(source_ports, input_ports, strict=True)
        ],
    )

    meta = graph.node_by_id[target.uuid]
    assert meta.inferred_port_roles == {input_ports[2].id: 'additive'}
    assert meta.input_port_ids_for_roles('additive') == (input_ports[2].id,)
    assert graph.shape_rule_compilation.rules_by_node[target.uuid] == (
        SameShapeRule(inputs=(input_ports[2].id,), output=output_id),
    )


def test_missing_role_port_is_a_diagnostic_not_an_error() -> None:
    # A multiplicative node with no output ports cannot resolve the 'output' role.
    factor_port = InputPortDef(id=uuid4(), unit=_unit('kg/vkm'))
    target = NodeSnapshot(
        uuid=uuid4(),
        identifier='target',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='simple.MultiplicativeNode'),
            input_ports=[factor_port],
        ),
    )
    source, source_port = _source_node('source', 'kg/vkm')
    graph = _build([source, target], [_edge(source, source_port, target, factor_port.id)])

    compilation = graph.shape_rule_compilation
    assert compilation.rules_by_node[target.uuid] == ()
    assert any(
        diagnostic.code == 'missing_role_port' and diagnostic.node_id == target.uuid for diagnostic in compilation.diagnostics
    )


class BrokenRuleTestNode(Node):
    @classmethod
    def shape_rules(cls, meta):  # noqa: ANN206
        return (SameShapeRule(inputs=(uuid4(),), output=meta.spec.output_ports[0].id),)


def test_invalid_class_rule_fails_naming_the_class() -> None:
    target = NodeSnapshot(
        uuid=uuid4(),
        identifier='broken',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='nodes.tests.test_shape_rules.BrokenRuleTestNode'),
            output_ports=[OutputPortDef(id=uuid4(), identifier='default', unit=_unit('t/a'))],
        ),
    )
    graph = _build([target], [])
    with pytest.raises(ShapeRuleError, match=r'BrokenRuleTestNode.*broken.*unknown input value'):
        _ = graph.shape_rule_compilation


class GWPLikeTestNode(Node):
    """Test-only consumes/produces exemplar until a production class needs DimensionTransformRule."""

    @classmethod
    def shape_rules(cls, meta):  # noqa: ANN206
        ghg = meta.graph.require_dimension('greenhouse_gases')
        return (
            DimensionTransformRule(
                input=meta.spec.input_ports[0].id,
                output=meta.spec.output_ports[0].id,
                requires=frozenset({ghg.id}),
                consumes=frozenset({ghg.id}),
            ),
        )


def test_dimension_transform_rule_compiles_against_the_dimension_registry() -> None:
    ghg = DimensionMeta(id=uuid4(), identifier='greenhouse_gases')
    input_port = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    target = NodeSnapshot(
        uuid=uuid4(),
        identifier='gwp',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='nodes.tests.test_shape_rules.GWPLikeTestNode'),
            input_ports=[input_port],
            output_ports=[OutputPortDef(id=uuid4(), identifier='default', unit=_unit('t/a'))],
        ),
    )
    source, source_port = _source_node('source', 't/a')
    edges = [_edge(source, source_port, target, input_port.id)]

    graph = _build([source, target], edges, dimensions=(ghg,))
    (rule,) = graph.shape_rule_compilation.rules_by_node[target.uuid]
    assert isinstance(rule, DimensionTransformRule)
    assert rule.consumes == {ghg.id}

    without_dimension = _build([source, target], edges)
    with pytest.raises(ValueError, match='no dimension'):
        _ = without_dimension.shape_rule_compilation


def test_rule_model_invariants() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError, match='subset of requires'):
        DimensionTransformRule(input=uuid4(), output=uuid4(), consumes=frozenset({uuid4()}))
    shared = uuid4()
    with pytest.raises(pydantic.ValidationError, match='disjoint'):
        DimensionTransformRule(
            input=uuid4(),
            output=uuid4(),
            requires=frozenset({shared}),
            produces=frozenset({shared}),
        )
    with pytest.raises(pydantic.ValidationError):
        SameShapeRule(inputs=(), output=uuid4())


def test_pipeline_operations_compile_through_intermediates() -> None:
    node_uuid = uuid4()
    port_a = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    port_b = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    output_id = uuid4()
    spec = NodeSpec(
        input_ports=[port_a, port_b],
        output_ports=[OutputPortDef(id=output_id, identifier='default', unit=_unit('t/a'))],
    )
    operations: list[AnyOperationSpec] = [
        AddOperationSpec(input=PortInputRef(port=port_a.id), values=[PortInputRef(port=port_b.id)], result_id='sum'),
        MultiplyOperationSpec(
            input=IntermediateInputRef(ref='sum'),
            values=[ScalarValue(value=2.0, dimensionless=True)],
            result_id='output',
        ),
    ]

    rules, notes = compile_pipeline_operations(
        node_uuid=node_uuid,
        spec=spec,
        operations=operations,
        output_port_id=output_id,
        output_ref='output',
    )

    assert notes == ()
    intermediate = uuid5(node_uuid, 'pipeline-intermediate:sum')
    assert rules == (
        SameShapeRule(inputs=(port_a.id, port_b.id), output=intermediate),
        ProductShapeRule(inputs=(intermediate,), output=output_id),
    )

    # The compiled chain passes node-rule validation: intermediates resolve without cycles.
    meta = NodeMeta(id=node_uuid, identifier='pipeline', node_class_path='nodes.node.Node', spec=spec)
    from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec

    graph = InstanceGraph(instance_id=uuid4(), metadata=InstanceMetadata(), spec=InstanceModelSpec(), nodes=(meta,))
    _validate_node_rules(graph, meta, rules)


def test_pipeline_divide_compiles_the_divisor_as_an_inverse_input() -> None:
    from nodes.pipeline.ops.arithmetic import DivideOperationSpec

    port_a = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    port_b = InputPortDef(id=uuid4(), unit=_unit('cap'))
    output_id = uuid4()
    spec = NodeSpec(
        input_ports=[port_a, port_b],
        output_ports=[OutputPortDef(id=output_id, identifier='default', unit=_unit('t/a/cap'))],
    )
    rules, notes = compile_pipeline_operations(
        node_uuid=uuid4(),
        spec=spec,
        operations=[DivideOperationSpec(input=PortInputRef(port=port_a.id), other=PortInputRef(port=port_b.id))],
        output_port_id=output_id,
    )
    assert notes == ()
    assert rules == (ProductShapeRule(inputs=(port_a.id,), inverse_inputs=(port_b.id,), output=output_id),)

    # A scalar divisor is shape-neutral: only the numerator remains.
    rules, notes = compile_pipeline_operations(
        node_uuid=uuid4(),
        spec=spec,
        operations=[
            DivideOperationSpec(input=PortInputRef(port=port_a.id), other=ScalarValue(value=2.0, dimensionless=True)),
        ],
        output_port_id=output_id,
    )
    assert rules == (ProductShapeRule(inputs=(port_a.id,), output=output_id),)


def test_product_rule_validates_inverse_inputs_too() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError, match='at least one operand'):
        ProductShapeRule(output=uuid4())

    target = NodeSnapshot(
        uuid=uuid4(),
        identifier='broken',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='nodes.tests.test_shape_rules.BrokenInverseRuleTestNode'),
            input_ports=[InputPortDef(id=uuid4(), unit=_unit('t/a'))],
            output_ports=[OutputPortDef(id=uuid4(), identifier='default', unit=_unit('t/a'))],
        ),
    )
    graph = _build([target], [])
    with pytest.raises(ShapeRuleError, match=r'unknown input value'):
        _ = graph.shape_rule_compilation


class BrokenInverseRuleTestNode(Node):
    @classmethod
    def shape_rules(cls, meta):  # noqa: ANN206
        return (
            ProductShapeRule(
                inputs=(meta.spec.input_ports[0].id,),
                inverse_inputs=(uuid4(),),
                output=meta.spec.output_ports[0].id,
            ),
        )


def test_pipeline_dataset_reference_suppresses_the_rule_with_a_note() -> None:
    port_a = InputPortDef(id=uuid4(), unit=_unit('t/a'))
    output_id = uuid4()
    spec = NodeSpec(
        input_ports=[port_a],
        output_ports=[OutputPortDef(id=output_id, identifier='default', unit=_unit('t/a'))],
    )
    rules, notes = compile_pipeline_operations(
        node_uuid=uuid4(),
        spec=spec,
        operations=[
            MultiplyOperationSpec(input=PortInputRef(port=port_a.id), values=[DatasetInputRef(dataset='some_dataset')]),
        ],
        output_port_id=output_id,
    )
    assert rules == ()
    assert len(notes) == 1
    assert 'not compilable' in notes[0]


def test_inferred_roles_are_derived_and_recomputed_after_reload() -> None:
    factor_port = InputPortDef(id=uuid4(), unit=_unit('kg/vkm'))
    target, _output_id = _multiplicative_target([factor_port])
    source, source_port = _source_node('source', 'kg/vkm')
    graph = _build([source, target], [_edge(source, source_port, target, factor_port.id)])
    assert graph.node_by_id[target.uuid].inferred_port_roles == {factor_port.id: 'factors'}

    dumped = _dump_graph(graph)
    assert b'inferred_port_roles' not in dumped  # derived state stays out of the serialized graph

    reloaded = _load_graph(dumped)
    assert reloaded == graph
    assert reloaded.node_by_id[target.uuid].inferred_port_roles == {factor_port.id: 'factors'}
    assert [d.code for d in reloaded.node_by_id[target.uuid].port_role_diagnostics] == ['inferred_port_role']
    assert reloaded.shape_rule_compilation.rules_by_node == graph.shape_rule_compilation.rules_by_node


def test_multiplicative_and_additive_defaults_match_designed_creation_shape() -> None:
    factors = MultiplicativeNode.factors_port
    additive = MultiplicativeNode.additive_port
    assert factors.repeatable
    assert not factors.multi
    assert factors.effective_default_count == 2
    assert additive.multi
    assert not additive.repeatable
    assert additive.effective_default_count == 1
    assert AdditiveNode.additive_port.multi
    assert AdditiveNode.impute_port.effective_default_count == 0
