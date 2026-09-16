import pytest

from kausal_common.i18n.pydantic import TranslatedString

from nodes.explanation_inputs import ExplainedEdge, ExplainedNode, ExplainedParam
from nodes.explanations import BasketRule, NodeExplanationSystem, explanation_to_html
from nodes.simple import AdditiveNode
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def _make_context(identifier: str):
    instance = InstanceFactory.create(id=identifier, name=identifier)
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name=identifier)
    return instance.context


def _explain(node_id: str, nodes: list[ExplainedNode]):
    context = _make_context(f'basket-impute-{node_id}')
    for node_def in nodes:
        if node_def.id == node_id:
            continue  # The node under test doesn't need a real Node instance; its inputs do.
        assert node_def.unit is not None
        node = AdditiveNode(
            id=node_def.id,
            context=context,
            name=TranslatedString(node_def.id, default_language='en'),
            unit=unit_registry.parse_units(node_def.unit),
            quantity=node_def.quantity,
        )
        context.add_node(node)
    nes = NodeExplanationSystem(context, nodes)
    nes.generate_input_baskets()
    context.node_explanation_system = nes
    node_def = next(n for n in nodes if n.id == node_id)
    return BasketRule().explain(node_def, context)


def _base_nodes() -> list[ExplainedNode]:
    return [
        ExplainedNode(id='source_a', node_class='simple.AdditiveNode', unit='kWh', quantity='energy'),
        ExplainedNode(id='impute_source', node_class='simple.AdditiveNode', unit='kWh', quantity='energy'),
    ]


def _render_parameter_explanation(value: float) -> str:
    context = _make_context(f'parameter-explanation-{type(value).__name__}')
    node = ExplainedNode(
        id='target',
        node_class='simple.ChpAction',
        unit='%',
        quantity='fraction',
        params=(
            ExplainedParam(id='operations', value='add'),
            ExplainedParam(id='t_supply', value=value),
        ),
    )
    nes = NodeExplanationSystem(context, [node])
    context.node_explanation_system = nes
    nes.generate_input_baskets()
    explanation = nes.generate_explanations()['target']
    return ''.join(explanation_to_html(explanation))


def test_integral_float_parameter_has_source_neutral_explanation():
    integer_explanation = _render_parameter_explanation(373)
    float_explanation = _render_parameter_explanation(373.0)

    assert float_explanation == integer_explanation
    assert '373.0' not in float_explanation


def test_impute_description_absent_when_no_impute_tagged_input():
    nodes = [
        *_base_nodes(),
        ExplainedNode(
            id='no_impute_node',
            node_class='generic.GenericNode',
            unit='kWh',
            quantity='energy',
            params=(ExplainedParam(id='operations', value='get_single_dataset,impute'),),
            inputs=(ExplainedEdge(source_id='source_a'),),
        ),
    ]
    explanation = _explain('no_impute_node', nodes)
    assert not any('impute' in f.lower() for f in explanation.functions)


def test_impute_description_present_when_impute_tagged_input_exists():
    nodes = [
        *_base_nodes(),
        ExplainedNode(
            id='with_impute_node',
            node_class='generic.GenericNode',
            unit='kWh',
            quantity='energy',
            params=(ExplainedParam(id='operations', value='get_single_dataset,impute'),),
            inputs=(ExplainedEdge(source_id='source_a'), ExplainedEdge(source_id='impute_source', tags=('impute',))),
        ),
    ]
    explanation = _explain('with_impute_node', nodes)
    assert any('<b>impute</b>' in f for f in explanation.functions)
