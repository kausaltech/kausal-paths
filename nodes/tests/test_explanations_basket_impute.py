import pytest

from kausal_common.i18n.pydantic import TranslatedString

from nodes.explanations import BasketRule, NodeExplanationSystem
from nodes.simple import AdditiveNode
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def _make_context(identifier: str):
    instance = InstanceFactory.create(id=identifier, name=identifier)
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name=identifier)
    return instance.context


def _explain(node_id: str, all_node_configs: list[dict]):
    context = _make_context(f'basket-impute-{node_id}')
    for cfg in all_node_configs:
        if cfg['id'] == node_id:
            continue  # The node under test doesn't need a real Node instance; its inputs do.
        node = AdditiveNode(
            id=cfg['id'],
            context=context,
            name=TranslatedString(cfg['id'], default_language='en'),
            unit=unit_registry.parse_units(cfg['unit']),
            quantity=cfg['quantity'],
        )
        context.add_node(node)
    nes = NodeExplanationSystem(context, all_node_configs)
    nes.generate_input_baskets()
    context.node_explanation_system = nes
    node_config = next(n for n in all_node_configs if n['id'] == node_id)
    return BasketRule().explain(node_config, context)


def _base_configs() -> list[dict]:
    return [
        {'id': 'source_a', 'type': 'simple.AdditiveNode', 'unit': 'kWh', 'quantity': 'energy'},
        {'id': 'impute_source', 'type': 'simple.AdditiveNode', 'unit': 'kWh', 'quantity': 'energy'},
    ]


def test_impute_description_absent_when_no_impute_tagged_input():
    configs = [
        *_base_configs(),
        {
            'id': 'no_impute_node',
            'type': 'generic.GenericNode',
            'unit': 'kWh',
            'quantity': 'energy',
            'params': {'operations': 'get_single_dataset,impute'},
            'input_nodes': ['source_a'],
        },
    ]
    explanation = _explain('no_impute_node', configs)
    assert not any('impute' in f.lower() for f in explanation.functions)


def test_impute_description_present_when_impute_tagged_input_exists():
    configs = [
        *_base_configs(),
        {
            'id': 'with_impute_node',
            'type': 'generic.GenericNode',
            'unit': 'kWh',
            'quantity': 'energy',
            'params': {'operations': 'get_single_dataset,impute'},
            'input_nodes': ['source_a', {'id': 'impute_source', 'tags': ['impute']}],
        },
    ]
    explanation = _explain('with_impute_node', configs)
    assert any('<b>impute</b>' in f for f in explanation.functions)
