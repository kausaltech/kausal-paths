from __future__ import annotations

from pydantic import ValidationError

import pytest

from nodes.actions.action import ImpactGraphType, ImpactOverview, ImpactOverviewSpec
from nodes.normalization import Normalization, NormalizationSpec
from nodes.tests.factories import ContextFactory, NodeFactory
from nodes.visualizations import NodeVisualizations

pytestmark = pytest.mark.django_db


def test_node_visualizations_can_be_parsed_without_runtime_context():
    viz = NodeVisualizations.model_validate([
        {
            'kind': 'node',
            'node_id': 'some_node',
            'desired_outcome': 'decreasing',
            'dimensions': [{'id': 'sector'}],
        }
    ])

    entry = viz.root[0]
    assert entry.id == 'auto'


def test_node_visualizations_runtime_validation_still_assigns_ids():
    context = ContextFactory.create()
    node = NodeFactory.create(context=context)

    viz = NodeVisualizations.model_validate(
        [
            {
                'kind': 'node',
                'node_id': node.id,
                'desired_outcome': 'decreasing',
                'dimensions': [],
            }
        ],
        context=NodeVisualizations.ValidationContext(context=context, node=None, root_node=node),
    )

    entry = viz.root[0]
    assert entry.id.startswith(f'{node.id}:')


def test_impact_overview_can_be_bound_to_runtime_context():
    context = ContextFactory.create()
    effect_node = NodeFactory.create(context=context, quantity='energy')

    spec = ImpactOverviewSpec.model_validate({
        'graph_type': ImpactGraphType.SIMPLE_EFFECT,
        'effect_node_id': effect_node.id,
        'indicator_unit': 'kWh',
    })

    overview = ImpactOverview(spec, context)

    assert overview.effect_node is effect_node
    assert overview.cost_node is None


def test_impact_overview_rejects_forbidden_fields_for_graph_type():
    with pytest.raises(ValidationError, match='must not be used'):
        ImpactOverviewSpec.model_validate({
            'graph_type': ImpactGraphType.SIMPLE_EFFECT,
            'effect_node_id': 'effect_node',
            'indicator_unit': 'kWh',
            'cost_node_id': 'cost_node',
        })


def test_impact_overview_requires_fields_for_graph_type():
    with pytest.raises(ValidationError, match='must be given'):
        ImpactOverviewSpec.model_validate({
            'graph_type': ImpactGraphType.COST_EFFICIENCY,
            'effect_node_id': 'effect_node',
            'indicator_unit': 'EUR/kWh',
        })


def test_cost_benefit_allows_legacy_cost_node_id():
    overview = ImpactOverviewSpec.model_validate({
        'graph_type': ImpactGraphType.COST_BENEFIT,
        'effect_node_id': 'effect_node',
        'cost_node_id': 'legacy_cost_node',
        'indicator_unit': 'MEUR',
    })

    assert overview.cost_node_id == 'legacy_cost_node'


def test_normalization_spec_can_be_bound_to_runtime_context():
    context = ContextFactory.create()
    normalizer_node = NodeFactory.create(context=context, quantity='population')

    spec = NormalizationSpec.model_validate({
        'normalizer_node_id': normalizer_node.id,
        'quantities': [{'id': 'energy', 'unit': 'kWh/cap/a'}],
        'default': True,
    })

    normalization = Normalization(spec, context)

    assert normalization.spec.default is True
    assert normalization.normalizer_node is normalizer_node
