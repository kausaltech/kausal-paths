from __future__ import annotations

import pytest

from nodes.graph_layout import PrimaryLayoutClass
from nodes.models import InstanceConfig
from nodes.tests.factories import AdditiveActionFactory, ContextFactory, SimpleNodeFactory

pytestmark = pytest.mark.django_db


def test_node_graph_classifier_classifies_nodes_for_layout():
    ctx = ContextFactory.create()

    action = AdditiveActionFactory.create(context=ctx, id='action')
    core = SimpleNodeFactory.create(context=ctx, id='core')
    outcome = SimpleNodeFactory.create(context=ctx, id='outcome', is_outcome=True)
    context_root = SimpleNodeFactory.create(context=ctx, id='context_root')
    context_mid = SimpleNodeFactory.create(context=ctx, id='context_mid')
    ghost = SimpleNodeFactory.create(context=ctx, id='ghost')
    hub = SimpleNodeFactory.create(context=ctx, id='hub')
    hub_targets = [SimpleNodeFactory.create(context=ctx, id=f'hub_target_{idx}') for idx in range(7)]

    action.add_output_node(core)
    core.add_output_node(outcome)
    context_root.add_output_node(context_mid)
    context_mid.add_output_node(core)
    ghost.add_output_node(core)
    ghost.add_output_node(outcome)
    for target in hub_targets:
        hub.add_output_node(target)

    ctx.finalize_nodes()
    classifier = ctx.node_graph_classifier

    action_meta = classifier.for_node(action.id)
    core_meta = classifier.for_node(core.id)
    outcome_meta = classifier.for_node(outcome.id)
    context_root_meta = classifier.for_node(context_root.id)
    context_mid_meta = classifier.for_node(context_mid.id)
    ghost_meta = classifier.for_node(ghost.id)
    hub_meta = classifier.for_node(hub.id)

    assert action_meta.primary_class == PrimaryLayoutClass.ACTION
    assert outcome_meta.primary_class == PrimaryLayoutClass.OUTCOME
    assert core_meta.primary_class == PrimaryLayoutClass.CORE
    assert context_root_meta.primary_class == PrimaryLayoutClass.GHOSTABLE_CONTEXT_SOURCE
    assert context_mid_meta.primary_class == PrimaryLayoutClass.CONTEXT_SOURCE
    assert ghost_meta.primary_class == PrimaryLayoutClass.GHOSTABLE_CONTEXT_SOURCE
    assert ghost_meta.ghost_targets == ['core', 'outcome']
    assert hub_meta.is_hub is True
    assert hub_meta.total_degree == 7
    assert classifier.actions == ['action']
    assert classifier.outcomes == ['outcome']
    assert 'core' in classifier.core_nodes
    assert 'ghost' in classifier.ghostable_context_sources
    assert 'context_mid' in classifier.context_sources
    assert 'hub' not in classifier.main_graph_node_ids
    assert 'ghost' not in classifier.main_graph_node_ids
    assert action_meta.topological_layer < core_meta.topological_layer < outcome_meta.topological_layer
    assert context_mid_meta.avg_outgoing_span == 1.0


def test_espoo_2026_graph_layout_smoke():
    try:
        ic = InstanceConfig.objects.get(identifier='espoo-2026')
    except InstanceConfig.DoesNotExist:
        pytest.skip('espoo-2026 instance config not available in test database')

    instance = ic.get_instance()
    classifier = instance.context.node_graph_classifier

    assert classifier.ghostable_context_sources
    for action in instance.context.get_actions():
        assert action.id in classifier.main_graph_node_ids
    for outcome in instance.context.get_outcome_nodes():
        assert outcome.id in classifier.main_graph_node_ids


def test_node_graph_clusterer_finds_action_regions_and_bridge_nodes():
    ctx = ContextFactory.create()

    heat_action = AdditiveActionFactory.create(context=ctx, id='heat_action', node_group='Heating')
    traffic_action = AdditiveActionFactory.create(context=ctx, id='traffic_action', node_group='Transport')
    heat_core = SimpleNodeFactory.create(context=ctx, id='heat_core', node_group='Heating')
    traffic_core = SimpleNodeFactory.create(context=ctx, id='traffic_core', node_group='Transport')
    outcome = SimpleNodeFactory.create(context=ctx, id='outcome', is_outcome=True)
    shared_factor = SimpleNodeFactory.create(context=ctx, id='shared_factor', node_group='Electricity')

    heat_action.add_output_node(heat_core)
    traffic_action.add_output_node(traffic_core)
    heat_core.add_output_node(outcome)
    traffic_core.add_output_node(outcome)
    shared_factor.add_output_node(heat_core)
    shared_factor.add_output_node(traffic_core)

    ctx.finalize_nodes()
    clusterer = ctx.node_graph_clusterer

    heat_cluster = clusterer.for_node('heat_action').cluster_id
    traffic_cluster = clusterer.for_node('traffic_action').cluster_id
    shared_meta = clusterer.for_node('shared_factor')

    assert heat_cluster is not None
    assert traffic_cluster is not None
    assert heat_cluster != traffic_cluster
    assert clusterer.for_node('heat_core').cluster_id == heat_cluster
    assert clusterer.for_node('traffic_core').cluster_id == traffic_cluster
    assert shared_meta.cluster_id in {heat_cluster, traffic_cluster}
    assert shared_meta.cluster_confidence < 0.75
    assert set(shared_meta.neighboring_clusters) == {heat_cluster, traffic_cluster} - {shared_meta.cluster_id}
    assert shared_meta.is_bridge is True
    assert len(clusterer.clusters) == 2
