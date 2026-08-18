import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from django.core.cache import cache
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

import pytest

from kausal_common.datasets.tests.factories import DatasetFactory, DatasetMetricFactory

from paths.context import PathsObjectCache

from nodes.defs.binding_def import DatasetBindingDef, EdgeBindingDef, NodePortRef
from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec
from nodes.defs.node_defs import NodeSpec
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.instance_graph import InstanceGraph, NodeMeta, build_instance_graph
from nodes.instance_graph_cache import (
    _dump_graph,
    _load_graph,
    get_instance_graph,
    resolve_instance_source,
)
from nodes.instance_serialization import EdgeSnapshot, InstanceSnapshot, NodeSnapshot, build_instance_snapshot
from nodes.models import NodeInputPortBinding, PreferredInstanceSource
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def _snapshot() -> InstanceSnapshot:
    source_id, target_id = uuid4(), uuid4()
    output_id, input_id = uuid4(), uuid4()
    unit = unit_registry.parse_units('t/a')
    return InstanceSnapshot(
        metadata=InstanceMetadata(uuid=uuid4(), identifier='test', name='Test'),
        spec=InstanceModelSpec(),
        nodes=[
            NodeSnapshot(
                uuid=source_id,
                identifier='source',
                spec=NodeSpec(output_ports=[OutputPortDef(id=output_id, unit=unit)]),
            ),
            NodeSnapshot(
                uuid=target_id,
                identifier='target',
                spec=NodeSpec(input_ports=[InputPortDef(id=input_id, unit=unit)]),
            ),
        ],
        bindings=[
            EdgeSnapshot(
                uuid=uuid4(),
                from_node=source_id,
                from_port=output_id,
                to_node=target_id,
                to_port=input_id,
            )
        ],
    )


def _database_config():
    instance = InstanceFactory.create()
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        spec=InstanceModelSpec(),
    )


def test_graph_round_trip_binds_children_and_omits_derived_state() -> None:
    graph = build_instance_graph(_snapshot())

    assert graph.nodes[0].graph is graph
    assert graph.bindings[0].graph is graph
    assert graph.topological_order == tuple(node.id for node in graph.nodes)

    dumped = _dump_graph(graph)
    public = json.loads(dumped)
    assert 'node_by_id' not in public
    assert 'topological_order' not in public
    assert 'diagnostics' not in public
    assert '_graph' not in public
    assert all('_graph' not in node for node in public['nodes'])
    assert all('_graph' not in binding for binding in public['bindings'])

    reloaded = _load_graph(dumped)
    assert reloaded == graph
    assert reloaded.model_dump(mode='json') == graph.model_dump(mode='json')
    assert reloaded.nodes[0].graph is reloaded
    assert reloaded.bindings[0].graph is reloaded


def test_graph_owned_values_reject_unbound_navigation_rebinding_and_copy() -> None:
    node = NodeMeta(
        id=uuid4(),
        identifier='node',
        node_class_path='nodes.simple.AdditiveNode',
        spec=NodeSpec(),
    )
    with pytest.raises(RuntimeError, match='not bound'):
        _ = node.graph

    first = InstanceGraph(instance_id=uuid4(), metadata=InstanceMetadata(), spec=InstanceModelSpec(), nodes=(node,))
    second = InstanceGraph(instance_id=uuid4(), metadata=InstanceMetadata(), spec=InstanceModelSpec())
    with pytest.raises(RuntimeError, match='already bound'):
        node._bind_graph(second)
    with pytest.raises(RuntimeError, match='cannot be copied'):
        node.model_copy()
    assert node.graph is first


def test_invalid_references_are_diagnostics_but_cycles_are_not_orderable() -> None:
    graph = build_instance_graph(_snapshot())
    source, target = graph.nodes
    broken = EdgeBindingDef(
        id=uuid4(),
        port_ref=NodePortRef(node_uuid=source.id, node_id='source', port_id=uuid4()),
        from_ref=NodePortRef(
            node_uuid=target.id,
            node_id='target',
            port_id=target.spec.input_ports[0].id,
        ),
    )
    graph_data = graph.model_dump(mode='python')
    graph_data['bindings'] = [*graph_data['bindings'], broken.model_dump(mode='python')]
    cyclic = InstanceGraph.model_validate(graph_data)

    codes = {diagnostic.code for diagnostic in cyclic.diagnostics}
    assert {'unknown_input_port', 'unknown_output_port', 'directed_cycle'} <= codes
    with pytest.raises(ValueError, match='directed cycle'):
        _ = cyclic.topological_order


def test_draft_graph_cache_round_trips_through_l2_and_invalidates(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _database_config()
    object_cache = PathsObjectCache()
    source = resolve_instance_source(config, PreferredInstanceSource.DRAFT)
    cache.delete(source.cache_key)

    first = get_instance_graph(config, PreferredInstanceSource.DRAFT, object_cache=object_cache)
    object_cache.instance_graphs.clear()
    monkeypatch.setattr(
        'nodes.instance_graph_cache._build_graph',
        lambda *_args, **_kwargs: pytest.fail('L2 cache miss'),
    )
    second = get_instance_graph(
        config,
        PreferredInstanceSource.DRAFT,
        object_cache=object_cache,
        snapshot_loader=lambda: pytest.fail('L2 hit must not load a snapshot'),
    )
    assert second is not first
    assert second.model_dump(mode='json') == first.model_dump(mode='json')

    old_key = source.cache_key
    config.cache_invalidated_at = timezone.now() + timedelta(seconds=1)
    config.save(update_fields=['cache_invalidated_at'])
    updated = resolve_instance_source(config, PreferredInstanceSource.DRAFT)
    assert updated.cache_key != old_key


def test_source_keys_track_published_revision_and_yaml_content(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _database_config()
    config.publish_instance()
    config.refresh_from_db()
    published = resolve_instance_source(config, PreferredInstanceSource.PUBLISHED)

    config.cache_invalidated_at = timezone.now() + timedelta(seconds=1)
    config.save(update_fields=['cache_invalidated_at'])
    assert resolve_instance_source(config, PreferredInstanceSource.PUBLISHED) == published

    config.publish_instance()
    config.refresh_from_db()
    assert resolve_instance_source(config, PreferredInstanceSource.PUBLISHED) != published

    config.config_source = 'yaml'
    monkeypatch.setattr(type(config), 'get_yaml_config_entrypoint', lambda _self: Path('test.yaml').resolve())
    yaml_config = SimpleNamespace(meta=SimpleNamespace(mtime_hash='first'))
    monkeypatch.setattr('nodes.instance_graph_cache._load_yaml_config', lambda _config: yaml_config)
    first_yaml = resolve_instance_source(config, PreferredInstanceSource.DRAFT)
    yaml_config.meta.mtime_hash = 'second'
    assert resolve_instance_source(config, PreferredInstanceSource.DRAFT) != first_yaml


def test_graph_construction_does_not_query_dataset_payloads() -> None:
    config = _database_config()
    unit = unit_registry.parse_units('t/a')
    port_id = uuid4()
    node = NodeConfigFactory.create(
        instance=config,
        spec=NodeSpec(input_ports=[InputPortDef(id=port_id, unit=unit)]),
    )
    dataset = DatasetFactory.create(identifier='structural', scope=config)
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='value', unit='t/a')
    NodeInputPortBinding.objects.create(
        instance=config,
        node=node,
        port_id=port_id,
        dataset=dataset,
        metric=metric,
    )

    with CaptureQueriesContext(connection) as queries:
        graph = get_instance_graph(config, PreferredInstanceSource.DRAFT, refresh=True)

    assert len(graph.datasets) == 1
    binding = graph.bindings[0]
    assert isinstance(binding, DatasetBindingDef)
    assert binding.target_node.id == node.uuid
    assert binding.target_port.id == port_id
    assert binding.dataset.id == dataset.uuid
    assert binding.metric.id == metric.uuid
    sql = '\n'.join(query['sql'].lower() for query in queries.captured_queries)
    assert 'data_point' not in sql
    assert 'datapoint' not in sql


def test_snapshot_catalog_keeps_dataset_references_stable_across_renames() -> None:
    config = _database_config()
    unit = unit_registry.parse_units('t/a')
    port_id = uuid4()
    node = NodeConfigFactory.create(
        instance=config,
        spec=NodeSpec(input_ports=[InputPortDef(id=port_id, unit=unit)]),
    )
    dataset = DatasetFactory.create(identifier='before-dataset', scope=config)
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='before_metric', unit='t/a')
    stored = NodeInputPortBinding.objects.create(
        instance=config,
        node=node,
        port_id=port_id,
        dataset=dataset,
        metric=metric,
    )
    snapshot = build_instance_snapshot(config)

    dataset.identifier = 'after-dataset'
    dataset.save(update_fields=['identifier'])
    metric.name = 'after_metric'
    metric.save(update_fields=['name'])

    graph = build_instance_graph(snapshot)
    binding = graph.binding_by_id[stored.uuid]
    assert isinstance(binding, DatasetBindingDef)
    assert binding.dataset.id == dataset.uuid
    assert binding.dataset.identifier == 'before-dataset'
    assert binding.metric.id == metric.uuid
    assert binding.metric.identifier == 'before_metric'
