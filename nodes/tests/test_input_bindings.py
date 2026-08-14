"""
Tests for the unified ``NodeInputPortBinding`` mirror.

The mirror is derived from the authoritative ``NodeEdge`` / ``DatasetPort``
tables at every write boundary. What matters: binding UUIDs are the legacy
row UUIDs (identity survives rebuilds), positions reproduce the order
``build_instance_graph()`` observes (edges first, then dataset ports), and
the ``annotate_ports()`` projection serves the same defs the graph would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from kausal_common.datasets.tests.factories import DatasetFactory, DatasetMetricFactory

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import DatasetPortSpec, InputDatasetDef
from nodes.input_bindings import sync_input_bindings
from nodes.models import DatasetPort, NodeConfig, NodeEdge, NodeInputPortBinding
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset as DatasetModel

    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


@pytest.fixture
def ic() -> InstanceConfig:
    instance = InstanceFactory.create()
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        spec=InstanceModelSpec(years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030)),
    )


def _make_dataset_port(
    ic: InstanceConfig,
    node: NodeConfig,
    port_id,
    *,
    dataset: DatasetModel | None = None,
    dataset_index: int = 0,
) -> DatasetPort:
    if dataset is None:
        dataset = DatasetFactory.create(identifier=f'ds_{uuid4().hex[:8]}', scope=ic)
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='Value', label='Value', unit='kt/a')
    return DatasetPort.objects.create(
        instance=ic,
        node=node,
        port_id=port_id,
        dataset=dataset,
        metric=metric,
        spec=DatasetPortSpec.from_input_dataset(InputDatasetDef(id=dataset.identifier or 'ds', forecast_from=2025)),
        dataset_index=dataset_index,
    )


def test_mirror_interleaves_edges_before_dataset_rows_on_a_shared_port(ic: InstanceConfig):
    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source_a = NodeConfigFactory.create(instance=ic, identifier='source_a')
    source_b = NodeConfigFactory.create(instance=ic, identifier='source_b')
    port_id = uuid4()

    # Created dataset-first and b-before-a: edges come first regardless of the
    # dataset row predating them, in their creation order — the authored order
    # the YAML runtime observes — not in source-node pk order.
    row = _make_dataset_port(ic, target, port_id)
    edge_b = NodeEdge.objects.create(
        instance=ic, from_node=source_b, from_port=uuid4(), to_node=target, to_port=port_id, tags=['b']
    )
    edge_a = NodeEdge.objects.create(
        instance=ic, from_node=source_a, from_port=uuid4(), to_node=target, to_port=port_id, tags=['a']
    )

    changed = sync_input_bindings(ic)
    assert changed == 3

    bindings = list(NodeInputPortBinding.objects.filter(instance=ic).order_by('position'))
    assert [b.uuid for b in bindings] == [edge_b.uuid, edge_a.uuid, row.uuid]
    assert [b.position for b in bindings] == [0, 1, 2]

    assert bindings[0].source_node_id == source_b.pk
    assert bindings[0].source_port_id == edge_b.from_port
    assert bindings[0].dataset_id is None
    assert bindings[0].tags == ['b']

    assert bindings[2].source_node_id is None
    assert bindings[2].dataset_id == row.dataset_id
    assert bindings[2].metric_id == row.metric_id
    assert [op.kind for op in bindings[2].transformations] == [op.kind for op in row.spec.transformations]


def test_mirror_positions_match_the_instance_graph(ic: InstanceConfig):
    from nodes.instance_graph import build_instance_graph
    from nodes.instance_serialization import build_instance_snapshot

    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source = NodeConfigFactory.create(instance=ic, identifier='source')
    shared_port = uuid4()
    other_port = uuid4()

    NodeEdge.objects.create(instance=ic, from_node=source, from_port=uuid4(), to_node=target, to_port=shared_port)
    _make_dataset_port(ic, target, shared_port, dataset_index=0)
    _make_dataset_port(ic, target, other_port, dataset_index=1)

    sync_input_bindings(ic)

    graph = build_instance_graph(build_instance_snapshot(ic))
    graph_positions = {binding.id: binding.position for binding in graph.bindings}
    mirror_positions = {b.uuid: b.position for b in NodeInputPortBinding.objects.filter(instance=ic)}

    assert mirror_positions == graph_positions


def test_sync_is_idempotent_and_preserves_rows(ic: InstanceConfig):
    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source = NodeConfigFactory.create(instance=ic, identifier='source')
    NodeEdge.objects.create(instance=ic, from_node=source, from_port=uuid4(), to_node=target, to_port=uuid4())

    assert sync_input_bindings(ic) == 1
    first = NodeInputPortBinding.objects.get(instance=ic)

    assert sync_input_bindings(ic) == 0
    second = NodeInputPortBinding.objects.get(instance=ic)
    assert (second.pk, second.uuid) == (first.pk, first.uuid)


def test_removing_a_binding_renumbers_but_keeps_identity(ic: InstanceConfig):
    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source = NodeConfigFactory.create(instance=ic, identifier='source')
    port_id = uuid4()
    edge = NodeEdge.objects.create(instance=ic, from_node=source, from_port=uuid4(), to_node=target, to_port=port_id)
    row = _make_dataset_port(ic, target, port_id)
    sync_input_bindings(ic)

    survivor_before = NodeInputPortBinding.objects.get(instance=ic, uuid=row.uuid)
    assert survivor_before.position == 1

    edge.delete()
    sync_input_bindings(ic)

    assert not NodeInputPortBinding.objects.filter(instance=ic, uuid=edge.uuid).exists()
    survivor = NodeInputPortBinding.objects.get(instance=ic, uuid=row.uuid)
    assert survivor.position == 0, 'positions are renumbered densely'
    assert survivor.pk == survivor_before.pk, 'reordering does not make a new binding'


def test_annotate_ports_serves_defs_from_the_mirror(ic: InstanceConfig):
    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source = NodeConfigFactory.create(instance=ic, identifier='source')
    port_id = uuid4()
    edge = NodeEdge.objects.create(
        instance=ic, from_node=source, from_port=uuid4(), to_node=target, to_port=port_id, tags=['additive']
    )
    row = _make_dataset_port(ic, target, port_id)
    sync_input_bindings(ic)

    annotated = NodeConfig.objects.get_queryset().filter(pk=target.pk).annotate_ports().get()

    (edge_def,) = annotated.port_edge_bindings
    assert edge_def.id == edge.uuid
    assert edge_def.position == 0
    assert edge_def.port_ref.node_uuid == target.uuid
    assert edge_def.port_ref.port_id == port_id
    assert edge_def.from_ref.node_uuid == source.uuid
    assert edge_def.from_ref.port_id == edge.from_port
    assert edge_def.tags == ['additive']

    (dataset_def,) = annotated.port_dataset_bindings
    assert dataset_def.id == row.uuid
    assert dataset_def.position == 1
    assert dataset_def.dataset_uuid == row.dataset.uuid
    assert dataset_def.metric_uuid == row.metric.uuid
    assert dataset_def.forecast_from == 2025

    # The source side sees its outgoing edge through the same projection.
    source_annotated = NodeConfig.objects.get_queryset().filter(pk=source.pk).annotate_ports().get()
    assert [b.id for b in source_annotated.port_edge_bindings] == [edge.uuid]
    assert source_annotated.port_dataset_bindings == []


def test_change_operation_resyncs_the_mirror_when_bindings_were_touched(ic: InstanceConfig):
    from nodes.change_ops import change_operation, record_change

    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source = NodeConfigFactory.create(instance=ic, identifier='source')

    with change_operation(ic, user=None, action='edge.create'):
        edge = NodeEdge.objects.create(instance=ic, from_node=source, from_port=uuid4(), to_node=target, to_port=uuid4())
        record_change(edge, action='edge.create', before=None, after=edge.serializable_data())

    assert NodeInputPortBinding.objects.filter(instance=ic, uuid=edge.uuid).exists()


def test_change_operation_skips_the_mirror_for_unrelated_writes(ic: InstanceConfig, monkeypatch: pytest.MonkeyPatch):
    import nodes.input_bindings as input_bindings_module
    from nodes.change_ops import change_operation, record_change

    calls: list[InstanceConfig] = []

    def record_call(ic: InstanceConfig) -> int:
        calls.append(ic)
        return 0

    monkeypatch.setattr(input_bindings_module, 'sync_input_bindings', record_call)

    with change_operation(ic, user=None, action='instance.update'):
        record_change(ic, action='instance.update', before={}, after={})

    assert calls == []
