"""
Tests for the authoritative ``NodeInputPortBinding`` table and its write service.

What matters: ``reconcile_input_bindings()`` preserves row identity (UUID and
pk) across full rewrites, per-port positions stay dense and ordered through
the row-level helpers, ``build_instance_snapshot()`` reads bindings and
positions straight from the rows, and the ``annotate_ports()`` projection
serves the same defs the graph builder derives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from kausal_common.datasets.tests.factories import DatasetFactory, DatasetMetricFactory

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import DatasetPortSpec, InputDatasetDef
from nodes.input_bindings import (
    compact_port_positions,
    next_dataset_index,
    next_port_position,
    reconcile_input_bindings,
)
from nodes.models import NodeConfig, NodeInputPortBinding
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


def _make_edge_binding(
    ic: InstanceConfig,
    source: NodeConfig,
    target: NodeConfig,
    port_id,
    *,
    position: int = 0,
    tags: list[str] | None = None,
) -> NodeInputPortBinding:
    return NodeInputPortBinding.objects.create(
        instance=ic,
        node=target,
        port_id=port_id,
        position=position,
        source_node=source,
        source_port_id=uuid4(),
        tags=tags or [],
    )


def _make_dataset_binding(
    ic: InstanceConfig,
    node: NodeConfig,
    port_id,
    *,
    dataset: DatasetModel | None = None,
    dataset_index: int = 0,
    position: int = 0,
) -> NodeInputPortBinding:
    if dataset is None:
        dataset = DatasetFactory.create(identifier=f'ds_{uuid4().hex[:8]}', scope=ic)
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='Value', label='Value', unit='kt/a')
    spec = DatasetPortSpec.from_input_dataset(InputDatasetDef(id=dataset.identifier or 'ds', forecast_from=2025))
    return NodeInputPortBinding.objects.create(
        instance=ic,
        node=node,
        port_id=port_id,
        position=position,
        dataset=dataset,
        metric=metric,
        transformations=list(spec.transformations),
        tags=list(spec.tags),
        dataset_spec=spec,
        dataset_index=dataset_index,
    )


def test_reconcile_is_idempotent_and_preserves_rows(ic: InstanceConfig):
    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source = NodeConfigFactory.create(instance=ic, identifier='source')
    existing = _make_edge_binding(ic, source, target, uuid4())

    desired = [
        NodeInputPortBinding(
            instance=ic,
            node=target,
            port_id=existing.port_id,
            position=0,
            source_node=source,
            source_port_id=existing.source_port_id,
            uuid=existing.uuid,
        )
    ]
    assert reconcile_input_bindings(ic, desired) == 0

    # A changed field updates the row in place.
    desired[0].tags = ['additive']
    assert reconcile_input_bindings(ic, desired) == 1
    row = NodeInputPortBinding.objects.get(instance=ic)
    assert (row.pk, row.uuid) == (existing.pk, existing.uuid)
    assert row.tags == ['additive']


def test_reconcile_creates_and_deletes_by_uuid(ic: InstanceConfig):
    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source = NodeConfigFactory.create(instance=ic, identifier='source')
    doomed = _make_edge_binding(ic, source, target, uuid4())

    fresh = NodeInputPortBinding(
        instance=ic,
        node=target,
        port_id=uuid4(),
        position=0,
        source_node=source,
        source_port_id=uuid4(),
    )
    assert reconcile_input_bindings(ic, [fresh]) == 2
    assert not NodeInputPortBinding.objects.filter(instance=ic, uuid=doomed.uuid).exists()
    assert NodeInputPortBinding.objects.filter(instance=ic, uuid=fresh.uuid).exists()


def test_reconcile_position_swap_is_legal(ic: InstanceConfig):
    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source = NodeConfigFactory.create(instance=ic, identifier='source')
    port_id = uuid4()
    first = _make_edge_binding(ic, source, target, port_id, position=0)
    second = _make_edge_binding(ic, source, target, port_id, position=1)

    swapped = [
        NodeInputPortBinding(
            instance=ic,
            node=target,
            port_id=port_id,
            position=1 - row.position,
            source_node=source,
            source_port_id=row.source_port_id,
            uuid=row.uuid,
        )
        for row in (first, second)
    ]
    assert reconcile_input_bindings(ic, swapped) == 2
    positions = {row.uuid: row.position for row in NodeInputPortBinding.objects.filter(instance=ic)}
    assert positions == {first.uuid: 1, second.uuid: 0}


def test_position_helpers(ic: InstanceConfig):
    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source = NodeConfigFactory.create(instance=ic, identifier='source')
    port_id = uuid4()

    assert next_port_position(target, port_id) == 0
    kept = _make_edge_binding(ic, source, target, port_id, position=0)
    doomed = _make_edge_binding(ic, source, target, port_id, position=1)
    tail = _make_edge_binding(ic, source, target, port_id, position=2)
    assert next_port_position(target, port_id) == 3

    doomed.delete()
    compact_port_positions(target, [port_id])
    positions = {row.uuid: row.position for row in NodeInputPortBinding.objects.filter(node=target)}
    assert positions == {kept.uuid: 0, tail.uuid: 1}


def test_next_dataset_index_counts_only_dataset_bindings(ic: InstanceConfig):
    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source = NodeConfigFactory.create(instance=ic, identifier='source')
    assert next_dataset_index(target) == 0
    _make_edge_binding(ic, source, target, uuid4())
    assert next_dataset_index(target) == 0
    _make_dataset_binding(ic, target, uuid4(), dataset_index=2)
    assert next_dataset_index(target) == 3


def test_snapshot_reads_bindings_and_positions_from_rows(ic: InstanceConfig):
    from nodes.instance_graph import build_instance_graph
    from nodes.instance_serialization import DatasetPortSnapshot, EdgeSnapshot, build_instance_snapshot

    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source = NodeConfigFactory.create(instance=ic, identifier='source')
    shared_port = uuid4()
    other_port = uuid4()

    edge = _make_edge_binding(ic, source, target, shared_port, position=0, tags=['b'])
    row = _make_dataset_binding(ic, target, shared_port, position=1)
    other = _make_dataset_binding(ic, target, other_port, dataset_index=1, position=0)

    snapshot = build_instance_snapshot(ic)
    by_uuid = {b.uuid: b for b in snapshot.bindings}
    assert by_uuid.keys() == {edge.uuid, row.uuid, other.uuid}
    assert by_uuid[edge.uuid].position == 0
    assert by_uuid[row.uuid].position == 1
    assert by_uuid[other.uuid].position == 0
    edge_snap = by_uuid[edge.uuid]
    assert isinstance(edge_snap, EdgeSnapshot)
    assert edge_snap.tags == ['b']
    row_snap = by_uuid[row.uuid]
    assert isinstance(row_snap, DatasetPortSnapshot)
    assert row_snap.dataset_index == 0
    assert row_snap.spec.forecast_from == 2025

    graph = build_instance_graph(snapshot)
    graph_positions = {binding.id: binding.position for binding in graph.bindings}
    assert graph_positions == {edge.uuid: 0, row.uuid: 1, other.uuid: 0}


def test_annotate_ports_serves_defs_from_the_rows(ic: InstanceConfig):
    target = NodeConfigFactory.create(instance=ic, identifier='consumer')
    source = NodeConfigFactory.create(instance=ic, identifier='source')
    port_id = uuid4()
    edge = _make_edge_binding(ic, source, target, port_id, position=0, tags=['additive'])
    row = _make_dataset_binding(ic, target, port_id, position=1)

    annotated = NodeConfig.objects.get_queryset().filter(pk=target.pk).annotate_ports().get()

    (edge_def,) = annotated.port_edge_bindings
    assert edge_def.id == edge.uuid
    assert edge_def.position == 0
    assert edge_def.port_ref.node_uuid == target.uuid
    assert edge_def.port_ref.port_id == port_id
    assert edge_def.from_ref.node_uuid == source.uuid
    assert edge_def.from_ref.port_id == edge.source_port_id
    assert edge_def.tags == ['additive']

    (dataset_def,) = annotated.port_dataset_bindings
    assert dataset_def.id == row.uuid
    assert dataset_def.position == 1
    assert row.dataset is not None
    assert row.metric is not None
    assert dataset_def.dataset_uuid == row.dataset.uuid
    assert dataset_def.metric_uuid == row.metric.uuid
    assert dataset_def.forecast_from == 2025

    # The source side sees its outgoing edge through the same projection.
    source_annotated = NodeConfig.objects.get_queryset().filter(pk=source.pk).annotate_ports().get()
    assert [b.id for b in source_annotated.port_edge_bindings] == [edge.uuid]
    assert source_annotated.port_dataset_bindings == []
