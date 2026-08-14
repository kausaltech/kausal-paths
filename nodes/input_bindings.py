"""
Transitional mirror maintenance for the unified ``NodeInputPortBinding`` table.

``NodeEdge`` and ``DatasetPort`` are still the authoritative stores for what
is bound to a node's input ports; ``NodeInputPortBinding`` is a derived
mirror of both, giving every delivered value one identity (the legacy row's
UUID) and one shared per-port ``position``. Write paths call
``sync_input_bindings()`` at their transaction boundary; once writes move to
the unified table, authority flips and this module goes away.

Position assignment goes through ``ordered_binding_snapshots()`` — the same
function ``build_instance_graph()`` uses — so the mirror can never disagree
with graph-derived binding order.
"""

from typing import TYPE_CHECKING

from django.db import transaction

from nodes.instance_serialization import (
    DatasetPortSnapshot,
    EdgeSnapshot,
    dataset_port_qs_for,
    edge_qs_for,
    ordered_binding_snapshots,
)
from nodes.models import DatasetPort, NodeConfig, NodeEdge, NodeInputPortBinding

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django.db.models import Model

    from nodes.models import InstanceConfig

#: Writes touching these models can change what the mirror derives; the
#: ``change_operation`` exit hook resyncs when one of them was recorded.
#: ``NodeConfig`` is included because deleting a node cascades its bindings
#: without per-row change records.
BINDING_SOURCE_MODELS: tuple[type, ...] = (NodeEdge, DatasetPort, NodeConfig)

_MIRROR_FIELDS = (
    'node_id',
    'port_id',
    'position',
    'source_node_id',
    'source_port_id',
    'dataset_id',
    'metric_id',
    'transformations',
    'tags',
)


def models_affect_input_bindings(model_classes: Iterable[type[Model]]) -> bool:
    return any(issubclass(cls, BINDING_SOURCE_MODELS) for cls in model_classes)


def _desired_bindings(ic: InstanceConfig) -> list[NodeInputPortBinding]:
    """Derive the mirror rows the legacy tables currently imply, in position order."""
    edge_rows = list(edge_qs_for(ic))
    port_rows = list(dataset_port_qs_for(ic))
    edges_by_uuid = {e.uuid: e for e in edge_rows}
    ports_by_uuid = {p.uuid: p for p in port_rows}

    edge_snapshots = [EdgeSnapshot.from_model(e) for e in edge_rows]
    port_snapshots = [DatasetPortSnapshot.from_model(p) for p in port_rows]

    desired: list[NodeInputPortBinding] = []
    for item, position in ordered_binding_snapshots(edge_snapshots, port_snapshots):
        assert item.uuid is not None  # from_model always carries the row UUID
        if isinstance(item, DatasetPortSnapshot):
            port = ports_by_uuid[item.uuid]
            desired.append(
                NodeInputPortBinding(
                    uuid=port.uuid,
                    instance=ic,
                    node_id=port.node_id,
                    port_id=port.port_id,
                    position=position,
                    dataset_id=port.dataset_id,
                    metric_id=port.metric_id,
                    transformations=list(port.spec.transformations),
                    tags=list(port.spec.tags),
                )
            )
            continue
        edge = edges_by_uuid[item.uuid]
        desired.append(
            NodeInputPortBinding(
                uuid=edge.uuid,
                instance=ic,
                node_id=edge.to_node_id,
                port_id=edge.to_port,
                position=position,
                source_node_id=edge.from_node_id,
                source_port_id=edge.from_port,
                transformations=list(edge.transformations or []),
                tags=list(edge.tags or []),
            )
        )
    return desired


def _mirror_state(binding: NodeInputPortBinding) -> tuple[object, ...]:
    return tuple(getattr(binding, field) for field in _MIRROR_FIELDS)


def sync_input_bindings(ic: InstanceConfig) -> int:
    """
    Reconcile the ``NodeInputPortBinding`` mirror with the legacy tables.

    Idempotent and UUID-preserving: a binding keeps its identity across
    reorders and rebuilds because the row UUID comes from the legacy row.
    Returns the number of rows created, updated or deleted. Position swaps
    within one transaction are legal — the position uniqueness constraint
    is deferred to commit.
    """
    desired = _desired_bindings(ic)
    desired_by_uuid = {b.uuid: b for b in desired}
    with transaction.atomic():
        existing_by_uuid = {b.uuid: b for b in NodeInputPortBinding.objects.filter(instance=ic)}

        removed_uuids = existing_by_uuid.keys() - desired_by_uuid.keys()
        to_create = [b for uuid, b in desired_by_uuid.items() if uuid not in existing_by_uuid]
        to_update: list[NodeInputPortBinding] = []
        for uuid, wanted in desired_by_uuid.items():
            current = existing_by_uuid.get(uuid)
            if current is None or _mirror_state(current) == _mirror_state(wanted):
                continue
            for field in _MIRROR_FIELDS:
                setattr(current, field, getattr(wanted, field))
            to_update.append(current)

        if removed_uuids:
            NodeInputPortBinding.objects.filter(instance=ic, uuid__in=removed_uuids).delete()
        if to_update:
            NodeInputPortBinding.objects.bulk_update(to_update, _MIRROR_FIELDS)
        if to_create:
            NodeInputPortBinding.objects.bulk_create(to_create)

    return len(removed_uuids) + len(to_update) + len(to_create)
