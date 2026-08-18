"""
Write service for the unified ``NodeInputPortBinding`` table.

``NodeInputPortBinding`` is the authoritative store for what is bound to a
node's input ports; the legacy ``NodeEdge`` / ``DatasetPort`` tables are dead
(kept empty until their removal in plan step 11). Snapshot-driven writers
(sync, import) build the full desired row set and call
``reconcile_input_bindings()``; row-level editors (GraphQL mutations) write
rows directly and keep per-port positions dense with the position helpers.

Position assignment for snapshot-driven writers goes through
``ordered_binding_snapshots()`` — the same function the snapshot upgrader
uses — so no writer can disagree with the canonical delivery order.
"""

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from django.db import transaction
from django.db.models import Max

from nodes.models import NodeInputPortBinding

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nodes.models import InstanceConfig, NodeConfig

_ROW_FIELDS = (
    'node_id',
    'port_id',
    'position',
    'source_node_id',
    'source_port_id',
    'dataset_id',
    'metric_id',
    'transformations',
    'tags',
    'dataset_spec',
    'dataset_index',
)


def _row_state(binding: NodeInputPortBinding) -> tuple[object, ...]:
    return tuple(getattr(binding, field) for field in _ROW_FIELDS)


def reconcile_input_bindings(ic: InstanceConfig, desired: list[NodeInputPortBinding]) -> int:
    """
    Reconcile the instance's binding rows with the desired state.

    Idempotent and UUID-preserving: rows are matched by UUID, updated in
    place, created when new and deleted when gone, so a surviving binding
    keeps both its UUID and its pk across a full rewrite. Returns the number
    of rows created, updated or deleted. Position swaps within one
    transaction are legal — the position uniqueness constraint is deferred
    to commit.
    """
    # The uuid column is db_default-populated, so an unsaved row without an
    # authored/preserved identity carries a sentinel, not a UUID — mint one
    # here or every fresh row would collapse onto the same dict key.
    for binding in desired:
        if not isinstance(binding.uuid, UUID):
            binding.uuid = uuid4()
    desired_by_uuid = {b.uuid: b for b in desired}
    with transaction.atomic():
        existing_by_uuid = {b.uuid: b for b in NodeInputPortBinding.objects.filter(instance=ic)}

        removed_uuids = existing_by_uuid.keys() - desired_by_uuid.keys()
        to_create = [b for uuid, b in desired_by_uuid.items() if uuid not in existing_by_uuid]
        to_update: list[NodeInputPortBinding] = []
        for uuid, wanted in desired_by_uuid.items():
            current = existing_by_uuid.get(uuid)
            if current is None or _row_state(current) == _row_state(wanted):
                continue
            for field in _ROW_FIELDS:
                setattr(current, field, getattr(wanted, field))
            to_update.append(current)

        if removed_uuids:
            NodeInputPortBinding.objects.filter(instance=ic, uuid__in=removed_uuids).delete()
        if to_update:
            NodeInputPortBinding.objects.bulk_update(to_update, _ROW_FIELDS)
        if to_create:
            NodeInputPortBinding.objects.bulk_create(to_create)

    return len(removed_uuids) + len(to_update) + len(to_create)


def next_port_position(nc: NodeConfig, port_id: UUID) -> int:
    """Position for a binding appended to the port."""
    highest = NodeInputPortBinding.objects.filter(node=nc, port_id=port_id).aggregate(highest=Max('position'))['highest']
    return 0 if highest is None else highest + 1


def compact_port_positions(nc: NodeConfig, port_ids: Iterable[UUID]) -> None:
    """
    Renumber the ports' bindings densely (0..n-1), preserving relative order.

    Call after deletes; the uniqueness constraint is deferred, so in-transaction
    swaps are legal.
    """
    for port_id in set(port_ids):
        rows = list(NodeInputPortBinding.objects.filter(node=nc, port_id=port_id).order_by('position'))
        changed = []
        for position, row in enumerate(rows):
            if row.position != position:
                row.position = position
                changed.append(row)
        if changed:
            NodeInputPortBinding.objects.bulk_update(changed, ['position'])


def next_dataset_index(nc: NodeConfig) -> int:
    """Next free binding-group index on the node's ``input_dataset_instances`` list."""
    highest = NodeInputPortBinding.objects.filter(node=nc, dataset__isnull=False).aggregate(highest=Max('dataset_index'))[
        'highest'
    ]
    return 0 if highest is None else highest + 1
