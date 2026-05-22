"""
GraphQL types for the change-tracking audit surface.

Exposes ``InstanceChangeOperation`` + ``InstanceModelLogEntry`` rows
emitted by ``nodes.change_ops.change_operation`` / ``record_change``
to API consumers.

``EditableEntity`` is the interface implemented by types that can be
tracked in the change log (Node, NodeEdge, DatasetPort, …). It carries
``uuid`` + ``changeHistory`` so the UI can fetch per-entity history
without narrowing via ``... on``. The interface is also the ``target``
type on ``InstanceModelLogEntryType``.

Target *kind* is reported as an enum (``ChangeTargetKind``) rather than
a ContentType string so API consumers don't have to deal with
implementation-detail model names.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

import strawberry as sb

from paths import gql

from nodes.models import (
    InstanceChangeOperation,
    InstanceModelLogEntry,
)

if TYPE_CHECKING:
    from django.db.models import Model


@sb.enum
class ChangeTargetKind(Enum):
    """
    Discriminator for the affected entity in a change entry.

    Stable API values — maps to ORM models internally but callers
    never see the ``app_label.model`` strings.
    """

    NODE = 'node'
    EDGE = 'edge'
    DATASET_PORT = 'dataset_port'
    DIMENSION = 'dimension'
    DIMENSION_CATEGORY = 'dimension_category'
    DATA_POINT = 'data_point'
    INSTANCE = 'instance'
    UNKNOWN = 'unknown'


# (app_label, model) → enum. Anything missing maps to UNKNOWN so new
# target types don't error out the query surface — they just render as
# unknown until someone teaches the mapping.
_CT_TO_KIND: dict[tuple[str, str], ChangeTargetKind] = {
    ('nodes', 'nodeconfig'): ChangeTargetKind.NODE,
    ('nodes', 'nodeedge'): ChangeTargetKind.EDGE,
    ('nodes', 'datasetport'): ChangeTargetKind.DATASET_PORT,
    ('nodes', 'instanceconfig'): ChangeTargetKind.INSTANCE,
    ('datasets', 'dimension'): ChangeTargetKind.DIMENSION,
    ('datasets', 'dimensioncategory'): ChangeTargetKind.DIMENSION_CATEGORY,
    ('datasets', 'datapoint'): ChangeTargetKind.DATA_POINT,
}


def _resolve_target_kind(entry: InstanceModelLogEntry) -> ChangeTargetKind:
    ct = entry.content_type
    if ct is None:
        return ChangeTargetKind.UNKNOWN
    return _CT_TO_KIND.get((ct.app_label, ct.model), ChangeTargetKind.UNKNOWN)


def fetch_entity_history(
    django_model: type[Model],
    pk: int,
    *,
    limit: int,
    before: datetime | None = None,
) -> list[InstanceModelLogEntryType]:
    """
    Return log entries targeting a specific ORM row (by ContentType + pk).

    Used by the ``EditableEntity.change_history`` resolver on each
    implementing type. Results are newest-first.
    """
    from django.contrib.contenttypes.models import ContentType

    ct = ContentType.objects.get_for_model(django_model)
    qs = (
        InstanceModelLogEntry.objects
        .filter(content_type=ct, object_id=str(pk))
        .select_related('operation', 'content_type')
        .order_by('-id')
    )
    if before is not None:
        qs = qs.filter(created_at__lt=before)
    return [InstanceModelLogEntryType.from_model(e) for e in qs[:limit]]


@sb.interface
class EditableEntity:
    """
    Shared surface for entities participating in Trailhead's change log.

    Implementing types are at minimum the editable ORM children of an
    InstanceConfig: ``Node``, ``NodeEdge``, ``DatasetPort``. Each carries
    a stable ``uuid`` and a per-entity ``changeHistory`` query.

    ``uuid`` is always populated: DB-backed entities return their
    persisted uuid, runtime-only (YAML) entities fall back to a stable
    uuidv5 derived from their identifier. For the latter,
    ``changeHistory`` returns ``[]`` since no DB row exists to carry
    entries.

    Permission gating: the ``changeHistory`` resolver checks the viewer's
    ``change`` permission on the hosting InstanceConfig; denied readers
    see an empty list. One place for the perm check across all editable
    types — analogous to the ``editor`` sub-type pattern but without
    forcing every read-side query through the edit-time surface.
    """

    uuid: UUID

    @sb.field(description='Row-level change history for this entity, newest first.')
    @staticmethod
    def change_history(
        root: EditableEntity,
        info: gql.Info,
        limit: int = 50,
        before: datetime | None = None,
    ) -> list[InstanceModelLogEntryType]:
        # Each implementing type overrides this with a concrete lookup
        # (django model + pk + instance perm check). The base raises so
        # missing overrides fail loudly rather than silently returning [].
        raise NotImplementedError


@sb.type
class InstanceModelLogEntryType:
    """One row-level change recorded under an ``InstanceChangeOperation``."""

    uuid: UUID
    action: str = sb.field(description="Dotted action id, e.g. 'node.update'.")
    created_at: datetime
    target_uuid: UUID | None = sb.field(description='UUID of the affected entity. Survives deletion of the entity.')

    # Private handle for resolvers below.
    _entry: sb.Private[InstanceModelLogEntry]

    @sb.field(description='Discriminator for the affected entity.')
    @staticmethod
    def target_kind(root: InstanceModelLogEntryType) -> ChangeTargetKind:
        return _resolve_target_kind(root._entry)

    @sb.field(
        graphql_type=EditableEntity | None,
        description='The affected entity if it still exists, null if deleted.',
    )
    @staticmethod
    def target(root: InstanceModelLogEntryType, info: gql.Info) -> Any:
        return _resolve_target(root._entry)

    @sb.field(graphql_type=sb.scalars.JSON | None, description='State prior to the change. Null for create operations.')
    @staticmethod
    def before(root: InstanceModelLogEntryType) -> dict[str, Any] | None:
        return (root._entry.data or {}).get('before')

    @sb.field(graphql_type=sb.scalars.JSON | None, description='State after the change. Null for delete operations.')
    @staticmethod
    def after(root: InstanceModelLogEntryType) -> dict[str, Any] | None:
        return (root._entry.data or {}).get('after')

    @classmethod
    def from_model(cls, entry: InstanceModelLogEntry) -> InstanceModelLogEntryType:
        raw_uuid = (entry.data or {}).get('target_uuid')
        target_uuid: UUID | None = UUID(raw_uuid) if raw_uuid else None
        return cls(
            uuid=entry.uuid,
            action=entry.action,
            created_at=entry.created_at,
            target_uuid=target_uuid,
            _entry=entry,
        )


def _resolve_target(entry: InstanceModelLogEntry) -> Any:
    """
    Resolve the GFK target of a log entry to its GQL object, or ``None``.

    Returns ``None`` when the row no longer exists (e.g. after a delete)
    or when the target kind has no GQL representation yet. The ``before``
    snapshot in the entry data carries what was there for UI fallback.
    """
    from nodes.graphql.types.graph import NodeEdgeType
    from nodes.models import DatasetPort, NodeConfig, NodeEdge

    ct = entry.content_type
    if ct is None or entry.object_id is None:
        return None
    model = ct.model_class()
    if model is None:
        return None
    try:
        pk = int(entry.object_id)
    except TypeError, ValueError:
        return None

    if model is NodeConfig:
        nc = NodeConfig.objects.filter(pk=pk).select_related('instance').first()
        if nc is None:
            return None
        # Resolve via the runtime Node so the UI gets a real Node /
        # ActionNode object it can introspect. The EditableEntity
        # interface only needs uuid + changeHistory, which both
        # concrete Node types provide.
        ic = nc.instance
        instance = ic.get_instance()
        return instance.context.nodes.get(nc.identifier)

    if model is NodeEdge:
        edge = NodeEdge.objects.filter(pk=pk).select_related('from_node', 'to_node').first()
        return NodeEdgeType.from_node_edge(edge) if edge else None

    if model is DatasetPort:
        return DatasetPort.objects.filter(pk=pk).first()

    # Dimension / DimensionCategory / DataPoint / spec-embedded: no GQL
    # representation yet — surface the shape via ``before`` / ``after``.
    return None


@sb.type
class InstanceChangeOperationType:
    """One user-facing edit (grouping anchor for a set of log entries)."""

    uuid: UUID
    action: str = sb.field(description="Top-level action that triggered the operation, e.g. 'node.delete'.")
    source: str = sb.field(description='Transport that initiated the operation (graphql / rest / admin / cli / migration).')
    created_at: datetime
    user_email: str | None = sb.field(description='Email of the user who initiated the operation, or null for system.')
    superseded_by_uuid: UUID | None = sb.field(description='UUID of the operation that undid this one, if any.')

    _operation: sb.Private[InstanceChangeOperation]

    @sb.field(description='Row-level entries bundled under this operation, in insertion order.')
    @staticmethod
    def entries(root: InstanceChangeOperationType) -> list[InstanceModelLogEntryType]:
        entries = InstanceModelLogEntry.objects.filter(operation=root._operation).select_related('content_type').order_by('id')
        return [InstanceModelLogEntryType.from_model(e) for e in entries]

    @classmethod
    def from_model(cls, op: InstanceChangeOperation) -> InstanceChangeOperationType:
        return cls(
            uuid=op.uuid,
            action=op.action,
            source=op.source,
            created_at=op.created_at,
            user_email=op.user.email if op.user else None,
            superseded_by_uuid=op.superseded_by.uuid if op.superseded_by else None,
            _operation=op,
        )
