"""
GraphQL types for the change-tracking audit surface.

Exposes ``InstanceChangeOperation`` + ``InstanceModelLogEntry`` rows
emitted by ``nodes.change_ops.change_operation`` / ``record_change``
to API consumers.

``EditableEntity`` is the interface implemented by types that can be
tracked in the change log (Node, input bindings, …). It carries
the stable ``uuid`` shared by change-log targets and is the ``target``
type on ``InstanceModelLogEntryType``. Entity-specific history lives on
the relevant editor or concrete type.

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
    from django.contrib.contenttypes.models import ContentType
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
    ACTION_GROUP = 'action_group'
    INSTANCE = 'instance'
    UNKNOWN = 'unknown'


# (app_label, model) → enum. Anything missing maps to UNKNOWN so new
# target types don't error out the query surface — they just render as
# unknown until someone teaches the mapping.
_CT_TO_KIND: dict[tuple[str, str], ChangeTargetKind] = {
    ('nodes', 'nodeconfig'): ChangeTargetKind.NODE,
    # Legacy binding tables; live entries carry 'nodeinputportbinding',
    # discriminated by the payload's source kind in _resolve_target_kind.
    ('nodes', 'nodeedge'): ChangeTargetKind.EDGE,
    ('nodes', 'datasetport'): ChangeTargetKind.DATASET_PORT,
    ('nodes', 'instanceconfig'): ChangeTargetKind.INSTANCE,
    ('datasets', 'dimension'): ChangeTargetKind.DIMENSION,
    ('datasets', 'dimensioncategory'): ChangeTargetKind.DIMENSION_CATEGORY,
    ('datasets', 'datapoint'): ChangeTargetKind.DATA_POINT,
}


def _resolve_target_kind(entry: InstanceModelLogEntry) -> ChangeTargetKind:
    if entry.action.startswith('action_group.'):
        return ChangeTargetKind.ACTION_GROUP
    ct = entry.content_type
    if ct is None:
        return ChangeTargetKind.UNKNOWN
    if (ct.app_label, ct.model) == ('nodes', 'nodeinputportbinding'):
        # The unified binding row is edge- or dataset-sourced; the payload
        # snapshot carries the source discriminator even after deletion.
        data = entry.data or {}
        snapshot = data.get('after') or data.get('before') or {}
        source_kind = (snapshot.get('source') or {}).get('kind')
        if source_kind == 'node':
            return ChangeTargetKind.EDGE
        if source_kind == 'dataset':
            return ChangeTargetKind.DATASET_PORT
        return ChangeTargetKind.UNKNOWN
    return _CT_TO_KIND.get((ct.app_label, ct.model), ChangeTargetKind.UNKNOWN)


def fetch_entity_history_by_uuid(
    django_model: type[Model] | tuple[type[Model], ...],
    target_uuid: UUID,
    info: gql.Info,
    *,
    limit: int,
    before: datetime | None = None,
) -> list[InstanceModelLogEntryType]:
    """
    Return authorized log entries targeting a stable entity UUID.

    UUID lookup survives deletion of the target row. Permission is checked
    against the owning InstanceConfig recorded by each operation, so this
    helper is safe even when a type later gains a new query path.
    """
    from django.contrib.contenttypes.models import ContentType

    models = django_model if isinstance(django_model, tuple) else (django_model,)
    cts = [ContentType.objects.get_for_model(model) for model in models]
    return _fetch_history_for_cts(cts, target_uuid, info, limit=limit, before=before)


def fetch_binding_history_by_uuid(
    target_uuid: UUID,
    info: gql.Info,
    *,
    limit: int,
    before: datetime | None = None,
) -> list[InstanceModelLogEntryType]:
    """
    Return authorized log entries for an input binding's stable UUID.

    Entries recorded before the unified-binding flip carry the legacy
    ``NodeEdge`` / ``DatasetPort`` content types under the same binding
    UUID. Those models are gone (plan step 11), so the legacy content-type
    rows are looked up by natural key — and may be absent on databases
    created after the removal.
    """
    from django.contrib.contenttypes.models import ContentType

    from nodes.models import NodeInputPortBinding

    cts = [ContentType.objects.get_for_model(NodeInputPortBinding)]
    cts.extend(ContentType.objects.filter(app_label='nodes', model__in=('nodeedge', 'datasetport')))
    return _fetch_history_for_cts(cts, target_uuid, info, limit=limit, before=before)


def _fetch_history_for_cts(
    cts: list[ContentType],
    target_uuid: UUID,
    info: gql.Info,
    *,
    limit: int,
    before: datetime | None = None,
) -> list[InstanceModelLogEntryType]:
    from kausal_common.users import user_or_none

    from nodes.models import InstanceConfig

    user = user_or_none(info.context.user)
    if user is None:
        return []

    permitted_instances = InstanceConfig.permission_policy().instances_user_has_permission_for(user, 'change')
    qs = (
        InstanceModelLogEntry.objects
        .filter(
            content_type__in=cts,
            target_uuid=target_uuid,
            operation__instance_config__in=permitted_instances,
        )
        .select_related('operation', 'operation__instance_config', 'content_type')
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
    InstanceConfig: ``Node`` and the input-binding types. Each carries
    a stable ``uuid``.

    ``uuid`` is always populated: DB-backed entities return their
    persisted uuid, while runtime-only (YAML) entities fall back to a stable
    uuidv5 derived from their identifier.
    """

    uuid: UUID


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
        if not root._entry.operation.instance_config.gql_action_allowed(info, 'change', raise_on_denied=False):
            return None
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
        raw_uuid = entry.target_uuid or (entry.data or {}).get('target_uuid')
        target_uuid: UUID | None = UUID(str(raw_uuid)) if raw_uuid else None
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
    from nodes.models import NodeConfig, NodeInputPortBinding

    ct = entry.content_type
    if ct is None or entry.object_id is None:
        return None
    try:
        pk = int(entry.object_id)
    except TypeError, ValueError:
        return None

    if (ct.app_label, ct.model) in {('nodes', 'nodeedge'), ('nodes', 'datasetport')}:
        # Pre-flip entries; the legacy models are gone (``ct.model_class()``
        # is None), but the binding UUID resolves through the unified table.
        return _resolve_binding_target(entry, pk, by_pk=False)

    model = ct.model_class()
    if model is None:
        return None

    if model is NodeConfig:
        nc = NodeConfig.objects.filter(pk=pk).select_related('instance').first()
        if nc is None:
            return None
        # Resolve via the runtime Node so the UI gets a real Node /
        # ActionNode object it can introspect through EditableEntity.
        ic = nc.instance
        instance = ic.get_instance()
        return instance.context.nodes.get(nc.identifier)

    if model is NodeInputPortBinding:
        return _resolve_binding_target(entry, pk, by_pk=True)

    # Dimension / DimensionCategory / DataPoint / spec-embedded: no GQL
    # representation yet — surface the shape via ``before`` / ``after``.
    return None


def _resolve_binding_target(entry: InstanceModelLogEntry, pk: int, *, by_pk: bool) -> Any:
    """
    Resolve a binding log entry to its GQL object through the unified table.

    Legacy-table entries resolve through the shared binding UUID; the legacy
    rows themselves are gone.
    """
    from nodes.graphql.types.graph import NodeEdgeType
    from nodes.models import NodeInputPortBinding

    qs = NodeInputPortBinding.objects.select_related('node', 'source_node', 'dataset', 'metric')
    if by_pk:
        binding = qs.filter(pk=pk).first()
    elif entry.target_uuid is not None:
        binding = qs.filter(uuid=entry.target_uuid).first()
    else:
        binding = None
    if binding is None:
        return None
    if binding.source_node_id is not None:
        return NodeEdgeType.from_input_binding(binding)
    from nodes.graphql.bindings import _to_gql

    return _to_gql(binding)


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
    def entries(root: InstanceChangeOperationType, info: gql.Info) -> list[InstanceModelLogEntryType]:
        ic = root._operation.instance_config
        if not ic.gql_action_allowed(info, 'change', raise_on_denied=False):
            return []
        entries = (
            InstanceModelLogEntry.objects
            .filter(operation=root._operation)
            .select_related('content_type', 'operation__instance_config')
            .order_by('id')
        )
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
