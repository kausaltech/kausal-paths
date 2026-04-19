"""
Change-operation context manager for Paths model-changing mutations.

A user-facing edit to an ``InstanceConfig`` opens exactly one
``InstanceChangeOperation`` via ``change_operation(...)``. All row-level
writes performed inside that block emit ``InstanceModelLogEntry`` rows
attached to the operation.

The active operation is carried in a module-level ``ContextVar``. This is
transport-neutral — GraphQL, REST, Wagtail admin and CLI callers can all
open an operation without any transport-specific plumbing. Higher-level
transport wrappers may *additionally* attach the operation to their
native carrier (``info.context.operation``, ``request.operation``) for
typed-access convenience, but the ContextVar is the ground truth.

Typical use::

    with change_operation(ic, user=user, action='node.update',
                          source=InstanceChangeSource.GRAPHQL):
        old = node.snapshot_data()
        node.name = new_name
        node.save()
        record_change(node, action='node.update',
                      before=old, after=node.snapshot_data())
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from django.contrib.contenttypes.models import ContentType
from django.db import transaction

from kausal_common.users import user_or_none

from nodes.models import (
    InstanceChangeOperation,
    InstanceChangeSource,
    InstanceConfig,
    InstanceModelLogEntry,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterator

    from django.db.models import Model

    from kausal_common.users import UserOrAnon


# The active operation for the current context (request, task, command).
# ``None`` means no operation is open — attempts to ``record_change`` in
# that state will raise.
_current_op: ContextVar[InstanceChangeOperation | None] = ContextVar(
    'instance_current_change_operation',
    default=None,
)


class NoActiveChangeOperation(RuntimeError):  # noqa: N818
    """Raised when record_change is called outside a change_operation block."""


def get_current_operation() -> InstanceChangeOperation:
    """Return the active ``InstanceChangeOperation`` or raise."""
    op = _current_op.get()
    if op is None:
        msg = 'No active InstanceChangeOperation in context. Wrap the write in a change_operation(...) block.'
        raise NoActiveChangeOperation(msg)
    return op


def current_operation_or_none() -> InstanceChangeOperation | None:
    """Return the active operation or ``None``. Use for conditional recording."""
    return _current_op.get()


@contextmanager
def change_operation(
    ic: InstanceConfig,
    *,
    user: UserOrAnon | None,
    action: str,
    source: InstanceChangeSource | str = InstanceChangeSource.GRAPHQL,
) -> Iterator[InstanceChangeOperation]:
    """
    Open an ``InstanceChangeOperation`` for the duration of the block.

    Wraps everything in a ``transaction.atomic()`` with a ``SELECT FOR
    UPDATE`` on the ``InstanceConfig`` row — this serializes concurrent
    mutations against the same instance and makes the version-token
    check (phase 3) race-free.

    Nested ``change_operation`` calls for the same instance reuse the
    outer operation (allows nested resolvers like the dataset editor to
    participate in the parent operation without plumbing). A nested call
    targeting a *different* ``InstanceConfig`` raises.
    """
    existing = _current_op.get()
    if existing is not None:
        if existing.instance_config.pk != ic.pk:
            msg = (
                f'Nested change_operation targets a different InstanceConfig '
                f'({ic.identifier}) than the active operation '
                f'({existing.instance_config.identifier}).'
            )
            raise RuntimeError(msg)
        # Reuse the outer operation; don't open a new transaction, the
        # outer block already holds the lock.
        yield existing
        return

    # Django's User type is not imported here; the FK on
    # InstanceChangeOperation already validates.
    user_obj = user_or_none(user)

    src = source.value if isinstance(source, InstanceChangeSource) else source

    with transaction.atomic():
        # Serialize against concurrent mutations.
        InstanceConfig.objects.select_for_update().filter(pk=ic.pk).first()

        op = InstanceChangeOperation.objects.create(
            instance_config=ic,
            user=user_obj,
            action=action,
            source=src,
        )
        token = _current_op.set(op)
        try:
            yield op
        finally:
            _current_op.reset(token)


def record_change(
    obj: Model,
    *,
    action: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    target_uuid: uuid.UUID | str | None = None,
) -> InstanceModelLogEntry:
    """
    Emit one ``InstanceModelLogEntry`` under the active operation.

    ``obj`` is the affected ORM row. Its ``pk`` and ``ContentType`` form
    the GFK; ``target_uuid`` (defaulting to ``obj.uuid``) is recorded in
    the payload to survive row deletion.

    ``before`` / ``after`` are the snapshot dicts produced by
    ``serializable_data()`` or ``snapshot_data()`` helpers:

    * ``before=None``  → create  (payload's ``before`` is null)
    * ``after=None``   → delete  (payload's ``after``  is null; GFK may
                                  dangle after the deletion commits, but
                                  ``target_uuid`` lets undo find the row)
    * both present     → update
    """
    op = get_current_operation()

    if target_uuid is None:
        # UUIDIdentifiedModel.uuid is the canonical choice; fall back to pk
        # so non-uuid targets (e.g. InstanceConfig itself, for spec edits)
        # still produce readable entries.
        target_uuid = getattr(obj, 'uuid', None) or obj.pk

    return InstanceModelLogEntry.objects.create(
        operation=op,
        content_type=ContentType.objects.get_for_model(type(obj)),
        object_id=str(obj.pk) if obj.pk is not None else None,
        action=action,
        data={
            'target_uuid': str(target_uuid),
            'before': before,
            'after': after,
        },
    )
