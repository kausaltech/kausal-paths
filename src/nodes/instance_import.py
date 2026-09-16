"""
Load a standalone ``InstanceExport`` document into this database.

``import_instance`` (``nodes.instance_serialization``) populates an
``InstanceConfig`` that already exists and is empty; this module owns the step
before it: deciding which row the document lands in, creating it when needed,
and refusing when the target already holds a model.
"""

from typing import TYPE_CHECKING
from uuid import UUID

from django.db import transaction

if TYPE_CHECKING:
    from nodes.instance_serialization import InstanceExport
    from nodes.models import InstanceConfig
    from orgs.models import Organization


class InstanceImportError(ValueError):
    """The document cannot be imported as asked; the message says why."""


def _parse_uuid(value: str) -> UUID | None:
    try:
        return UUID(value)
    except ValueError:
        return None


def resolve_organization(ref: str | None) -> Organization:
    """
    Pick the organization a newly created instance belongs to.

    ``ref`` is an organization UUID or exact name. Without it, the only
    organization in the database is used; with several, the caller must choose.
    """
    from orgs.models import Organization

    qs = Organization.objects.all()
    if ref:
        ref_uuid = _parse_uuid(ref)
        org = qs.filter(uuid=ref_uuid).first() if ref_uuid is not None else qs.filter(name=ref).first()
        if org is None:
            raise InstanceImportError(f'No organization matches {ref!r} (by UUID or exact name)')
        return org
    candidates = list(qs[:2])
    if not candidates:
        raise InstanceImportError('No organizations exist in this database; create one first')
    if len(candidates) > 1:
        raise InstanceImportError('Several organizations exist; say which one with an organization UUID or name')
    return candidates[0]


def _unique_instance_name(preferred: str, identifier: str) -> str:
    from nodes.models import InstanceConfig

    if not InstanceConfig.objects.filter(name=preferred).exists():
        return preferred
    return f'{preferred} ({identifier})'


def import_instance_export(
    export: InstanceExport,
    *,
    identifier: str | None = None,
    organization: str | None = None,
    name: str | None = None,
) -> InstanceConfig:
    """
    Create a database-sourced ``InstanceConfig`` from ``export``, or fill an empty one.

    The target is ``identifier`` or, by default, the identifier recorded in the
    document. A fresh row keeps the document's instance UUID when no other row
    holds it, so a downloaded instance stays the same entity locally; a row
    that already has nodes is refused, because ``import_instance`` creates
    node rows unconditionally and would duplicate identifiers.
    """
    from nodes.instance_serialization import import_instance
    from nodes.models import InstanceConfig

    meta = export.instance.metadata
    target = identifier or meta.identifier
    if not target:
        raise InstanceImportError('The document names no instance identifier; pass one explicitly')

    with transaction.atomic():
        ic = InstanceConfig.objects.filter(identifier=target).first()
        if ic is None:
            ic = InstanceConfig(
                identifier=target,
                name=_unique_instance_name(name or str(meta.name) or target, target),
                organization=resolve_organization(organization),
                primary_language=meta.primary_language,
                other_languages=list(meta.other_languages),
                config_source='database',
            )
            if not InstanceConfig.objects.filter(uuid=meta.uuid).exists():
                ic.uuid = meta.uuid
            ic.save()
        else:
            node_count = ic.nodes.count()
            if node_count:
                raise InstanceImportError(
                    f'Instance {target!r} already has {node_count} nodes; import into a new identifier instead',
                )
            if name:
                ic.name = _unique_instance_name(name, target)
                ic.save(update_fields=['name'])
        import_instance(ic, export)

    ic.refresh_from_db()
    return ic
