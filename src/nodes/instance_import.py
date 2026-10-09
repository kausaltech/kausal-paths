"""
Load a standalone ``InstanceExport`` document into this database.

``import_instance`` (``nodes.instance_serialization``) populates an
``InstanceConfig`` that already exists and is empty; this module owns the step
before it: creating the row under the document's own uuid, and refusing when
the instance is already here unless asked to replace it.
"""

from typing import TYPE_CHECKING, cast
from uuid import UUID

from django.db import transaction
from django.db.models import SET_NULL

from kausal_common.i18n.pydantic import TranslatedString, get_modeltrans_attrs_from_str

if TYPE_CHECKING:
    from django.db.models import Model
    from django.db.models.fields.reverse_related import ForeignObjectRel

    from kausal_common.i18n.pydantic import I18nString

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


type _References = list[tuple[type[Model], str, list[int]]]


def _references_to(instance: InstanceConfig) -> _References:
    """
    Return the rows that refer to ``instance`` by a foreign key its deletion would null.

    A replaced instance is the same instance: the framework whose template it is, the copies
    made from it and the users who selected it keep referring to it.
    """
    references: _References = []
    for relation in instance._meta.get_fields(include_hidden=True):
        if not (relation.auto_created and not relation.concrete and getattr(relation, 'on_delete', None) is SET_NULL):
            continue
        model = cast('type[Model]', relation.related_model)
        field_name = cast('ForeignObjectRel', relation).field.name
        pks = list(model._default_manager.filter(**{field_name: instance}).values_list('pk', flat=True))
        if pks:
            references.append((model, field_name, pks))
    return references


def _restore_references(references: _References, instance: InstanceConfig) -> None:
    for model, field_name, pks in references:
        model._default_manager.filter(pk__in=pks).update(**{field_name: instance})


def _instance_names(explicit: str | None, exported: I18nString | None, language: str) -> tuple[str | None, dict[str, str]]:
    """Return the instance's name in ``language`` and its translations: the explicit one alone, else the export's."""
    if explicit:
        return explicit, {}
    if exported is None:
        return None, {}
    if not isinstance(exported, TranslatedString):
        return str(exported), {}
    primary, translations = get_modeltrans_attrs_from_str(exported, 'name', language, strict=False)
    return primary, translations


def import_instance_export(
    export: InstanceExport,
    *,
    identifier: str | None = None,
    organization: str | None = None,
    name: str | None = None,
    replace: bool = False,
) -> InstanceConfig:
    """
    Create a database-sourced ``InstanceConfig`` from ``export``, keeping every uuid it holds.

    The instance must not be here: its uuid and the target identifier (``identifier``, or
    the one in the document) must both be free. With ``replace``, an instance of the same
    uuid is deleted first, with its model, datasets and pages; an identifier held by a
    different instance is refused even then. To copy an instance within one database,
    rekey the export instead (`InstanceExport.rekeyed`).
    """
    from nodes.instance_serialization import import_instance
    from nodes.models import InstanceConfig

    meta = export.instance.metadata
    target = identifier or meta.identifier
    if not target:
        raise InstanceImportError('The document names no instance identifier; pass one explicitly')

    with transaction.atomic():
        existing = InstanceConfig.objects.filter(uuid=meta.uuid).first()
        previous_organization = None
        references: _References = []
        if existing is not None:
            if not replace:
                raise InstanceImportError(
                    f'Instance {meta.uuid} is already here as {existing.identifier!r}; replace it, or copy the document'
                )
            heirs = list(
                InstanceConfig.objects
                .filter(template_revision__object_id=str(existing.pk))
                .exclude(pk=existing.pk)
                .values_list('identifier', flat=True)
            )
            if heirs:
                # Their pins name this row's revisions, which a replacement does not keep.
                msg = f'Instance {existing.identifier!r} is the template of {sorted(heirs)}; it cannot be replaced'
                raise InstanceImportError(msg)
            previous_organization = existing.organization
            references = _references_to(existing)
            existing.delete()
        holder = InstanceConfig.objects.filter(identifier=target).first()
        if holder is not None:
            raise InstanceImportError(
                f'Identifier {target!r} belongs to another instance ({holder.uuid}); import into another one'
            )
        names, name_i18n = _instance_names(name, meta.name, meta.primary_language)
        ic = InstanceConfig(
            uuid=meta.uuid,
            identifier=target,
            name=_unique_instance_name(names or target, target),
            i18n=name_i18n,
            organization=previous_organization
            if organization is None and previous_organization
            else resolve_organization(organization),
            primary_language=meta.primary_language,
            other_languages=list(meta.other_languages),
            config_source='database',
        )
        ic.save()
        try:
            import_instance(ic, export)
        except ValueError as e:
            raise InstanceImportError(str(e)) from e
        if existing is not None:
            _restore_references(references, ic)

    ic.refresh_from_db()
    return ic
