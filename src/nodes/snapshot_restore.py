"""Restore editable model definitions without flattening template inheritance."""

from typing import TYPE_CHECKING, Any

from django.db import transaction

from kausal_common.datasets.models import Dataset

from nodes.instance_serialization import InstanceExport, NodePortSource, _import_bindings
from nodes.legacy_specs import authoring_snapshot_from_legacy
from nodes.models import InputPortBindingSet, InstanceConfig, NodeConfig, NodeLayout, NodeLayoutSource
from nodes.snapshot_base import apply_translated

if TYPE_CHECKING:
    from uuid import UUID

    from nodes.instance_serialization import InstanceSnapshot


@transaction.atomic
def restore_instance_definition(instance: InstanceConfig, snapshot: InstanceSnapshot) -> None:
    """Restore local declarations, graph selections and template pin; dataset bodies have their own lifecycle."""
    if snapshot.snapshot_kind == 'legacy' and snapshot.template_revision_id is not None:
        snapshot = authoring_snapshot_from_legacy(snapshot)
    if snapshot.snapshot_kind == 'composed':
        raise ValueError('Restore requires an authoring snapshot, not a flattened template composition')
    if snapshot.metadata.uuid != instance.uuid:
        raise ValueError('Snapshot belongs to another instance')
    snapshot.resolve()  # Verify the template pin before changing any draft rows.
    locked = InstanceConfig.objects.select_for_update().get(pk=instance.pk)
    locked.binding_overrides.all().delete()
    locked.input_bindings.all().delete()
    nodes = _restore_nodes(locked, snapshot)
    datasets = {
        item.identifier: item
        for item in Dataset.objects.filter(
            uuid__in=[item.id for item in snapshot.all_datasets()],
        )
        if item.identifier is not None
    }
    _import_bindings(locked, InstanceExport(instance=snapshot), nodes, datasets)
    for override in snapshot.binding_overrides:
        InputPortBindingSet.objects.create(
            instance=locked, node_uuid=override.node_uuid, port_uuid=override.port_uuid, bindings=override.bindings
        )
    # A frozen inherited source need not have a live ORM node. Retain that
    # selection in the UUID-based binding set instead of dropping it on restore.
    inherited_ports = {
        (item.node_id, item.port_id)
        for item in snapshot.bindings
        if isinstance(item.source, NodePortSource) and item.source.node_id not in nodes
    }
    for node_id, port_id in inherited_ports:
        bindings = [item for item in snapshot.bindings if (item.node_id, item.port_id) == (node_id, port_id)]
        locked.input_bindings.filter(node__uuid=node_id, port_id=port_id).delete()
        InputPortBindingSet.objects.update_or_create(
            instance=locked, node_uuid=node_id, port_uuid=port_id, defaults={'bindings': bindings}
        )
    _restore_metadata(locked, snapshot)
    locked.invalidate_cache()
    instance.refresh_from_db()


def _restore_nodes(instance: InstanceConfig, snapshot: InstanceSnapshot) -> dict[UUID, NodeConfig]:
    nodes: dict[UUID, NodeConfig] = {}
    for item in snapshot.nodes:
        fields: dict[str, Any] = {}
        translations: dict[str, str] = {}
        for field in ('name', 'short_name', 'short_description', 'description', 'goal'):
            value = getattr(item, field)
            if value is None:
                fields[field] = ''
            else:
                apply_translated(fields, translations, value, field, snapshot.metadata.primary_language)
        fields.update(
            identifier=item.identifier,
            color=item.color,
            order=item.order,
            is_visible=item.is_visible,
            is_editable=item.is_editable if item.is_editable is not None else True,
            body=item.body or [],
            i18n=translations,
            spec=item.spec,
            is_stale=False,
        )
        node = NodeConfig.objects.with_spec().filter(instance=instance, uuid=item.uuid).first()
        if node is None:
            if NodeConfig.objects.filter(uuid=item.uuid).exists():
                raise ValueError('Snapshot node UUID belongs to another instance')
            node = NodeConfig.objects.create(instance=instance, uuid=item.uuid, **fields)
        else:
            NodeConfig.objects.filter(pk=node.pk).update(**fields)
        nodes[item.uuid] = node
        if item.layout is None:
            NodeLayout.objects.filter(node=node).delete()
        else:
            NodeLayout.objects.update_or_create(
                node=node,
                defaults={
                    'x': item.layout.x,
                    'y': item.layout.y,
                    'source': NodeLayoutSource(item.layout.source),
                },
            )
    instance.nodes.exclude(uuid__in=nodes).update(is_stale=True)
    for item in snapshot.nodes:
        NodeConfig.objects.filter(pk=nodes[item.uuid].pk).update(
            indicator_node=nodes.get(item.indicator_node) if item.indicator_node else None,
        )
    return nodes


def _restore_metadata(instance: InstanceConfig, snapshot: InstanceSnapshot) -> None:
    instance.spec = snapshot.spec.model_copy(deep=True)
    instance.template_revision_id = snapshot.template_revision_id
    instance.node_settings = snapshot.node_settings
    instance.primary_language = snapshot.metadata.primary_language
    instance.other_languages = snapshot.metadata.other_languages
    fields: dict[str, Any] = {}
    translations: dict[str, str] = {}
    for field in ('name', 'owner', 'lead_title', 'lead_paragraph'):
        value = getattr(snapshot.metadata, field)
        if value is None:
            fields[field] = ''
        else:
            apply_translated(fields, translations, value, field, instance.primary_language)
    for field, value in fields.items():
        setattr(instance, field, value)
    instance.i18n = translations
    instance.save(
        update_fields=[
            'spec',
            'template_revision',
            'node_settings',
            'primary_language',
            'other_languages',
            'name',
            'owner',
            'lead_title',
            'lead_paragraph',
            'i18n',
        ]
    )
