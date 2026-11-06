"""Persist explicit ports for legacy edges that serve several operands."""

from typing import Any
from uuid import uuid5

from django.db import migrations
from django.utils import timezone


def forwards(apps: Any, schema_editor: Any) -> None:
    from kausal_common.i18n.pydantic import set_i18n_context

    from nodes.instance_graph import node_class_for_type_config
    from nodes.instance_serialization import InputBindingSnapshot, NodePortSource
    from nodes.legacy_port_roles import split_legacy_edge_roles

    alias = schema_editor.connection.alias
    Node = apps.get_model('nodes', 'NodeConfig')
    Binding = apps.get_model('nodes', 'NodeInputPortBinding')
    Instance = apps.get_model('nodes', 'InstanceConfig')
    changed_instances: set[int] = set()
    nodes = Node.objects.using(alias).filter(spec__type_config__kind='simple', is_stale=False).select_related('instance')
    for node in nodes:
        with set_i18n_context(node.instance.primary_language, other_languages=node.instance.other_languages or []):
            roles = node_class_for_type_config(node.spec.type_config).legacy_edge_role_fanout
        if not roles:
            continue
        rows = list(Binding.objects.using(alias).filter(node_id=node.pk).select_related('source_node'))
        # A dataset input cannot be the legacy edge shared by distinct operands.
        dataset_ports = {row.port_id for row in rows if row.source_node_id is None}
        edge_rows = [row for row in rows if row.source_node_id is not None and row.port_id not in dataset_ports]
        bindings = [
            InputBindingSnapshot(
                uuid=row.uuid,
                node_id=node.uuid,
                port_id=row.port_id,
                position=row.position,
                source=NodePortSource(node_id=row.source_node.uuid, port_id=row.source_port_id),
                tags=row.tags or [],
                transformations=row.transformations or [],
            )
            for row in edge_rows
        ]
        upgraded_spec, upgraded_bindings = split_legacy_edge_roles(node.spec, bindings, roles)
        if upgraded_spec is node.spec:
            continue
        originals = {row.uuid: row for row in edge_rows}
        for binding in upgraded_bindings:
            if binding.uuid in originals:
                continue
            original = next(
                row for row in edge_rows if any(uuid5(row.uuid, f'legacy-edge-role:{role}') == binding.uuid for role in roles)
            )
            values = {
                field.attname: getattr(original, field.attname)
                for field in Binding._meta.concrete_fields
                if not field.primary_key and field.name != 'uuid'
            }
            values['port_id'] = binding.port_id
            Binding.objects.using(alias).create(uuid=binding.uuid, **values)
        Node.objects.using(alias).filter(pk=node.pk).update(spec=upgraded_spec)
        changed_instances.add(node.instance_id)
    Instance.objects.using(alias).filter(pk__in=changed_instances).update(cache_invalidated_at=timezone.now())


class Migration(migrations.Migration):
    dependencies = [('nodes', '0089_disable_certification_municipal_overlap')]

    operations = [migrations.RunPython(forwards, migrations.RunPython.noop)]
