import json
from typing import Any
from uuid import UUID, uuid3

from django.db import migrations


def _as_dict(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def _action_group_uuid(instance_uuid: Any, identifier: str) -> UUID:
    return uuid3(UUID(str(instance_uuid)), f'action-group:{identifier}')


def add_action_group_uuids(apps, schema_editor):
    InstanceConfig = apps.get_model('nodes', 'InstanceConfig')
    NodeConfig = apps.get_model('nodes', 'NodeConfig')
    quote = schema_editor.quote_name

    group_maps = {}
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            f'SELECT id, uuid, spec FROM {quote(InstanceConfig._meta.db_table)} WHERE spec IS NOT NULL'  # noqa: S608
        )
        for instance_pk, instance_uuid, raw_spec in cursor.fetchall():
            spec = _as_dict(raw_spec)
            groups = spec.get('action_groups') or []
            group_map = {}
            changed = False
            for group in groups:
                identifier = group['id']
                group_uuid = group.get('uuid') or str(_action_group_uuid(instance_uuid, identifier))
                group['uuid'] = group_uuid
                group_map[str(identifier)] = str(group_uuid)
                changed = True
            group_maps[instance_pk] = (UUID(str(instance_uuid)), group_map)
            if changed:
                cursor.execute(
                    f'UPDATE {quote(InstanceConfig._meta.db_table)} SET spec = %s WHERE id = %s',  # noqa: S608
                    [json.dumps(spec), instance_pk],
                )

        cursor.execute(
            f'SELECT id, instance_id, spec FROM {quote(NodeConfig._meta.db_table)} WHERE spec IS NOT NULL'  # noqa: S608
        )
        for node_pk, instance_pk, raw_spec in cursor.fetchall():
            spec = _as_dict(raw_spec)
            type_config = spec.get('type_config') or {}
            group_ref = type_config.get('group')
            if type_config.get('kind') != 'action' or group_ref is None:
                continue
            instance_uuid, group_map = group_maps[instance_pk]
            type_config['group'] = group_map.get(
                str(group_ref),
                str(_action_group_uuid(instance_uuid, group_ref)),
            )
            cursor.execute(
                f'UPDATE {quote(NodeConfig._meta.db_table)} SET spec = %s WHERE id = %s',  # noqa: S608
                [json.dumps(spec), node_pk],
            )


def remove_action_group_uuids(apps, schema_editor):
    InstanceConfig = apps.get_model('nodes', 'InstanceConfig')
    NodeConfig = apps.get_model('nodes', 'NodeConfig')
    quote = schema_editor.quote_name

    group_maps = {}
    instance_specs = {}
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            f'SELECT id, spec FROM {quote(InstanceConfig._meta.db_table)} WHERE spec IS NOT NULL'  # noqa: S608
        )
        for instance_pk, raw_spec in cursor.fetchall():
            spec = _as_dict(raw_spec)
            group_map = {
                str(group['uuid']): group['id']
                for group in spec.get('action_groups') or []
                if group.get('uuid') is not None
            }
            group_maps[instance_pk] = group_map
            instance_specs[instance_pk] = spec

        cursor.execute(
            f'SELECT id, instance_id, spec FROM {quote(NodeConfig._meta.db_table)} WHERE spec IS NOT NULL'  # noqa: S608
        )
        for node_pk, instance_pk, raw_spec in cursor.fetchall():
            spec = _as_dict(raw_spec)
            type_config = spec.get('type_config') or {}
            group_ref = type_config.get('group')
            if type_config.get('kind') != 'action' or group_ref is None:
                continue
            identifier = group_maps.get(instance_pk, {}).get(str(group_ref))
            if identifier is None:
                continue
            type_config['group'] = identifier
            cursor.execute(
                f'UPDATE {quote(NodeConfig._meta.db_table)} SET spec = %s WHERE id = %s',  # noqa: S608
                [json.dumps(spec), node_pk],
            )

        for instance_pk, spec in instance_specs.items():
            groups = spec.get('action_groups') or []
            if not groups:
                continue
            for group in groups:
                group.pop('uuid', None)
            cursor.execute(
                f'UPDATE {quote(InstanceConfig._meta.db_table)} SET spec = %s WHERE id = %s',  # noqa: S608
                [json.dumps(spec), instance_pk],
            )


class Migration(migrations.Migration):
    dependencies = [
        ('nodes', '0070_alter_nodeinputportbinding_transformations'),
    ]

    operations = [
        migrations.RunPython(add_action_group_uuids, remove_action_group_uuids),
    ]
