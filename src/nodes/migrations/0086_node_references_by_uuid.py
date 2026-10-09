"""
Refer to an action's parent and hook targets by node uuid instead of identifier.

Stored specs are read with SQL: loading them through the model would validate the
identifiers against the new uuid type and fail. A target is looked up among the nodes of
the action's own instance, then among those of the template it inherits from. Instance
revisions are upgraded on read (snapshot schema v16).
"""

import json
from typing import Any

from django.db import migrations

from nodes.instance_serialization import upgrade_node_references_v16


def _uuids_by_identifier(cursor: Any, instance_ids: list[int]) -> dict[str, str]:
    cursor.execute(
        'SELECT identifier, uuid FROM nodes_nodeconfig WHERE instance_id = ANY(%s) AND identifier IS NOT NULL',
        [instance_ids],
    )
    return {identifier: str(uuid) for identifier, uuid in cursor.fetchall()}


def forwards(apps: Any, schema_editor: Any) -> None:
    InstanceConfig = apps.get_model('nodes', 'InstanceConfig')
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, instance_id, spec FROM nodes_nodeconfig
            WHERE spec -> 'type_config' ->> 'kind' = 'action'
              AND jsonb_path_exists(spec, '$.type_config ? (@.parent != null || (@.hooks.type() == "array" && @.hooks.size() > 0))')
            """
        )
        rows = cursor.fetchall()
        lookups: dict[int, dict[str, str]] = {}
        for pk, instance_id, stored in rows:
            spec = json.loads(stored) if isinstance(stored, str) else stored
            if instance_id not in lookups:
                instance_ids = [instance_id]
                ic = InstanceConfig.objects.filter(pk=instance_id).select_related('template_revision').first()
                if ic is not None and ic.template_revision is not None:
                    instance_ids.append(int(ic.template_revision.object_id))
                lookups[instance_id] = _uuids_by_identifier(cursor, instance_ids)
            node = {'identifier': pk, 'spec': spec}
            if upgrade_node_references_v16([node], lookups[instance_id]):
                cursor.execute('UPDATE nodes_nodeconfig SET spec = %s WHERE id = %s', [json.dumps(spec), pk])


class Migration(migrations.Migration):
    dependencies = [
        ('nodes', '0085_dataset_snapshot_v2'),
    ]

    operations = [
        migrations.RunPython(forwards, migrations.RunPython.noop),
    ]
