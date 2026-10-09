"""
Remove ``pipeline`` and ``extra.other`` from stored node specs.

Both fields are gone from ``NodeSpec`` (snapshot schema v17). Every stored spec carries
them, as ``null`` and ``{}``, and ``NodeSpec`` forbids unknown keys, so they have to go
before a spec can load. Instance revisions are upgraded on read.
"""

from typing import Any

from django.db import migrations


def forwards(apps: Any, schema_editor: Any) -> None:
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT count(*) FROM nodes_nodeconfig
            WHERE jsonb_typeof(spec -> 'pipeline') NOT IN ('null')
               OR (spec -> 'extra' -> 'other') NOT IN ('{}'::jsonb, 'null'::jsonb)
            """
        )
        (with_content,) = cursor.fetchone()
        if with_content:
            raise RuntimeError(f'{with_content} node specs hold a pipeline or extra.other; nothing reads them, but look before dropping')
        cursor.execute(
            """
            UPDATE nodes_nodeconfig
            SET spec = (spec - 'pipeline') #- '{extra,other}'
            WHERE spec ? 'pipeline' OR spec -> 'extra' ? 'other'
            """
        )


class Migration(migrations.Migration):
    dependencies = [
        ('nodes', '0086_node_references_by_uuid'),
    ]

    operations = [
        migrations.RunPython(forwards, migrations.RunPython.noop),
    ]
