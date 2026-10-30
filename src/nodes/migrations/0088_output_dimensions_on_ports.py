"""
Move the node-level ``output_dimensions`` of stored node specs onto their output ports.

``NodeSpec.output_dimensions`` is gone (snapshot schema v18): a node's output dimensions are an
assertion on each output port. Every stored spec carries the old key and ``NodeSpec`` forbids
unknown keys, so it has to move before a spec can load. Instance revisions are upgraded on read.
"""

import json
from typing import Any

from django.db import migrations


def forwards(apps: Any, schema_editor: Any) -> None:
    from nodes.instance_serialization import move_output_dimensions_to_ports_v18

    # Historical SchemaFields still use today's Pydantic classes, which no longer
    # accept the key; read and write the JSON directly.
    with schema_editor.connection.cursor() as cursor:
        cursor.execute("SELECT id, spec FROM nodes_nodeconfig WHERE spec ? 'output_dimensions'")
        rows = cursor.fetchall()
        for pk, spec in rows:
            data = json.loads(spec) if isinstance(spec, str) else spec
            move_output_dimensions_to_ports_v18(data)
            cursor.execute('UPDATE nodes_nodeconfig SET spec = %s::jsonb WHERE id = %s', [json.dumps(data), pk])


def backwards(apps: Any, schema_editor: Any) -> None:
    # The node-level list becomes the union of the port assertions, and an inferred port
    # becomes a scalar: before v18 there was no way to say "inferred".
    with schema_editor.connection.cursor() as cursor:
        cursor.execute("SELECT id, spec FROM nodes_nodeconfig WHERE spec IS NOT NULL AND NOT spec ? 'output_dimensions'")
        rows = cursor.fetchall()
        for pk, spec in rows:
            data = json.loads(spec) if isinstance(spec, str) else spec
            dimensions: dict[str, None] = {}
            for port in data.get('output_ports') or []:
                port['dimensions'] = port.get('dimensions') or []
                dimensions.update(dict.fromkeys(port['dimensions']))
            data['output_dimensions'] = list(dimensions)
            cursor.execute('UPDATE nodes_nodeconfig SET spec = %s::jsonb WHERE id = %s', [json.dumps(data), pk])


class Migration(migrations.Migration):
    dependencies = [
        ('nodes', '0087_drop_dead_node_spec_fields'),
    ]

    operations = [
        migrations.RunPython(forwards, backwards),
    ]
