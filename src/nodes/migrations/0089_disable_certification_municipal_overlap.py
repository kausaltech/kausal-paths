"""Municipal facilities are entered separately in both certification instances."""

import json
from typing import Any

from django.db import migrations


def forwards(apps: Any, schema_editor: Any) -> None:
    # Read raw JSON: historical SchemaFields use the current Pydantic schema.
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, spec FROM nodes_instanceconfig WHERE identifier IN ('bisko', 'pruefstadt-bisko') AND spec IS NOT NULL"
        )
        for pk, raw_spec in cursor.fetchall():
            spec = json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec
            changed = False
            for parameter in spec.get('params', []):
                if parameter.get('local_id') == 'municipal_facilities_included_in_commerce' and parameter.get('value') is True:
                    parameter['value'] = False
                    changed = True
            if changed:
                cursor.execute(
                    'UPDATE nodes_instanceconfig SET spec = %s::jsonb, cache_invalidated_at = CURRENT_TIMESTAMP WHERE id = %s',
                    [json.dumps(spec), pk],
                )


class Migration(migrations.Migration):
    dependencies = [('nodes', '0088_output_dimensions_on_ports')]

    operations = [migrations.RunPython(forwards, migrations.RunPython.noop)]
