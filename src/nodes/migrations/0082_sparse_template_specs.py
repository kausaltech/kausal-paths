"""
Retire copied template specs and formula parameters without rewriting published revisions.

YAML-sourced instances are migrated too: they run from their YAML, but their database copy is
what `--source db`, `copy_instance --mode yaml` and a later switch to the database load, and an
unmigrated copy no longer builds. A copy the upgrader refuses is reported for re-syncing instead
of stopping the migration, since `sync_instance_to_db` rebuilds it from the YAML anyway.
"""

import json

from django.db import migrations
from kausal_common.i18n.pydantic import set_i18n_context
from pydantic import TypeAdapter


def migrate_specs(apps, schema_editor):
    from nodes.defs.instance_defs import InstanceModelSpec
    from nodes.instance_serialization import InheritedNodeSettings, InstanceSnapshot
    from nodes.legacy_specs import local_spec_from_template, migrate_inherited_node_settings, upgrade_formula_specs_v13

    # Historical SchemaFields still use today's Pydantic classes. Read and write
    # JSON directly so fields retired by later migrations survive until then.
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            'SELECT id, identifier, config_source, spec, node_settings, '
            'template_revision_id, primary_language, other_languages '
            'FROM nodes_instanceconfig WHERE spec IS NOT NULL'
        )
        instances = cursor.fetchall()
    settings_adapter = TypeAdapter(list[InheritedNodeSettings])
    for (
        pk,
        identifier,
        config_source,
        stored_spec,
        stored_settings,
        template_revision_id,
        primary_language,
        other_languages,
    ) in instances:
        spec = json.loads(stored_spec) if isinstance(stored_spec, str) else stored_spec
        node_settings = json.loads(stored_settings) if isinstance(stored_settings, str) else stored_settings
        with schema_editor.connection.cursor() as cursor:
            cursor.execute(
                'SELECT id, identifier, spec FROM nodes_nodeconfig WHERE instance_id = %s AND spec IS NOT NULL',
                [pk],
            )
            nodes = [
                {'pk': node_pk, 'identifier': node_identifier, 'spec': json.loads(raw) if isinstance(raw, str) else raw}
                for node_pk, node_identifier, raw in cursor.fetchall()
            ]
        payload = {'spec': spec, 'nodes': nodes}
        try:
            upgrade_formula_specs_v13(payload, discard_captured_defaults=True)
        except ValueError as error:
            if config_source == 'database':
                raise
            print(f'\n  {identifier}: database copy not migrated ({error}); re-sync it with sync_instance_to_db')
            continue
        local = InstanceModelSpec.model_validate(spec)
        if template_revision_id is not None:
            Revision = apps.get_model('wagtailcore', 'Revision')
            revision = Revision.objects.using(schema_editor.connection.alias).get(pk=template_revision_id)
            with set_i18n_context(primary_language, other_languages=other_languages):
                base = InstanceSnapshot.from_serialized_data(revision.content['model_snapshot']['structured'])
                local = local_spec_from_template(local, base)
                settings = migrate_inherited_node_settings(local, settings_adapter.validate_python(node_settings), base)
                node_settings = [setting.model_dump(mode='json') for setting in settings]
        with schema_editor.connection.cursor() as cursor:
            for node in nodes:
                cursor.execute('UPDATE nodes_nodeconfig SET spec = %s WHERE id = %s', [json.dumps(node['spec']), node['pk']])
            cursor.execute(
                'UPDATE nodes_instanceconfig SET spec = %s, node_settings = %s WHERE id = %s',
                [json.dumps(local.model_dump(mode='json')), json.dumps(node_settings), pk],
            )


class Migration(migrations.Migration):
    dependencies = [('nodes', '0081_datasetmaterialization_validation_payload_version')]
    operations = [migrations.RunPython(migrate_specs)]
