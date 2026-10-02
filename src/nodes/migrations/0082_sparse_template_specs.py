"""Retire copied template specs and formula parameters without rewriting published revisions."""

from typing import cast

from django.db import migrations
from kausal_common.i18n.pydantic import set_i18n_context


def migrate_specs(apps, schema_editor):
    from nodes.defs.instance_defs import InstanceModelSpec
    from nodes.instance_serialization import InstanceSnapshot
    from nodes.legacy_specs import local_spec_from_template, migrate_inherited_node_settings, upgrade_formula_specs_v13
    from nodes.models import InstanceConfig

    InstanceConfigModel = apps.get_model('nodes', 'InstanceConfig')
    NodeConfig = apps.get_model('nodes', 'NodeConfig')
    for instance_untyped in (
        InstanceConfigModel.objects.using(schema_editor.connection.alias).filter(config_source='database').iterator()
    ):
        instance = cast('InstanceConfig', instance_untyped)
        if instance.spec is None:
            continue
        spec = instance.spec.model_dump(mode='json')
        nodes = [
            {'identifier': node.identifier, 'spec': node.spec.model_dump(mode='json')}
            for node in NodeConfig.objects.using(schema_editor.connection.alias).filter(instance=instance).defer(None)
            if node.spec is not None
        ]
        payload = {'spec': spec, 'nodes': nodes}
        upgrade_formula_specs_v13(payload, discard_captured_defaults=True)
        for node in nodes:
            NodeConfig.objects.using(schema_editor.connection.alias).filter(
                instance=instance,
                identifier=node['identifier'],
            ).update(spec=node['spec'])
        local = InstanceModelSpec.model_validate(spec)
        node_settings = instance.node_settings
        if instance.template_revision_id is not None:
            with set_i18n_context(instance.primary_language, other_languages=instance.other_languages):
                base = InstanceSnapshot.from_serialized_data(instance.template_revision.content['model_snapshot']['structured'])
                local = local_spec_from_template(local, base)
                node_settings = migrate_inherited_node_settings(local, node_settings, base)
        InstanceConfig.objects.using(schema_editor.connection.alias).filter(pk=instance.pk).update(
            spec=local.model_dump(mode='json'),
            node_settings=[setting.model_dump(mode='json') for setting in node_settings],
        )


class Migration(migrations.Migration):
    dependencies = [('nodes', '0081_datasetmaterialization_validation_payload_version')]
    operations = [migrations.RunPython(migrate_specs)]
