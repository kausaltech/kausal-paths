"""
Give every dataset a scope.

Before a910cd46 (2026-06-01), `sync_dataset_placeholder` created external
placeholders without a scope; they reached their instance only through their
schema's `DatasetSchemaScope`. Scope each one to the instance its schema is
scoped to, when there is exactly one.
"""

from django.db import migrations


def scope_placeholders(apps, schema_editor):
    ContentType = apps.get_model('contenttypes', 'ContentType')
    Dataset = apps.get_model('datasets', 'Dataset')
    DatasetSchemaScope = apps.get_model('datasets', 'DatasetSchemaScope')
    instance_ct = ContentType.objects.get_for_model(apps.get_model('nodes', 'InstanceConfig'))

    for dataset in Dataset.objects.filter(scope_content_type__isnull=True, schema__isnull=False):
        scopes = list(DatasetSchemaScope.objects.filter(schema_id=dataset.schema_id))
        if len(scopes) != 1 or scopes[0].scope_content_type_id != instance_ct.pk:
            continue
        dataset.scope_content_type_id = instance_ct.pk
        dataset.scope_id = scopes[0].scope_id
        dataset.save(update_fields=['scope_content_type', 'scope_id'])


class Migration(migrations.Migration):
    dependencies = [
        ('nodes', '0076_binding_metric_references'),
        ('datasets', '0037_source_reference_targets_exactly_one'),
        ('contenttypes', '0002_remove_content_type_name'),
    ]

    run_before = [
        ('datasets', '0038_dataset_scope_not_null'),
    ]

    operations = [
        migrations.RunPython(scope_placeholders, migrations.RunPython.noop),
    ]
