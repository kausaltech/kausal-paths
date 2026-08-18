from django.db import migrations


def truncate_legacy_binding_tables(apps, schema_editor):
    """Empty the legacy binding tables now that NodeInputPortBinding is authoritative.

    Every row was mirrored (same UUID, spec and dataset_index carried over in
    0068), so nothing is lost. Removing the rows matters beyond hygiene: the
    DatasetPort rows hold PROTECTed references to DatasetMetric, and stale
    copies would block metric maintenance forever. The empty tables await
    removal in plan step 11.
    """
    apps.get_model('nodes', 'NodeEdge').objects.all().delete()
    apps.get_model('nodes', 'DatasetPort').objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ('nodes', '0068_inputbinding_dataset_spec'),
    ]

    operations = [
        migrations.RunPython(truncate_legacy_binding_tables, migrations.RunPython.noop),
    ]
