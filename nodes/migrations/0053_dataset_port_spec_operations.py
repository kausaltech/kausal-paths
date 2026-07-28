"""
Convert ``DatasetPort.spec`` from flat filter fields to a transform pipeline.

A data migration rather than a re-sync: database-backed instances cannot be
regenerated from YAML, since they may carry admin edits that exist nowhere else.

The conversion itself lives on ``DatasetPortSpec``, which applies it when
reading a spec that still has the flat fields. So reading a row is already
correct before this migration runs — deploy order does not matter — and all this
does is write the converted shape back, so that the read-time conversion can be
removed once no old rows remain.
"""

from django.db import migrations
from django.db.migrations.exceptions import IrreversibleError


def forwards(apps, schema_editor):
    """
    Normalize stored specs to the pipeline shape.

    ``DatasetPortSpec`` also converts legacy specs when it reads them, so this
    is not what makes old rows work — it is what lets that read-time conversion
    be deleted later.
    """
    DatasetPort = apps.get_model('nodes', 'DatasetPort')
    updated = []
    for port in DatasetPort.objects.all().iterator():
        spec = port.spec
        if spec is None:
            continue
        as_dict = spec if isinstance(spec, dict) else spec.model_dump(mode='json')
        if 'operations' in as_dict:
            continue
        port.spec = as_dict
        updated.append(port)
    if updated:
        DatasetPort.objects.bulk_update(updated, ['spec'], batch_size=500)


def backwards(apps, schema_editor):
    """
    Irreversible.

    The pipeline can express orderings the flat fields cannot, so converting
    back would silently drop them.
    """
    raise IrreversibleError('DatasetPort.spec operations cannot be converted back to flat filter fields')


class Migration(migrations.Migration):
    dependencies = [
        ('nodes', '0052_alter_instanceconfig_other_languages'),
    ]

    operations = [
        migrations.RunPython(forwards, backwards),
    ]
