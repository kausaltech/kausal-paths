"""Name the scheme within its framework; retain scheme/level UUIDs and evidence."""

from typing import TYPE_CHECKING

from django.db import migrations

if TYPE_CHECKING:
    from django.apps.registry import Apps
    from django.db.backends.base.schema import BaseDatabaseSchemaEditor


def rename_schemes(apps: Apps, schema_editor: BaseDatabaseSchemaEditor, old: str, new: str) -> None:
    schemes = apps.get_model('frameworks', 'DataQualityScheme')
    database = schema_editor.connection.alias
    rows = schemes.objects.using(database).filter(framework__identifier='bisko', identifier=old)
    for scheme in rows:
        if (
            schemes.objects
            .using(database)
            .filter(framework_id=scheme.framework_id, identifier=new, version=scheme.version)
            .exists()
        ):
            raise RuntimeError(f'Cannot rename BISKO scheme {old}: {new} version {scheme.version} already exists')
    rows.update(identifier=new)


def forwards(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    rename_schemes(apps, schema_editor, 'bisko', 'quality')


def backwards(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    rename_schemes(apps, schema_editor, 'quality', 'bisko')


class Migration(migrations.Migration):
    dependencies = [('frameworks', '0033_submission_events')]
    operations = [migrations.RunPython(forwards, backwards)]
