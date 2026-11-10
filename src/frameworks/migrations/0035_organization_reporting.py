"""Declare reporting units on the framework; BISKO keeps the classes that were hardcoded until now."""

from typing import TYPE_CHECKING

import django.core.serializers.json
import django_pydantic_field.fields
import frameworks.models.framework
from django.db import migrations, models

if TYPE_CHECKING:
    from django.apps.registry import Apps
    from django.db.backends.base.schema import BaseDatabaseSchemaEditor


def declare_bisko_reporting_units(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    framework_model = apps.get_model('frameworks', 'Framework')
    for framework in framework_model.objects.using(schema_editor.connection.alias).filter(identifier='bisko'):
        framework.reporting = frameworks.models.framework.ReportingSpec(
            units=frameworks.models.framework.ReportingUnitSpec(
                classes=['de_municipality', 'de_district_free_city'], identifier_namespace='ags'
            )
        )
        framework.save(update_fields=['reporting'])


class Migration(migrations.Migration):
    dependencies = [
        ('frameworks', '0034_bisko_quality_scheme_identifier'),
    ]

    operations = [
        migrations.AddField(
            model_name='framework',
            name='reporting',
            field=django_pydantic_field.fields.PydanticSchemaField(
                config=None,
                default=frameworks.models.framework.ReportingSpec,
                encoder=django.core.serializers.json.DjangoJSONEncoder,
                schema=frameworks.models.framework.ReportingSpec,
            ),
        ),
        migrations.AddField(
            model_name='frameworkorganizationroot',
            name='territorial_reference_date',
            field=models.DateField(
                blank=True, help_text='The date whose administrative boundaries the tree under this root describes.', null=True
            ),
        ),
        migrations.RunPython(declare_bisko_reporting_units, migrations.RunPython.noop),
    ]
