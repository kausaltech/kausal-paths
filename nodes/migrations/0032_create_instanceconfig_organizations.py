import django.db.models.deletion
from django.db import migrations, models
from treebeard.mp_tree import MP_Node


def create_dummy_organizations(apps, schema_editor):
    InstanceConfig = apps.get_model('nodes', 'instanceconfig')  # noqa:N806
    Organization = apps.get_model('orgs', 'organization')  # noqa:N806
    if Organization.objects.exists():
        raise Exception("Cannot create dummy organizations because some organizations already exist.")
    for index, instance in enumerate(InstanceConfig.objects.all(), 1):
        org = Organization.objects.create(
            name=instance.name,
            depth=1,
            numchild=0,
            path=MP_Node._get_path('', 1, index),
        )
        instance.organization = org
        instance.save()


class Migration(migrations.Migration):

    dependencies = [
        ('nodes', '0031_alter_nodeconfig_color'),
        ('orgs', '0002_organizationmetadataadmin_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='instanceconfig',
            name='organization',
            field=models.ForeignKey(null=True, blank=True, on_delete=django.db.models.deletion.PROTECT, to='orgs.organization'),
        ),
        migrations.RunPython(create_dummy_organizations),
    ]
