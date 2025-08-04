from django.db import migrations
from treebeard.mp_tree import MP_Node


def create_dummy_organizations(apps, schema_editor):
    InstanceConfig = apps.get_model('nodes', 'instanceconfig')
    Organization = apps.get_model('orgs', 'organization')
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
        ('nodes', '0032_add_instanceconfig_organization'),
    ]

    operations = [
        migrations.RunPython(create_dummy_organizations),
    ]
