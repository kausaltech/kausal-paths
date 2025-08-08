from django.db import migrations
from treebeard.mp_tree import MP_Node


def create_initial_organizations(apps, schema_editor):
    InstanceConfig = apps.get_model('nodes', 'instanceconfig')
    Organization = apps.get_model('orgs', 'organization')
    FrameworkConfig = apps.get_model('frameworks', 'frameworkconfig')

    if Organization.objects.exists():
        raise Exception("Cannot create initial organizations because some organizations already exist.")

    kausal_org = Organization.objects.create(
        name="Kausal",
        depth=1,
        numchild=0,
        path=MP_Node._get_path('', 1, 1),  # type: ignore
    )

    netzerocities_org = Organization.objects.create(
        name="NetZeroCities",
        depth=1,
        numchild=0,
        path=MP_Node._get_path('', 1, 2),  # type: ignore
    )

    for instance in InstanceConfig.objects.all():
        has_framework = FrameworkConfig.objects.filter(instance_config=instance).exists()
        if has_framework:
            instance.organization = netzerocities_org
        else:
            instance.organization = kausal_org
        instance.save()


class Migration(migrations.Migration):

    dependencies = [
        ('nodes', '0032_add_instanceconfig_organization'),
    ]

    operations = [
        migrations.RunPython(create_initial_organizations),
    ]
