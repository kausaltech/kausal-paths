from django.db import migrations


def enable_user_management(raw_spec):
    if hasattr(raw_spec, 'model_dump'):
        spec = raw_spec.model_dump(mode='json')
    else:
        spec = dict(raw_spec)

    features = dict(spec.get('features') or {})
    features['enable_user_management'] = True
    spec['features'] = features
    return spec


def backfill_instance_user_management(apps, schema_editor):
    FrameworkConfig = apps.get_model('frameworks', 'FrameworkConfig')
    InstanceConfig = apps.get_model('nodes', 'InstanceConfig')

    configs = FrameworkConfig.objects.select_related('instance_config').filter(
        framework__enable_user_management=True,
        instance_config__spec__isnull=False,
    )
    for framework_config in configs.iterator():
        instance_config = framework_config.instance_config
        spec = enable_user_management(instance_config.spec)
        InstanceConfig.objects.filter(pk=instance_config.pk).update(spec=spec)


class Migration(migrations.Migration):
    dependencies = [
        ('frameworks', '0025_move_instance_years_to_spec'),
    ]

    operations = [
        migrations.RunPython(backfill_instance_user_management, migrations.RunPython.noop),
    ]
