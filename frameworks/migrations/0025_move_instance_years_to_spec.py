from django.db import migrations


def update_instance_year_spec(raw_spec, *, baseline_year, target_year, default_target_year):
    if raw_spec is None:
        spec: dict = {}
    elif hasattr(raw_spec, 'model_dump'):
        spec = raw_spec.model_dump(mode='json')
    else:
        spec = dict(raw_spec)

    years = dict(spec.get('years') or {})
    years['reference'] = baseline_year
    if target_year is not None:
        years['target'] = target_year
    elif years.get('target') is None:
        years['target'] = default_target_year
    spec['years'] = years
    return spec


def move_instance_years_to_spec(apps, schema_editor):
    FrameworkConfig = apps.get_model('frameworks', 'FrameworkConfig')
    InstanceConfig = apps.get_model('nodes', 'InstanceConfig')

    configs = FrameworkConfig.objects.select_related('framework', 'instance_config').filter(
        instance_config__config_source='yaml',
    )
    for framework_config in configs.iterator():
        instance_config = framework_config.instance_config
        defaults = framework_config.framework.defaults
        if hasattr(defaults, 'model_dump'):
            defaults = defaults.model_dump(mode='json')
        target_defaults = (defaults or {}).get('target_year') or {}
        spec = update_instance_year_spec(
            instance_config.spec,
            baseline_year=framework_config.baseline_year,
            target_year=framework_config.target_year,
            default_target_year=target_defaults.get('default'),
        )
        InstanceConfig.objects.filter(pk=instance_config.pk).update(spec=spec)


class Migration(migrations.Migration):
    dependencies = [
        ('frameworks', '0024_set_root_page_home_label'),
        ('nodes', '0072_delete_legacy_binding_tables'),
    ]

    operations = [
        migrations.RunPython(move_instance_years_to_spec, migrations.RunPython.noop),
        migrations.RemoveField(
            model_name='frameworkconfig',
            name='baseline_year',
        ),
        migrations.RemoveField(
            model_name='frameworkconfig',
            name='target_year',
        ),
    ]
