from django.db import migrations

RETIRED_KINDS = ('allowed_combinations',)


def refuse_retired_rules(apps, schema_editor):
    """
    Report deprecated requirements and refuse rules that no longer parse.

    Required combinations remain supported so deployment can finish before models are
    re-synced to shapes. A closed shape replaces allowed-combination rules; those must
    still be converted explicitly before deployment.
    """
    rule_model = apps.get_model('datasets', 'DatasetMetricValidationRule')
    rule_rows = rule_model.objects.using(schema_editor.connection.alias)
    deprecated = rule_rows.filter(rule__kind='required_combinations')
    if deprecated.exists():
        names = sorted({f'{rule.metric.schema_id}/{rule.metric.name}' for rule in deprecated.select_related('metric')})
        print(
            '\n  Deprecated required_combinations rules remain (schema id/metric): '
            f'{", ".join(names)}. Re-sync these models to shapes before removing compatibility support.'
        )
    retired = rule_rows.filter(rule__kind__in=RETIRED_KINDS)
    if retired.exists():
        names = sorted({
            f'{rule.metric.schema_id}/{rule.metric.name}: {rule.rule["kind"]}' for rule in retired.select_related('metric')
        })
        raise RuntimeError(f'Dataset validation rules of a retired kind remain (schema id/metric): {", ".join(names)}')


class Migration(migrations.Migration):
    dependencies = [
        ('datasets', '0040_dimension_category_short_label'),
        ('paths_datasets', '0002_plausibility_ranges'),
    ]

    operations = [
        migrations.RunPython(refuse_retired_rules, migrations.RunPython.noop),
    ]
