from django.db import migrations

RETIRED_KINDS = ('required_combinations', 'allowed_combinations')


def refuse_retired_rules(apps, schema_editor):
    """
    Stop rather than drop rules of a retired kind; they no longer parse.

    Required combinations are now required groups of a shape, checked at the input port the
    dataset feeds, and a closed shape refuses values outside it without a rule. Move any such
    rule into a shape (docs/architecture/shapes.md) and delete it, then migrate again.
    """
    rule_model = apps.get_model('datasets', 'DatasetMetricValidationRule')
    retired = rule_model.objects.filter(rule__kind__in=RETIRED_KINDS)
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
