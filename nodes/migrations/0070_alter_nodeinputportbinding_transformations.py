import json
import typing
from typing import Any

import django.core.serializers.json
import django_pydantic_field.compat.django
import django_pydantic_field.fields
from django.db import migrations

import nodes.defs.transform_def


TEMPORAL_FLAG_KINDS = ('interpolate', 'backfill', 'extend')


def _as_dict(spec: Any) -> dict[str, Any]:
    if hasattr(spec, 'model_dump'):
        return spec.model_dump(mode='json')
    return dict(spec or {})


def _write_rows(schema_editor: Any, rows: list[tuple[Any, dict[str, Any], list[dict[str, Any]]]]) -> None:
    if not rows:
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.executemany(
            """
            UPDATE nodes_nodeinputportbinding
            SET dataset_spec = %s::jsonb, transformations = %s::jsonb
            WHERE id = %s
            """,
            [(json.dumps(spec), json.dumps(transformations), pk) for pk, spec, transformations in rows],
        )


def flags_to_transformations(apps, schema_editor):
    """Move transitional dataset flags into the authoritative ordered pipeline."""
    NodeInputPortBinding = apps.get_model('nodes', 'NodeInputPortBinding')
    rows = []
    for pk, stored_spec in NodeInputPortBinding.objects.filter(dataset_id__isnull=False).values_list('pk', 'dataset_spec'):
        spec = _as_dict(stored_spec)
        transformations = list(spec.get('transformations') or [])
        kinds = {op.get('kind') for op in transformations}
        transformations.extend({'kind': kind} for kind in TEMPORAL_FLAG_KINDS if spec.pop(kind, False) and kind not in kinds)
        spec['transformations'] = transformations
        rows.append((pk, spec, transformations))
    _write_rows(schema_editor, rows)


def transformations_to_flags(apps, schema_editor):
    """Restore the pre-0070 encoding for a code rollback."""
    NodeInputPortBinding = apps.get_model('nodes', 'NodeInputPortBinding')
    rows = []
    for pk, stored_spec in NodeInputPortBinding.objects.filter(dataset_id__isnull=False).values_list('pk', 'dataset_spec'):
        spec = _as_dict(stored_spec)
        transformations = list(spec.get('transformations') or [])
        kinds = {op.get('kind') for op in transformations}
        for kind in TEMPORAL_FLAG_KINDS:
            spec[kind] = kind in kinds
        transformations = [op for op in transformations if op.get('kind') not in TEMPORAL_FLAG_KINDS]
        spec['transformations'] = transformations
        rows.append((pk, spec, transformations))
    _write_rows(schema_editor, rows)


class Migration(migrations.Migration):
    dependencies = [
        ('nodes', '0069_truncate_legacy_binding_tables'),
    ]

    operations = [
        migrations.AlterField(
            model_name='nodeinputportbinding',
            name='transformations',
            field=django_pydantic_field.fields.PydanticSchemaField(
                blank=True,
                config=None,
                default=list,
                encoder=django.core.serializers.json.DjangoJSONEncoder,
                schema=django_pydantic_field.compat.django.GenericContainer(
                    list,
                    (
                        django_pydantic_field.compat.django.GenericContainer(
                            typing.Union,
                            (
                                nodes.defs.transform_def.FilterDimensionOp,
                                nodes.defs.transform_def.AssignDimensionOp,
                                nodes.defs.transform_def.DropNullsOp,
                                nodes.defs.transform_def.FilterTemporalOp,
                                nodes.defs.transform_def.FilterColumnOp,
                                nodes.defs.transform_def.RenameColumnOp,
                                nodes.defs.transform_def.RenameItemOp,
                                nodes.defs.transform_def.SetForecastFromOp,
                                nodes.defs.transform_def.EnsureUnitOp,
                                nodes.defs.transform_def.InterpolateOp,
                                nodes.defs.transform_def.BackfillOp,
                                nodes.defs.transform_def.ExtendOp,
                                nodes.defs.transform_def.SelectMetricOp,
                                nodes.defs.transform_def.IndexTemporalOp,
                                nodes.defs.transform_def.RemapLegacyYearsOp,
                                nodes.defs.transform_def.TagOperationOp,
                                nodes.defs.transform_def.SelectCategoriesTransformation,
                                nodes.defs.transform_def.AssignCategoryTransformation,
                                nodes.defs.transform_def.FlattenTransformation,
                            ),
                        ),
                    ),
                ),
            ),
        ),
        migrations.RunPython(flags_to_transformations, transformations_to_flags),
    ]
