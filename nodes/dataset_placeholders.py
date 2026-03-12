from __future__ import annotations

from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType

from kausal_common.datasets.models import (
    Dataset,
    DatasetMetric,
    DatasetSchema,
    DatasetSchemaDimension,
    DatasetSchemaScope,
    Dimension,
    DimensionCategory,
    DimensionScope,
)
from kausal_common.i18n.pydantic import TranslatedString

from nodes.units import unit_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from nodes.context import Context
    from nodes.dimensions import Dimension as DimensionSpec, DimensionCategory as DimensionCategorySpec
    from nodes.models import InstanceConfig
    from nodes.units import Unit


def _report(reporter: Callable[[str], None] | None, message: str) -> None:
    if reporter is not None:
        reporter(message)


def _iter_mappable_index_columns(
    index_columns: Sequence[str | object], ds_id: str, reporter: Callable[[str], None] | None
) -> list[str]:
    cols: list[str] = []
    for col in index_columns:
        if isinstance(col, str):
            cols.append(col)
            continue
        _report(
            reporter,
            f"Skipping non-column external dataset index descriptor {col!r} for placeholder '{ds_id}'.",
        )
    return cols


def make_external_dataset_ref(ctx: Context, ds_id: str) -> dict[str, str | None] | None:
    repo = ctx.dataset_repo_spec
    if repo is None:
        return None
    return {
        'repo_url': repo.url,
        'commit': repo.commit,
        'dataset_id': ds_id,
    }


def _create_dataset_schema(
    instance_config: InstanceConfig,
    default_language: str,
    name_i18n: dict[str, str] | None,
) -> DatasetSchema:
    schema = DatasetSchema(
        time_resolution=DatasetSchema.TimeResolution.YEARLY,
    )
    if name_i18n is not None:
        if default_language not in name_i18n:
            # Fallback on whatever we find
            name_i18n[default_language] = next(iter(name_i18n.values()))
        name = TranslatedString(default_language=default_language, **name_i18n)
        name.set_modeltrans_field(schema, 'name', default_language)
    schema.save()
    DatasetSchemaScope.objects.create(
        schema=schema,
        scope_content_type=ContentType.objects.get_for_model(instance_config),
        scope_id=instance_config.pk,
    )
    return schema


def _create_metric(
    col: str,
    unit: Unit | str,
    schema: DatasetSchema,
    default_language: str,
    label_i18n: dict[str, str] | None,
) -> DatasetMetric:
    if isinstance(unit, str):
        unit = unit_registry.parse_units(unit)
    metric = DatasetMetric(schema=schema, name=col, label=col, unit=str(unit))
    if label_i18n is not None:
        if default_language not in label_i18n:
            # Fallback on whatever we find
            label_i18n[default_language] = next(iter(label_i18n.values()))
        label = TranslatedString(default_language=default_language, **label_i18n)
        label.set_modeltrans_field(metric, 'label', default_language)
    metric.save()
    return metric


def _create_dimension_category(
    dimension: Dimension,
    default_language: str,
    spec: DimensionCategorySpec,
) -> DimensionCategory:
    cat = DimensionCategory(dimension=dimension, identifier=spec.id)
    label = spec.label
    assert isinstance(label, TranslatedString)
    label.set_modeltrans_field(cat, 'label', default_language)
    cat.save()
    return cat


def _create_dimension(
    schema: DatasetSchema,
    instance_config: InstanceConfig,
    default_language: str,
    spec: DimensionSpec,
) -> Dimension:
    dimension = Dimension()
    label = spec.label
    assert isinstance(label, TranslatedString)
    label.set_modeltrans_field(dimension, 'name', default_language)
    dimension.save()
    DatasetSchemaDimension.objects.create(schema=schema, dimension=dimension)
    for cat_spec in spec.categories:
        _create_dimension_category(
            dimension=dimension,
            default_language=default_language,
            spec=cat_spec,
        )
    DimensionScope.objects.create(
        dimension=dimension,
        scope_content_type=ContentType.objects.get_for_model(instance_config),
        scope_id=instance_config.pk,
        identifier=spec.id,
    )
    return dimension


def _get_or_create_dimension(
    schema: DatasetSchema,
    instance_config: InstanceConfig,
    default_language: str,
    spec: DimensionSpec,
) -> Dimension:
    scope = (
        DimensionScope.objects
        .filter(
            scope_content_type=ContentType.objects.get_for_model(instance_config),
            scope_id=instance_config.pk,
            identifier=spec.id,
        )
        .select_related('dimension')
        .first()
    )
    if scope is None:
        return _create_dimension(
            schema=schema,
            instance_config=instance_config,
            default_language=default_language,
            spec=spec,
        )
    DatasetSchemaDimension.objects.get_or_create(schema=schema, dimension=scope.dimension)
    return scope.dimension


def sync_dataset_placeholder(
    instance_config: InstanceConfig,
    ctx: Context,
    ds_id: str,
    *,
    force: bool = False,
    reporter: Callable[[str], None] | None = None,
) -> tuple[Dataset | None, bool]:
    try:
        dvc_ds = ctx.load_dvc_dataset(ds_id)
    except Exception as e:
        _report(reporter, f"Error loading DVC dataset '{ds_id}': {e}")
        return None, False

    dvc_metadata = dvc_ds.metadata or {}
    metric_units = dvc_ds.units or {}
    index_columns = _iter_mappable_index_columns(dvc_ds.index_columns or [], ds_id, reporter)

    dataset_lookup = dict(
        identifier=ds_id,
    )
    existing = Dataset.objects.qs.for_scope(instance_config).filter(identifier=ds_id).select_related('schema').first()
    if existing is not None:
        if existing.is_external_placeholder and force:
            schema = existing.schema
            if schema is not None and schema.datasets.count() > 1:
                raise RuntimeError(f"Dataset '{existing}' cannot be recreated because its schema is shared with other datasets.")
            existing.delete()
            if schema is not None:
                schema.delete()
        else:
            _report(
                reporter,
                f"Dataset '{existing}' with identifier '{ds_id}' already exists for instance '{instance_config}'; "
                + 'skipping placeholder creation.',
            )
            return existing, False

    schema = _create_dataset_schema(
        instance_config=instance_config,
        default_language=ctx.instance.default_language,
        name_i18n=dvc_metadata.get('name'),
    )
    dataset = Dataset.objects.create(
        **dataset_lookup,
        schema=schema,
        external_ref=make_external_dataset_ref(ctx, ds_id),
        is_external_placeholder=True,
    )

    metrics_meta = {
        (m.get('column_id') or m.get('id')): m for m in dvc_metadata.get('metrics') or [] if m.get('column_id') or m.get('id')
    }
    for col, unit in metric_units.items():
        _create_metric(
            col=col,
            unit=unit,
            schema=schema,
            default_language=ctx.instance.default_language,
            label_i18n=metrics_meta.get(col, {}).get('label'),
        )

    for col in index_columns:
        if col not in ctx.dimensions:
            _report(
                reporter,
                f"Skipping external dataset index column '{col}' for placeholder '{ds_id}' "
                + 'because it does not map to a model dimension.',
            )
            continue
        _get_or_create_dimension(
            schema=schema,
            instance_config=instance_config,
            default_language=ctx.instance.default_language,
            spec=ctx.dimensions[col],
        )

    _report(reporter, f"Created external placeholder dataset '{ds_id}'")
    return dataset, True


def sync_instance_dataset_placeholders(
    instance_config: InstanceConfig,
    ctx: Context,
    *,
    force: bool = False,
    reporter: Callable[[str], None] | None = None,
) -> list[str]:
    created_dataset_ids: list[str] = []
    for ds_id in sorted(ctx.get_all_dvc_dataset_ids()):
        _dataset, created = sync_dataset_placeholder(
            instance_config,
            ctx,
            ds_id,
            force=force,
            reporter=reporter,
        )
        if created:
            created_dataset_ids.append(ds_id)
    return created_dataset_ids
