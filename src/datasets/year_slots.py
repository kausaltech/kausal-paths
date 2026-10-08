"""Create blank annual input cells from an instance dataset's declared shape."""

from datetime import date
from typing import TYPE_CHECKING

from django.db import transaction

from kausal_common.datasets.models import DataPoint, DataPointDimensionCategory, Dataset, DimensionCategory

from datasets.materialization import refresh_dataset_materialization
from datasets.shape_domain import dataset_category_domain

if TYPE_CHECKING:
    from uuid import UUID


def _point_layouts(source: Dataset, year: int, metric_ids: set[int]) -> set[tuple[int, tuple[int, ...]]]:
    source_year = (
        source.data_points.filter(date__year__lte=year).order_by('-date').values_list('date__year', flat=True).first()
        or source.data_points.order_by('date').values_list('date__year', flat=True).first()
    )
    if source_year is None:
        return set()
    return {
        (point.metric_id, tuple(sorted(cat.pk for cat in point.dimension_categories.all())))
        for point in source.data_points.filter(date__year=source_year).prefetch_related('dimension_categories')
        if point.metric_id in metric_ids
    }


@transaction.atomic
def ensure_empty_year(dataset: Dataset, year: int, *, prototype: Dataset | None = None) -> int:
    """Add missing null cells for a year; preserve observations and existing blank cells."""
    dataset = Dataset.objects.select_for_update().get(pk=dataset.pk)
    schema = dataset.schema
    if schema is None or not schema.is_editable or schema.time_resolution != schema.TimeResolution.YEARLY:
        return 0
    metrics = {metric.pk: metric for metric in schema.metrics.all() if 'quality_of' not in (metric.spec or {})}
    if not metrics:
        return 0

    # The dataset's shape is the model's explicit row layout. Without declared combinations,
    # an existing year (or the template) supplies the observed layout.
    layouts: set[tuple[int, tuple[int, ...]]] = set()
    domain = dataset_category_domain(dataset)
    if domain is not None and domain.combinations:
        uuids: set[UUID] = {uuid for combo in domain.combinations for uuid in combo.categories.values()}
        category_ids = dict(DimensionCategory.objects.filter(uuid__in=uuids).values_list('uuid', 'pk'))
        for combo in domain.combinations:
            categories = tuple(sorted(category_ids[uuid] for uuid in combo.categories.values()))
            layouts.update((metric_id, categories) for metric_id in metrics)

    layouts.update(_point_layouts(dataset, year, set(metrics)))
    if prototype is not None and prototype.schema_id == schema.pk:
        layouts.update(_point_layouts(prototype, year, set(metrics)))

    if not layouts and not schema.dimensions.exists():
        layouts.update((metric_id, ()) for metric_id in metrics)
    if not layouts:
        return 0

    existing = {
        (point.metric_id, tuple(sorted(cat.pk for cat in point.dimension_categories.all())))
        for point in dataset.data_points.filter(date__year=year).prefetch_related('dimension_categories')
    }
    missing = sorted(layouts - existing)
    if not missing:
        return 0
    points = DataPoint.objects.bulk_create([
        DataPoint(dataset=dataset, metric=metrics[metric_id], date=date(year, 1, 1), value=None)
        for metric_id, _categories in missing
    ])
    DataPointDimensionCategory.objects.bulk_create([
        DataPointDimensionCategory(data_point=point, dimension_category_id=category_id)
        for point, (_metric_id, categories) in zip(points, missing, strict=True)
        for category_id in categories
    ])
    refresh_dataset_materialization(dataset, touch=False)
    return len(points)
