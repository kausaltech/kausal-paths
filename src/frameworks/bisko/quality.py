"""Make BISKO quality metrics projections of per-point grades."""

from collections import defaultdict
from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from django.db.models import Q

from kausal_common.datasets.models import DataPoint, Dataset, DatasetMetric

from datasets.materialization import refresh_dataset_materialization
from frameworks.evidence import QUALITY_OF_SPEC_KEY, get_evidence, set_evidence
from frameworks.models import DataQualityLevel, Framework
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from datetime import date
    from decimal import Decimal


def _point_key(point: DataPoint) -> tuple[date, frozenset[int]]:
    return point.date, frozenset(category.pk for category in point.dimension_categories.all())


@transaction.atomic
def provision_bisko_quality_projections(framework: Framework) -> None:
    """
    Convert legacy scores before linking a shared schema's quality metric.

    A schema can be shared by the template, municipalities, and instances still
    awaiting framework conversion. Its metric spec affects all of them, so every
    dataset using it must be converted before the projection is enabled.
    """
    template = framework.template_instance
    assert template is not None
    instance_ids = [template.pk, *framework.configs.values_list('instance_config_id', flat=True)]
    scoped = Dataset.objects.filter(
        Q(scope_content_type=ContentType.objects.get_for_model(InstanceConfig), scope_id__in=instance_ids)
        | Q(scope_content_type=ContentType.objects.get_for_model(Framework), scope_id=framework.pk),
        schema__metrics__name='quality',
    )
    schema_ids = set(scoped.values_list('schema_id', flat=True)) - {None}
    levels = list(DataQualityLevel.objects.filter(scheme__framework=framework, scheme__identifier='quality', scheme__version='1'))
    by_score: dict[Decimal, list[DataQualityLevel]] = defaultdict(list)
    for level in levels:
        by_score[level.score].append(level)

    for schema_id in sorted(schema_ids):
        _provision_schema(schema_id, by_score)


def _provision_schema(schema_id: int, by_score: dict[Decimal, list[DataQualityLevel]]) -> None:
    metrics = list(DatasetMetric.objects.filter(schema_id=schema_id))
    quality = [metric for metric in metrics if metric.name == 'quality']
    values = [metric for metric in metrics if metric.name != 'quality']
    if len(quality) != 1 or len(values) != 1:
        raise ValueError(f'BISKO schema {schema_id} needs one quality metric and one value metric.')
    quality_metric, value_metric = quality[0], values[0]
    projection = (quality_metric.spec or {}).get(QUALITY_OF_SPEC_KEY)
    if projection is not None and projection != str(value_metric.uuid):
        raise ValueError(f'BISKO schema {schema_id} projects quality onto a different metric.')

    datasets = list(Dataset.objects.filter(schema_id=schema_id))
    changes = [change for dataset in datasets for change in _legacy_grades(dataset, quality_metric, value_metric, by_score)]
    for target, level in changes:
        set_evidence(target, quality_level=level, user=None)
    if projection is None:
        quality_metric.spec = {**(quality_metric.spec or {}), QUALITY_OF_SPEC_KEY: str(value_metric.uuid)}
        quality_metric.save(update_fields=['spec'])
    if changes or projection is None:
        for dataset in datasets:
            refresh_dataset_materialization(dataset)


def _legacy_grades(
    dataset: Dataset,
    quality_metric: DatasetMetric,
    value_metric: DatasetMetric,
    by_score: dict[Decimal, list[DataQualityLevel]],
) -> list[tuple[DataPoint, DataQualityLevel]]:
    points = list(
        DataPoint.objects
        .filter(dataset=dataset, metric_id__in=[quality_metric.pk, value_metric.pk])
        .select_related('evidence__quality_level')
        .prefetch_related('dimension_categories')
    )
    value_points = {_point_key(point): point for point in points if point.metric_id == value_metric.pk}
    changes: list[tuple[DataPoint, DataQualityLevel]] = []
    for point in points:
        if point.metric_id != quality_metric.pk or point.value is None:
            continue
        matching = by_score.get(point.value, [])
        target = value_points.get(_point_key(point))
        if len(matching) != 1 or target is None:
            raise ValueError(
                f'{dataset.identifier}: quality value {point.value} has no unique BISKO grade '
                'or matching value point; migrate it explicitly.'
            )
        existing = get_evidence(target)
        if existing is not None and existing.quality_level_id not in (None, matching[0].pk):
            raise ValueError(f'{dataset.identifier}: existing grade conflicts with legacy quality value.')
        if existing is None or existing.quality_level_id is None:
            changes.append((target, matching[0]))
    return changes
