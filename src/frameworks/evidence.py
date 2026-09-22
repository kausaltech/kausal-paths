"""Reading and writing data-point evidence, independent of the API that carries the request."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from django.core.exceptions import ObjectDoesNotExist, ValidationError

from frameworks.models import DataEvidenceKind, DataPointEvidence, DataQualityLevel, DataQualityScheme, Framework

if TYPE_CHECKING:
    from uuid import UUID

    from django.db.models import QuerySet

    import polars as pl

    from kausal_common.datasets.models import DataPoint, Dataset, DatasetMetric

    from users.models import User


class _Unchanged:
    def __repr__(self) -> str:
        return 'UNCHANGED'


UNCHANGED: Final = _Unchanged()
"""Leave a field as it is; distinct from `None`, which clears it."""


def framework_for_dataset(dataset: Dataset) -> Framework | None:
    """
    Return the framework whose vocabularies apply to `dataset`, if any.

    That is the framework the dataset is scoped to, the framework of the member
    instance it belongs to, or the framework whose template instance it belongs to.
    """
    from nodes.models import InstanceConfig

    scope = dataset.scope
    if isinstance(scope, Framework):
        return scope
    if not isinstance(scope, InstanceConfig):
        return None
    if scope.has_framework_config():
        return scope.framework_config.framework
    return Framework.objects.filter(template_instance=scope).first()


def quality_schemes_for_dataset(dataset: Dataset) -> QuerySet[DataQualityScheme]:
    framework = framework_for_dataset(dataset)
    if framework is None:
        return DataQualityScheme.objects.none()
    return framework.quality_schemes.prefetch_related('levels')


def resolve_quality_level(dataset: Dataset, level_uuid: UUID) -> DataQualityLevel:
    level = DataQualityLevel.objects.filter(uuid=level_uuid, scheme__in=quality_schemes_for_dataset(dataset)).first()
    if level is None:
        raise ValidationError(f'Quality level {level_uuid} is not available for this dataset')
    return level


def get_evidence(data_point: DataPoint) -> DataPointEvidence | None:
    try:
        return data_point.evidence  # type: ignore[attr-defined]
    except ObjectDoesNotExist:
        return None


def evidence_snapshot(data_point: DataPoint) -> dict[str, Any] | None:
    """Evidence part of a data point's change-history snapshot."""
    evidence = get_evidence(data_point)
    if evidence is None:
        return None
    level = evidence.quality_level
    return {
        'kind': evidence.kind,
        'quality_level_uuid': str(level.uuid) if level is not None else None,
    }


def set_evidence(
    data_point: DataPoint,
    *,
    kind: DataEvidenceKind | _Unchanged | None = UNCHANGED,
    quality_level: DataQualityLevel | _Unchanged | None = UNCHANGED,
    user: User | None,
) -> DataPointEvidence | None:
    """
    Apply an evidence change to `data_point` and return the resulting row.

    Clearing both fields removes the row, since an assertion of nothing is no
    assertion. The level must already be resolved against the dataset (see
    `resolve_quality_level`). Call `validate_evidence` afterwards, once the
    value is final too.
    """
    evidence = get_evidence(data_point)
    new_kind = evidence.kind if evidence is not None else None
    new_level = evidence.quality_level if evidence is not None else None
    if not isinstance(kind, _Unchanged):
        new_kind = kind
    if not isinstance(quality_level, _Unchanged):
        new_level = quality_level

    if new_kind is None and new_level is None:
        if evidence is not None:
            evidence.delete()
            data_point._state.fields_cache.pop('evidence', None)
        return None

    if evidence is None:
        evidence = DataPointEvidence(data_point=data_point, created_by=user)
    if evidence.pk is not None and (evidence.kind, evidence.quality_level_id) == (
        new_kind,
        new_level.pk if new_level is not None else None,
    ):
        return evidence
    evidence.kind = new_kind
    evidence.quality_level = new_level
    evidence.last_modified_by = user
    evidence.save()
    data_point.evidence = evidence  # type: ignore[attr-defined]
    return evidence


def validate_evidence(data_point: DataPoint) -> None:
    """
    Check the invariants that tie evidence to the value it describes.

    A confirmed zero is a statement about a zero; it does not survive the value
    changing. The caller must clear or change the kind in the same write.
    """
    evidence = get_evidence(data_point)
    if evidence is None or evidence.kind != DataEvidenceKind.EXPLICIT_ZERO:
        return
    if data_point.value is None or data_point.value != 0:
        raise ValidationError('A confirmed zero requires the value to be zero')


QUALITY_OF_SPEC_KEY: Final = 'quality_of'
"""
`DatasetMetric.spec` key marking a metric as the numeric projection of another metric's grades.

Its value is the UUID of the graded metric. The projected column is derived from
evidence whenever the dataset is read into a dataframe; stored data points of the
projected metric are ignored, so evidence stays the only authority.
"""


def quality_projections(dataset: Dataset) -> dict[str, str]:
    """Map projected metric UUID -> graded metric UUID, for the metrics of `dataset`'s schema."""
    from kausal_common.datasets.models import DatasetMetric

    if dataset.schema_id is None:
        return {}
    result: dict[str, str] = {}
    for uuid, spec in DatasetMetric.objects.filter(schema_id=dataset.schema_id, spec__has_key=QUALITY_OF_SPEC_KEY).values_list(
        'uuid', 'spec'
    ):
        result[str(uuid)] = str(spec[QUALITY_OF_SPEC_KEY])
    return result


def is_projected_metric(metric: DatasetMetric) -> bool:
    return QUALITY_OF_SPEC_KEY in (metric.spec or {})


def project_quality_columns(dataset: Dataset, df: pl.DataFrame) -> pl.DataFrame:
    """
    Replace projected metrics' rows in a long data-point frame with evidence scores.

    `df` is the long frame `DBDataset.deserialize_df` builds before pivoting: one
    row per data point, with its `id`, `value`, metric UUID in `metric` and the
    metric's column name in `metric_name`. Each graded data point of the graded
    metric yields one row of the projected metric at the same coordinates; an
    ungraded point yields none, so the projected cell is null rather than zero.
    """
    import polars as pl

    projections = quality_projections(dataset)
    if not projections:
        return df

    scores = DataPointEvidence.objects.filter(data_point__dataset=dataset, quality_level__isnull=False).values_list(
        'data_point_id', 'quality_level__score'
    )
    score_df = pl.DataFrame(
        [(dp_id, float(score)) for dp_id, score in scores],
        schema={'id': pl.Int64, '_score': pl.Float64},
        orient='row',
    )
    names = dict(df.select('metric', 'metric_name').unique().iter_rows())

    parts = [df.filter(~pl.col('metric').is_in(list(projections)))]
    for projected, graded in projections.items():
        projected_name = names.get(projected)
        if projected_name is None:
            from kausal_common.datasets.models import DatasetMetric

            metric = DatasetMetric.objects.get(uuid=projected)
            projected_name = metric.name or metric.label or projected
        parts.append(
            df
            .filter(pl.col('metric') == graded)
            .join(score_df, on='id', how='inner')
            .with_columns(
                pl.col('_score').alias('value'),
                pl.lit(projected).alias('metric'),
                pl.lit(projected_name).alias('metric_name'),
                pl.lit(None, dtype=pl.Int64).alias('id'),
            )
            .drop('_score')
            .select(df.columns)
        )
    return pl.concat(parts, how='vertical')
