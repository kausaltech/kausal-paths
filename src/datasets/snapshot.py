"""
Dataset revision and portable export snapshot shapes.

A `DatasetSnapshot` is a dataset's structure (a `DatasetMeta`, the same entry the
instance graph catalogs) and its body: one entry per data point, each carrying its
own uuid and the comments, evidence and source references attached to it. Everything
is referred to by uuid; column names and category labels are a property of the frame
built from it (`DatasetSnapshot.to_frame`), never of the stored form.
"""

from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING, Literal, Self
from uuid import UUID

from django.db.models import CharField, F, Value
from django.db.models.functions import Cast, Coalesce, NullIf
from pydantic import BaseModel, Field

import polars as pl

from common import polars as ppl
from nodes.constants import YEAR_COLUMN
from nodes.defs.graph import DatasetMeta  # noqa: TC001 - Pydantic field
from nodes.snapshot_base import ModelSnapshot
from nodes.units import unit_registry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kausal_common.datasets.models import (
        DataPointComment,
        Dataset,
        DatasetMetric,
        DatasetSourceReference,
        DataSource,
    )

    from frameworks.models import DataPointEvidence
    from nodes.defs.graph import DimensionMeta
    from nodes.models import InstanceConfig


def metric_column_id(metric: DatasetMetric) -> str:
    """
    Resolve the dataframe column a metric maps to.

    ``DatasetMetric.name`` *is* the column name and ``label`` is display text, so the
    two agree for imported metrics: ``load_dvc_dataset`` and ``dataset_placeholders``
    both create them as ``name=label=<column>``. A metric authored in the editor has
    no ``name`` at all, and its column is built from the label — see the
    ``Coalesce(name, label, uuid)`` in ``DBDataset.deserialize_df``
    (``datasets/runtime/db.py``), which is the writer this function has to agree with.
    Build SQL selectors with ``metric_column_id_expr()``, its twin below.

    **Falling through to the uuid is never a working answer.** No dataframe column is
    ever named after a metric uuid, so a selector built from one cannot match: it
    fails in ``select_metric`` as "Column '<uuid>' not found". It is kept only so a
    metric carrying neither name nor label still yields a stable string instead of
    ``None``.
    """
    return metric.name or metric.label or str(metric.uuid)


def metric_column_id_expr(prefix: str = '') -> Coalesce:
    """
    Return the SQL twin of ``metric_column_id()`` for a ``DatasetMetric`` queryset.

    ``prefix`` is the lookup path to the metric (``'metric__'`` from a binding row).
    Empty strings fall through as in the Python version, which tests truthiness.
    """
    return Coalesce(
        NullIf(F(f'{prefix}name'), Value('')),
        NullIf(F(f'{prefix}label'), Value('')),
        Cast(f'{prefix}uuid', output_field=CharField()),
    )


class DataSourceSnapshot(ModelSnapshot['DataSource']):
    """A data source cited by the dataset or by its data points."""

    id: UUID
    name: str
    edition: str | None = None
    authority: str | None = None
    description: str | None = None
    url: str | None = None

    @classmethod
    def from_model(cls, obj: DataSource) -> Self:
        return cls(
            id=obj.uuid,
            name=obj.name,
            edition=obj.edition,
            authority=obj.authority,
            description=obj.description,
            url=obj.url,
        )


class SourceReferenceSnapshot(ModelSnapshot['DatasetSourceReference']):
    """A citation of a data source, by the dataset or by the data point it is listed under."""

    id: UUID
    data_source: UUID

    @classmethod
    def from_model(cls, obj: DatasetSourceReference) -> Self:
        return cls(id=obj.uuid, data_source=obj.data_source.uuid)


class DataPointCommentSnapshot(ModelSnapshot['DataPointComment']):
    """A (non-soft-deleted) comment on a data point. Users are referenced by uuid."""

    id: UUID
    text: str
    is_sticky: bool = False
    is_review: bool = False
    review_state: str | None = None
    resolved_at: str | None = None  # ISO 8601
    created_by: str | None = None  # user uuid
    last_modified_by: str | None = None  # user uuid
    resolved_by: str | None = None  # user uuid

    @classmethod
    def from_model(cls, obj: DataPointComment) -> Self:
        return cls(
            id=obj.uuid,
            text=obj.text,
            is_sticky=obj.is_sticky,
            is_review=obj.is_review,
            review_state=obj.review_state,
            resolved_at=obj.resolved_at.isoformat() if obj.resolved_at else None,
            created_by=str(obj.created_by.uuid) if obj.created_by else None,
            last_modified_by=str(obj.last_modified_by.uuid) if obj.last_modified_by else None,
            resolved_by=str(obj.resolved_by.uuid) if obj.resolved_by else None,
        )


class QualityLevelRef(BaseModel):
    """
    A framework quality grade, by UUID and by authored identity.

    The UUID is exact within one deployment; the identifiers let a snapshot
    resolve against a framework provisioned elsewhere. The score is the grade's
    as it stood when the snapshot was taken, so a frozen revision computes the
    quality columns it computed then.
    """

    uuid: str
    scheme: str
    scheme_version: str
    level: str
    score: float | None = None


class DataPointEvidenceSnapshot(ModelSnapshot['DataPointEvidence']):
    """What is asserted about one data point's value. Users are referenced by uuid."""

    kind: str | None = None
    quality_level: QualityLevelRef | None = None
    created_by: str | None = None  # user uuid
    last_modified_by: str | None = None  # user uuid

    @classmethod
    def from_model(cls, obj: DataPointEvidence) -> Self:
        level = obj.quality_level
        return cls(
            kind=obj.kind,
            quality_level=QualityLevelRef(
                uuid=str(level.uuid),
                scheme=level.scheme.identifier,
                scheme_version=level.scheme.version,
                level=level.identifier,
                score=float(level.score) if level.score is not None else None,
            )
            if level is not None
            else None,
            created_by=str(obj.created_by.uuid) if obj.created_by else None,
            last_modified_by=str(obj.last_modified_by.uuid) if obj.last_modified_by else None,
        )


class DataPointSnapshot(BaseModel):
    """One data point, with everything attached to it."""

    id: UUID
    date: date
    metric: UUID
    categories: dict[UUID, UUID] = Field(default_factory=dict)
    """Dimension uuid to category uuid; a dimension the point is not classified by is absent."""
    value: float | None = None
    """A null value is a cell that exists and holds no number, which is not an absent cell."""
    comments: list[DataPointCommentSnapshot] = Field(default_factory=list)
    evidence: DataPointEvidenceSnapshot | None = None
    sources: list[SourceReferenceSnapshot] = Field(default_factory=list)


type CellLabels = tuple[int, str, tuple[str, ...]]
"""A cell as a frame names it: year, metric column and the sorted category labels."""


class DatasetFrameError(ValueError):
    """The snapshot refers to a dimension or category the given catalog does not have."""


class DatasetSnapshot(ModelSnapshot['Dataset']):
    """
    A dataset: its structure and every data point, keyed by uuid.

    The Wagtail revision payload of a `Dataset`, its materialization (the content
    the current runtime reads, and the content hash) and the dataset body of an
    `InstanceExport` are all this.
    """

    schema_version: Literal[2] = 2
    meta: DatasetMeta
    dimension_columns: dict[UUID, str] = Field(default_factory=dict)
    """
    A dimension's column in the external source, where it differs from the dimension's identifier.

    External-source metadata (see decision 13 of docs/plans/node-owned-datasets.md):
    the frame built from the snapshot uses it, so that the dataset reads as it did
    when it was imported.
    """
    points: list[DataPointSnapshot] = Field(default_factory=list)
    data_sources: list[DataSourceSnapshot] = Field(default_factory=list)
    """Every data source cited, by the dataset or by a data point."""
    source_references: list[SourceReferenceSnapshot] = Field(default_factory=list)
    """The dataset's own citations; a data point's are listed under the data point."""

    @classmethod
    def from_model(cls, obj: Dataset, instance_config: InstanceConfig | None = None) -> Self:
        from kausal_common.datasets.models import (
            DataPoint,
            DataPointComment,
            DataPointDimensionCategory,
            DatasetSchemaDimension,
            DatasetSourceReference,
        )

        from datasets.catalogue import dataset_meta_from_model
        from frameworks.models import DataPointEvidence

        instance = instance_config or obj.scope_instance
        meta = dataset_meta_from_model(obj, primary_language=instance.primary_language or 'en')
        dimension_columns = {
            schema_dimension.dimension.uuid: schema_dimension.column_name
            for schema_dimension in DatasetSchemaDimension.objects.filter(schema=obj.schema).select_related('dimension')
            if schema_dimension.column_name
        }

        sources: dict[UUID, DataSourceSnapshot] = {}

        def cite(reference: DatasetSourceReference) -> SourceReferenceSnapshot:
            if reference.data_source.uuid not in sources:
                sources[reference.data_source.uuid] = DataSourceSnapshot.from_model(reference.data_source)
            return SourceReferenceSnapshot.from_model(reference)

        dataset_references = [
            cite(reference)
            for reference in DatasetSourceReference.objects.filter(dataset=obj).select_related('data_source').order_by('uuid')
        ]
        points: list[DataPointSnapshot] = []
        if not obj.is_external_placeholder:
            categories: defaultdict[int, dict[UUID, UUID]] = defaultdict(dict)
            for point_pk, dimension_uuid, category_uuid in DataPointDimensionCategory.objects.filter(
                data_point__dataset=obj
            ).values_list('data_point_id', 'dimension_category__dimension__uuid', 'dimension_category__uuid'):
                categories[point_pk][dimension_uuid] = category_uuid
            comments: defaultdict[int, list[DataPointCommentSnapshot]] = defaultdict(list)
            for comment in (
                DataPointComment.objects  # the default manager leaves out soft-deleted comments
                .filter(data_point__dataset=obj)
                .select_related('created_by', 'last_modified_by', 'resolved_by')
                .order_by('created_at', 'uuid')
            ):
                assert comment.data_point_id is not None
                comments[comment.data_point_id].append(DataPointCommentSnapshot.from_model(comment))
            point_sources: defaultdict[int, list[SourceReferenceSnapshot]] = defaultdict(list)
            for reference in (
                DatasetSourceReference.objects.filter(data_point__dataset=obj).select_related('data_source').order_by('uuid')
            ):
                assert reference.data_point_id is not None
                point_sources[reference.data_point_id].append(cite(reference))
            evidence = {
                item.data_point_id: DataPointEvidenceSnapshot.from_model(item)
                for item in DataPointEvidence.objects.filter(data_point__dataset=obj).select_related(
                    'quality_level__scheme', 'created_by', 'last_modified_by'
                )
            }
            metric_order = {metric.id: index for index, metric in enumerate(meta.metrics)}
            for pk, uuid, point_date, value, metric_uuid in DataPoint.objects.filter(dataset=obj).values_list(
                'pk', 'uuid', 'date', 'value', 'metric__uuid'
            ):
                points.append(
                    DataPointSnapshot(
                        id=uuid,
                        date=point_date,
                        metric=metric_uuid,
                        categories=categories.get(pk, {}),
                        value=float(value) if value is not None else None,
                        comments=comments.get(pk, []),
                        evidence=evidence.get(pk),
                        sources=point_sources.get(pk, []),
                    )
                )
            # An order that does not depend on row pks, so that the same data hashes alike in any database.
            points.sort(
                key=lambda point: (
                    point.date,
                    metric_order.get(point.metric, len(metric_order)),
                    str(point.metric),
                    sorted(str(category) for category in point.categories.values()),
                    str(point.id),
                )
            )
        return cls(
            meta=meta,
            dimension_columns=dimension_columns,
            points=points,
            data_sources=sorted(sources.values(), key=lambda source: str(source.id)),
            source_references=dataset_references,
        )

    def _columns(self, dimensions: Mapping[UUID, DimensionMeta]) -> dict[UUID, tuple[str, dict[UUID, str]]]:
        """Each declared dimension's frame column, and its categories' labels in it."""
        result: dict[UUID, tuple[str, dict[UUID, str]]] = {}
        for dimension_id in self.meta.declared_dimension_ids:
            dimension = dimensions.get(dimension_id)
            if dimension is None:
                raise DatasetFrameError(
                    f'Dataset {self.meta.identifier or self.meta.id} refers to unknown dimension {dimension_id}'
                )
            column = self.dimension_columns.get(dimension_id) or dimension.identifier
            result[dimension_id] = (
                column,
                {category.id: category.identifier or str(category.id) for category in dimension.categories},
            )
        return result

    def _labels(self, point: DataPointSnapshot, columns: Mapping[UUID, tuple[str, dict[UUID, str]]]) -> dict[str, str]:
        labels: dict[str, str] = {}
        for dimension_id, category_id in point.categories.items():
            if dimension_id not in columns:
                continue  # classified by a dimension the schema no longer declares; the frame has no column for it
            column, categories = columns[dimension_id]
            label = categories.get(category_id)
            if label is None:
                raise DatasetFrameError(
                    f'Dataset {self.meta.identifier or self.meta.id}: data point {point.id} '
                    f'refers to unknown category {category_id}'
                )
            labels[column] = label
        return labels

    def to_frame(self, dimensions: Mapping[UUID, DimensionMeta]) -> ppl.PathsDataFrame:
        """
        Build the frame the computation reads: one row per year and categories, one column per metric.

        ``dimensions`` is the catalog the frame's column names and category labels come
        from, normally the dimensions of the graph that reads the dataset. It is the same
        frame `DBDataset.deserialize_df` builds from the live rows: a metric whose grades
        another metric holds (`DatasetMetricMeta.quality_of`) reads the scores of the
        graded metric's evidence instead of data points of its own, and a metric without
        data points is a typed, empty column.
        """
        columns = self._columns(dimensions)
        dim_columns = [column for column, _ in columns.values()]
        metric_columns = {metric.id: metric.identifier or str(metric.id) for metric in self.meta.metrics}
        projections = {metric.id: metric.quality_of for metric in self.meta.metrics if metric.quality_of is not None}
        graded = set(projections.values())
        rows: dict[tuple[int, tuple[str | None, ...]], dict[str, object]] = {}
        for point in self.points:
            if point.metric in projections or point.metric not in metric_columns:
                continue
            labels = self._labels(point, columns)
            key = (point.date.year, tuple(labels.get(column) for column in dim_columns))
            row = rows.setdefault(key, {YEAR_COLUMN: point.date.year, **{column: labels.get(column) for column in dim_columns}})
            column = metric_columns[point.metric]
            if column in row and row[column] is not None and point.value is None:
                continue  # a duplicate cell; the valued one wins
            row[column] = point.value
            if point.metric in graded:
                level = point.evidence.quality_level if point.evidence is not None else None
                for projected, of in projections.items():
                    if of == point.metric and level is not None and level.score is not None:
                        row[metric_columns[projected]] = level.score
        schema: dict[str, pl.DataType | type[pl.DataType]] = {
            YEAR_COLUMN: pl.Int64,
            **dict.fromkeys(dim_columns, pl.String),
            **dict.fromkeys(metric_columns.values(), pl.Float64),
        }
        df = pl.from_dicts(list(rows.values()), schema=schema).sort([YEAR_COLUMN, *dim_columns])
        if dim_columns:
            df = df.with_columns([pl.col(column).cast(pl.Categorical) for column in dim_columns])
        meta = ppl.DataFrameMeta(
            units={metric_columns[metric.id]: unit_registry.parse_units(metric.unit) for metric in self.meta.metrics},
            primary_keys=[YEAR_COLUMN, *dim_columns],
        )
        return ppl.to_ppdf(df, meta)

    def cell_grades(self, dimensions: Mapping[UUID, DimensionMeta]) -> dict[CellLabels, QualityLevelRef]:
        """Return the grade of every graded data point, by the cell the frame from `to_frame` names it with."""
        columns = self._columns(dimensions)
        metric_columns = {metric.id: metric.identifier or str(metric.id) for metric in self.meta.metrics}
        grades: dict[CellLabels, QualityLevelRef] = {}
        for point in self.points:
            if point.evidence is None or point.evidence.quality_level is None or point.metric not in metric_columns:
                continue
            labels = tuple(sorted(self._labels(point, columns).values()))
            grades[point.date.year, metric_columns[point.metric], labels] = point.evidence.quality_level
        return grades
