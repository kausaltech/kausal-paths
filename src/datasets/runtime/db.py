"""Datasets stored in the database, read live or from a serialized payload."""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Context as DecimalContext, Decimal
from typing import TYPE_CHECKING, Any, NamedTuple, Self, cast, override

from django.contrib.postgres.expressions import ArraySubquery
from django.db.models import F, OuterRef
from django.db.models.fields import CharField
from django.db.models.functions import Cast, Coalesce, JSONObject

import polars as pl

from kausal_common.logging.errors import capture_error

from common import polars as ppl
from datasets.runtime.base import DatasetWithFilters
from nodes.constants import (
    YEAR_COLUMN,
)
from nodes.units import Unit, unit_registry

if TYPE_CHECKING:
    from django.db import models

    from kausal_common.datasets.models import (
        DataPoint,
        Dataset as DBDatasetModel,
        DatasetMetric,
        Dimension,
        DimensionCategory,
    )

    from datasets.payloads import DatasetPayloadRef, DatasetPayloadStore
    from datasets.snapshot import CellLabels, QualityLevelRef
    from nodes.context import Context
    from nodes.defs.node_defs import InputDatasetDef


@dataclass
class SerializedDBDataset(DatasetWithFilters):
    """DB dataset loaded from a current or immutable serialized payload."""

    payload_ref: DatasetPayloadRef | None = None
    payload_store: DatasetPayloadStore | None = None

    @classmethod
    def from_def(
        cls,
        ds_def: InputDatasetDef,
        context: Context,
        *,
        payload_ref: DatasetPayloadRef,
        payload_store: DatasetPayloadStore,
    ) -> Self:
        kwargs = super().kwargs_from_def(ds_def)
        cls.apply_forecast_defaults(kwargs, context, payload_ref.forecast_from)
        return cls(
            id=ds_def.id,
            context=context,
            **kwargs,
            payload_ref=payload_ref,
            payload_store=payload_store,
        )

    @override
    def load_internal(self) -> ppl.PathsDataFrame:
        if self.df is not None:
            return self.df
        assert self.payload_ref is not None
        assert self.payload_store is not None
        df = self.payload_store.get_dataframe(self.payload_ref).copy()
        from frameworks.qualifiers import attach_evidence_qualifiers

        grades = self.payload_store.get_cell_grades(self.payload_ref)
        df = attach_evidence_qualifiers(df, grades, self.context.qualifiers, portable=True)
        df = self._filter_and_process_df(df)
        df = self.after_transformations(df)
        self.df = df
        return df

    def hash_data(self) -> dict[str, Any]:
        assert self.payload_ref is not None
        return {
            'source': {
                'dataset_uuid': self.payload_ref.dataset_uuid,
                'generation': self.payload_ref.generation,
                'content_hash': self.payload_ref.content_hash,
            },
            'pipeline': self.pipeline_hash_data(),
        }

    def get_unit(self) -> Unit:
        df = self.load_internal()
        meta = df.get_meta()
        if len(meta.units) == 1:
            return next(iter(meta.units.values()))
        raise Exception('Dataset %s does not have a single unit' % self.id)


@dataclass
class DBDataset(DatasetWithFilters):
    """Dataset that is loaded from the admin UI's Dataset model."""

    db_dataset_id: str | None = None
    db_dataset_obj: DBDatasetModel | None = None
    unit: Unit | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.db_dataset_obj is None:
            from kausal_common.datasets.models import Dataset as DBDatasetModel

            assert self.db_dataset_id is not None
            self.db_dataset_obj = DBDatasetModel.objects.get(uuid=self.db_dataset_id)

    @classmethod
    def from_def(cls, ds_def: InputDatasetDef, context: Context, db_dataset_obj: DBDatasetModel) -> Self:
        kwargs = super().kwargs_from_def(ds_def)
        # A DB dataset can declare its own forecast year, which bindings inherit when they
        # don't override it (see `_promote_dataset_forecast_defaults`). Forecast synthesis is
        # an operation, so the default has to enter the pipeline — setting the field alone
        # would do nothing.
        cls.apply_forecast_defaults(kwargs, context, (db_dataset_obj.spec or {}).get('forecast_from'))
        return cls(
            id=ds_def.id,
            context=context,
            **kwargs,
            db_dataset_obj=db_dataset_obj,
        )

    @override
    def load_internal(self) -> ppl.PathsDataFrame:
        if self.df is not None:
            return self.df

        ds_obj = self.db_dataset_obj
        if ds_obj is None:
            raise Exception('Admin dataset not loaded')
        df = self.context.db_dataset_dfs.get(ds_obj.pk)
        if df is None:
            df = self.deserialize_df(ds_obj)
            self.context.db_dataset_dfs[ds_obj.pk] = df
        df = df.copy()
        from frameworks.qualifiers import attach_evidence_qualifiers

        grades = self.cell_grades(ds_obj) if self.context.qualifiers.assessments else {}
        df = attach_evidence_qualifiers(df, grades, self.context.qualifiers)
        df = self._filter_and_process_df(df)
        df = self.after_transformations(df)
        self.df = df
        return df

    def hash_data(self) -> dict[str, Any]:
        obj = self.db_dataset_obj
        assert obj is not None
        return {
            'source': {'obj_pk': obj.pk, 'updated_at': str(obj.last_modified_at)},
            'pipeline': self.pipeline_hash_data(),
        }

    def get_unit(self) -> Unit:
        df = self.load_internal()
        meta = df.get_meta()
        if len(meta.units) == 1:
            return next(iter(meta.units.values()))
        raise Exception('Dataset %s does not have a single unit' % self.id)

    @classmethod
    def deserialize_df(cls, ds_in: DBDatasetModel, *, include_data_point_primary_keys: bool = False) -> ppl.PathsDataFrame:
        from kausal_common.datasets.models import (
            DataPoint,
            Dataset as DBDatasetModel,
            DatasetMetric,
            DimensionCategory,
        )

        # dim_cats = DimensionCategory.objects.filter(data_points=OuterRef('pk')).values(
        #     json=JSONObject(
        #         dim_id=Coalesce(F('dimension__identifier'), Cast('dimension__uuid', output_field=CharField())),
        #         cat_id=Coalesce(F('identifier'), Cast('uuid', output_field=CharField())),
        #     )
        # )

        dims = [(dimension.uuid, column) for dimension, column in cls.dimension_columns(ds_in)]
        dim_anns = {
            str(dim[1]): DimensionCategory.objects
            .filter(dimension__uuid=dim[0])
            .filter(data_points=OuterRef('pk'))
            .annotate(cat_id=Coalesce(F('identifier'), Cast('uuid', output_field=CharField())))
            .values_list('cat_id', flat=True)
            for dim in dims
        }

        dps = (
            DataPoint.objects
            .filter(dataset=OuterRef('pk'))
            .order_by()
            .distinct('id')
            .values(
                json=JSONObject(
                    id=F('id'),
                    **{YEAR_COLUMN: F('date__year')},
                    value=F('value'),
                    metric=F('metric__uuid'),
                    # dim_cats=ArraySubquery(dim_cats),
                    **dim_anns,
                ),
            )
        )

        # This names the metric columns, so anything building a *selector* for them
        # has to resolve the name the same way, or it asks for a column that is not
        # there: use `metric_column_id()` / `metric_column_id_expr()` from
        # `datasets/snapshot.py`.
        from datasets.snapshot import metric_column_id_expr

        metrics = DatasetMetric.objects.filter(schema=OuterRef('schema')).values(
            json=JSONObject(
                uuid=F('uuid'),
                name=metric_column_id_expr(),
                unit=F('unit'),
            )
        )
        ds = DBDatasetModel.objects.filter(id=ds_in.pk).annotate(dps=ArraySubquery(dps), metrics=ArraySubquery(metrics)).first()
        assert ds is not None
        dp_list = cast('list[dict[str, Any]]', ds.dps)  # type: ignore
        df_schema = {
            'id': pl.Int64,
            YEAR_COLUMN: pl.Int64,
            'value': pl.Float64,
            'metric': pl.Utf8,
            **{str(dim[1]): pl.Utf8 for dim in dims},
        }
        df = pl.DataFrame(dp_list, schema=df_schema, orient='row')
        mdf = pl.DataFrame(ds.metrics)  # type: ignore
        df = df.join(mdf.select(pl.col('uuid').alias('metric'), pl.col('name').alias('metric_name')), on='metric', how='left')

        from frameworks.evidence import project_quality_columns

        df = project_quality_columns(ds_in, df)

        dim_ids = [str(dim[1]) for dim in dims]

        id_map = None
        if include_data_point_primary_keys:
            id_map = df.select([YEAR_COLUMN, *dim_ids, 'metric_name', 'id'])

        index_cols = [YEAR_COLUMN, *dim_ids]
        uniq_cols = [*index_cols, 'metric']
        df = df.with_columns(pl.col('metric_name').alias('metric')).drop('metric_name', 'id').sort(uniq_cols)

        dupes = df.group_by(uniq_cols).agg(pl.count().alias('_count')).filter(pl.col('_count') > 1)
        if len(dupes) > 0:
            extra = dupes.head().to_dicts()
            capture_error(
                'Dataset %s (pk %d) has %s duplicate rows' % (ds_in.identifier, ds_in.pk, len(dupes)),
                extras={'example_rows': extra},
            )
            # Filter out duplicate rows, keeping the first one
            df = df.group_by(uniq_cols).first()

        df = df.pivot(on='metric', index=[YEAR_COLUMN, *dim_ids], values='value')
        # A metric without data points gets no pivot column, but it is still part of the
        # schema: a dataset nobody has filled in yet reads as typed, empty columns.
        missing = [m['name'] for m in ds.metrics if m['name'] not in df.columns]  # type: ignore
        if missing:
            df = df.with_columns([pl.lit(None, dtype=pl.Float64).alias(name) for name in missing])

        if include_data_point_primary_keys and id_map is not None:
            id_pivoted = id_map.pivot(on='metric_name', on_columns=[YEAR_COLUMN, *dim_ids], values='id')
            id_pivoted = id_pivoted.rename({
                col: f'_dp_pk_{col}' for col in id_pivoted.columns if col not in [YEAR_COLUMN, *dim_ids]
            })
            df = df.join(id_pivoted, on=[YEAR_COLUMN, *dim_ids], how='left', nulls_equal=True)

        if dim_ids:
            df = df.with_columns([pl.col(dim_id).cast(pl.Categorical) for dim_id in dim_ids])

        meta = ppl.DataFrameMeta(
            units={
                m['name']: unit_registry.parse_units(m['unit'])
                for m in ds.metrics  # type: ignore
            },
            primary_keys=[YEAR_COLUMN, *dim_ids],
        )

        pdf = ppl.to_ppdf(df, meta)

        return pdf

    @classmethod
    def cell_grades(cls, ds: DBDatasetModel) -> dict[CellLabels, QualityLevelRef]:
        """
        Return the grade of every graded data point, by the cell `deserialize_df` names it with.

        The live twin of `DatasetSnapshot.cell_grades`: a cell is its year, its metric's
        column and its categories' labels in the dimensions `dimension_columns` names.
        """
        from datasets.snapshot import DataPointEvidenceSnapshot, metric_column_id
        from frameworks.models import DataPointEvidence

        declared = {dimension.pk for dimension, _ in cls.dimension_columns(ds)}
        evidence = (
            DataPointEvidence.objects
            .filter(data_point__dataset=ds, quality_level__isnull=False)
            .select_related('data_point__metric', 'quality_level__scheme')
            .prefetch_related('data_point__dimension_categories')
        )
        grades: dict[CellLabels, QualityLevelRef] = {}
        for item in evidence:
            point = item.data_point
            labels = tuple(
                sorted(
                    category.identifier or str(category.uuid)
                    for category in point.dimension_categories.all()
                    if category.dimension_id in declared
                )
            )
            level = DataPointEvidenceSnapshot.from_model(item).quality_level
            assert level is not None
            grades[point.date.year, metric_column_id(point.metric), labels] = level
        return grades

    @staticmethod
    def dimension_columns(ds: DBDatasetModel) -> list[tuple[Dimension, str]]:
        """
        Name the frame column of each of the dataset's dimensions, in schema order.

        The column is the schema dimension's ``column_name``, else the identifier the
        dimension is scoped under, else its uuid. A dimension scoped to several instances
        under different identifiers takes the one of the dataset's own instance.

        `deserialize_df` names its columns with this and `upsert_df` resolves them with
        it, so a frame read from a dataset can be written back to it.
        """
        from kausal_common.datasets.models import DatasetSchemaDimension

        schema_dimensions = (
            DatasetSchemaDimension.objects
            .filter(schema=ds.schema)
            .select_related('dimension')
            .prefetch_related('dimension__scopes')
            .order_by('id')
        )
        result: list[tuple[Dimension, str]] = []
        for schema_dimension in schema_dimensions:
            dimension = schema_dimension.dimension
            column = schema_dimension.column_name
            if not column:
                scopes = sorted((s for s in dimension.scopes.all() if s.identifier), key=lambda s: s.pk)
                if len({s.identifier for s in scopes}) > 1:
                    instance = ds.scope_instance
                    own = [s for s in scopes if s.scope_id == instance.pk and isinstance(s.scope, type(instance))]
                    scopes = own or scopes
                column = next((s.identifier for s in scopes if s.identifier), None) or str(dimension.uuid)
            result.append((dimension, column))
        return result

    @classmethod
    def upsert_df(cls, ds: DBDatasetModel, df: ppl.PathsDataFrame) -> DataPointUpsert:  # noqa: C901, PLR0912, PLR0915
        """
        Make the dataset's data points match ``df``, keeping the identity of every cell that survives.

        A cell is a (metric, year, categories) coordinate, with the frame's columns resolved
        by the rule `deserialize_df` names them with. A cell present on both sides keeps its
        data point, and with it its uuid, comments, evidence and source references; only its
        value is written, and only when it changed. A cell only in ``df`` gets a new data
        point. A data point whose cell is not in ``df`` is deleted, along with whatever
        hangs off it: afterwards the dataset holds exactly what ``df`` does.

        A valueless cell is a null data point, not a missing one. It reads as "no data"
        (`DataAvailabilityNode` tests `is_not_null()`) while still existing as a cell a city
        can see, comment on and fill in, with its categories and provenance. Dropping it
        forced template datasets to ship zeros, and a pre-filled 0 cannot be told from a
        confirmed one -- which is what BISKO's Pruefschritt 1.4 checks.

        The frame must carry every dimension column of the dataset's schema and only metric
        columns of its metrics. Other columns (a ``Source`` or ``Comment`` column, say) are
        not read.

        Rows may repeat a cell. A table that is wide by metric has to split a row to give
        two metrics of one (year, categories) different provenance, and each half then
        holds the other metric's cell empty. So a repeated cell takes its value from the
        row that has one, else from the first row that addresses it; two different values
        for one cell are an error. Only the row a cell was taken from is in
        `DataPointUpsert.points`, so provenance can be attached from that row alone. A
        caller wanting a particular row to win among empty ones puts it first.
        """
        from django.utils import timezone

        from kausal_common.datasets.models import DataPoint, DataPointDimensionCategory, DimensionCategory

        from datasets.snapshot import metric_column_id

        assert ds.schema is not None
        meta = df.get_meta()
        columns = cls.dimension_columns(ds)
        known_columns = {column for _, column in columns}
        unknown_columns = sorted(set(meta.dim_ids) - known_columns)
        if unknown_columns:
            raise ValueError(f'Dataset {ds.identifier or ds.uuid} has no dimension for column(s) {", ".join(unknown_columns)}')
        metrics = {metric_column_id(metric): metric for metric in ds.schema.metrics.all()}
        unknown_metrics = sorted(set(meta.metric_cols) - metrics.keys())
        if unknown_metrics:
            raise ValueError(f'Dataset {ds.identifier or ds.uuid} has no metric for column(s) {", ".join(unknown_metrics)}')

        categories_by_column = {
            column: {category.identifier or str(category.uuid): category for category in dimension.categories.all()}
            for dimension, column in columns
            if column in meta.dim_ids
        }
        value_field = cast('models.DecimalField', DataPoint._meta.get_field('value'))
        assert value_field.max_digits is not None
        assert value_field.decimal_places is not None
        db_context = DecimalContext(prec=value_field.max_digits)
        db_quantum = Decimal(1).scaleb(-value_field.decimal_places)

        def db_value(value: float | None) -> Decimal | None:
            # The value as the database stores it (Django's `format_number`), so an unchanged
            # cell compares equal to what was read back.
            if value is None or math.isnan(value):  # NaN is how a frame can spell an empty cell
                return None
            return db_context.create_decimal_from_float(value).quantize(db_quantum, context=db_context)

        point_categories: defaultdict[int, set[int]] = defaultdict(set)
        for point_pk, category_pk in DataPointDimensionCategory.objects.filter(data_point__dataset=ds).values_list(
            'data_point_id', 'dimension_category_id'
        ):
            point_categories[point_pk].add(category_pk)
        existing: dict[CellKey, DataPoint] = {}
        surplus: list[int] = []
        for point in DataPoint.objects.filter(dataset=ds).order_by('id'):
            key = (point.metric_id, point.date.year, frozenset(point_categories[point.pk]))
            if key in existing:
                surplus.append(point.pk)  # a duplicate cell; `deserialize_df` reads only the first
            else:
                existing[key] = point

        incoming: dict[CellKey, IncomingCell] = {}
        for index, row in enumerate(df.iter_rows(named=True)):
            year = int(row[YEAR_COLUMN])
            row_categories: list[DimensionCategory] = []
            for column, categories in categories_by_column.items():
                label = row[column]
                if label is None or label == '':
                    continue
                category = categories.get(str(label))
                if category is None:
                    raise DimensionCategory.DoesNotExist(
                        f"Dimension category '{label}' not found for column '{column}' of dataset {ds.identifier or ds.uuid}"
                    )
                row_categories.append(category)
            category_pks = frozenset(category.pk for category in row_categories)
            for column in meta.metric_cols:
                metric = metrics[column]
                key = (metric.pk, year, category_pks)
                cell = IncomingCell(index, column, db_value(row[column]), metric, date(year, 1, 1), row_categories)
                current = incoming.get(key)
                if current is None or (current.value is None and cell.value is not None):
                    incoming[key] = cell
                elif cell.value is not None and cell.value != current.value:
                    raise ValueError(
                        f'Dataset {ds.identifier or ds.uuid}: rows {current.row} and {index} give the cell '
                        f'{column} {year} {row_categories} two values ({current.value} and {cell.value})'
                    )

        result = DataPointUpsert()
        new_points: list[DataPoint] = []
        new_point_categories: list[list[DimensionCategory]] = []
        changed: list[DataPoint] = []
        now = timezone.now()
        for key, cell in incoming.items():
            point = existing.get(key)
            if point is None:
                point = DataPoint(dataset=ds, date=cell.date, metric=cell.metric, value=cell.value)
                new_points.append(point)
                new_point_categories.append(cell.categories)
            elif point.value != cell.value:
                point.value = cell.value
                point.last_modified_at = now
                changed.append(point)
            else:
                result.unchanged += 1
            result.points[cell.row, cell.column] = point

        stale = [point.pk for key, point in existing.items() if key not in incoming] + surplus
        if stale:
            DataPoint.objects.filter(pk__in=stale).delete()
            result.deleted = len(stale)
        if changed:
            DataPoint.objects.bulk_update(changed, ['value', 'last_modified_at'], batch_size=UPSERT_BATCH_SIZE)
            result.updated = len(changed)
        if new_points:
            DataPoint.objects.bulk_create(new_points, batch_size=UPSERT_BATCH_SIZE)
            DataPointDimensionCategory.objects.bulk_create(
                [
                    DataPointDimensionCategory(data_point=point, dimension_category=category)
                    for point, categories in zip(new_points, new_point_categories, strict=True)
                    for category in categories
                ],
                batch_size=UPSERT_BATCH_SIZE,
            )
            result.created = len(new_points)
        return result


UPSERT_BATCH_SIZE = 2000

type CellKey = tuple[int, int, frozenset[int]]
"""A data point's cell: metric pk, year and dimension category pks."""


class IncomingCell(NamedTuple):
    """One cell of a frame given to `DBDataset.upsert_df`, and the row it was taken from."""

    row: int
    column: str
    value: Decimal | None
    metric: DatasetMetric
    date: date
    categories: list[DimensionCategory]


@dataclass
class DataPointUpsert:
    """What `DBDataset.upsert_df` did, and the data point each incoming cell landed on."""

    points: dict[tuple[int, str], DataPoint] = field(default_factory=dict)
    """The data point of each cell, by (frame row index, metric column)."""
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
