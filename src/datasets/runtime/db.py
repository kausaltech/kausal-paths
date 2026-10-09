"""Datasets stored in the database, read live or from a serialized payload."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast, override

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
    from kausal_common.datasets.models import Dataset as DBDatasetModel

    from datasets.payloads import DatasetPayloadRef, DatasetPayloadStore
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
        from datasets.snapshot import DataPointEvidenceSnapshot
        from frameworks.qualifiers import attach_evidence_qualifiers

        evidence = [
            DataPointEvidenceSnapshot.model_validate(item)
            for item in self.payload_store.get_content(self.payload_ref).get('evidence', [])
        ]
        df = attach_evidence_qualifiers(df, evidence, self.context.qualifiers, portable=True)
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
        from datasets.transfer import export_dataset_evidence
        from frameworks.qualifiers import attach_evidence_qualifiers

        evidence = export_dataset_evidence(ds_obj) if self.context.qualifiers.assessments else []
        df = attach_evidence_qualifiers(df, evidence, self.context.qualifiers)
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
            DatasetSchemaDimension,
            DimensionCategory,
        )

        # dim_cats = DimensionCategory.objects.filter(data_points=OuterRef('pk')).values(
        #     json=JSONObject(
        #         dim_id=Coalesce(F('dimension__identifier'), Cast('dimension__uuid', output_field=CharField())),
        #         cat_id=Coalesce(F('identifier'), Cast('uuid', output_field=CharField())),
        #     )
        # )

        dims = (
            DatasetSchemaDimension.objects
            .filter(schema=ds_in.schema)
            .annotate(
                dim_uuid=F('dimension__uuid'),
                dim_id=Coalesce(
                    F('column_name'),
                    F('dimension__scopes__identifier'),
                    Cast('dimension__uuid', output_field=CharField()),
                ),
            )
            .order_by('id')
            .distinct('id')
            .values_list('dim_uuid', 'dim_id')
        )
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
