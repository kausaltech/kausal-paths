from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, override

import polars as pl
from pint import DimensionalityError

from common import polars as ppl
from frameworks.models import MeasureDataPoint
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.datasets import DVCDataset, GenericDataset

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset as DBDatasetModel

    from nodes.context import Context, FrameworkConfigData
    from nodes.defs.node_defs import InputDatasetDef

ENABLE_UNIT_CONVERSION = True


MEASURE_DATAPOINT_SCHEMA: dict[str, type[pl.DataType]] = {
    'uuid': pl.String,
    'MeasureYear': pl.Int64,
    'MeasureValue': pl.Float64,
    'MeasureDefaultValue': pl.Float64,
    'MeasureUnit': pl.String,
}


def query_all_measure_datapoints(fwd: FrameworkConfigData | None) -> pl.DataFrame:
    """
    Query the DB for every MeasureDataPoint under the given framework config.

    This is the single query behind `Context.measure_datapoints`; per-dataset
    lookups slice its result in memory via `collect_measure_datapoints` instead
    of issuing their own query (avoids an N+1 across a model run).

    Returns a plain DataFrame with columns: uuid, MeasureYear, MeasureValue,
    MeasureDefaultValue, MeasureUnit. `uuid` is in DB (hyphenated) canonical
    form. No unit conversion is applied. Returns an empty DataFrame with the
    same schema if fwd is None.
    """
    from django.db.models import TextField
    from django.db.models.functions import Cast

    from frameworks.models import Measure

    if fwd is None:
        return pl.DataFrame(schema=MEASURE_DATAPOINT_SCHEMA)

    measures = Measure.objects.filter(framework_config=fwd.id)
    dps = (
        MeasureDataPoint.objects
        .filter(measure__in=measures)
        .annotate(uuid=Cast('measure__measure_template__uuid', output_field=TextField()))
        .values_list('uuid', 'year', 'value', 'default_value', 'measure__measure_template__unit')
    )
    return pl.DataFrame(data=list(dps), schema=MEASURE_DATAPOINT_SCHEMA, orient='row')


def collect_measure_datapoints(
    context: Context,
    uuids: list[str],
) -> pl.DataFrame:
    """
    Return MeasureDataPoints for the given UUIDs, sliced from the context cache.

    Filters `context.measure_datapoints` (fetched with a single DB query per
    context) instead of querying the DB per call. `uuids` must be in DB
    (hyphenated) canonical form.

    Returns a plain DataFrame with columns: uuid, MeasureYear, MeasureValue,
    MeasureDefaultValue, MeasureUnit. No unit conversion is applied — callers
    that need values in a specific unit should convert themselves.
    Returns an empty DataFrame with the same schema if uuids is empty.
    """
    if not uuids:
        return pl.DataFrame(schema=MEASURE_DATAPOINT_SCHEMA)
    return context.measure_datapoints.filter(pl.col('uuid').is_in(uuids))


@dataclass
class FrameworkMeasureDVCDataset(DVCDataset):
    measure_data_point_years: list[int] = field(default_factory=list)

    def hash_data(self) -> dict[str, Any]:
        data = super().hash_data()
        if self.context.framework_config_data:
            data['framework_config_updated'] = str(self.context.framework_config_data.last_modified_at)
        return data

    def _override_with_measure_datapoints(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        context = self.context
        fwd = context.framework_config_data
        ref_year = context.instance.reference_year
        if fwd is None:
            # FIXME This is needed because DatasetNode and other nodes have a different sequence of operations.
            if 'prepare_gpc_dataset' in self.tags:
                drop_cols = [col for col in ['UUID', 'Sector'] if col in df.columns]
                df = df.drop(drop_cols)
                df = df.set_unit(VALUE_COLUMN, df['Unit'].unique()[0]).drop('Unit')
            # Legacy DVC datasets encode the reference year as Year=0, which
            # DatasetWithFilters._filter_and_process_df converts to ref_year-1 (e.g. 2017).
            # Remap to ref_year so get_correct_baseline (measure_data_baseline_year_only) keeps
            # the row instead of filtering it out.
            target_year = context.instance.target_year
            df = df.with_columns(
                pl
                .when(pl.col(YEAR_COLUMN) == ref_year - 1)
                .then(pl.lit(ref_year))
                .when(pl.col(YEAR_COLUMN) == target_year - 1)
                .then(pl.lit(target_year))
                .otherwise(pl.col(YEAR_COLUMN))
                .alias(YEAR_COLUMN),
            )
            # Add flag columns so _observed_only_extend_all falls through to
            # model_default=True and extends values to the full time range.
            df = df.with_columns([
                pl.lit(value=False).alias('ObservedDataPoint'),
                pl.lit(value=False).alias('FromMeasureDataPoint'),
            ])
            return df

        df = df.with_columns(
            pl.when(pl.col('UUID') == 'ADD UUID HERE').then(pl.lit(None)).otherwise(pl.col('UUID')).alias('UUID'),
        )

        uuids = df['UUID'].unique().to_list()
        dpdf = collect_measure_datapoints(context, uuids).rename({'uuid': 'UUID'})

        meta = df.get_meta()
        df_cols = df.columns

        # Check if there are multiple years for the same UUID. If so, we
        # know it's not just the baseline year.
        unique_years_by_sector = df.group_by('Sector').agg(pl.col('Year').unique().len().alias('NrSectorYears'))

        jdf = df.join(dpdf, on='UUID', how='outer')
        jdf = jdf.join(unique_years_by_sector, on='Sector', how='left')

        # Convert units
        diff_unit = jdf.filter(pl.col('MeasureUnit') != pl.col('Unit')).select(['MeasureUnit', 'Unit']).unique()
        conversions: list[tuple[str, str, float]] = []
        for m_unit_s, ds_unit_s in diff_unit.rows():
            m_unit = context.unit_registry.parse_units(m_unit_s)
            ds_unit = context.unit_registry.parse_units(ds_unit_s)
            cf = context.unit_registry._get_conversion_factor(m_unit._units, ds_unit._units)
            if isinstance(cf, DimensionalityError):
                raise cf
            assert not isinstance(cf, complex)
            conversions.append((m_unit_s, ds_unit_s, float(cf)))
        if conversions and ENABLE_UNIT_CONVERSION:
            conv_df = pl.DataFrame(data=conversions, schema=('MeasureUnit', 'Unit', 'ConversionFactor'), orient='row')
            jdf = jdf.join(conv_df, on=['MeasureUnit', 'Unit'], how='left')
            jdf = jdf.with_columns(
                [
                    pl.col('MeasureDefaultValue') * pl.col('ConversionFactor').fill_null(1.0),
                    pl.col('MeasureValue') * pl.col('ConversionFactor').fill_null(1.0),
                    pl.col('Unit').alias('MeasureUnit'),
                ],
            )

        target_year = context.instance.target_year
        jdf = jdf.with_columns([
            pl
            .when(pl.col('NrSectorYears') == 1)
            .then(
                pl.coalesce(['MeasureYear', YEAR_COLUMN]),
            )
            .otherwise(pl.col(YEAR_COLUMN))
            .alias(YEAR_COLUMN),
            pl.coalesce(['MeasureValue', 'MeasureDefaultValue', 'Value']).alias('Value'),
            pl.coalesce(['MeasureUnit', 'Unit']).alias('Unit'),
            (pl.col('MeasureValue').is_not_null() | pl.col('MeasureDefaultValue').is_not_null()).alias('FromMeasureDataPoint'),
            ((pl.col('MeasureValue').is_not_null()) & (pl.col('UUID').is_not_null())).alias('ObservedDataPoint'),
        ])
        # Legacy DVC datasets encode reference/target years as Year=0/100. After
        # DatasetWithFilters._filter_and_process_df these become ref_year-1 and target_year-1.
        # If MeasureYear from the DB also carries these off-by-one values, remap them so that
        # get_correct_baseline (measure_data_baseline_year_only) keeps the rows.
        jdf = jdf.with_columns(
            pl
            .when(pl.col(YEAR_COLUMN) == ref_year - 1)
            .then(pl.lit(ref_year))
            .when(pl.col(YEAR_COLUMN) == target_year - 1)
            .then(pl.lit(target_year))
            .otherwise(pl.col(YEAR_COLUMN))
            .alias(YEAR_COLUMN),
        )
        out_cols = [*df_cols, 'FromMeasureDataPoint', 'ObservedDataPoint']
        df = ppl.to_ppdf(jdf.select(out_cols), meta=meta)
        # FIXME This is needed because DatasetNode and other nodes have a different sequence of operations.
        if 'prepare_gpc_dataset' in self.tags:
            drop_cols = [col for col in ['UUID', 'Sector'] if col in df.columns]
            df = df.drop(drop_cols)
            df = df.set_unit(VALUE_COLUMN, df['Unit'].unique()[0]).drop('Unit')

        return df

    def post_process(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = super().post_process(df)
        if 'UUID' not in df.columns:
            raise Exception("Dataset must have a 'UUID' column")
        df = self._override_with_measure_datapoints(df)
        return df

    # def load(self, context: Context) -> ppl.PathsDataFrame:
    #    df = super().load(context)
    #    df = self._override_with_measure_datapoints(context, df)
    #    return df


@dataclass
class FrameworkMeasureDVCDataset2(DVCDataset):
    measure_data_point_years: list[int] = field(default_factory=list)
    db_dataset_obj: DBDatasetModel | None = field(default=None)

    @classmethod
    def from_def(  # type: ignore[override]
        cls,
        ds_def: InputDatasetDef,
        context: Context,
        *,
        db_dataset_obj: DBDatasetModel | None = None,
    ) -> Self:
        obj = super().from_def(ds_def, context)
        obj.db_dataset_obj = db_dataset_obj
        return obj

    @override
    def load_internal(self) -> ppl.PathsDataFrame:
        if self.db_dataset_obj is None:
            return super().load_internal()
        cached = self.cache_get()
        if cached is not None:
            return cached
        from nodes.datasets import DBDataset

        df = DBDataset.deserialize_df(self.db_dataset_obj)
        df = self._filter_and_process_df(df)
        df = self.post_process(df)
        if self.cache_key:
            self.cache_set(df)
        return df

    @property  # Override parent @cached_property: key must vary per scenario
    def cache_key(self) -> str:
        base = self.get_cache_key()
        use_obs = bool(self.context.get_parameter_value('use_observations', required=False) or False)
        return f'{base}:use_obs={use_obs}'

    def hash_data(self) -> dict[str, Any]:
        data = super().hash_data()
        if self.context.framework_config_data:
            data['framework_config_updated'] = str(self.context.framework_config_data.last_modified_at)
        return data

    def _override_with_measure_datapoints(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        context = self.context
        fwd = context.framework_config_data
        ref_year = context.instance.reference_year
        df = df.paths._drop_unnecessary_levels(df, context)

        if fwd is None or 'uuid' not in df.columns:
            drop_cols = [col for col in ['uuid', 'Sector'] if col in df.columns]
            if drop_cols:
                df = df.drop(drop_cols)
            df = df.with_columns([
                pl.lit(value=False).alias('ObservedDataPoint'),
                pl.lit(value=False).alias('FromMeasureDataPoint'),
                (pl.col(YEAR_COLUMN) > ref_year).alias(FORECAST_COLUMN),
            ])
            return df

        uuids = df['uuid'].unique().to_list()
        dpdf = collect_measure_datapoints(context, uuids)

        # Convert measure values to the dataset unit. After a UUID-keyed join there is
        # usually exactly one MeasureUnit, so a single ensure_unit() call is sufficient.
        ds_unit = df.get_unit(VALUE_COLUMN)
        mu_strs = dpdf['MeasureUnit'].drop_nulls().unique().to_list()
        if len(mu_strs) > 1:  # FIXME This is needed for SCurveAction that has several units in one metric.
            if all(s in ['%', 'dimensionless'] for s in mu_strs):
                dpdf = dpdf.with_columns([
                    pl
                    .when(pl.col('MeasureUnit') == '%')
                    .then(pl.col('MeasureValue') * 0.01)
                    .otherwise(pl.col('MeasureValue'))
                    .alias('MeasureValue'),
                    pl
                    .when(pl.col('MeasureUnit') == '%')
                    .then(pl.col('MeasureDefaultValue') * 0.01)
                    .otherwise(pl.col('MeasureDefaultValue'))
                    .alias('MeasureDefaultValue'),
                    pl.lit('dimensionless').alias('MeasureUnit'),
                ])
                mu_strs = ['dimensionless']
            else:
                raise ValueError(self, 'Measure dataset tries to use more than one unit at a time.')

        if mu_strs:
            mu = context.unit_registry.parse_units(mu_strs[0])
            dpdf = (
                ppl
                .to_ppdf(
                    dpdf,
                    meta=ppl.DataFrameMeta(units={'MeasureValue': mu, 'MeasureDefaultValue': mu}, primary_keys=['uuid']),
                )
                .ensure_unit('MeasureValue', ds_unit)
                .ensure_unit('MeasureDefaultValue', ds_unit)
            )

        # NaN is not null in Polars, so coalesce would pick up NaN as a "real" value and
        # propagate it through interpolation. Treat NaN measure values as null (missing data)
        # so the coalesce falls through to the DVC baseline instead.
        dpdf = dpdf.with_columns([
            pl.col('MeasureValue').fill_nan(None),
            pl.col('MeasureDefaultValue').fill_nan(None),
        ])

        # Drop datapoints where both value and default_value are null — they provide no
        # actual data. Without this filter, they still participate in the year-override join
        # and cause single-row DVC baselines to appear at wrong years with the DVC fallback
        # value (rather than not appearing at all).
        dpdf = dpdf.filter(pl.col('MeasureValue').is_not_null() | pl.col('MeasureDefaultValue').is_not_null())

        meta = df.get_meta()
        df_cols = df.columns

        baseline_year = context.instance.reference_year

        # If a uuid appears at only one year in the DVC data, MeasureYear (from the DB
        # entry) overrides the DVC year — this handles the "single baseline row per uuid"
        # pattern. UUIDs that span multiple years keep their DVC years unchanged.
        unique_years_by_uuid = df.group_by('uuid').agg(pl.col('Year').unique().len().alias('NrUUIDYears'))

        # When not in observation/progress-tracking mode, restrict DB data points to the
        # reference year only. Without this, a single-year DVC baseline row (NrUUIDYears==1)
        # expands into one row per observed year via the left join, pushing max_hist_year
        # forward and masking future goal targets for cities with recent observed data.
        use_obs = bool(context.get_parameter_value('use_observations', required=False) or False)
        if not use_obs and not dpdf.is_empty():
            dpdf = dpdf.filter(pl.col('MeasureYear') == baseline_year)

        # Left join: only keep rows present in the DVC data (dpdf rows without a DVC
        # counterpart would introduce phantom rows with null Year/dims).
        jdf = df.join(dpdf, on='uuid', how='left')
        jdf = jdf.join(unique_years_by_uuid, on='uuid', how='left')

        jdf = jdf.with_columns([
            # Only override Year when the uuid has a single year in the DVC data AND
            # that year is the baseline year — the "single reference-row" pattern where
            # MeasureYear meaningfully shifts when the value applies. UUIDs that already
            # carry a specific calendar year (≠ baseline) are left unchanged.
            pl
            .when((pl.col('NrUUIDYears') == 1) & (pl.col(YEAR_COLUMN) == baseline_year))
            .then(pl.coalesce(['MeasureYear', YEAR_COLUMN]))
            .otherwise(pl.col(YEAR_COLUMN))
            .alias(YEAR_COLUMN),
            pl.coalesce(['MeasureValue', 'MeasureDefaultValue', 'Value']).alias('Value'),
            (pl.col('MeasureValue').is_not_null() | pl.col('MeasureDefaultValue').is_not_null()).alias('FromMeasureDataPoint'),
            ((pl.col('MeasureValue').is_not_null()) & (pl.col('uuid').is_not_null())).alias('ObservedDataPoint'),
        ])
        max_hist = context.instance.maximum_historical_year or ref_year
        has_db_data = pl.col('ObservedDataPoint') | pl.col('FromMeasureDataPoint')
        obs_expr = (has_db_data & (pl.col(YEAR_COLUMN) <= max_hist)) | (pl.col(YEAR_COLUMN) == ref_year)
        jdf = jdf.with_columns(~obs_expr.alias(FORECAST_COLUMN))

        out_cols = [c for c in [*df_cols, 'FromMeasureDataPoint', 'ObservedDataPoint'] if c != 'uuid']
        out_cols = out_cols if FORECAST_COLUMN in out_cols else [*out_cols, FORECAST_COLUMN]
        new_pks = [pk for pk in meta.primary_keys if pk != 'uuid']
        df = ppl.to_ppdf(
            jdf.select(out_cols),
            meta=ppl.DataFrameMeta(primary_keys=new_pks, units=meta.units),
        )
        return df

    def get_observation_years(self) -> list[int]:
        """
        Return all years with MeasureDataPoints, regardless of use_observations.

        Reads UUIDs from the raw DVC file (before post-processing) and queries
        the DB directly, so the result is independent of the use_observations
        parameter and of whether the processed df is in cache.
        """
        cached: list[int] | None = getattr(self, '_cached_observation_years', None)
        if cached is not None:
            return cached
        fwd = self.context.framework_config_data
        if fwd is None:
            return []
        ds_id = self.input_dataset or self.id
        dvc_ds = self.context.load_dvc_dataset(ds_id)
        if dvc_ds.df is None or 'uuid' not in dvc_ds.df.columns:
            return []
        uuids = dvc_ds.df['uuid'].cast(pl.String).str.replace_all('_', '-').drop_nulls().unique().to_list()
        if not uuids:
            return []
        dpdf = collect_measure_datapoints(self.context, uuids)
        # Exclude rows where both value and default_value are null — those have no
        # data to contribute and should not be counted as observation years.
        dpdf = dpdf.filter(pl.col('MeasureValue').is_not_null() | pl.col('MeasureDefaultValue').is_not_null())
        result = dpdf['MeasureYear'].drop_nulls().unique().sort().to_list()
        self._cached_observation_years: list[int] = result
        return result

    def post_process(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = super().post_process(df)
        if 'uuid' in df.columns:
            # df = df.rename({'uuid': 'UUID'})
            df = df.with_columns(pl.col('uuid').cast(pl.String).str.replace_all('_', '-'))

        if 'uuid' not in df.columns:
            raise Exception("Dataset must have a 'UUID' column")
        df = self._override_with_measure_datapoints(df)
        return df


@dataclass
class ObservationDataset(DVCDataset):
    """
    DVCDataset that overlays user observations from MeasureDataPoints.

    UUID must be present as a dimension in the loaded DVC dataset. After loading,
    queries DB for MeasureDataPoints matching those UUIDs and adds two boolean
    columns to the result:

    - ``observed``: True where the user entered a MeasureDataPoint value.
    - ``placeholder``: True where only a default_value (comparable-city average) exists.

    Value is set to: coalesce(user value, default value, DVC value).

    The UUID dimension is kept in the output so that ``ObservableNode``'s
    ``apply_observations`` operation can use it before dropping it.
    """

    def hash_data(self) -> dict[str, Any]:
        data = super().hash_data()
        if self.context.framework_config_data:
            data['framework_config_updated'] = str(self.context.framework_config_data.last_modified_at)
        return data

    def _overlay_observations(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:  # noqa: C901,PLR0912,PLR0915
        """Query DB for observations and overlay onto DVC data, adding observed/placeholder columns."""
        context = self.context
        fwd = context.framework_config_data

        # --- 1. Filter to rows where uuid and Value are both non-null ----------------
        if 'uuid' not in df.columns:
            df = df.with_columns([pl.lit(value=False).alias('observed'), pl.lit(value=False).alias('placeholder')])
            return df

        # Rows with null uuid have DVC values but no DB linkage (e.g. reference-year
        # baseline rows whose uuid only exists for the target year).  Keep them as
        # passthrough — observed=False, value unchanged — and re-attach at every exit.
        _passthrough = df.filter(pl.col('uuid').is_null() & pl.col(VALUE_COLUMN).is_not_null()).with_columns([
            pl.lit(value=False).alias('observed'),
            pl.lit(value=False).alias('placeholder'),
        ])
        df = df.filter(pl.col('uuid').is_not_null() & pl.col(VALUE_COLUMN).is_not_null())

        # Drop all-null dimension columns (e.g. pollutant/cost_type that don't apply
        # to this metric).
        meta = df.get_meta()
        all_null_dims = [d for d in df.dim_ids if d != 'uuid' and df[d].null_count() == len(df)]
        if all_null_dims:
            new_pks = [pk for pk in meta.primary_keys if pk not in all_null_dims]
            df = ppl.to_ppdf(
                df.drop(all_null_dims),
                meta=ppl.DataFrameMeta(
                    primary_keys=new_pks,
                    units=meta.units,
                ),
            )
            meta = df.get_meta()

        # --- 2. Add default False flags (will be overwritten where DB data exists) ---
        df = df.with_columns([
            pl.lit(value=False).alias('observed'),
            pl.lit(value=False).alias('placeholder'),
        ])

        if fwd is None:
            return self._reattach_passthrough(df, _passthrough)

        # --- 3. Look up MeasureDataPoints by UUID -------------------------------------
        # DVC stores UUIDs with underscores; DB uses hyphens.
        dvc_uuids = df['uuid'].unique().drop_nulls().to_list()
        db_uuids = [u.replace('_', '-') for u in dvc_uuids]

        dpdf = collect_measure_datapoints(context, db_uuids)
        if dpdf.is_empty():
            return self._reattach_passthrough(df, _passthrough)

        # Build obs DataFrame (convert hyphen UUIDs back to underscore format)
        obs_raw: pl.DataFrame = dpdf.select(
            pl.col('uuid').str.replace_all('-', '_', literal=True),
            pl.col('MeasureYear').alias(YEAR_COLUMN),
            pl.col('MeasureValue').alias('_obs_value'),
            pl.col('MeasureDefaultValue').alias('_obs_default'),
            pl.col('MeasureUnit').alias('_obs_unit'),
        )
        # DVC files may store uuid as categorical; cast obs_raw to match for join compatibility.
        uuid_dtype = df.schema['uuid']
        if obs_raw.schema['uuid'] != uuid_dtype:
            obs_raw = obs_raw.with_columns(pl.col('uuid').cast(uuid_dtype))

        # --- 4. Unit conversion -------------------------------------------------------
        ds_unit_str = str(df.get_unit(VALUE_COLUMN))
        unique_units = obs_raw['_obs_unit'].drop_nulls().unique().to_list()
        conversions: list[tuple[str, float]] = []
        for m_unit_s in unique_units:
            if m_unit_s == ds_unit_str:
                continue  # no conversion needed
            try:
                m_unit = context.unit_registry.parse_units(m_unit_s)
                ds_unit_obj = context.unit_registry.parse_units(ds_unit_str)
                cf = context.unit_registry._get_conversion_factor(m_unit._units, ds_unit_obj._units)
                if isinstance(cf, DimensionalityError):
                    raise cf  # noqa: TRY301
                if isinstance(cf, complex):
                    raise TypeError('Unexpected complex conversion factor')  # noqa: TRY301
                conversions.append((m_unit_s, float(cf)))
            except Exception:  # noqa: S110
                pass  # leave unconverted; mismatch will surface elsewhere
        if conversions:
            conv_df = pl.DataFrame(conversions, schema=['_obs_unit', '_conv_factor'], orient='row')
            obs_raw = obs_raw.join(conv_df, on='_obs_unit', how='left')
            obs_raw = obs_raw.with_columns([
                (pl.col('_obs_value') * pl.col('_conv_factor').fill_null(1.0)).alias('_obs_value'),
                (pl.col('_obs_default') * pl.col('_conv_factor').fill_null(1.0)).alias('_obs_default'),
            ]).drop('_conv_factor')
        obs_raw = obs_raw.drop('_obs_unit')

        ref_year = context.instance.reference_year

        # --- 5. Add pre-reference observation rows (years not in DVC data) -----------
        pre_ref_obs = obs_raw.filter(pl.col(YEAR_COLUMN) < ref_year)
        if len(pre_ref_obs) > 0:
            # Use ref_year rows as a dimension template (drop Year, Value, flags)
            ref_rows = df.filter(pl.col(YEAR_COLUMN) == ref_year)
            if len(ref_rows) > 0:
                template = ref_rows.select([
                    c
                    for c in ref_rows.columns
                    if c
                    not in [
                        YEAR_COLUMN,
                        VALUE_COLUMN,
                        'observed',
                        'placeholder',
                        FORECAST_COLUMN,
                    ]
                ])
                # Cross-join template with pre-ref obs on uuid
                extra = template.join(
                    pre_ref_obs.rename({YEAR_COLUMN: '_pre_year'}),
                    on='uuid',
                    how='inner',
                )
                extra_cols = [
                    pl.col('_pre_year').alias(YEAR_COLUMN),
                    pl.coalesce(['_obs_value', '_obs_default', pl.lit(None, dtype=pl.Float64)]).alias(VALUE_COLUMN),
                    pl.col('_obs_value').is_not_null().alias('observed'),
                    (pl.col('_obs_value').is_null() & pl.col('_obs_default').is_not_null()).alias('placeholder'),
                    # Mark FORECAST=True so DatasetReduceAction._get_metric_data('historical')
                    # excludes these rows when computing max_hist_year, keeping the DVC
                    # reference year as the action baseline.
                    pl.lit(value=True).alias(FORECAST_COLUMN),
                ]
                extra = extra.with_columns(extra_cols).drop(['_pre_year', '_obs_value', '_obs_default'])
                if FORECAST_COLUMN not in df.columns:
                    df = ppl.to_ppdf(df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN)), meta=meta)
                df = ppl.to_ppdf(
                    pl.concat([df.select(extra.columns), extra]),
                    meta=meta,
                )

        # --- 6. Overlay in-range observations (years already in df) ------------------
        in_range_obs = obs_raw.filter(pl.col(YEAR_COLUMN) >= ref_year)
        if len(in_range_obs) > 0:
            dvc_years = set(df[YEAR_COLUMN].to_list())
            obs_years = set(in_range_obs[YEAR_COLUMN].to_list())
            years_overlap = dvc_years & obs_years

            if years_overlap:
                # Normal case: DVC and DB share year values, join on both uuid + year.
                joined = ppl.to_ppdf(
                    df.join(
                        in_range_obs.select(['uuid', YEAR_COLUMN, '_obs_value', '_obs_default']),
                        on=['uuid', YEAR_COLUMN],
                        how='left',
                    ),
                    meta=meta,
                )
            else:
                # Lookup-table case: DVC has a single placeholder year (originally Year=0,
                # transformed to reference_year) that doesn't appear in DB observations.
                # Fall back to UUID-only join using the latest available observation.
                latest_obs = (
                    obs_raw
                    .sort(YEAR_COLUMN, descending=True)
                    .group_by('uuid')
                    .first()
                    .select(['uuid', '_obs_value', '_obs_default'])
                )
                uuid_dtype = df.schema['uuid']
                if latest_obs.schema['uuid'] != uuid_dtype:
                    latest_obs = latest_obs.with_columns(pl.col('uuid').cast(uuid_dtype))
                joined = ppl.to_ppdf(
                    df.join(latest_obs, on='uuid', how='left'),
                    meta=meta,
                )

            df = ppl.to_ppdf(
                joined.with_columns([
                    pl.coalesce(['_obs_value', '_obs_default', VALUE_COLUMN]).alias(VALUE_COLUMN),
                    pl.col('_obs_value').is_not_null().alias('observed'),
                    (pl.col('_obs_value').is_null() & pl.col('_obs_default').is_not_null()).alias('placeholder'),
                ]).drop(['_obs_value', '_obs_default']),
                meta=meta,
            )

            # Observation years not present in the DVC data only make sense to add when
            # the normal (overlap) path was taken. In the lookup-table / no-overlap case
            # the latest observation already overwrites every DVC row via the UUID-only
            # join above, so adding extra rows would duplicate data and corrupt datasets
            # that legitimately have future-only DVC years (e.g. action scenario targets).
            extra_obs_years = obs_years - dvc_years if years_overlap else set()
            if extra_obs_years:
                extra_obs = in_range_obs.filter(pl.col(YEAR_COLUMN).is_in(sorted(extra_obs_years)))
                ref_rows = df.filter(pl.col(YEAR_COLUMN) == ref_year)
                if len(ref_rows) > 0:
                    template = ref_rows.select([
                        c
                        for c in ref_rows.columns
                        if c not in [YEAR_COLUMN, VALUE_COLUMN, 'observed', 'placeholder', FORECAST_COLUMN]
                    ])
                    extra = template.join(
                        extra_obs.rename({YEAR_COLUMN: '_extra_year'}),
                        on='uuid',
                        how='inner',
                    )
                    extra_cols = [
                        pl.col('_extra_year').alias(YEAR_COLUMN),
                        pl.coalesce(['_obs_value', '_obs_default', pl.lit(None, dtype=pl.Float64)]).alias(VALUE_COLUMN),
                        pl.col('_obs_value').is_not_null().alias('observed'),
                        (pl.col('_obs_value').is_null() & pl.col('_obs_default').is_not_null()).alias('placeholder'),
                        # Mark FORECAST=True so DatasetReduceAction._get_metric_data('historical')
                        # excludes these rows when computing max_hist_year, keeping the DVC
                        # reference year as the action baseline.
                        pl.lit(value=True).alias(FORECAST_COLUMN),
                    ]
                    extra = extra.with_columns(extra_cols).drop(['_extra_year', '_obs_value', '_obs_default'])
                    if FORECAST_COLUMN not in df.columns:
                        df = ppl.to_ppdf(df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN)), meta=meta)
                    df = ppl.to_ppdf(
                        pl.concat([df.select(extra.columns), extra]),
                        meta=meta,
                    )

        return self._reattach_passthrough(df, _passthrough)

    @staticmethod
    def _reattach_passthrough(df: ppl.PathsDataFrame, passthrough: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        """Re-attach null-uuid passthrough rows that were split off in _overlay_observations step 1."""
        if passthrough.is_empty():
            return df
        if df.is_empty():
            # No uuid-linked rows survived; return passthrough with its original schema intact.
            # (When df is empty, all_null_dims cleanup would have dropped all dimension columns,
            # making df.columns an unreliable guide for schema alignment.)
            return passthrough
        # Align passthrough columns to df (all_null_dims may have been dropped from df).
        shared_cols = [c for c in passthrough.columns if c in df.columns]
        pt = passthrough.select(shared_cols)
        meta = df.get_meta()
        return ppl.to_ppdf(
            pl.concat([df, pt], how='diagonal_relaxed').sort(YEAR_COLUMN),
            meta=meta,
        )

    def post_process(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = super().post_process(df)
        if 'uuid' not in df.columns:
            return df
        df = self._overlay_observations(df)
        return df


@dataclass
class CityDataset(GenericDataset, ObservationDataset):
    """
    GenericDataset that overlays city-specific DB values (via MeasureDataPoints) before return.

    Used with the ``city_data`` tag on datasets consumed by plain GenericNode and action
    nodes that need city-specific DB values without being ObservableNodes. The consuming
    node receives clean data with city-specific values already in the Value column.

    Inherits from GenericDataset (for proper metric-column setup via _transform_data /
    _index_data) and ObservationDataset (for _overlay_observations). The overlay is
    injected between _filter_and_process_df and _transform_data so it runs before the
    PathsDataFrame metric columns are finalised.
    """

    def load_internal(self) -> ppl.PathsDataFrame:
        cached_df = self.cache_get()
        if cached_df is not None:
            return cached_df

        ds_id = self.input_dataset or self.id
        dvc_ds = self.context.load_dvc_dataset(ds_id)
        assert dvc_ds.df is not None
        df = self._convert_dvc_dataset(dvc_ds)
        df = self._filter_and_process_df(df)

        # Apply city-specific DB overlay when uuid column is present.
        # If the overlay empties the df (e.g. all uuid values were null so no DB lookup
        # was possible), fall back to the original DVC data.
        if 'uuid' in df.columns:
            df_before = df
            df = self._overlay_observations(df)
            if df.is_empty() and not df_before.is_empty():
                df = df_before
            df = df.drop([c for c in ['uuid', 'observed', 'placeholder'] if c in df.columns])

        df = self._transform_data(df)
        if FORECAST_COLUMN not in df.columns:
            df = df.with_columns(pl.lit(False).alias(FORECAST_COLUMN))  # noqa: FBT003
        self.interpolate = True
        if self.interpolate:
            df = self._linear_interpolate(df)
        df = self._index_data(df)
        if self.context.sample_size > 0:
            df = self._sample(df)
        self.cache_set(df)
        return df
