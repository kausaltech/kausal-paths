from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl
from pint import DimensionalityError

from common import polars as ppl
from frameworks.models import MeasureDataPoint
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.datasets import DVCDataset

ENABLE_UNIT_CONVERSION = True


@dataclass
class FrameworkMeasureDVCDataset(DVCDataset):
    measure_data_point_years: list[int] = field(default_factory=list)

    def hash_data(self) -> dict[str, Any]:
        data = super().hash_data()
        if self.context.framework_config_data:
            data['framework_config_updated'] = str(self.context.framework_config_data.last_modified_at)
        return data

    def _override_with_measure_datapoints(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        from django.db.models import TextField
        from django.db.models.functions import Cast

        from frameworks.models import Measure

        context = self.context
        fwd = context.framework_config_data
        if fwd is None:
            # FIXME This is needed because DatasetNode and other nodes have a different sequence of operations.
            if 'prepare_gpc_dataset' in self.tags:
                drop_cols = [col for col in ['UUID', 'Sector'] if col in df.columns]
                df = df.drop(drop_cols)
                df = df.set_unit(VALUE_COLUMN, df['Unit'].unique()[0]).drop('Unit')
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
        measures = Measure.objects.filter(framework_config=fwd.id).filter(measure_template__uuid__in=uuids)
        dps = (
            MeasureDataPoint.objects
            .filter(measure__in=measures)
            .annotate(uuid=Cast('measure__measure_template__uuid', output_field=TextField()))
            .values_list('uuid', 'year', 'value', 'default_value', 'measure__measure_template__unit')
        )
        schema = (
            ('UUID', pl.String),
            ('MeasureYear', pl.Int64),
            ('MeasureValue', pl.Float64),
            ('MeasureDefaultValue', pl.Float64),
            ('MeasureUnit', pl.String),
        )
        dpdf = ppl.PathsDataFrame(data=list(dps), schema=schema, orient='row')

        meta = df.get_meta()
        df_cols = df.columns

        baseline_year = context.instance.reference_year
        df = df.with_columns(  # FIXME Does this not already happen in load()? So this is redundant.
            pl.when(pl.col('Year').lt(100)).then(pl.col('Year') + baseline_year).otherwise(pl.col('Year')).alias('Year'),
        )
        # Duplicates may occur when baseline year overlaps with existing data points.
        df = ppl.to_ppdf(df.unique(subset=meta.primary_keys, keep='last', maintain_order=True), meta=meta)

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
class ObservationDataset(DVCDataset):
    """
    DVCDataset that overlays user observations from MeasureDataPoints.

    UUID must be present as a dimension in the loaded DVC dataset (use
    ``drop_col: false`` in the YAML filter so uuid is retained). After loading,
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
        from django.db.models import TextField
        from django.db.models.functions import Cast

        from frameworks.models import Measure

        context = self.context
        fwd = context.framework_config_data

        # --- 1. Filter to rows where uuid and Value are both non-null ----------------
        if 'uuid' not in df.columns:
            df = df.with_columns([pl.lit(value=False).alias('observed'), pl.lit(value=False).alias('placeholder')])
            return df
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
            return df

        # --- 3. Query DB for MeasureDataPoints by UUID --------------------------------
        # DVC stores UUIDs with underscores; DB uses hyphens.
        dvc_uuids = df['uuid'].unique().drop_nulls().to_list()
        db_uuids = [u.replace('_', '-') for u in dvc_uuids]

        measures = Measure.objects.filter(
            framework_config=fwd.id,
            measure_template__uuid__in=db_uuids,
        )
        raw_dps = list(
            MeasureDataPoint.objects
            .filter(measure__in=measures)
            .annotate(uuid_str=Cast('measure__measure_template__uuid', output_field=TextField()))
            .values_list('uuid_str', 'year', 'value', 'default_value', 'measure__measure_template__unit')
        )
        if not raw_dps:
            return df

        # Build obs DataFrame (convert hyphen UUIDs back to underscore format)
        obs_raw: pl.DataFrame = pl.DataFrame(
            {
                'uuid': [r[0].replace('-', '_') for r in raw_dps],
                YEAR_COLUMN: [r[1] for r in raw_dps],
                '_obs_value': [r[2] for r in raw_dps],
                '_obs_default': [r[3] for r in raw_dps],
                '_obs_unit': [r[4] for r in raw_dps],
            },
        )

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
                extra = extra.with_columns([
                    pl.col('_pre_year').alias(YEAR_COLUMN),
                    pl.coalesce(['_obs_value', pl.lit(None, dtype=pl.Float64)]).alias(VALUE_COLUMN),
                    pl.col('_obs_value').is_not_null().alias('observed'),
                    (pl.col('_obs_value').is_null() & pl.col('_obs_default').is_not_null()).alias('placeholder'),
                    pl.lit(value=False).alias(FORECAST_COLUMN),
                ]).drop(['_pre_year', '_obs_value', '_obs_default'])
                df = ppl.to_ppdf(
                    pl.concat([df.select(extra.columns), extra]),
                    meta=meta,
                )

        # --- 6. Overlay in-range observations (years already in df) ------------------
        in_range_obs = obs_raw.filter(pl.col(YEAR_COLUMN) >= ref_year)
        if len(in_range_obs) > 0:
            joined = ppl.to_ppdf(
                df.join(
                    in_range_obs.select(['uuid', YEAR_COLUMN, '_obs_value', '_obs_default']),
                    on=['uuid', YEAR_COLUMN],
                    how='left',
                ),
                meta=meta,
            )
            df = ppl.to_ppdf(
                joined.with_columns([
                    pl.coalesce(['_obs_value', VALUE_COLUMN]).alias(VALUE_COLUMN),
                    pl.col('_obs_value').is_not_null().alias('observed'),
                    (pl.col('_obs_value').is_null() & pl.col('_obs_default').is_not_null()).alias('placeholder'),
                ]).drop(['_obs_value', '_obs_default']),
                meta=meta,
            )

        return df

    def post_process(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = super().post_process(df)
        if 'uuid' not in df.columns:
            return df
        df = self._overlay_observations(df)
        return df
