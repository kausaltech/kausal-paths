from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl
from pint import DimensionalityError

from common import polars as ppl
from frameworks.models import MeasureDataPoint
from nodes.constants import YEAR_COLUMN
from nodes.context import Context
from nodes.datasets import DVCDataset

if TYPE_CHECKING:
    from nodes.context import Context


ENABLE_UNIT_CONVERSION = True


@dataclass
class FrameworkMeasureDVCDataset(DVCDataset):
    def hash_data(self, context: Context) -> dict[str, Any]:
        data = super().hash_data(context)
        if context.framework_config_data:
            data['framework_config_updated'] = str(context.framework_config_data.last_modified_at)
        return data

    def _override_with_measure_datapoints(self, context: Context, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        from django.db.models import TextField
        from django.db.models.functions import Cast

        from frameworks.models import Measure

        fwd = context.framework_config_data
        if fwd is None:
            return df

        df = df.with_columns(
            pl.when(pl.col('UUID') == 'ADD UUID HERE').then(pl.lit(None)).otherwise(pl.col('UUID')).alias('UUID'),
        )

        uuids = df['UUID'].unique().to_list()
        measures = (
            Measure.objects.filter(framework_config=fwd.id).filter(measure_template__uuid__in=uuids).select_related('template')
        )
        dps = (
            MeasureDataPoint.objects.filter(measure__in=measures).exclude(value__isnull=True)
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
        dpdf = pl.DataFrame(data=list(dps), schema=schema, orient='row')

        meta = df.get_meta()
        df_cols = df.columns
        df_cols.remove('UUID')

        baseline_year = context.instance.reference_year
        df = df.with_columns(
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
        conversions = []
        for m_unit_s, ds_unit_s in diff_unit.rows():
            m_unit = context.unit_registry(m_unit_s)
            ds_unit = context.unit_registry(ds_unit_s)
            cf = context.unit_registry._get_conversion_factor(m_unit._units, ds_unit._units)
            if isinstance(cf, DimensionalityError):
                raise cf
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

        jdf = jdf.with_columns(
            [
                pl.when(pl.col('NrSectorYears') == 1).then(
                    pl.coalesce(['MeasureYear', YEAR_COLUMN]),
                ).otherwise(pl.col(YEAR_COLUMN)).alias(YEAR_COLUMN),
                pl.coalesce(['MeasureValue', 'MeasureDefaultValue', 'Value']).alias('Value'),
                pl.coalesce(['MeasureUnit', 'Unit']).alias('Unit'),
            ],
        )

        df = ppl.to_ppdf(jdf.select(df_cols), meta=meta)
        return df

    def post_process(self, context: Context | None, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = super().post_process(context, df)
        assert context is not None
        if 'UUID' not in df.columns:
            raise Exception("Dataset must have a 'UUID' column")
        df = self._override_with_measure_datapoints(context, df)
        return df

    # def load(self, context: Context) -> ppl.PathsDataFrame:
    #    df = super().load(context)
    #    df = self._override_with_measure_datapoints(context, df)
    #    return df
