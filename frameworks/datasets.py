from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pint import DimensionalityError
import polars as pl

from common import polars as ppl
from frameworks.models import MeasureDataPoint
from nodes.constants import YEAR_COLUMN
from nodes.datasets import DVCDataset

if TYPE_CHECKING:
    from nodes.context import Context


@dataclass
class FrameworkMeasureDVCDataset(DVCDataset):
    def _override_with_measure_datapoints(self, context: Context, df: ppl.PathsDataFrame):
        from nodes.models import InstanceConfig
        from django.db.models import TextField
        from django.db.models.functions import Cast

        ic = InstanceConfig.objects.filter(identifier=context.instance.id).first()
        if ic is None:
            return df
        fwc = ic.framework_configs.first()
        if fwc is None:
            return df

        uuids = df['UUID'].unique()
        measures = fwc.measures.filter(measure_template__uuid__in=uuids).select_related('template')
        dps = (
            MeasureDataPoint.objects.filter(measure__in=measures)
            .annotate(uuid=Cast('measure__measure_template__uuid', output_field=TextField()))
            .values_list('uuid', 'year', 'value', 'measure__measure_template__unit')
        )
        schema = (
            ('UUID', pl.String),
            (YEAR_COLUMN, pl.Int64),
            ('MeasureValue', pl.Float64),
            ('MeasureUnit', pl.String)
        )
        df_cols = df.columns
        meta = df.get_meta()
        dpdf = pl.DataFrame(data=list(dps), schema=schema, orient='row')
        jdf = df.join(dpdf, on=['UUID'], how='left')

        # Convert units
        diff_unit = jdf.filter(pl.col('MeasureUnit') != pl.col('Unit')).select(['MeasureUnit', 'Unit']).unique()
        conversions = []
        for m_unit_s, ds_unit_s in diff_unit.rows():
            m_unit = context.unit_registry(m_unit_s)
            ds_unit = context.unit_registry(ds_unit_s)
            cf = context.unit_registry._get_conversion_factor(m_unit._units, ds_unit._units)
            if isinstance(cf, DimensionalityError):
                raise
            conversions.append((m_unit_s, ds_unit_s, float(cf)))
        if conversions:
            conv_df = pl.DataFrame(data=conversions, schema=('MeasureUnit', 'Unit', 'ConversionFactor'), orient='row')
            jdf = jdf.join(conv_df, on=['MeasureUnit', 'Unit'], how='left')
            jdf = jdf.with_columns([pl.col('MeasureValue') * pl.col('ConversionFactor').fill_null(1.0)])

        jdf = jdf.with_columns([
            pl.coalesce(['MeasureValue', 'Value']).alias('Value'),
        ])
        df = ppl.to_ppdf(jdf.select(df_cols), meta=meta)
        return df


    def load(self, context: Context) -> ppl.PathsDataFrame:
        df = super().load(context)
        if 'UUID' not in df.columns:
            raise Exception("Dataset must have a 'UUID' column")
        # FIXME: Disable this for now in order not to break the model
        df = self._override_with_measure_datapoints(context, df)
        return df
