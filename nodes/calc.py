from typing import Tuple, TypeAlias
import polars as pl
import pandas as pd
import pint_pandas

from nodes.dimensions import Dimension
from nodes.node import NodeMetric
from common import polars as ppl

from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN


pl.Config.set_tbl_rows(100)


def nafill_all_forecast_years(df: pd.DataFrame, end_year: int) -> pd.DataFrame:
    pldf = ppl.from_pandas(df)
    meta = pldf.get_meta()
    pldf = (
        pldf.paths.to_wide(meta=meta)
            .paths.make_forecast_rows(end_year)
            .paths.to_narrow()
    )
    df = pldf.paths.to_pandas(meta=meta)
    return df


def extend_last_historical_value_pl(df: ppl.PathsDataFrame, end_year: int) -> ppl.PathsDataFrame:
    df = df.paths.to_wide()
    if FORECAST_COLUMN not in df.columns:
        df = df.with_columns([pl.lit(False).alias(FORECAST_COLUMN)])
    df = df.paths.make_forecast_rows(end_year)
    last_hist_year = df.filter(pl.col(FORECAST_COLUMN).eq(False))[YEAR_COLUMN].max()
    df = df.paths.nafill_pad()
    fc = pl.when(pl.col(YEAR_COLUMN) > last_hist_year).then(True).otherwise(False)
    df = df.with_columns([fc.alias(FORECAST_COLUMN)])
    df = df.paths.to_narrow()
    return df

def extend_last_historical_value(df: pd.DataFrame, end_year: int) -> pd.DataFrame:
    pdf = ppl.from_pandas(df)
    pdf = extend_last_historical_value_pl(pdf, end_year)
    df = pdf.paths.to_pandas()
    return df


AR5GWP100 = {
    'co2': 1.0,
    'ch4': 28.0,
    'n2o': 265.0,
    'hfc': 1300.0,
    'pfc': 6630.0,
    'sf6': 23500.0,
    'nf3': 16100.0,
    'co2_biogen': 0.0,
}


def convert_to_co2e(df: ppl.PathsDataFrame, dim_id: str) -> ppl.PathsDataFrame:
    if len(df.metric_cols) > 1:
        raise Exception("Only one metric column supported")
    metric_col = df.metric_cols[0]

    gwp_items = list(AR5GWP100.items())
    gwp = pl.DataFrame(gwp_items, schema=[dim_id, 'gwp_factor'])
    gdf = ppl.to_ppdf(gwp, meta=ppl.DataFrameMeta(units={}, primary_keys=[dim_id]))
    gdf = gdf.set_unit('gwp_factor', 'dimensionless')

    df = df.paths.join_over_index(gdf, how='left', index_from='left')
    if df['gwp_factor'].null_count():
        raise Exception("Some greenhouse gases failed to convert")
    df = df.multiply_cols([metric_col, 'gwp_factor'], metric_col)
    df = df.drop(columns='gwp_factor')
    df = df.paths.sum_over_dims([dim_id])
    return df


