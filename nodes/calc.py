import polars as pl
import pandas as pd

from common import polars as ppl

from .constants import FORECAST_COLUMN, YEAR_COLUMN


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
    if FORECAST_COLUMN not in df.columns:
        df = df.with_columns([pl.lit(False).alias(FORECAST_COLUMN)])
    if all(df[FORECAST_COLUMN]):  # Nothing to extend if there are no historical values
        return df
    df = df.paths.to_wide()
    df = df.paths.make_forecast_rows(end_year)
    last_hist_year = df.filter(pl.col(FORECAST_COLUMN).eq(False))[YEAR_COLUMN].max()
    df = df.paths.nafill_pad()
    if last_hist_year is not None:
        fc_cond = pl.col(YEAR_COLUMN) > last_hist_year
    else:
        fc_cond = pl.lit(True)  # noqa: FBT003
    fc = pl.when(fc_cond).then(True).otherwise(False)  # noqa: FBT003
    df = df.with_columns([fc.alias(FORECAST_COLUMN)])
    df = df.paths.to_narrow()
    return df

def extend_last_historical_value(df: pd.DataFrame, end_year: int) -> pd.DataFrame:
    pdf = ppl.from_pandas(df)
    pdf = extend_last_historical_value_pl(pdf, end_year)
    df = pdf.paths.to_pandas()
    return df


AR5GWP100 = {
    'co2_eq': 1.0,
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
    if dim_id not in df.primary_keys:
        raise Exception("Greenhouse gas column '%s' not in primary keys (%s)" % (dim_id, df.primary_keys))
    metric_col = df.metric_cols[0]

    gwp_items = list(AR5GWP100.items())
    gwp = pl.DataFrame(gwp_items, schema=[dim_id, 'gwp_factor'], orient='row')
    gdf = ppl.to_ppdf(gwp, meta=ppl.DataFrameMeta(units={}, primary_keys=[dim_id]))
    gdf = gdf.set_unit('gwp_factor', 'dimensionless')

    df = df.paths.join_over_index(gdf, how='left', index_from='left')
    if df['gwp_factor'].null_count():
        print(df)
        raise Exception("Some greenhouse gases failed to convert")
    df = df.multiply_cols([metric_col, 'gwp_factor'], metric_col)
    df = df.drop('gwp_factor')
    df = df.paths.sum_over_dims([dim_id])
    return df
