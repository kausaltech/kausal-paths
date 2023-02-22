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
