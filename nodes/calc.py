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
    pldf = (
        pldf.paths.to_wide(dimensions={}, metrics={})
            .paths.make_forecast_rows(end_year)
            .paths.to_narrow()
    )

    return df


def extend_last_historical_value(
    df: pd.DataFrame, end_year: int,
    dimensions: dict[str, Dimension] | None = None,
    metrics: dict[str, NodeMetric] | None = None
) -> pd.DataFrame:
    pdf = ppl.from_pandas(df)
    meta = pdf.get_meta()
    pdf = (
        pdf.paths.to_wide(dimensions=dimensions or {}, metrics=metrics or {}, meta=meta)
            .paths.make_forecast_rows(end_year)
    )
    last_hist_year = pdf.filter(pl.col(FORECAST_COLUMN) == False)[YEAR_COLUMN].max()
    pdf = pdf.paths.nafill_pad()
    fc = pl.when(pl.col(YEAR_COLUMN) > last_hist_year).then(True).otherwise(False)
    pdf = ppl.to_ppdf(pdf.with_column(fc.alias(FORECAST_COLUMN)))
    pdf = pdf.paths.to_narrow()
    df = pdf.paths.to_pandas(meta=meta)
    return df
