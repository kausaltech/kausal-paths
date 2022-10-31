from typing import Tuple
import pandas as pd

from .constants import FORECAST_COLUMN, VALUE_COLUMN


def _generate_forecast_rows(df: pd.DataFrame, target_year: int) -> Tuple[pd.DataFrame, int]:
    df = df[~df[FORECAST_COLUMN]]
    last_hist_year = df.index.max()
    df = df.reindex(df.index.append(pd.RangeIndex(last_hist_year + 1, target_year + 1)))
    df.loc[df.index > last_hist_year, FORECAST_COLUMN] = True
    df[FORECAST_COLUMN] = df[FORECAST_COLUMN].astype(bool)
    return df, last_hist_year + 1


def nafill_all_forecast_years(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    df, _ = _generate_forecast_rows(df, target_year)
    return df


def extend_last_historical_value(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    df, first_forecast_year = _generate_forecast_rows(df, target_year)
    dt = df.dtypes[VALUE_COLUMN]
    row_selector = df.index >= first_forecast_year - 1
    df.loc[row_selector, VALUE_COLUMN] = df.loc[row_selector, VALUE_COLUMN]\
        .astype('float64').fillna(method='pad').astype(dt)  # type: ignore
    return df
