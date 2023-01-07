from typing import Tuple
import pandas as pd
import pint_pandas

from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN


def _generate_forecast_rows(df: pd.DataFrame, end_year: int) -> Tuple[pd.DataFrame, int]:
    if isinstance(df.index, pd.MultiIndex):
        names = df.index.names
        df = df.reset_index().set_index(names)
        years = df.index.get_level_values(YEAR_COLUMN).unique().sort_values()
        last_hist_year = years[-1]
        years = years.append(pd.RangeIndex(last_hist_year + 1, end_year + 1))
        new_idx = df.index.reindex(years, level=YEAR_COLUMN)[0]
        new_idx = pd.MultiIndex.from_product(new_idx.levels)
        df = df.reindex(new_idx)
        df.loc[df.index.get_level_values(YEAR_COLUMN) > last_hist_year, FORECAST_COLUMN] = True
    else:
        df = df[~df[FORECAST_COLUMN]]
        last_hist_year = df.index.max()
        df = df.reindex(df.index.append(pd.RangeIndex(last_hist_year + 1, end_year + 1)))
        df.loc[df.index > last_hist_year, FORECAST_COLUMN] = True
    df[FORECAST_COLUMN] = df[FORECAST_COLUMN].astype(bool)
    return df, last_hist_year + 1


def nafill_all_forecast_years(df: pd.DataFrame, end_year: int) -> pd.DataFrame:
    df, _ = _generate_forecast_rows(df, end_year)
    return df


def extend_last_historical_value(df: pd.DataFrame, end_year: int) -> pd.DataFrame:
    df, first_forecast_year = _generate_forecast_rows(df, end_year)

    dtypes = df.dtypes.copy()
    if isinstance(df.index, pd.MultiIndex):
        # Year must be the last multi-index level for fillna() to work
        multi_names = list(df.index.names)
        no_year = multi_names.copy()
        no_year.remove(YEAR_COLUMN)
        df = df.unstack(no_year)  # type: ignore
        row_selector = df.index.get_level_values(YEAR_COLUMN) >= first_forecast_year - 1
    else:
        row_selector = df.index >= first_forecast_year - 1
        multi_names = None
        no_year = None

    for col, dt in list(df.dtypes.items()):
        if isinstance(col, tuple):
            if col[0] == FORECAST_COLUMN:
                continue
        else:
            if col == FORECAST_COLUMN:
                continue

        rows = df.loc[row_selector, col]  # type: ignore
        if hasattr(rows, 'pint'):
            s = rows.pint.m
        else:
            s = rows.astype('float64')

        df.loc[row_selector, col] = s.fillna(method='pad').astype(dt)  # type: ignore

    if multi_names:
        df = df.stack(no_year)  # type: ignore
        df = df.reset_index().set_index(multi_names)
        for col, dt in dtypes.items():
            assert isinstance(col, str)
            if isinstance(dt, pint_pandas.PintType):
                df[col] = df[col].astype(dt)

    return df
