import pandas as pd

from .constants import FORECAST_COLUMN


def nafill_all_forecast_years(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    df = df[~df[FORECAST_COLUMN]]
    last_hist_year = df.index.max()
    df = df.reindex(df.index.append(pd.RangeIndex(last_hist_year + 1, target_year + 1)))
    df.loc[df.index > last_hist_year, FORECAST_COLUMN] = True
    df[FORECAST_COLUMN] = df[FORECAST_COLUMN].astype(bool)
    return df
