from __future__ import annotations

import pandas as pd
import pint
import pint_pandas
from typing import Iterable, List, Tuple
from dataclasses import dataclass
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN


# Use the pyarrow parquet engine because it's faster to start.
pd.set_option('io.parquet.engine', 'pyarrow')


@dataclass
class Dataset:
    id: str
    input_dataset: str = None
    column: str = None
    filters: Iterable = None
    groupby: dict = None

    # The year from which the time series is a forecast
    forecast_from: int = None
    fixed_data: pd.DataFrame = None

    def load(self, context):
        if self.fixed_data is not None:
            return self.fixed_data

        if self.input_dataset:
            df = context.load_dataset(self.input_dataset)
            if self.filters:
                for d in self.filters:
                    col = d['column']
                    val = d['value']
                    df = df[df[col] == val]

            if self.groupby:
                g = self.groupby
                df = df.groupby([g['index_column'], g['columns_from']])[g['value_column']].sum()
                df = df.unstack(g['columns_from'])

            return df

        df = context.load_dataset(self.id)
        cols = df.columns
        if self.column:
            if self.column not in cols:
                available = ', '.join(cols)
                raise Exception(
                    "Column '%s' not found in dataset '%s'. Available columns: %s" % (
                        self.column, self.id, available
                    )
                )
            assert self.column in cols
            if YEAR_COLUMN in cols:
                df = df.set_index(YEAR_COLUMN)
            if FORECAST_COLUMN in cols:
                df = df.rename(columns={self.column: VALUE_COLUMN})
                cols = [VALUE_COLUMN, FORECAST_COLUMN]
            elif self.forecast_from is not None:
                df = pd.DataFrame(df[self.column])
                df = df.rename(columns={self.column: VALUE_COLUMN})
                df['Forecast'] = False
                df.loc[df.index >= self.forecast_from, 'Forecast'] = True
                return df
            else:
                return df[self.column]

        return df[cols]

    @classmethod
    def fixed_multi_values_to_df(kls, data):
        series = []
        for d in data:
            vals = d['values']
            s = pd.Series(data=[x[1] for x in vals], index=[x[0] for x in vals], name=d['id'])
            series.append(s)
        df = pd.concat(series, axis=1)
        df.index.name = YEAR_COLUMN
        df = df.reset_index()
        return df

    @classmethod
    def from_fixed_values(
        kls, id: str,
        unit: pint.Unit,
        historical: List[Tuple[int, float]] = None,
        forecast: List[Tuple[int, float]] = None,
    ) -> Dataset:
        if historical:
            hdf = pd.DataFrame(historical, columns=[YEAR_COLUMN, VALUE_COLUMN])
            hdf[FORECAST_COLUMN] = False
        else:
            hdf = None

        if forecast:
            if isinstance(forecast[0], dict):
                fdf = kls.fixed_multi_values_to_df(forecast)
            else:
                fdf = pd.DataFrame(forecast, columns=[YEAR_COLUMN, VALUE_COLUMN])
            fdf[FORECAST_COLUMN] = True
        else:
            fdf = None

        if hdf is not None and fdf is not None:
            df = hdf.append(fdf)
        elif hdf is not None:
            df = hdf
        else:
            df = fdf

        df = df.set_index(YEAR_COLUMN)

        # Ensure value column has right units
        pt = pint_pandas.PintType(unit)
        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            df[col] = df[col].astype(float).astype(pt)

        ds = Dataset(id=id)
        ds.fixed_data = df

        return ds
