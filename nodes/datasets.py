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
            else:
                return df[self.column]

        return df[cols]

    @classmethod
    def from_fixed_values(
        kls, id: str,
        unit: pint.Unit,
        historical: List[Tuple[int, float]] = None,
        forecast: List[Tuple[int, float]] = None,
    ) -> Dataset:
        if historical:
            hdf = pd.DataFrame(historical, columns=[YEAR_COLUMN, VALUE_COLUMN])
            hdf['Forecast'] = False
        else:
            hdf = None

        if forecast:
            fdf = pd.DataFrame(forecast, columns=[YEAR_COLUMN, VALUE_COLUMN])
            fdf['Forecast'] = True
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
        df[VALUE_COLUMN] = df[VALUE_COLUMN].astype(float).astype(pt)

        ds = Dataset(id=id)
        ds.fixed_data = df

        return ds
