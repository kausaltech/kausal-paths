from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import orjson

import pandas as pd
import pint
import pint_pandas
from dvc_pandas import Dataset as DVCDataset

from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN

if TYPE_CHECKING:
    from .context import Context

# Use the pyarrow parquet engine because it's faster to start.
pd.set_option('io.parquet.engine', 'pyarrow')


@dataclass
class Dataset:
    id: str

    # If this dataset comes from dvc-pandas, we can customize the output
    # further by specifying a column, filters and groupby.
    # If `input_dataset` is not specified, we default to `id` being
    # the dvc-pandas dataset identifier.
    input_dataset: Optional[str] = None
    column: Optional[str] = None
    filters: Optional[list] = None
    groupby: Optional[dict] = None
    dropna: Optional[bool] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None

    # The year from which the time series becomes a forecast
    forecast_from: Optional[int] = None
    fixed_data: Optional[pd.DataFrame] = None

    df: Optional[pd.DataFrame] = field(init=False)
    hash: Optional[bytes] = field(init=False)
    dvc_dataset: Optional[DVCDataset] = field(init=False)

    def __post_init__(self):
        self.df = None
        self.hash = None
        self.dvc_dataset = None

    def load_dvc_dataset(self, context: Context) -> DVCDataset:
        if self.dvc_dataset is not None:
            return self.dvc_dataset
        if self.input_dataset:
            dvc_dataset_id = self.input_dataset
        else:
            dvc_dataset_id = self.id
        return context.load_dvc_dataset(dvc_dataset_id)

    def handle_output(self, df: Union[pd.DataFrame, pd.Series], ds_hash: str, context: Context):
        if self.max_year:
            df = df[df.index <= self.max_year]
        if self.min_year:
            df = df[df.index >= self.min_year]
        if self.dropna:
            df = df.dropna()

        context.cache.set(ds_hash, df)
        return df

    def load(self, context: Context) -> Union[pd.DataFrame, pd.Series]:
        if self.fixed_data is not None:
            return self.fixed_data

        if self.df is not None:
            return self.df

        ds_hash = self.calculate_hash(context).hex()
        obj = context.cache.get(ds_hash)
        if obj is not None and not context.skip_cache:
            self.df = obj
            return obj

        if self.input_dataset:
            self.dvc_dataset = context.load_dvc_dataset(self.input_dataset)
            df = self.dvc_dataset.df
            if self.filters:
                for d in self.filters:
                    col = d['column']
                    val = d['value']
                    df = df[df[col] == val]

            if self.groupby:
                g = self.groupby
                df = df.groupby([g['index_column'], g['columns_from']])[g['value_column']].sum()
                df = df.unstack(g['columns_from'])
        else:
            self.dvc_dataset = context.load_dvc_dataset(self.id)
            df = self.dvc_dataset.df

        cols = df.columns
        if self.column:
            if self.column not in cols:
                available = ', '.join(cols)
                raise Exception(
                    "Column '%s' not found in dataset '%s'. Available columns: %s" % (
                        self.column, self.id, available
                    )
                )
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
                return self.handle_output(df, ds_hash, context)
            else:
                s = df[self.column]
                return self.handle_output(s, ds_hash, context)

        df = df[cols]
        return self.handle_output(df, ds_hash, context)

    def get_copy(self, context: Context) -> Union[pd.DataFrame, pd.Series]:
        return self.load(context).copy()

    def calculate_hash(self, context: Context) -> bytes:
        if self.hash is not None:
            return self.hash
        extra_fields = [
            'input_dataset', 'column', 'filters', 'groupby', 'forecast_from',
            'max_year', 'min_year', 'dropna'
        ]
        d = {'id': self.id}
        for f in extra_fields:
            d[f] = getattr(self, f)

        if self.fixed_data is not None:
            df = self.fixed_data
            hash_val = str(pd.util.hash_pandas_object(df).sum())
        else:
            ds = self.load_dvc_dataset(context)
            hash_val = str(ds.modified_at.isoformat())
        d['hash'] = hash_val
        h = hashlib.md5(orjson.dumps(d)).digest()
        self.hash = h
        return h

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
        """Use `dimensionless` for `unit` if the quantities should be dimensionless."""
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
