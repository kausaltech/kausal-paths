from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
import orjson

import pandas as pd
import pint
import pint_pandas
from dvc_pandas import Dataset as DVCPandasDataset
from pint_pandas.pint_array import PintType

from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN

if TYPE_CHECKING:
    from .context import Context

# Use the pyarrow parquet engine because it's faster to start.
pd.set_option('io.parquet.engine', 'pyarrow')


@dataclass
class Dataset:
    id: str

    df: Optional[pd.DataFrame] = field(init=False)
    hash: Optional[bytes] = field(init=False)

    def __post_init__(self):
        self.df = None
        self.hash = None
        if getattr(self, 'unit', None) is None:
            self.unit = None

    def load(self, context: Context) -> Union[pd.DataFrame, pd.Series]:
        raise NotImplementedError()

    def hash_data(self, context: Context) -> dict[str, Any]:
        raise NotImplementedError()

    def calculate_hash(self, context: Context) -> bytes:
        if self.hash is not None:
            return self.hash
        d = {'id': self.id}
        d.update(self.hash_data(context))
        h = hashlib.md5(orjson.dumps(d)).digest()
        self.hash = h
        return h

    def get_copy(self, context: Context) -> Union[pd.DataFrame, pd.Series]:
        return self.load(context).copy()

    def get_unit(self, context: Context) -> pint.Unit:
        raise NotImplementedError()


@dataclass
class DVCDataset(Dataset):
    """Dataset that is loaded by dvc-pandas."""

    # The output can be customized further by specifying a column, filters and groupby.
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
    unit: Optional[pint.Unit] = None

    dvc_dataset: Optional[DVCPandasDataset] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.unit is not None:
            assert isinstance(self.unit, pint.Unit)
        self.dvc_dataset = None

    def _load_dvc_dataset(self, context: Context) -> DVCPandasDataset:
        if self.dvc_dataset is not None:
            return self.dvc_dataset
        if self.input_dataset:
            dvc_dataset_id = self.input_dataset
        else:
            dvc_dataset_id = self.id
        return context.load_dvc_dataset(dvc_dataset_id)

    def _process_output(self, df: Union[pd.DataFrame, pd.Series], ds_hash: str, context: Context):
        df = df.copy()
        if self.max_year:
            df = df[df.index <= self.max_year]  # type: ignore
        if self.min_year:
            df = df[df.index >= self.min_year]  # type: ignore
        if self.dropna:
            df = df.dropna()

        # If units are given as a constructor argument, ensure the dataset units match.
        if self.unit is not None and isinstance(df, pd.DataFrame):
            for col in df.columns:
                if col == FORECAST_COLUMN:
                    continue
                if hasattr(df[col], 'pint'):
                    assert df[col].pint.units == self.unit
                else:
                    df[col] = df[col].astype(PintType(self.unit))
        context.cache.set(ds_hash, df)
        return df

    def load(self, context: Context) -> Union[pd.DataFrame, pd.Series]:
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
                df[FORECAST_COLUMN] = False
                df.loc[df.index >= self.forecast_from, FORECAST_COLUMN] = True
                return self._process_output(df, ds_hash, context)
            else:
                s = df[self.column]
                return self._process_output(s, ds_hash, context)

        df = df[cols]
        return self._process_output(df, ds_hash, context)

    def get_unit(self, context: Context) -> pint.Unit:
        if self.unit:
            return self.unit
        df = self.load(context)
        if VALUE_COLUMN in df.columns:
            s = df[VALUE_COLUMN]
            if hasattr(s, 'pint'):
                self.unit = s.pint.units
                return self.unit
            raise Exception("Dataset %s does not have a unit" % self.id)
        else:
            raise Exception("Dataset %s does not have the value column" % self.id)

    def hash_data(self, context: Context) -> dict[str, Any]:
        extra_fields = [
            'input_dataset', 'column', 'filters', 'groupby', 'forecast_from',
            'max_year', 'min_year', 'dropna'
        ]
        d = {}
        for f in extra_fields:
            d[f] = getattr(self, f)

        ds = self._load_dvc_dataset(context)
        d['modified_at'] = str(ds.modified_at.isoformat())
        return d


@dataclass
class FixedDataset(Dataset):
    """Dataset from fixed values."""

    # Use `dimensionless` for `unit` if the quantities should be dimensionless.
    unit: pint.Unit
    historical: List[Tuple[int, float]] | None
    forecast: List[Tuple[int, float]] | None

    def _fixed_multi_values_to_df(self, data):
        series = []
        for d in data:
            vals = d['values']
            s = pd.Series(data=[x[1] for x in vals], index=[x[0] for x in vals], name=d['id'])
            series.append(s)
        df = pd.concat(series, axis=1)
        df.index.name = YEAR_COLUMN
        df = df.reset_index()
        return df

    def __post_init__(self):
        super().__post_init__()

        if self.historical:
            hdf = pd.DataFrame(self.historical, columns=[YEAR_COLUMN, VALUE_COLUMN])
            hdf[FORECAST_COLUMN] = False
        else:
            hdf = None

        if self.forecast:
            if isinstance(self.forecast[0], dict):
                fdf = self._fixed_multi_values_to_df(self.forecast)
            else:
                fdf = pd.DataFrame(self.forecast, columns=[YEAR_COLUMN, VALUE_COLUMN])
            fdf[FORECAST_COLUMN] = True
        else:
            fdf = None

        if hdf is not None and fdf is not None:
            df = hdf.append(fdf)
        elif hdf is not None:
            df = hdf
        else:
            df = fdf

        assert df is not None
        df = df.set_index(YEAR_COLUMN)

        # Ensure value column has right units
        pt = pint_pandas.PintType(self.unit)
        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            df[col] = df[col].astype(float).astype(pt)

        self.df = df

    def load(self, context: Context) -> Union[pd.DataFrame, pd.Series]:
        assert self.df is not None
        return self.df

    def hash_data(self, context: Context) -> dict[str, Any]:
        return dict(hash=int(pd.util.hash_pandas_object(self.df).sum()))

    def get_unit(self, context: Context) -> pint.Unit:
        return self.unit
