from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
import io
import json
import uuid
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, overload
import orjson

import pandas as pd
import polars as pl
import pint_pandas
from dvc_pandas import Dataset as DVCPandasDataset

from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.units import Unit
from common import polars as ppl

if TYPE_CHECKING:
    from .context import Context

# Use the pyarrow parquet engine because it's faster to start.
pd.set_option('io.parquet.engine', 'pyarrow')


@dataclass
class Dataset:
    id: str
    tags: list[str]
    df: Optional[ppl.PathsDataFrame] = field(init=False)
    hash: Optional[bytes] = field(init=False)

    def __post_init__(self):
        self.df = None
        self.hash = None
        if getattr(self, 'unit', None) is None:
            self.unit = None

    def load(self, context: Context) -> ppl.PathsDataFrame:
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

    def get_copy(self, context: Context) -> ppl.PathsDataFrame:
        return self.load(context).copy()

    def get_unit(self, context: Context) -> Unit:
        raise NotImplementedError()


@dataclass
class DVCDataset(Dataset):
    """Dataset that is loaded by dvc-pandas."""

    # The output can be customized further by specifying a column and filters.
    # If `input_dataset` is not specified, we default to `id` being
    # the dvc-pandas dataset identifier.
    input_dataset: Optional[str] = None
    column: Optional[str] = None
    filters: Optional[list] = None
    dropna: Optional[bool] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None

    # The year from which the time series becomes a forecast
    forecast_from: Optional[int] = None
    unit: Optional[Unit] = None

    dvc_dataset: Optional[DVCPandasDataset] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.unit is not None:
            assert isinstance(self.unit, Unit)
        self.dvc_dataset = None

    def _load_dvc_dataset(self, context: Context) -> DVCPandasDataset:
        if self.dvc_dataset is not None:
            return self.dvc_dataset
        if self.input_dataset:
            dvc_dataset_id = self.input_dataset
        else:
            dvc_dataset_id = self.id
        return context.load_dvc_dataset(dvc_dataset_id)

    def _process_output(self, df: ppl.PathsDataFrame, ds_hash: str, context: Context) -> ppl.PathsDataFrame:
        if self.max_year:
            df = df.filter(pl.col(YEAR_COLUMN) <= self.max_year)
        if self.min_year:
            df = df.filter(pl.col(YEAR_COLUMN) >= self.min_year)
        if self.dropna:
            df = df.drop_nulls()

        # If units are given as a constructor argument, ensure the dataset units match.
        if self.unit is not None:
            for col in df.columns:
                if col in [FORECAST_COLUMN, YEAR_COLUMN, *df.dim_ids]:
                    continue
                if col in df.metric_cols:
                    df = df.ensure_unit(col, self.unit)
                else:
                    df = df.set_unit(col, self.unit)

        context.cache.set(ds_hash, df)
        return df

    def load(self, context: Context) -> ppl.PathsDataFrame:
        if self.df is not None:
            return self.df

        ds_hash = self.calculate_hash(context).hex()
        obj = context.cache.get(ds_hash)
        if obj is not None and not context.skip_cache:
            self.df = obj
            return obj

        if self.input_dataset:
            ds_id = self.input_dataset
        else:
            ds_id = self.id

        self.dvc_dataset = context.load_dvc_dataset(ds_id)
        df = ppl.from_pandas(self.dvc_dataset.df.copy())

        if self.filters:
            for d in self.filters:
                if 'column' in d:
                    col = d['column']
                    val = d['value']
                    df = df.filter(pl.col(col) == val)
                    df = df.drop(col)
                elif 'dimension' in d:
                    dim_id = d['dimension']
                    dim = context.dimensions[dim_id]
                    if 'groups' in d:
                        grp_ids = d['groups']
                        grp_s = dim.ids_to_groups(dim.series_to_ids_pl(df[dim_id]))
                        df = df.filter(grp_s.is_in(grp_ids))
                    elif 'categories' in d:
                        cat_ids = d['categories']
                        df = df.filter(pl.col(dim_id).is_in(cat_ids))
                    elif 'assign_category' in d:
                        cat_id = d['assign_category']
                        assert dim_id not in df.dim_ids
                        assert cat_id in dim.cat_map
                        df = df.with_columns(pl.lit(cat_id).alias(dim_id)).add_to_index(dim_id)
                    flatten = d.get('flatten', False)
                    if flatten:
                        df = df.paths.sum_over_dims(dim_id)

        cols = df.columns

        if self.column:
            if self.column not in cols:
                available = ', '.join(cols)  # type: ignore
                raise Exception(
                    "Column '%s' not found in dataset '%s'. Available columns: %s" % (
                        self.column, self.id, available
                    )
                )
            df = df.with_columns(pl.col(self.column).alias(VALUE_COLUMN))
            cols = [YEAR_COLUMN, VALUE_COLUMN, *df.dim_ids]

        if YEAR_COLUMN in cols and YEAR_COLUMN not in df.primary_keys:
            df = df.add_to_index(YEAR_COLUMN)

        if FORECAST_COLUMN in df.columns:
            cols.append(FORECAST_COLUMN)
        elif self.forecast_from is not None:
            df = df.with_columns([
                pl.when(pl.col(YEAR_COLUMN) >= self.forecast_from)
                    .then(pl.lit(True))
                    .otherwise(pl.lit(False))
                .alias(FORECAST_COLUMN)
            ])
            cols.append(FORECAST_COLUMN)

        df = df.select(cols)
        ppl._validate_ppdf(df)
        ret = self._process_output(df, ds_hash, context)
        return ret

    def get_unit(self, context: Context) -> Unit:
        if self.unit:
            return self.unit
        df = self.load(context)
        if VALUE_COLUMN in df.columns:
            meta = df.get_meta()
            if VALUE_COLUMN not in meta.units:
                raise Exception("Dataset %s does not have a unit" % self.id)
            return meta.units[VALUE_COLUMN]
        else:
            raise Exception("Dataset %s does not have the value column" % self.id)

    def hash_data(self, context: Context) -> dict[str, Any]:
        extra_fields = [
            'input_dataset', 'column', 'filters', 'forecast_from',
            'max_year', 'min_year', 'dropna'
        ]
        d = {}
        for f in extra_fields:
            d[f] = getattr(self, f)

        d['commit_id'] = context.dataset_repo.commit_id
        d['dvc_id'] = self.input_dataset or self.id
        return d


@dataclass
class FixedDataset(Dataset):
    """Dataset from fixed values."""

    # Use `dimensionless` for `unit` if the quantities should be dimensionless.
    unit: Unit
    historical: Optional[List[Tuple[int, float]]]
    forecast: Optional[List[Tuple[int, float]]]

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
            df = pd.concat([hdf, fdf])
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

        self.df = ppl.from_pandas(df)

    def load(self, context: Context) -> ppl.PathsDataFrame:
        assert self.df is not None
        return self.df

    def hash_data(self, context: Context) -> dict[str, Any]:
        assert self.df is not None
        df = self.df.to_pandas()
        return dict(hash=int(pd.util.hash_pandas_object(df).sum()))

    def get_unit(self, context: Context) -> Unit:
        assert self.unit is not None
        return self.unit


@dataclass
class JSONDataset(Dataset):
    data: dict
    unit: Unit | None
    df: ppl.PathsDataFrame = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        df, units = JSONDataset.deserialize_df(self.data, return_units=True)
        self.df = ppl.from_pandas(df)
        if len(units) == 1:
            self.unit = list(units.values())[0]

    def load(self, context: Context) -> ppl.PathsDataFrame:
        assert self.df is not None
        return self.df

    def hash_data(self, context: Context) -> dict[str, Any]:
        df = self.df.to_pandas()
        return dict(hash=int(pd.util.hash_pandas_object(df).sum()))

    def get_unit(self, context: Context) -> Unit | None:
        return self.unit

    @overload
    @classmethod
    def deserialize_df(cls, value: dict, return_units: Literal[False] = False) -> pd.DataFrame: ...

    @overload
    @classmethod
    def deserialize_df(cls, value: dict, return_units: Literal[True]) -> Tuple[pd.DataFrame, dict[str, Unit]]: ...

    @classmethod
    def deserialize_df(cls, value: dict, return_units: bool = False):
        sio = io.StringIO(json.dumps(value))
        df = pd.read_json(sio, orient='table')
        units = {}
        for f in value['schema']['fields']:
            unit = f.get('unit')
            col = f['name']
            if unit is not None:
                pt = pint_pandas.PintType(unit)
                df[col] = df[col].astype(float).astype(pt)
                units[col] = unit
        if return_units:
            return (df, units)
        return df

    @classmethod
    def serialize_df(cls, df: pd.DataFrame, add_uuids: bool = False) -> dict:
        units = {}
        df = df.copy()
        for col in df.columns:
            if hasattr(df[col], 'pint'):
                units[col] = str(df[col].pint.units)
                df[col] = df[col].pint.m

        d = json.loads(df.to_json(orient='table'))
        fields = d['schema']['fields']
        for f in fields:
            if f['name'] in units:
                f['unit'] = units[f['name']]

        if add_uuids:
            for row in d['data']:
                uv = row.get('uuid')
                if not uv:
                    row['uuid'] = str(uuid.uuid4())
            for f in fields:
                if f['name'] == 'uuid':
                    break
            else:
                f = dict(name='uuid', type='string')
                fields.append(f)
            f['format'] = 'uuid'

        return d
