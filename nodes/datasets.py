from __future__ import annotations

import hashlib
import io
import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import orjson
import pandas as pd
import pint_pandas
import polars as pl

from common import polars as ppl
from nodes.units import Unit

from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN

if TYPE_CHECKING:
    from .context import Context

# Use the pyarrow parquet engine because it's faster to start.
pd.set_option('io.parquet.engine', 'pyarrow')


@dataclass
class Dataset:
    id: str
    tags: list[str]
    interpolate: bool = field(init=False)
    df: ppl.PathsDataFrame | None = field(init=False)
    hash: bytes | None = field(init=False)

    def __post_init__(self):
        self.df = None
        self.hash = None
        self.interpolate = False
        if getattr(self, 'unit', None) is None:
            self.unit = None

    def load(self, context: Context) -> ppl.PathsDataFrame:
        raise NotImplementedError()

    def hash_data(self, context: Context) -> dict[str, Any]:
        raise NotImplementedError()

    def calculate_hash(self, context: Context) -> bytes:
        if self.hash is not None:
            return self.hash
        d = {'id': self.id, 'interpolate': self.interpolate}
        d.update(self.hash_data(context))
        h = hashlib.md5(orjson.dumps(d, option=orjson.OPT_SORT_KEYS), usedforsecurity=False).digest()
        self.hash = h
        return h

    def _linear_interpolate(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        years = df[YEAR_COLUMN].unique().sort()
        min_year = years.min()
        assert isinstance(min_year, int)
        max_year = years.max()
        assert isinstance(max_year, int)
        df = df.paths.to_wide()
        years_df = pl.DataFrame(data=range(min_year, max_year + 1), schema=[YEAR_COLUMN])
        meta = df.get_meta()
        zdf = years_df.join(df, on=YEAR_COLUMN, how='left').sort(YEAR_COLUMN)
        df = ppl.to_ppdf(zdf, meta=meta)
        cols = [pl.col(col).interpolate() for col in df.metric_cols]
        if FORECAST_COLUMN in df.columns:
            cols.append(pl.col(FORECAST_COLUMN).fill_null(strategy='forward'))
        df = df.with_columns(cols)
        df = df.paths.to_narrow()
        return df

    def post_process(self, df: ppl.PathsDataFrame):
        if self.interpolate:
            df = self._linear_interpolate(df)
        return df

    def get_copy(self, context: Context) -> ppl.PathsDataFrame:
        df = self.load(context)
        return df.copy()

    def get_unit(self, context: Context) -> Unit:
        raise NotImplementedError()


@dataclass
class DVCDataset(Dataset):
    """Dataset that is loaded by dvc-pandas."""

    # The output can be customized further by specifying a column and filters.
    # If `input_dataset` is not specified, we default to `id` being
    # the dvc-pandas dataset identifier.
    input_dataset: str | None = None
    column: str | None = None
    filters: list | None = None
    dropna: bool | None = None
    min_year: int | None = None
    max_year: int | None = None

    # The year from which the time series becomes a forecast
    forecast_from: int | None = None
    unit: Unit | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.unit is not None:
            assert isinstance(self.unit, Unit)

    def _process_output(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
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

        return df

    def load(self, context: Context) -> ppl.PathsDataFrame:
        obj = None
        cache_key: str | None
        if not context.skip_cache:
            ds_hash = self.calculate_hash(context).hex()
            cache_key = 'ds:%s:%s' % (self.id, ds_hash)
            res = context.cache.get(cache_key)
            if res.is_hit:
                obj = res.obj
        else:
            cache_key = None

        if obj is not None:
            self.df = obj
            return obj

        if self.input_dataset:
            ds_id = self.input_dataset
        else:
            ds_id = self.id

        dvc_ds = context.load_dvc_dataset(ds_id)
        assert dvc_ds.df is not None
        df = ppl.from_dvc_dataset(dvc_ds)
        if self.filters:
            for d in self.filters:
                if 'column' in d:
                    col = d['column']
                    val = d.get('value', None)
                    vals = d.get('values', [])
                    drop = d.get('drop_col', True)
                    if vals:
                        df = df.filter(pl.col(col).is_in(vals))
                    if val:  # Column dropping is possible without value filtering
                        df = df.filter(pl.col(col) == val)
                    if drop:
                        df = df.drop(col)
                elif 'dimension' in d:
                    dim_id = d['dimension']
                    if 'groups' in d:
                        dim = context.dimensions[dim_id]
                        grp_ids = d['groups']
                        grp_s = dim.ids_to_groups(dim.series_to_ids_pl(df[dim_id]))
                        df = df.filter(grp_s.is_in(grp_ids))
                    elif 'categories' in d:
                        cat_ids = d['categories']
                        df = df.filter(pl.col(dim_id).is_in(cat_ids))
                    elif 'assign_category' in d:
                        cat_id = d['assign_category']
                        if dim_id in context.dimensions:
                            dim = context.dimensions[dim_id]
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
                        self.column, self.id, available,
                    ),
                )
            df = df.with_columns(pl.col(self.column).alias(VALUE_COLUMN))
            cols = [YEAR_COLUMN, VALUE_COLUMN, *df.dim_ids]

        if YEAR_COLUMN in cols:
            if YEAR_COLUMN not in df.primary_keys:
                df = df.add_to_index(YEAR_COLUMN)
            if len(df.filter(pl.col(YEAR_COLUMN).lt(100))) > 0:
                baseline_year = context.instance.maximum_historical_year
                if baseline_year is None:
                    raise Exception(
                        'The maximum_historical_year from instance is not given. It is needed by dataset %s to define the baseline for relative data.' % self.id)  # noqa: E501
                df = df.with_columns(
                    pl.when(pl.col(YEAR_COLUMN) < 100)
                    .then(pl.col(YEAR_COLUMN) + baseline_year)
                    .otherwise(pl.col(YEAR_COLUMN)).alias(YEAR_COLUMN),
                )
                df = df.with_columns(pl.col(YEAR_COLUMN).cast(int).alias(YEAR_COLUMN))

                # Duplicates may occur when baseline year overlaps with existing data points.
                meta = df.get_meta()
                df = ppl.to_ppdf(df.unique(subset=meta.primary_keys, keep='last',
                                           maintain_order=True), meta=meta)

        if FORECAST_COLUMN in df.columns:
            cols.append(FORECAST_COLUMN)
        elif self.forecast_from is not None:
            df = df.with_columns(
                pl.when(pl.col(YEAR_COLUMN) >= self.forecast_from)
                    .then(pl.lit(True))
                    .otherwise(pl.lit(False))
                .alias(FORECAST_COLUMN)
            )
            cols.append(FORECAST_COLUMN)

        df = df.select(cols)
        ppl._validate_ppdf(df)

        df = self._process_output(df)
        df = self.post_process(df)
        if cache_key:
            context.cache.set(cache_key, df, no_expiry=True)

        return df

    def get_unit(self, context: Context) -> Unit:
        if self.unit:
            return self.unit
        df = self.load(context)
        if VALUE_COLUMN in df.columns:
            meta = df.get_meta()
            if VALUE_COLUMN not in meta.units:
                raise Exception("Dataset %s does not have a unit" % self.id)
            return meta.units[VALUE_COLUMN]
        raise Exception("Dataset %s does not have the value column" % self.id)

    def hash_data(self, context: Context) -> dict[str, Any]:
        extra_fields = [
            'input_dataset', 'column', 'filters', 'dropna',
            'forecast_from', 'max_year', 'min_year', 'interpolate',
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
    historical: list[tuple[int, float]] | None
    forecast: list[tuple[int, float]] | None
    use_interpolation: bool = False

    def _fixed_multi_values_to_df(self, data) -> pd.DataFrame:
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

        if self.use_interpolation:
            self.interpolate = True

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

        df: pd.DataFrame | None
        if hdf is not None and fdf is not None:
            df = pd.concat([hdf, fdf])
        elif hdf is not None:
            df = hdf
        else:
            df = fdf

        assert df is not None, "Both historical and forecast data are None"
        df = df.set_index(YEAR_COLUMN)

        # Ensure value column has right units
        pt = pint_pandas.PintType(self.unit)
        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            df[col] = df[col].astype(float).astype(pt)

        self.df = self.post_process(ppl.from_pandas(df))

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
        self.df = JSONDataset.deserialize_df(self.data) # type: ignore
        meta = self.df.get_meta()
        if len(meta.units) == 1:
            self.unit = next(iter(meta.units.values()))

    def load(self, context: Context) -> ppl.PathsDataFrame:
        assert self.df is not None
        return self.post_process(self.df)

    def hash_data(self, context: Context) -> dict[str, Any]:
        df = self.df.to_pandas()
        return dict(hash=int(pd.util.hash_pandas_object(df).sum()))

    def get_unit(self, context: Context) -> Unit:
        return cast(Unit, self.unit)

    @classmethod
    def deserialize_df(cls, value: dict) -> ppl.PathsDataFrame:
        sio = io.StringIO(json.dumps(value))
        df = pd.read_json(sio, orient='table')
        for f in value['schema']['fields']:
            unit = f.get('unit')
            col = f['name']
            if unit is not None:
                pt = pint_pandas.PintType(unit)
                df[col] = df[col].astype(float).astype(pt)
        return ppl.from_pandas(df)

    @classmethod
    def serialize_df(cls, pdf: ppl.PathsDataFrame, add_uuids: bool = False) -> dict:
        units = {}
        df = pdf.to_pandas()
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
