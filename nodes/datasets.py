from __future__ import annotations

import hashlib
import io
import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import orjson
import polars as pl
from numpy.random import default_rng  # TODO Could call Generator to give hints about rng attributes but requires code change

from common import polars as ppl
from nodes.calc import extend_last_historical_value_pl
from nodes.units import Unit, unit_registry

from .constants import FORECAST_COLUMN, UNCERTAINTY_COLUMN, VALUE_COLUMN, YEAR_COLUMN

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas import DataFrame as PandasDataFrame

    from datasets.models import Dataset as DBDatasetModel

    from .context import Context


@dataclass
class Dataset(ABC):
    id: str
    tags: list[str]
    interpolate: bool = field(init=False)
    df: ppl.PathsDataFrame | None = field(init=False)
    hash: bytes | None = field(init=False)
    rng: Callable = field(init=False)

    def __post_init__(self):
        self.df = None
        self.hash = None
        self.interpolate = False
        self.rng = default_rng() # type: ignore
        if getattr(self, 'unit', None) is None:
            self.unit = None

    @abstractmethod
    def load(self, context: Context) -> ppl.PathsDataFrame:
        raise NotImplementedError()

    @abstractmethod
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

    def get_cache_key(self, context: Context) -> str:
        ds_hash = self.calculate_hash(context).hex()
        return 'ds:%s:%s' % (self.id, ds_hash)

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

    def post_process(self, context: Context | None, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:  # pyright: ignore[reportUnusedParameter]
        if self.interpolate:
            df = self._linear_interpolate(df)
        return df

    def get_copy(self, context: Context) -> ppl.PathsDataFrame:
        df = self.load(context)
        return df.copy()

    def get_unit(self, context: Context) -> Unit:  # pyright: ignore[reportUnusedParameter]
        raise NotImplementedError()

    def interpret(self, df: ppl.PathsDataFrame, context: Context) -> ppl.PathsDataFrame:
        size = context.sample_size
        cols = []
        for col in df.columns:
            # FIXME Invent a generic way to ignore sampling when content is not probabilities
            if (col not in [FORECAST_COLUMN, 'Unit', 'UUID', 'muni'] + df.primary_keys and
                isinstance(df[col].dtype, pl.String)):
                cols += [col]
        if size == 0 or len(cols) == 0:
            # TODO Whether too use uncertainties depends on the node
            # df = df.with_columns(pl.lit('median').cast(pl.Categorical).alias(UNCERTAINTY_COLUMN))
            return df

        meta = df.get_meta()
        meta.primary_keys += [UNCERTAINTY_COLUMN]
        df = df.with_columns(pl.arange(0, len(df)).alias('row_index'))

        out = None
        for col in cols:
            dfc = pl.DataFrame()
            for i in range(len(df)):
                dist_string = df[col][i]
                s = self.get_sample(dist_string, size)
                median_value = np.median(s)
                dfi = pl.DataFrame({
                    'row_index': [i] * (size + 1),
                    UNCERTAINTY_COLUMN: ['median'] + [str(num) for num in range(size)],
                    col: [median_value] + list(s),
                })
                dfi = dfi.with_columns(pl.col('row_index').cast(pl.Int64))
                dfc = pl.concat([dfc, dfi])
            if out is None:
                out = dfc
            else:
                out = out.join(dfc, how='inner', on=['row_index', UNCERTAINTY_COLUMN])

        df = df.drop(cols)
        df = df.join(out, how='inner', on='row_index').drop('row_index') # type: ignore
        df = ppl.to_ppdf(df, meta=meta)
        df = df.with_columns(pl.col(UNCERTAINTY_COLUMN).cast(pl.Categorical))

        return df

    def loguniform(self, match, size) -> list:
        low = np.log(float(match.group(1)))
        high = np.log(float(match.group(2)))
        return np.exp(self.rng.uniform(low, high, size)).tolist() # type: ignore
    def uniform(self, match, size) -> list:
        low = float(match.group(1))
        high = float(match.group(2))
        return self.rng.uniform(low, high, size).tolist() # type: ignore
    def lognormal_plusminus(self, match, size) -> list:
        mean_lognormal = float(match.group(1))
        std_lognormal = float(match.group(2))
        sigma = np.sqrt(np.log(1 + (std_lognormal ** 2) / (mean_lognormal ** 2)))
        mu = np.log(mean_lognormal) - (sigma ** 2) / 2
        return self.rng.lognormal(mu, sigma, size).tolist() # type: ignore
    def normal_plusminus(self, match, size) -> list:
        loc = float(match.group(1))
        scale = float(match.group(2))
        return self.rng.normal(loc, scale, size).tolist() # type: ignore
    def normal_interval(self, match, size) -> list:
        loc = float(match.group(1))
        lower = float(match.group(2))
        upper = float(match.group(3))
        scale = (upper - lower) / 2 / 1.959963984540054
        return self.rng.normal(loc, scale, size).tolist() # type: ignore
    def beta(self, match, size) -> list:
        a = float(match.group(1))
        b = float(match.group(2))
        return self.rng.beta(a, b, size).tolist() # type: ignore
    def poisson(self, match, size) -> list:
        lam = float(match.group(1))
        s = self.rng.poisson(lam, size).tolist() # type: ignore
        return [float(v) for v in s]
    def exponential(self, match, size) -> list:
        mean = float(match.group(1))
        return self.rng.exponential(scale=mean, size=size).tolist() # type: ignore
    def problist(self, match, size) -> list:
        s = match.group(1)
        s = [float(x) for x in s.split(',')]
        return self.rng.choice(s, size, replace=True).tolist() # type: ignore
    def scalar(self, match, size) -> list:
        value = float(match.group(1))
        return [value] * size

    def get_sample(self, dist_string: str, size: int) -> list:
        pos = r'(\d*.?\d+)'
        real = r'(\-?\d*.?\d+)'
        real2 = r'\-?\d*.?\d+'
        expressions = {
            'Loguniform': r'%s-%s\(log\)' % (pos, pos),  # low - high (log)
            'Uniform': r'%s-%s' % (real, real),  # low - high
            'Lognormal_plusminus': r'%s(?:\+-|±)%s\(log\)' % (pos, pos),  # mean +- sd (log)
            'Normal_plusminus': r'%s(?:\+-|±)%s' % (real, pos),  # mean +- sd
            'Normal_interval': r'%s\(%s,%s\)' % (real, real, real),  # mean (lower - upper) for 95 % CI
            'Beta': r'(?i)beta\(%s,%s\)' % (pos, pos),  # Beta(a, b)
            'Poisson': r'(?i)poisson\(%s\)' % pos,  # Poisson(lambda)
            'Exponential': r'(?i)exponential\(%s\)' % pos,  # Exponential(mean)
            'Problist': r'\[(%s(,%s)*)\]' % (real2, real2),  # [x1, x2, ... , xn]
            'Scalar': real,  # value
        }
        functions = {
            'Loguniform': self.loguniform,
            'Uniform': self.uniform,
            'Lognormal_plusminus': self.lognormal_plusminus,
            'Normal_plusminus': self.normal_plusminus,
            'Normal_interval': self.normal_interval,
            'Beta': self.beta,
            'Poisson': self.poisson,
            'Exponential': self.exponential,
            'Problist': self.problist,
            'Scalar': self.scalar,
        }

        dist_string = dist_string.replace(' ', '')
        for key, regex in expressions.items():  # noqa: B007
            match = re.search(regex, dist_string)
            if match:
                break
        else:
            raise LookupError(self, f"String '{dist_string}' does not match any distribution.")
        s = functions[key](match, size)
        return s


@dataclass
class DatasetWithFilters(Dataset):
    column: str | None = None
    filters: list | None = None
    dropna: bool | None = None
    min_year: int | None = None
    max_year: int | None = None

    # The year from which the time series becomes a forecast
    forecast_from: int | None = None

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

    def _filter_df(self, context: Context, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:  # noqa: C901
        if not self.filters:
            return df

        for d in self.filters:
            if 'column' in d:
                col = d['column']
                val = d.get('value', None)
                vals = d.get('values', [])
                ref = d.get('ref', None)
                drop = d.get('drop_col', True)
                if vals:
                    df = df.filter(pl.col(col).is_in(vals))
                if val:
                    df = df.filter(pl.col(col) == val)
                if ref:
                    pval = context.get_parameter_value(ref, required=True)
                    if isinstance(pval, float):
                        pval = int(pval)
                    val = str(pval)
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

        return df

    def _filter_and_process_df(self, context: Context, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = self._filter_df(context, df)
        cols = df.columns

        if self.column:
            if self.column not in cols:
                available = ', '.join(cols)  # type: ignore
                raise Exception(
                    "Column '%s' not found in dataset '%s'. Available columns: %s"
                    % (
                        self.column,
                        self.id,
                        available,
                    ),
                )
            df = df.with_columns(pl.col(self.column).alias(VALUE_COLUMN))
            cols = [YEAR_COLUMN, VALUE_COLUMN, *df.dim_ids]

        if YEAR_COLUMN in cols:
            if YEAR_COLUMN not in df.primary_keys:
                df = df.add_to_index(YEAR_COLUMN)
            if len(df.filter(pl.col(YEAR_COLUMN) < 200)) > 0:
                baseline_year = context.instance.reference_year
                if baseline_year is None:
                    raise Exception(
                        'The reference_year from instance is not given. ' +
                        'It is needed by dataset %s to define the baseline for relative data.'
                        % self.id,
                    )
                df = df.with_columns(
                    pl.when(pl.col(YEAR_COLUMN) < 90)
                    .then(pl.col(YEAR_COLUMN) + pl.lit(baseline_year))
                    .otherwise(pl.col(YEAR_COLUMN))
                    .alias(YEAR_COLUMN),
                )
                target_year = context.instance.target_year
                df = df.with_columns(
                    pl.when((pl.col(YEAR_COLUMN) >= 90) & (pl.col(YEAR_COLUMN) < 200))
                    .then(pl.col(YEAR_COLUMN) + pl.lit(target_year) - pl.lit(100))
                    .otherwise(pl.col(YEAR_COLUMN))
                    .alias(YEAR_COLUMN),
                )
                df = df.with_columns(pl.col(YEAR_COLUMN).cast(int).alias(YEAR_COLUMN))

                # Duplicates may occur when baseline year overlaps with existing data points.
                meta = df.get_meta()
                df = ppl.to_ppdf(df.unique(subset=meta.primary_keys, keep='last', maintain_order=True), meta=meta)

        if FORECAST_COLUMN in df.columns:
            cols.append(FORECAST_COLUMN)
        elif self.forecast_from is not None:
            df = df.with_columns(
                pl.when(pl.col(YEAR_COLUMN) >= self.forecast_from)
                .then(pl.lit(value=True))
                .otherwise(pl.lit(value=False))
                .alias(FORECAST_COLUMN),
            )
            cols.append(FORECAST_COLUMN)

        df = df.select(cols)
        ppl.validate_ppdf(df)

        df = self._process_output(df)
        return df


@dataclass
class DVCDataset(DatasetWithFilters):
    """Dataset that is loaded by dvc-pandas."""

    # The output can be customized further by specifying a column and filters.
    # If `input_dataset` is not specified, we default to `id` being
    # the dvc-pandas dataset identifier.
    input_dataset: str | None = None
    unit: Unit | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.unit is not None:
            assert isinstance(self.unit, Unit)

    def load(self, context: Context) -> ppl.PathsDataFrame:
        obj = None
        cache_key: str | None
        if not context.skip_cache:
            cache_key = self.get_cache_key(context)
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

        df = self._filter_and_process_df(context, df)
        df = self.post_process(context, df)
        df = self.interpret(df, context)
        if cache_key:
            context.cache.set(cache_key, df, expiry=0)

        return df

    def get_unit(self, context: Context) -> Unit:
        if self.unit:
            return self.unit
        df = self.load(context)
        if VALUE_COLUMN in df.columns:
            meta = df.get_meta()
            if VALUE_COLUMN not in meta.units:
                raise Exception('Dataset %s does not have a unit' % self.id)
            return meta.units[VALUE_COLUMN]
        raise Exception('Dataset %s does not have the value column' % self.id)

    def hash_data(self, context: Context) -> dict[str, Any]:
        extra_fields = [
            'input_dataset',
            'column',
            'filters',
            'dropna',
            'forecast_from',
            'max_year',
            'min_year',
            'interpolate',
        ]
        d = {}
        for f in extra_fields:
            d[f] = getattr(self, f)

        d['commit_id'] = context.dataset_repo.commit_id
        d['dvc_id'] = self.input_dataset or self.id
        return d


@dataclass
class GenericDataset(DVCDataset):
    """Dataset that already filters for relevant columns."""

    # Supported languages: Czech, Danish, English, Finnish, German, Latvian, Polish, Swedish
    characterlookup = str.maketrans(
        {
            '.': '',
            ',': '',
            ':': '',
            '-': '',
            '(': '',
            ')': '',
            ' ': '_',
            '/': '_',
            '&': 'and',
            'ä': 'a',
            'å': 'a',
            'ą': 'a',
            'á': 'a',
            'ā': 'a',
            'ć': 'c',
            'č': 'c',
            'ď': 'd',
            'ę': 'e',
            'é': 'e',
            'ě': 'e',
            'ē': 'e',
            'ģ': 'g',
            'í': 'i',
            'ī': 'i',
            'ķ': 'k',
            'ł': 'l',
            'ļ': 'l',
            'ń': 'n',
            'ň': 'n',
            'ņ': 'n',
            'ö': 'o',
            'ø': 'o',
            'ó': 'o',
            'ř': 'r',
            'ś': 's',
            'š': 's',
            'ť': 't',
            'ü': 'u',
            'ú': 'u',
            'ů': 'u',
            'ū': 'u',
            'ý': 'y',
            'ź': 'z',
            'ż': 'z',
            'ž': 'z',
            'æ': 'ae',
            'ß': 'ss',
        },
    )

    def implement_unit_col(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        """Create separate metric columns for each unique unit in the DataFrame."""
        if 'Unit' not in df.columns:
            return df

        unique_units = df['Unit'].unique()
        if len(unique_units) == 1:
            df = df.set_unit(VALUE_COLUMN, unique_units[0])
            df = df.drop('Unit')
            return df

        meta = df.get_meta()
        result = df.copy()

        new_units = meta.units.copy()
        if VALUE_COLUMN in new_units:
            del new_units[VALUE_COLUMN]

        # Create a new metric column for each unit
        for unit_str in unique_units:
            column_name = f"{VALUE_COLUMN}_{unit_str.replace('/', '_per_')}"

            # Create a filtered column with values only where Unit matches
            result = result.with_columns(
                pl.when(pl.col('Unit') == unit_str)
                .then(pl.col(VALUE_COLUMN))
                .otherwise(None)
                .alias(column_name)
            )

            # Add unit to metadata
            unit = unit_registry.parse_units(unit_str)
            new_units[column_name] = unit

        result = result.drop([VALUE_COLUMN, 'Unit'])
        new_meta = ppl.DataFrameMeta(
            primary_keys=meta.primary_keys,
            units=new_units
        )

        return ppl.to_ppdf(result, meta=new_meta)

    # -----------------------------------------------------------------------------------
    def convert_names_to_ids(self, df: ppl.PathsDataFrame, context: Context) -> ppl.PathsDataFrame:
        exset = {YEAR_COLUMN, VALUE_COLUMN, FORECAST_COLUMN, UNCERTAINTY_COLUMN, 'Unit', 'UUID'}
        exset |= {col for col in df.columns if col.startswith(f"{VALUE_COLUMN}_")}

        # Convert index level names from labels to IDs.
        collookup = {}
        for col in list(set(df.columns) - exset):
            collookup[col] = col.lower().translate(self.characterlookup)
        df = df.rename(collookup)

        # Convert levels within each index level from labels to IDs.
        if 'scope' in df.columns:
            catlookup = {}
            for cat in df['scope'].unique():
                catlookup[cat] = cat.lower().replace(' ', '').replace('.', '')
            df = df.with_columns(df['scope'].replace(catlookup))
            exset.add('scope')

        for col in list(set(df.columns) - exset):
            catlookup = {}
            for cat in df[col].unique():
                catlookup[cat] = cat.lower().translate(self.characterlookup)
            df = df.with_columns(df[col].replace(catlookup))

            if col in context.dimensions:
                for cat in context.dimensions[col].categories:
                    if cat.aliases:
                        df = df.with_columns(context.dimensions[col].series_to_ids_pl(df[col]))
                        break
        return df

    # -----------------------------------------------------------------------------------
    def drop_unnecessary_levels(self, df: ppl.PathsDataFrame, droplist: list) -> ppl.PathsDataFrame:
        # Drop filter levels and empty dimension levels.
        drops = [d for d in droplist if d in df.columns]

        for col in list(set(df.columns) - set(drops)):
            vals = df[col].unique().to_list()
            if vals in [['.'], [None]]:
                drops.append(col)

        df = df.drop(drops)
        return df

    # -----------------------------------------------------------------------------------
    def load(self, context: Context) -> ppl.PathsDataFrame:
        df = DVCDataset.load(self, context)
        df = self.drop_unnecessary_levels(df, droplist=['Description', 'Quantity']) # TODO Maybe filter by quantity?
        df = self.implement_unit_col(df)
        df = self.convert_names_to_ids(df, context)

        if FORECAST_COLUMN not in df.columns:
            df = df.with_columns(pl.lit(False).alias(FORECAST_COLUMN))  # noqa: FBT003

        self.interpolate = True
        df = self._linear_interpolate(df)
        df = extend_last_historical_value_pl(df, end_year=context.instance.model_end_year)
        return df


@dataclass
class FixedDataset(Dataset):
    """Dataset from fixed values."""

    # Use `dimensionless` for `unit` if the quantities should be dimensionless.
    unit: Unit
    historical: list[tuple[int, float]] | None
    forecast: list[tuple[int, float]] | None
    use_interpolation: bool = False

    def _fixed_multi_values_to_df(self, data) -> PandasDataFrame:
        series = []
        for d in data:
            vals = d['values']
            s = self.pd.Series(data=[x[1] for x in vals], index=[x[0] for x in vals], name=d['id'])
            series.append(s)
        df = self.pd.concat(series, axis=1)
        df.index.name = YEAR_COLUMN
        df = df.reset_index()
        return df

    def __post_init__(self):
        super().__post_init__()
        import pandas as pd

        self.pd = pd

        if self.use_interpolation:
            self.interpolate = True

        if self.historical:
            hdf = pl.DataFrame(self.historical, orient='row', schema=[YEAR_COLUMN, VALUE_COLUMN])
            hdf = hdf.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
        else:
            hdf = None
        if self.forecast:
            fdf = pl.DataFrame(self.forecast, orient='row', schema=[YEAR_COLUMN, VALUE_COLUMN])
            fdf = fdf.with_columns(pl.lit(value=True).alias(FORECAST_COLUMN))
        else:
            fdf = None

        if hdf is not None and fdf is not None:
            df = pl.concat([hdf, fdf])
        elif hdf is not None:
            df = hdf
        else:
            assert fdf is not None
            df = fdf

        assert df is not None, 'Both historical and forecast data are None'

        # Ensure value column has right units
        pdf = ppl.to_ppdf(df)
        pdf = pdf.set_unit(VALUE_COLUMN, self.unit)
        pdf = pdf.add_to_index(YEAR_COLUMN)
        pdf = self.post_process(None, pdf)

        self.df = pdf

    def load(self, context: Context) -> ppl.PathsDataFrame:
        # FIXME Cache does not work properly now but does not cause error.
        obj = None
        cache_key: str | None
        if not context.skip_cache:
            cache_key = self.get_cache_key(context)
            res = context.cache.get(cache_key)
            if res.is_hit:
                obj = res.obj
        else:
            cache_key = None

        if obj is not None:
            self.df = obj
            return obj

        df = self.df
        assert df is not None
        df = self.interpret(df, context) # FIXME If all are scalars, do not create interpret dimension.
        self.df = df
        if cache_key:
            context.cache.set(cache_key, df, expiry=0)
        return self.df

    def hash_data(self, context: Context) -> dict[str, Any]:
        assert self.df is not None
        df = self.df.to_pandas()
        return dict(
            hash=int(self.pd.util.hash_pandas_object(df).sum()),
            sample_size=context.sample_size
        )

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
        self.df = JSONDataset.deserialize_df(self.data)  # type: ignore[override]
        meta = self.df.get_meta()
        if len(meta.units) == 1:
            self.unit = next(iter(meta.units.values()))

    def load(self, context: Context) -> ppl.PathsDataFrame:
        assert self.df is not None
        return self.post_process(context, self.df)

    def hash_data(self, context: Context) -> dict[str, Any]:
        import pandas as pd

        df = self.df.to_pandas()
        return dict(hash=int(pd.util.hash_pandas_object(df).sum()))

    def get_unit(self, context: Context) -> Unit:
        return cast(Unit, self.unit)

    @classmethod
    def deserialize_df(cls, value: dict) -> ppl.PathsDataFrame:
        import pandas as pd
        from pint_pandas import PintType

        sio = io.StringIO(json.dumps(value))
        df = pd.read_json(sio, orient='table')
        for f in value['schema']['fields']:
            unit = f.get('unit')
            col = f['name']
            if unit is not None:
                pt = PintType(unit)
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


@dataclass
class DBDataset(DatasetWithFilters):
    """Dataset that is loaded from the admin UI's Dataset model."""

    db_dataset_id: str | None = None
    db_dataset_obj: DBDatasetModel | None = None
    unit: Unit | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.db_dataset_id is not None:
            from datasets.models import Dataset as DBDatasetModel
            self.db_dataset_obj = DBDatasetModel.objects.get(uuid=self.db_dataset_id)

    def load(self, context: Context) -> ppl.PathsDataFrame:
        if self.df is not None:
            return self.df

        admin_ds = self.db_dataset_obj
        if admin_ds is None:
            raise Exception('Admin dataset not loaded')
        df = JSONDataset.deserialize_df(admin_ds.table)
        df = self.post_process(context, df)
        self.df = df
        return df

    def hash_data(self, context: Context) -> dict[str, Any]:
        obj = self.db_dataset_obj
        assert obj is not None
        return dict(obj_pk=obj.pk, updated_at=str(obj.updated_at))

    def get_unit(self, context: Context) -> Unit:
        df = self.load(context)
        meta = df.get_meta()
        if len(meta.units) == 1:
            return next(iter(meta.units.values()))
        raise Exception('Dataset %s does not have a single unit' % self.id)
