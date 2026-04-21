from __future__ import annotations

import hashlib
import inspect
import io
import json
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from functools import cache, cached_property, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Concatenate, Literal, Self, TypedDict, cast, override

from pydantic import TypeAdapter

import numpy as np
import orjson
import polars as pl
from loguru import logger
from numpy.random import default_rng  # TODO Could call Generator to give hints about rng attributes but requires code change
from numpy.typing import NDArray

from kausal_common.logging.errors import capture_error
from kausal_common.perf.perf_context import PerfKind, estimate_size_bytes

from common import polars as ppl
from nodes.calc import extend_last_historical_value_pl
from nodes.exceptions import DatasetError
from nodes.units import Unit, unit_registry

from .constants import FORECAST_COLUMN, UNCERTAINTY_COLUMN, VALUE_COLUMN, YEAR_COLUMN

if TYPE_CHECKING:
    import dvc_pandas
    from pandas import DataFrame as PandasDataFrame
    from rich.repr import RichReprResult

    from kausal_common.datasets.models import Dataset as DBDatasetModel
    from kausal_common.perf.perf_context import PerfAttrs, PerfSpanEntry

    from nodes.defs.node_defs import (
        ColumnDatasetFilterDef,
        DimensionDatasetFilterDef,
        InputDatasetDef,
        InputDatasetFilterDef,
        RenameColumnDatasetFilterDef,
        RenameItemDatasetFilterDef,
    )

    from .context import Context


type DatasetMethod[DS: Dataset, **P, R] = Callable[Concatenate[DS, P], R]


def measure_dataset_call[DS: Dataset, **P, R](
    event_name: str,
    *,
    capture_df_result: bool = True,
    capture_df_arg: bool = False,
) -> Callable[[DatasetMethod[DS, P, R]], DatasetMethod[DS, P, R]]:
    def decorator(fn: DatasetMethod[DS, P, R]) -> DatasetMethod[DS, P, R]:
        @wraps(fn)
        def wrapped(self: DS, *args: P.args, **kwargs: P.kwargs) -> R:
            dataset = self
            assert isinstance(dataset, Dataset)
            context = dataset.context
            attrs = dataset.get_span_attrs()
            with context.perf_context.exec_named(
                kind=PerfKind.DATASET,
                id=dataset.id,
                op=event_name.removeprefix('dataset.'),
                attrs=attrs,
            ) as event:
                result = fn(self, *args, **kwargs)
                if capture_df_result:
                    assert isinstance(result, ppl.PathsDataFrame)
                    dataset.set_dataframe_span_attrs(event, result, kind='result')
                if capture_df_arg:
                    df = args[0]
                    assert isinstance(df, ppl.PathsDataFrame)
                    dataset.set_dataframe_span_attrs(event, df, kind='arg')
            return result

        return wrapped

    return decorator


class DatasetKwargs(TypedDict):
    tags: list[str]
    output_dimensions: list[str] | None
    interpolate: bool


@dataclass
class Dataset(ABC):
    _class_hash: ClassVar[bytes | None] = None

    id: str
    context: Context
    _: KW_ONLY
    tags: list[str] = field(default_factory=list)
    output_dimensions: list[str] | None = field(default=None)

    interpolate: bool = False
    df: ppl.PathsDataFrame | None = field(init=False, repr=False, default=None)
    hash: bytes | None = field(init=False, repr=False, default=None)

    def __rich_repr__(self) -> RichReprResult:
        yield 'id', self.id
        if self.df is None:
            yield 'df', '<not loaded>'
        else:
            yield 'columns', len(self.df.columns)
            yield 'rows', len(self.df)

    def __post_init__(self):  # noqa: B027
        pass

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        cls._class_hash = cls.get_class_hash()

    @classmethod
    def kwargs_from_def(cls, ds_def: InputDatasetDef) -> DatasetKwargs:
        return DatasetKwargs(
            tags=ds_def.tags,
            output_dimensions=ds_def.output_dimensions,
            interpolate=ds_def.interpolate,
        )

    @abstractmethod
    def load_internal(self) -> ppl.PathsDataFrame:
        """
        Load the dataset into a PathsDataFrame.

        This method is only for subclassess to implement. Do not call this directly, call `.get_copy()` instead.
        """
        raise NotImplementedError()

    @abstractmethod
    def hash_data(self) -> dict[str, Any]:
        """Return subclass-specific data to include in the hash."""
        raise NotImplementedError()

    @classmethod
    def get_class_hash(cls) -> bytes:
        h = hashlib.md5(usedforsecurity=False)
        for parent_class in cls.mro():
            if parent_class is object:
                continue
            try:
                class_file = inspect.getfile(parent_class)
            except TypeError:
                continue
            mod_mtime = Path(class_file).stat().st_mtime_ns
            h.update(str(mod_mtime).encode('ascii'))
        return h.digest()

    def calculate_hash(self) -> bytes:
        if self.hash is not None:
            return self.hash
        class_hash = type(self)._class_hash
        if class_hash is None:
            class_hash = type(self).get_class_hash()
            type(self)._class_hash = class_hash
        d = {'id': self.id, 'interpolate': self.interpolate, 'class_hash': class_hash.hex()}
        d.update(self.hash_data())
        h = hashlib.md5(orjson.dumps(d, option=orjson.OPT_SORT_KEYS), usedforsecurity=False).digest()
        self.hash = h
        return h

    def get_cache_key(self) -> str:
        ds_hash = self.calculate_hash().hex()
        return 'ds:%s:%s' % (self.id, ds_hash)

    def get_span_attrs(self) -> PerfAttrs:
        return {
            'dataset.id': self.id,
        }

    def set_dataframe_span_attrs(
        self, event: PerfSpanEntry[Any] | None, df: ppl.PathsDataFrame, kind: Literal['arg', 'result'] | None = None
    ) -> None:
        if event is None:
            return
        midfix = f'{kind}.' if kind is not None else ''
        event.set_attr(f'dataset.{midfix}rows', len(df))
        event.set_attr(f'dataset.{midfix}columns', len(df.columns))
        event.set_attr(f'dataset.{midfix}in_memory.bytes', estimate_size_bytes(df))

    @measure_dataset_call('dataset.interpolate')
    def _linear_interpolate(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        if YEAR_COLUMN not in df.columns:
            raise DatasetError(
                self, f"'{YEAR_COLUMN}' does not exist in dataset '{self.id}'. Available columns: {', '.join(df.columns)}."
            )
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

    def post_process(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        if self.interpolate:
            df = self._linear_interpolate(df)
        return df

    @measure_dataset_call('dataset.get')
    def get_copy(self) -> ppl.PathsDataFrame:
        df = self.load_internal()
        return df.copy()

    @cached_property
    def sampler(self) -> DatasetSampler:
        return DatasetSampler()

    @measure_dataset_call('dataset.sample')
    def _sample(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        return self.sampler.interpret(self, df)


@cache
def get_input_dataset_filter_adapter() -> TypeAdapter[InputDatasetFilterDef]:
    from nodes.defs.node_defs import InputDatasetFilterDef

    return TypeAdapter(InputDatasetFilterDef)


class FilterDatasetKwargs(DatasetKwargs):
    column: str | None
    filters: list[InputDatasetFilterDef] | None
    dropna: bool | None
    min_year: int | None
    max_year: int | None
    unit: Unit | None
    forecast_from: int | None


@dataclass
class DatasetWithFilters(Dataset, ABC):
    column: str | None = None
    filters: list[InputDatasetFilterDef] | None = None
    dropna: bool | None = None
    min_year: int | None = None
    max_year: int | None = None
    unit: Unit | None = None

    # The year from which the time series becomes a forecast
    forecast_from: int | None = None

    @classmethod
    def kwargs_from_def(cls, ds_def: InputDatasetDef) -> FilterDatasetKwargs:
        return FilterDatasetKwargs(
            **super().kwargs_from_def(ds_def),
            column=ds_def.column,
            filters=ds_def.filters,
            dropna=ds_def.dropna,
            min_year=ds_def.min_year,
            max_year=ds_def.max_year,
            unit=ds_def.unit,
            forecast_from=ds_def.forecast_from,
        )

    def __rich_repr__(self) -> RichReprResult:
        yield from super().__rich_repr__()
        if self.column is not None:
            yield 'column', self.column
        if self.filters is not None:
            yield 'filters', len(self.filters)

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

    def _filter_df(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        from nodes.defs.node_defs import (
            ColumnDatasetFilterDef,
            DimensionDatasetFilterDef,
            RenameColumnDatasetFilterDef,
            RenameItemDatasetFilterDef,
        )

        if not self.filters:
            return df

        df_orig = df
        for filter_def in self.filters:
            if isinstance(filter_def, ColumnDatasetFilterDef):
                df = self._column_filter(df, filter_def)
            elif isinstance(filter_def, DimensionDatasetFilterDef):
                df = self._dimension_filter(df, filter_def)
            elif isinstance(filter_def, RenameColumnDatasetFilterDef):
                continue
            else:
                assert isinstance(filter_def, RenameItemDatasetFilterDef)
                df = self._rename_item_filter(df, filter_def)

            if len(df) == 0:
                print(df_orig)
                print(self.filters)
                raise DatasetError(self, 'Nothing left after filtering. See original dataset above.')

        return df

    def _column_filter(self, df: ppl.PathsDataFrame, d: ColumnDatasetFilterDef) -> ppl.PathsDataFrame:
        context = self.context
        col = d.column
        val = d.value
        vals = d.values
        ref = d.ref
        drop = d.drop_col
        exclude = d.exclude
        flatten = d.flatten
        mask = None
        if vals:
            mask = pl.col(col).is_in(vals)
        if val:
            mask = pl.col(col) == val
        if ref:
            pval = context.get_parameter_value(ref, required=True)
            if isinstance(pval, float):
                pval = int(pval)
            val = str(pval)
            mask = pl.col(col) == val
        if mask is not None:
            if exclude:
                mask = ~mask
            df = df.filter(mask)

        if flatten:
            if VALUE_COLUMN in df.columns:
                df = df.filter(~pl.col(VALUE_COLUMN).is_nan())
            df = df.paths.sum_over_dims(col)

        elif drop:
            df = df.drop(col)
        return df

    def _dimension_filter(self, df: ppl.PathsDataFrame, d: DimensionDatasetFilterDef) -> ppl.PathsDataFrame:
        context = self.context
        dim_id = d.dimension
        if d.groups:
            dim = context.dimensions[dim_id]
            grp_ids = d.groups
            grp_s = dim.ids_to_groups(dim.series_to_ids_pl(df[dim_id]))
            df = df.filter(grp_s.is_in(grp_ids))
        elif d.categories:
            cat_ids = d.categories
            df = df.filter(pl.col(dim_id).is_in(cat_ids))
        elif d.assign_category is not None:
            cat_id = d.assign_category
            if dim_id in context.dimensions:
                dim = context.dimensions[dim_id]
                assert dim_id not in df.dim_ids
                assert cat_id in dim.cat_map
            df = df.with_columns(pl.lit(cat_id).alias(dim_id)).add_to_index(dim_id)
        flatten = d.flatten
        if flatten:
            if VALUE_COLUMN in df.columns:
                df = df.filter(~pl.col(VALUE_COLUMN).is_nan())
            df = df.paths.sum_over_dims(dim_id)
        return df

    def _rename_col_filter(self, df: ppl.PathsDataFrame, d: RenameColumnDatasetFilterDef) -> ppl.PathsDataFrame:
        col = d.rename_col
        val = d.value
        if col not in df.columns:
            raise DatasetError(self, f'Column {col} not found. Available columns are {df.columns}')
        if val:
            df = df.rename({col: val})
        return df

    def _rename_item_filter(self, df: ppl.PathsDataFrame, d: RenameItemDatasetFilterDef) -> ppl.PathsDataFrame:
        old = d.rename_item.split('|')
        if len(old) != 2:
            raise DatasetError(self, f"Rename item must have format 'col|item', now it is '{d.rename_item}'.")
        col = old[0]
        item = old[1]
        new_item = d.value
        if new_item == '':
            raise DatasetError(self, 'rename_item must have value.')
        # str.replace_all requires Utf8; cast if column is not string (e.g. Categorical)
        series = pl.col(col)
        if df.schema[col] != pl.Utf8:
            series = series.cast(pl.Utf8)
        df = df.with_columns(series.str.replace_all(re.escape(item), new_item))
        return df

    # Similar to Node._process_edge_output
    def _operate_tags(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        context = self.context

        # FIXME Don't let DatasetNodes get double preparation of gpc. Remove when you gte rid of DatasetNodes
        from nodes.gpc import DatasetNode

        tags = self.tags.copy()
        for n in context.nodes.values():
            if any(ds is self for ds in n.input_dataset_instances) and isinstance(n, DatasetNode):
                tags = [tag for tag in tags if tag != 'prepare_gpc_dataset']

        for tag in tags:
            if tag == 'ignore_content':
                logger.warning(f"Dataset {self.id} has tag 'ignore_content', which is not supported.")
            elif df.paths.has_operation(tag):
                df = df.paths.get_operation(tag)(df, context)
        return df

    @measure_dataset_call('dataset.filter', capture_df_result=True, capture_df_arg=True)
    def _filter_and_process_df(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:  # noqa: C901
        from nodes.defs.node_defs import RenameColumnDatasetFilterDef

        if self.filters is not None:
            for filter_def in self.filters:
                assert hasattr(filter_def, 'model_dump')
                if isinstance(filter_def, RenameColumnDatasetFilterDef):
                    df = self._rename_col_filter(df, filter_def)

        cols = list(df.columns)

        if self.column:
            if self.column not in cols:
                available = ', '.join(cols)
                raise DatasetError(
                    self,
                    "Column '%s' not found in dataset '%s'. Available columns: %s"
                    % (
                        self.column,
                        self.id,
                        available,
                    ),
                )
            df = df.with_columns(pl.col(self.column).alias(VALUE_COLUMN))
            cols = [YEAR_COLUMN, VALUE_COLUMN, *df.dim_ids]

        if YEAR_COLUMN in cols and YEAR_COLUMN not in df.primary_keys:
            df = df.add_to_index(YEAR_COLUMN)

        ldf = df.lazy()
        if YEAR_COLUMN in cols and not ldf.filter((pl.col(YEAR_COLUMN) < 200).first()).collect().is_empty():
            baseline_year = self.context.instance.reference_year
            if baseline_year is None:
                raise DatasetError(
                    self,
                    'The reference_year from instance is not given. '
                    + 'It is needed by dataset %s to define the baseline for relative data.' % self.id,
                )
            ldf = ldf.with_columns(
                pl
                .when(pl.col(YEAR_COLUMN) < 90)
                .then(pl.col(YEAR_COLUMN) + pl.lit(baseline_year))
                .otherwise(pl.col(YEAR_COLUMN))
                .alias(YEAR_COLUMN),
            )
            target_year = self.context.instance.target_year
            ldf = ldf.with_columns(
                pl
                .when((pl.col(YEAR_COLUMN) >= 90) & (pl.col(YEAR_COLUMN) < 200))
                .then(pl.col(YEAR_COLUMN) + pl.lit(target_year) - pl.lit(100))
                .otherwise(pl.col(YEAR_COLUMN))
                .alias(YEAR_COLUMN),
            )
            ldf = ldf.with_columns(pl.col(YEAR_COLUMN).cast(int).alias(YEAR_COLUMN))

            # FIXME Duplicates may occur when baseline year overlaps with existing data points.
            ldf = ldf.unique(subset=df.get_meta().primary_keys, keep='last', maintain_order=True)

        if FORECAST_COLUMN in df.columns:
            cols.append(FORECAST_COLUMN)
        elif self.forecast_from is not None:
            ldf = ldf.with_columns(
                pl
                .when(pl.col(YEAR_COLUMN) >= self.forecast_from)
                .then(pl.lit(value=True))
                .otherwise(pl.lit(value=False))
                .alias(FORECAST_COLUMN),
            )
            cols.append(FORECAST_COLUMN)

        ldf = ldf.select(cols)
        cdf = ldf.collect()

        df = ppl.to_ppdf(cdf, meta=df.get_meta().select(cols))
        df = self._filter_df(df)
        df = self._operate_tags(df)
        ppl.validate_ppdf(df)

        df = self._process_output(df)

        return df


type SampleRet = NDArray[np.float64]


class DatasetSampler:
    def __init__(self):
        self.rng = default_rng()

    def loguniform(self, match, size) -> SampleRet:
        low = np.log(float(match.group(1)))
        high = np.log(float(match.group(2)))
        return np.exp(self.rng.uniform(low, high, size))

    def uniform(self, match, size) -> SampleRet:
        low = float(match.group(1))
        high = float(match.group(2))
        return self.rng.uniform(low, high, size)

    def lognormal_plusminus(self, match, size) -> SampleRet:
        mean_lognormal = float(match.group(1))
        std_lognormal = float(match.group(2))
        sigma = np.sqrt(np.log(1 + (std_lognormal**2) / (mean_lognormal**2)))
        mu = np.log(mean_lognormal) - (sigma**2) / 2
        return self.rng.lognormal(mu, sigma, size)

    def normal_plusminus(self, match, size) -> SampleRet:
        loc = float(match.group(1))
        scale = float(match.group(2))
        return self.rng.normal(loc, scale, size)

    def normal_interval(self, match, size) -> SampleRet:
        loc = float(match.group(1))
        lower = float(match.group(2))
        upper = float(match.group(3))
        scale = (upper - lower) / 2 / 1.959963984540054
        return self.rng.normal(loc, scale, size)

    def beta(self, match, size) -> SampleRet:
        a = float(match.group(1))
        b = float(match.group(2))
        return self.rng.beta(a, b, size)

    def poisson(self, match, size) -> SampleRet:
        lam = float(match.group(1))
        return self.rng.poisson(lam, size)

    def exponential(self, match, size) -> SampleRet:
        mean = float(match.group(1))
        return self.rng.exponential(scale=mean, size=size)

    def problist(self, match, size) -> SampleRet:
        s = match.group(1)
        s = [float(x) for x in s.split(',')]
        return self.rng.choice(s, size, replace=True)

    def scalar(self, match, size) -> SampleRet:
        value = float(match.group(1))
        return np.repeat(value, size)

    def get_sample(self, dist_string: str, size: int) -> SampleRet:
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

    def interpret(self, dataset: Dataset, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        size = dataset.context.sample_size
        cols = []
        for col in df.columns:
            # FIXME Invent a generic way to ignore sampling when content is not probabilities
            if col not in [FORECAST_COLUMN, 'Unit', 'UUID', 'muni'] + df.primary_keys and isinstance(df[col].dtype, pl.String):
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
        df = df.join(out, how='inner', on='row_index').drop('row_index')  # type: ignore
        df = ppl.to_ppdf(df, meta=meta)
        df = df.with_columns(pl.col(UNCERTAINTY_COLUMN).cast(pl.Categorical))

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

    @classmethod
    def from_def(cls, ds_def: InputDatasetDef, context: Context) -> Self:
        return cls(
            id=ds_def.id,
            context=context,
            **super().kwargs_from_def(ds_def),
            input_dataset=ds_def.input_dataset,
        )

    def __rich_repr__(self) -> RichReprResult:
        yield from super().__rich_repr__()
        if self.input_dataset is not None and self.input_dataset != self.id:
            yield 'input_dataset', self.input_dataset

    def get_span_attrs(self) -> PerfAttrs:
        attrs = super().get_span_attrs()
        attrs['dataset.input.id'] = self.input_dataset or self.id
        return attrs

    @cached_property
    def cache_key(self) -> str | None:
        return self.get_cache_key()

    def cache_get(self) -> ppl.PathsDataFrame | None:
        if self.context.skip_cache:
            return None
        attrs = self.get_span_attrs()
        if self.cache_key is None:
            return None
        with self.context.perf_context.exec_named(
            kind=PerfKind.DATASET,
            id=self.id,
            op='cache_get',
            attrs=attrs,
        ) as event:
            res = self.context.cache.get(self.cache_key)
            if event is not None:
                event.set_attr('cache.hit', res.is_hit)
                event.set_attr('cache.kind', res.kind.name.lower())
                if res.obj is not None:
                    event.set_attr('dataset.in_memory.bytes', estimate_size_bytes(res.obj))

        if res.is_hit:
            if not isinstance(res.obj, ppl.PathsDataFrame):
                capture_error('Cached dataset %s (key: %s) is not a PathsDataFrame' % (self.id, self.cache_key))
                return None
            return res.obj
        return None

    def cache_set(self, df: ppl.PathsDataFrame) -> None:
        if self.cache_key is None:
            return
        attrs = self.get_span_attrs()
        with self.context.perf_context.exec_named(
            kind=PerfKind.DATASET,
            id=self.id,
            op='cache_set',
            attrs=attrs,
        ):
            self.context.cache.set(self.cache_key, df, expiry=0)

    @measure_dataset_call('dataset.dvc.convert')
    def _convert_dvc_dataset(self, dvc_ds: dvc_pandas.Dataset) -> ppl.PathsDataFrame:
        return ppl.from_dvc_dataset(dvc_ds)

    @override
    def load_internal(self) -> ppl.PathsDataFrame:
        obj = self.cache_get()
        if obj is not None:
            return obj

        if self.input_dataset:
            ds_id = self.input_dataset
        else:
            ds_id = self.id

        dvc_ds = self.context.load_dvc_dataset(ds_id)
        assert dvc_ds.df is not None
        df = self._convert_dvc_dataset(dvc_ds)
        df = self._filter_and_process_df(df)
        df = self.post_process(df)
        if self.context.sample_size > 0:
            df = self.sampler.interpret(self, df)
        if self.cache_key:
            self.cache_set(df)

        return df

    def get_unit(self) -> Unit:
        if self.unit:
            return self.unit
        df = self.load_internal()
        if VALUE_COLUMN in df.columns:
            meta = df.get_meta()
            if VALUE_COLUMN not in meta.units:
                raise DatasetError(self, 'Dataset %s does not have a unit' % self.id)
            return meta.units[VALUE_COLUMN]
        raise DatasetError(self, 'Dataset %s does not have the value column' % self.id)

    def hash_data(self) -> dict[str, Any]:
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
            value = getattr(self, f)
            if f == 'filters' and value is not None:
                value = [item.model_dump() if hasattr(item, 'model_dump') else item for item in value]
            d[f] = value

        if self.context.dataset_repo_spec is not None:
            d['commit_id'] = self.context.dataset_repo_spec.commit
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
            column_name = f'{VALUE_COLUMN}_{unit_str.replace("/", "_per_")}'

            # Create a filtered column with values only where Unit matches
            result = result.with_columns(
                pl.when(pl.col('Unit') == unit_str).then(pl.col(VALUE_COLUMN)).otherwise(None).alias(column_name)
            )

            # Add unit to metadata
            unit = unit_registry.parse_units(unit_str)
            new_units[column_name] = unit

        result = result.drop([VALUE_COLUMN, 'Unit'])
        new_meta = ppl.DataFrameMeta(primary_keys=meta.primary_keys, units=new_units)

        return ppl.to_ppdf(result, meta=new_meta)

    # -----------------------------------------------------------------------------------
    def convert_names_to_ids(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        context = self.context
        exset = {YEAR_COLUMN, VALUE_COLUMN, FORECAST_COLUMN, UNCERTAINTY_COLUMN, 'Unit', 'UUID'}
        exset |= {col for col in df.columns if col.startswith(f'{VALUE_COLUMN}_')}
        exset |= set(df.metric_cols)
        cols = list(set(df.columns) - exset)

        # Convert index level names from labels to IDs.
        collookup = {}
        for col in cols:
            collookup[col] = col.lower().translate(self.characterlookup)
        df = df.rename(collookup)

        for col in cols:
            if col in context.dimensions:
                df = df.with_columns(context.dimensions[col].series_to_ids_pl(df[col]))

        return df

    # -----------------------------------------------------------------------------------
    def drop_unnecessary_levels(self, df: ppl.PathsDataFrame, droplist: list[str]) -> ppl.PathsDataFrame:
        # Get all metric columns from the DataFrame's metadata
        metric_cols = list(df.get_meta().units.keys())

        # Only drop rows where all metric columns are null
        if metric_cols:
            null_condition = pl.lit(True)  # noqa: FBT003
            for col in metric_cols:
                null_condition = null_condition & pl.col(col).is_null()
            df = df.filter(~null_condition)

        # Drop filter levels and empty dimension levels.
        drops = [d for d in droplist if d in df.columns]

        for col in list(set(df.columns) - set(drops)):
            vals = df[col].unique().to_list()
            if vals in [['.'], [None]]:
                drops.append(col)

        df = df.drop(drops)
        return df

    # -----------------------------------------------------------------------------------

    @measure_dataset_call('dataset.transform')
    def _transform_data(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = self.drop_unnecessary_levels(df, droplist=['Description', 'Quantity'])
        df = self.implement_unit_col(df)
        return self.convert_names_to_ids(df)

    @measure_dataset_call('dataset.index')
    def _index_data(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        new_dims = [col for col, dtype in zip(df.columns, df.dtypes, strict=True) if dtype in [pl.Utf8(), pl.Categorical()]]
        return extend_last_historical_value_pl(
            df.add_to_index([dim for dim in new_dims if dim not in df.dim_ids]),
            end_year=self.context.instance.model_end_year,
        )

    @override
    def load_internal(self) -> ppl.PathsDataFrame:
        # Don't call DVCDataset.load directly since it does post_process too early
        # Instead, replicate the parts we need but with different ordering

        cached_df = self.cache_get()
        if cached_df is not None:
            return cached_df

        if self.input_dataset:
            ds_id = self.input_dataset
        else:
            ds_id = self.id

        dvc_ds = self.context.load_dvc_dataset(ds_id)
        assert dvc_ds.df is not None
        df = self._convert_dvc_dataset(dvc_ds)

        # First process data as DVCDataset would, but WITHOUT calling post_process
        df = self._filter_and_process_df(df)

        # Now do GenericDataset specific processing
        df = self._transform_data(df)

        # Only AFTER metric columns exist, handle interpolation
        if FORECAST_COLUMN not in df.columns:
            df = df.with_columns(pl.lit(False).alias(FORECAST_COLUMN))  # noqa: FBT003

        self.interpolate = True  # TODO Do we need this?
        if self.interpolate:
            df = self._linear_interpolate(df)

        df = self._index_data(df)

        # Finalize processing
        if self.context.sample_size > 0:
            df = self._sample(df)
        self.cache_set(df)

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
        pdf = self.post_process(pdf)

        self.df = pdf

    @override
    def load_internal(self) -> ppl.PathsDataFrame:
        df = self.df
        assert df is not None
        if self.context.sample_size > 0:
            df = self._sample(df)
        self.df = df
        return self.df

    def hash_data(self) -> dict[str, Any]:
        assert self.df is not None
        df = self.df.to_pandas()
        return dict(hash=int(self.pd.util.hash_pandas_object(df).sum()), sample_size=self.context.sample_size)

    def get_unit(self) -> Unit:
        assert self.unit is not None
        return self.unit


@dataclass
class JSONDataset(Dataset):
    data: dict[str, Any]
    unit: Unit | None
    df: ppl.PathsDataFrame = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.df = JSONDataset.deserialize_df(self.data)
        meta = self.df.get_meta()
        if len(meta.units) == 1:
            self.unit = next(iter(meta.units.values()))

    @override
    def load_internal(self) -> ppl.PathsDataFrame:
        assert self.df is not None
        return self.post_process(self.df)

    def hash_data(self) -> dict[str, Any]:
        import pandas as pd

        df = self.df.to_pandas()
        return dict(hash=int(pd.util.hash_pandas_object(df).sum()))

    def get_unit(self) -> Unit:
        return cast('Unit', self.unit)

    @classmethod
    def deserialize_df(cls, value: dict[str, Any]) -> ppl.PathsDataFrame:
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
    def serialize_df(cls, pdf: ppl.PathsDataFrame, add_uuids: bool = False) -> dict[str, Any]:
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
        if self.db_dataset_obj is None:
            from kausal_common.datasets.models import Dataset as DBDatasetModel

            assert self.db_dataset_id is not None
            self.db_dataset_obj = DBDatasetModel.objects.get(uuid=self.db_dataset_id)

    @classmethod
    def from_def(cls, ds_def: InputDatasetDef, context: Context, db_dataset_obj: DBDatasetModel) -> Self:
        return cls(
            id=ds_def.id,
            context=context,
            **super().kwargs_from_def(ds_def),
            db_dataset_obj=db_dataset_obj,
        )

    @override
    def load_internal(self) -> ppl.PathsDataFrame:
        if self.df is not None:
            return self.df

        ds_obj = self.db_dataset_obj
        if ds_obj is None:
            raise Exception('Admin dataset not loaded')
        df = self.deserialize_df(ds_obj)
        df = self._filter_and_process_df(df)
        df = self.post_process(df)
        self.df = df
        return df

    def hash_data(self) -> dict[str, Any]:
        obj = self.db_dataset_obj
        assert obj is not None
        return dict(obj_pk=obj.pk, updated_at=str(obj.last_modified_at))

    def get_unit(self) -> Unit:
        df = self.load_internal()
        meta = df.get_meta()
        if len(meta.units) == 1:
            return next(iter(meta.units.values()))
        raise Exception('Dataset %s does not have a single unit' % self.id)

    @classmethod
    def deserialize_df(cls, ds_in: DBDatasetModel, *, include_data_point_primary_keys: bool = False) -> ppl.PathsDataFrame:
        from django.contrib.postgres.expressions import ArraySubquery
        from django.db.models.expressions import F, OuterRef
        from django.db.models.fields import CharField
        from django.db.models.functions import Cast, Coalesce, JSONObject

        from kausal_common.datasets.models import (
            DataPoint,
            Dataset as DBDatasetModel,
            DatasetMetric,
            DatasetSchemaDimension,
            DimensionCategory,
        )

        # dim_cats = DimensionCategory.objects.filter(data_points=OuterRef('pk')).values(
        #     json=JSONObject(
        #         dim_id=Coalesce(F('dimension__identifier'), Cast('dimension__uuid', output_field=CharField())),
        #         cat_id=Coalesce(F('identifier'), Cast('uuid', output_field=CharField())),
        #     )
        # )

        dims = (
            DatasetSchemaDimension.objects
            .filter(schema=ds_in.schema)
            .annotate(
                dim_uuid=F('dimension__uuid'),
                dim_id=Coalesce(
                    F('column_name'),
                    F('dimension__scopes__identifier'),
                    Cast('dimension__uuid', output_field=CharField()),
                ),
            )
            .order_by('id')
            .distinct('id')
            .values_list('dim_uuid', 'dim_id')
        )
        dim_anns = {
            str(dim[1]): DimensionCategory.objects
            .filter(dimension__uuid=dim[0])
            .filter(data_points=OuterRef('pk'))
            .annotate(cat_id=Coalesce(F('identifier'), Cast('uuid', output_field=CharField())))
            .values_list('cat_id', flat=True)
            for dim in dims
        }

        dps = (
            DataPoint.objects
            .filter(dataset=OuterRef('pk'))
            .order_by()
            .distinct('id')
            .values(
                json=JSONObject(
                    id=F('id'),
                    **{YEAR_COLUMN: F('date__year')},
                    value=F('value'),
                    metric=F('metric__uuid'),
                    # dim_cats=ArraySubquery(dim_cats),
                    **dim_anns,
                ),
            )
        )

        metrics = DatasetMetric.objects.filter(schema=OuterRef('schema')).values(
            json=JSONObject(
                uuid=F('uuid'),
                name=Coalesce(F('name'), F('label'), Cast('uuid', output_field=CharField())),
                unit=F('unit'),
            )
        )
        ds = DBDatasetModel.objects.filter(id=ds_in.pk).annotate(dps=ArraySubquery(dps), metrics=ArraySubquery(metrics)).first()
        assert ds is not None
        dp_list = cast('list[dict[str, Any]]', ds.dps)  # type: ignore
        df_schema = {
            'id': pl.Int64,
            YEAR_COLUMN: pl.Int64,
            'value': pl.Float64,
            'metric': pl.Utf8,
            **{str(dim[1]): pl.Utf8 for dim in dims},
        }
        df = pl.DataFrame(dp_list, schema=df_schema, orient='row')
        mdf = pl.DataFrame(ds.metrics)  # type: ignore
        df = df.join(mdf.select(pl.col('uuid').alias('metric'), pl.col('name').alias('metric_name')), on='metric', how='left')

        dim_ids = [str(dim[1]) for dim in dims]

        id_map = None
        if include_data_point_primary_keys:
            id_map = df.select([YEAR_COLUMN, *dim_ids, 'metric_name', 'id'])

        index_cols = [YEAR_COLUMN, *dim_ids]
        uniq_cols = [*index_cols, 'metric']
        df = df.with_columns(pl.col('metric_name').alias('metric')).drop('metric_name', 'id').sort(uniq_cols)

        dupes = df.group_by(uniq_cols).agg(pl.count().alias('_count')).filter(pl.col('_count') > 1)
        if len(dupes) > 0:
            extra = dupes.head().to_dicts()
            capture_error(
                'Dataset %s (pk %d) has %s duplicate rows' % (ds_in.identifier, ds_in.pk, len(dupes)),
                extras={'example_rows': extra},
            )
            # Filter out duplicate rows, keeping the first one
            df = df.group_by(uniq_cols).first()

        df = df.pivot(on='metric', index=[YEAR_COLUMN, *dim_ids], values='value')  # noqa: PD010

        if include_data_point_primary_keys and id_map is not None:
            id_pivoted = id_map.pivot(on='metric_name', on_columns=[YEAR_COLUMN, *dim_ids], values='id')  # noqa: PD010
            id_pivoted = id_pivoted.rename({
                col: f'_dp_pk_{col}' for col in id_pivoted.columns if col not in [YEAR_COLUMN, *dim_ids]
            })
            df = df.join(id_pivoted, on=[YEAR_COLUMN, *dim_ids], how='left', nulls_equal=True)

        if dim_ids:
            df = df.with_columns([pl.col(dim_id).cast(pl.Categorical) for dim_id in dim_ids])

        meta = ppl.DataFrameMeta(
            units={
                m['name']: unit_registry.parse_units(m['unit'])
                for m in ds.metrics  # type: ignore
            },
            primary_keys=[YEAR_COLUMN, *dim_ids],
        )

        pdf = ppl.to_ppdf(df, meta)

        return pdf
