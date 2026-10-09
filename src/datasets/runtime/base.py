"""The runtime dataset interface the computation reads its inputs through."""

import hashlib
import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Concatenate, Literal, TypedDict

import numpy as np
import orjson
import polars as pl
from numpy.random import default_rng  # TODO Could call Generator to give hints about rng attributes but requires code change
from numpy.typing import NDArray

from kausal_common.deployment import get_deployment_build_id
from kausal_common.perf.perf_context import PerfKind, estimate_size_bytes

from common import polars as ppl, qualifiers
from nodes.constants import (
    FORECAST_COLUMN,
    UNCERTAINTY_COLUMN,
)
from nodes.defs.transform_def import (
    TEMPORAL_FILL_KINDS,
    PortTransformOp,
    forecast_from_transformations,
    unit_from_transformations,
    with_forecast_from,
)
from nodes.transforms import PipelineEnv, QualifierSource, apply_port_transformations

if TYPE_CHECKING:
    from rich.repr import RichReprResult

    from kausal_common.perf.perf_context import PerfAttrs, PerfSpanEntry

    from datasets.prepared import PreparedRecipe
    from nodes.context import Context
    from nodes.defs.node_defs import InputDatasetDef
    from nodes.units import Unit


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
    transformations: list[PortTransformOp]


@dataclass
class Dataset(ABC):
    _class_hash: ClassVar[bytes | None] = None

    id: str
    context: Context
    _: KW_ONLY
    tags: list[str] = field(default_factory=list)
    transformations: list[PortTransformOp] = field(default_factory=list)
    """The binding's complete ordered transformation recipe."""
    qualifier_source: QualifierSource | None = None
    """Where the selected values get their qualifiers; None for a dataset that says nothing about them."""
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
            transformations=ds_def.to_transformations(),
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
        d = {
            'id': self.id,
            'qualifier_version': qualifiers.QUALIFIER_VERSION,
            'transformations': [op.cache_hash_data(self.context) for op in self.transformations],
        }
        if build_id := get_deployment_build_id():
            d['build_id'] = build_id
        else:
            # Development workers share file mtimes, unlike a process-local namespace.
            # Transformation implementations are versioned by their operation classes.
            d['class_hash'] = class_hash.hex()
        d.update(self.hash_data())
        h = hashlib.md5(orjson.dumps(d, option=orjson.OPT_SORT_KEYS), usedforsecurity=False).digest()
        # Keep the hash only while nothing in it can change. A transformation that reads a
        # parameter (a `filter_column` with `ref`) hashes the parameter's current value, and
        # keeping the first hash served the frame for the first value after it changed.
        if not self.referenced_parameters():
            self.hash = h
        return h

    def referenced_parameters(self) -> list[str]:
        """Ids of the parameters this binding's transformations read."""
        return [param_id for op in self.transformations for param_id in op.referenced_parameters()]

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

    def dvc_source_id(self) -> str | None:
        """Identify an external DVC source, if this binding uses one."""
        return None

    def prepared_recipe(self) -> PreparedRecipe | None:
        """Describe the shareable source prefix, if this source supports one."""
        return None

    def post_process(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        """Compatibility entry point for datasets whose whole recipe runs after loading."""
        return self.after_transformations(self.apply_transformations(df))

    def after_transformations(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        """Subclass hook for source overlays that run after the binding recipe."""
        return df

    def before_temporal_fill(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        """Subclass hook for source overlays that need raw join keys temporal filling may remove."""
        return df

    def transformation_groups_at_temporal_fill(self) -> tuple[list[PortTransformOp], list[PortTransformOp]]:
        """Split the recipe before its first temporal fill operation without reordering it."""
        first_temporal = next(
            (index for index, op in enumerate(self.transformations) if op.kind in TEMPORAL_FILL_KINDS),
            len(self.transformations),
        )
        return self.transformations[:first_temporal], self.transformations[first_temporal:]

    def apply_transformations(
        self,
        df: ppl.PathsDataFrame,
        transformations: list[PortTransformOp] | None = None,
        *,
        metric_column: str | None = None,
    ) -> ppl.PathsDataFrame:
        """Execute a transformation list with this dataset as its source environment."""
        env = PipelineEnv(context=self.context, dataset=self, metric_column=metric_column)
        return apply_port_transformations(df, self.transformations if transformations is None else transformations, env)

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


class FilterDatasetKwargs(DatasetKwargs):
    column: str | None
    unit: Unit | None
    forecast_from: int | None


@dataclass
class DatasetWithFilters(Dataset, ABC):
    column: str | None = None
    """
    The metric column this binding selects, or None when it consumes the frame whole.

    Read by the ``select_metric`` operation, which says *where* in the pipeline
    the selection happens.
    """

    unit: Unit | None = None

    # The year from which the time series becomes a forecast
    forecast_from: int | None = None

    @classmethod
    def kwargs_from_def(cls, ds_def: InputDatasetDef) -> FilterDatasetKwargs:
        # A YAML-authored definition carries the legacy flat fields, a DB-backed
        # one carries the pipeline directly. Converting here means there is one
        # execution path, whichever the config source was.
        kwargs = super().kwargs_from_def(ds_def)
        return FilterDatasetKwargs(
            **kwargs,
            column=ds_def.column,
            unit=ds_def.unit if ds_def.transformations is None else unit_from_transformations(kwargs['transformations']),
            forecast_from=(
                ds_def.forecast_from
                if ds_def.transformations is None
                else forecast_from_transformations(kwargs['transformations'])
            ),
        )

    @staticmethod
    def apply_forecast_defaults(kwargs: FilterDatasetKwargs, context: Context, dataset_default: int | None = None) -> None:
        """
        Give a binding that does not say where the forecast begins the year from its dataset or instance.

        The binding's own year wins, then the dataset's, then — when the
        instance opts in with ``forecast_after_maximum_historical_year`` — the
        year after the instance's last historical year: the model's history
        ends there, whatever the data says. A ``Forecast`` column the data
        carries itself still wins over all of them (see ``set_forecast_from``).
        """
        if kwargs['forecast_from'] is not None:
            return
        year = dataset_default
        instance = context.instance
        last_historical = instance.maximum_historical_year
        if year is None and instance.features.forecast_after_maximum_historical_year and last_historical is not None:
            year = last_historical + 1
        if year is None:
            return
        kwargs['forecast_from'] = year
        kwargs['transformations'] = with_forecast_from(kwargs['transformations'], year)

    def __rich_repr__(self) -> RichReprResult:
        yield from super().__rich_repr__()
        if self.column is not None:
            yield 'column', self.column
        if self.transformations:
            yield 'transformations', len(self.transformations)

    def pipeline_hash_data(self) -> dict[str, Any]:
        """Return the complete recipe and runtime inputs used to materialize this binding."""
        data: dict[str, Any] = {
            'column': self.column,
            'unit': str(self.unit) if self.unit is not None else None,
            'forecast_from': self.forecast_from,
            'transformations': [op.cache_hash_data(self.context) for op in self.transformations],
        }
        # Only when set, so the cache keys of datasets without qualifiers stay as they were.
        if self.qualifier_source is not None:
            data['qualifiers'] = self.qualifier_source.hash_data()
        return data

    @measure_dataset_call('dataset.filter', capture_df_result=True, capture_df_arg=True)
    def _filter_and_process_df(self, df: ppl.PathsDataFrame, *, prepared_prefix_length: int = 0) -> ppl.PathsDataFrame:
        """Run the binding's transform pipeline over a freshly loaded frame."""
        before_temporal, from_temporal = self.transformation_groups_at_temporal_fill()
        df = self.apply_transformations(df, before_temporal[prepared_prefix_length:], metric_column=self.column)
        df = self.before_temporal_fill(df)
        df = self.apply_transformations(df, from_temporal, metric_column=self.column)
        ppl.validate_ppdf(df)
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
