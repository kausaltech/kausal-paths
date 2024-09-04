from __future__ import annotations

import pickle
import warnings
from collections import OrderedDict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Literal, Tuple, TypeVar, Union, cast

import loguru
import pandas as pd
import pint_pandas
import polars as pl
import redis

from kausal_common.debugging.perf import PerfCounter

from common import base32_crockford, polars as ppl
from nodes.perf import PerfStats

if TYPE_CHECKING:
    from types import TracebackType

    from nodes.units import CachingUnitRegistry as UnitRegistry


class PickledPintDataFrame:
    df: pd.DataFrame
    units: dict[str, str]

    def __init__(self, df, units):
        self.df = df
        self.units = units

    def to_df(self, ureg: UnitRegistry) -> pd.DataFrame:
        df = self.df
        for col, unit in self.units.items():
            pt = pint_pandas.PintType(ureg.parse_units(unit))
            df[col] = df[col].astype(pt)
        return df

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Union[pd.DataFrame, PickledPintDataFrame]:
        units = {}
        df = df.copy()
        for col in df.columns:
            if not hasattr(df[col], 'pint'):
                continue
            unit = df[col].pint.units
            units[col] = str(unit)
            df[col] = df[col].pint.m
        if not units:
            return df
        return PickledPintDataFrame(df, units)


class PickledPathsDataFrame:
    df: pl.DataFrame
    units: dict[str, str]
    primary_keys: list[str]

    def __init__(self, df, units, primary_keys):
        self.df = df
        self.units = units
        self.primary_keys = primary_keys

    def to_df(self, ureg: UnitRegistry) -> ppl.PathsDataFrame:
        df = self.df
        units = {}
        for col, unit in self.units.items():
            units[col] = ureg.parse_units(unit)
        meta = ppl.DataFrameMeta(units, self.primary_keys)
        return ppl.to_ppdf(df, meta=meta)

    @classmethod
    def from_df(cls, df: ppl.PathsDataFrame) -> PickledPathsDataFrame:
        meta = df.get_meta()
        pldf = pl.DataFrame._from_pydf(df._df)
        units = {col: str(unit) for col, unit in meta.units.items()}
        return cls(pldf, units, meta.primary_keys)


class LocalLRUCache:
    cache: OrderedDict[str, bytes]

    def __init__(self, max_size: int, log: loguru.Logger):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.current_size = 0
        self.discarded = 0
        self.log = log

    # we return the value of the key
    # that is queried in O(1) and return -1 if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key: str) -> bytes | None:
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    # first, we add / update the key by conventional methods.
    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key: str, value: bytes) -> None:
        value_len = len(value)
        if value_len > self.max_size:
            self.log.warning('Discarding object of %d KiB: %s' % (value_len // 1024, key))
            return
        self.cache[key] = value
        self.cache.move_to_end(key)
        self.current_size += len(value)
        while self.current_size > self.max_size:
            _, item = self.cache.popitem(last=False)
            nr = len(item)
            self.current_size -= nr
            self.discarded += nr
        assert self.current_size >= 0

    def clear(self):
        self.cache = OrderedDict()
        self.current_size = 0


class CacheKind(Enum):
    RUN = 1
    LOCAL = 2
    EXT = 3

    @cached_property
    def color(self):
        if self == self.RUN:
            return 'green'
        if self == self.LOCAL:
            return 'blue'
        return 'yellow'

CR = TypeVar('CR')

@dataclass(slots=True)
class CacheResult(Generic[CR]):
    is_hit: bool
    kind: CacheKind
    obj: CR

    @property
    def color(self) -> str:
        if not self.is_hit:
            if self.kind == CacheKind.EXT:
                return 'bold red'
            return 'red'
        if self.kind == CacheKind.RUN:
            return 'bright_green'
        if self.kind == CacheKind.LOCAL:
            return 'green'
        return 'blue'


class CacheRun:
    cache: Cache
    store: dict[str, Any]
    new_objs: list[tuple[str, Any]]
    stats: PerfStats
    ext_stats: PerfStats
    local_stats: PerfStats

    def __init__(self, cache: Cache):
        self.cache = cache
        self.store = {}
        self.new_objs = []
        self.obj_id = base32_crockford.gen_obj_id(self)
        self.stats = PerfStats()
        self.ext_stats = PerfStats()
        self.local_stats = PerfStats()
        self.cache.log.debug('Start execution run')

    def end(self):
        comp_time = self.cache.pc.measure()
        nr_new_objs = len(self.new_objs)
        if self.new_objs:
            self.cache.store_ext(self.new_objs)
            self.new_objs = []

        if comp_time < 50:
            style = 'green'
        elif comp_time < 200:
            style = 'yellow'
        elif comp_time < 1000:
            style = 'magenta'
        else:
            style = 'bold red'
        self.cache.log.info(
            'End execution run ([{style}]{comp_time:.2f}[/] ms, {nr_reqs} reqs ({nr_hits} hits, {nr_misses} misses); '
            'caching {nr_new_objs} new objects took {dump_time:.2f} ms)',
            style=style, comp_time=comp_time,
            nr_reqs=self.stats.nr_calls, nr_hits=self.stats.cache_hits,
            nr_misses=self.stats.cache_misses, nr_new_objs=nr_new_objs,
            dump_time=self.cache.pc.measure(),
        )
        if self.local_stats.nr_calls:
            self.cache.log.debug(
                'Cache stats:\nLocal cache: {local}\nExt cache: {ext}\n'
                'Objs in LRU {lru_count}, KiB in LRU: {lru_size}, discarded KiB: {lru_discarded}',
                local=repr(self.local_stats), ext=repr(self.ext_stats), lru_count=len(self.cache.local.cache),
                lru_size=self.cache.local.current_size // 1024, lru_discarded=self.cache.local.discarded // 1024,
            )

    def __del__(self):
        self.cache.log.debug('Run destroyed')

    def get(self, key: str) -> object | None:
        obj = self.store.get(key)
        if obj is None:
            self.stats.cache_misses += 1
            return None
        self.stats.cache_hits += 1
        return obj

    def add(self, key: str, obj: Any):  # noqa: ANN401
        assert key not in self.store
        self.store[key] = obj

    def add_ext(self, key: str, obj: Any):  # noqa: ANN401
        self.new_objs.append((key, obj))


LOCAL = 1
EXT = 2

REQ = 1
HIT = 2
MISS = 3


class Cache(AbstractContextManager):
    client: redis.Redis | None
    prefix: str

    local: LocalLRUCache
    run: CacheRun | None

    def __init__(self, ureg: UnitRegistry, redis_url: str | None = None, base_logger: loguru.Logger | None = None):
        if redis_url:
            self.client = redis.Redis.from_url(redis_url)
        else:
            self.client = None
        self.prefix = 'kausal-paths-model'
        self.timeout = 60 * 60
        self.ureg = ureg
        self.run = None
        self.log = base_logger or loguru.logger.bind(markup=True)
        self.obj_id = base32_crockford.gen_obj_id(self)
        self.pc = PerfCounter(f'cache {self.obj_id}')
        self.local = LocalLRUCache(32 * 1024 * 1024, self.log)
        redis_str = ''
        if self.client is None:
            redis_str = ', [warning]not using external cache[/]'
        else:
            redis_str = ' using Redis at [repr.url]%s[/]' % redis_url
        self.log.debug('Cache initialized{redis_state}', redis_state=redis_str)

    def __del__(self):
        #self.log.debug('<{}> Cache destroyed', self.obj_id)
        pass

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, __exc_type: type[BaseException] | None, __exc_value: BaseException | None, __traceback: TracebackType | None) -> bool | None:
        if __exc_type is not None:
            if self.run:
                self.run.new_objs = []
        self.end_run()
        return None

    def start_run(self):
        self.pc.measure()
        self.run = CacheRun(self)

    def end_run(self):
        if self.run is None:
            self.log.error('<{}> end_run() called while no run was active', self.obj_id)
            return
        self.run.end()
        self.run = None

    def store_ext(self, objs: list[Tuple[str, Any]]):
        if not self.client:
            return

        pipe = self.client.pipeline(transaction=False)

        for key, obj in objs:
            data = self.serialize_object(obj)
            pipe.set(key, data, ex=self.timeout)
            self.local.put(key, data)
        pipe.execute()

    def serialize_object(self, obj: Any) -> bytes:
        if isinstance(obj, ppl.PathsDataFrame):
            obj = PickledPathsDataFrame.from_df(obj)
        elif isinstance(obj, pd.DataFrame) and hasattr(obj, 'pint'):
            obj = PickledPintDataFrame.from_df(obj)
        data = pickle.dumps(obj)
        return data

    def deserialize_object(self, data: bytes) -> object:
        obj = pickle.loads(data)  # noqa: S301
        if isinstance(obj, PickledPathsDataFrame):
            return obj.to_df(self.ureg)
        if isinstance(obj, PickledPintDataFrame):
            return obj.to_df(self.ureg)
        return obj

    def get(self, key: str) -> CacheResult:
        full_key = '%s:%s' % (self.prefix, key)
        obj = None

        def record_event(cache: CacheKind, event: Literal['req', 'hit', 'miss']):
            if not self.run:
                return
            if cache == CacheKind.LOCAL:
                stats = self.run.local_stats
            else:
                stats = self.run.ext_stats
            if event == 'req':
                stats.nr_calls += 1
            elif event == 'hit':
                stats.cache_hits += 1
            elif event == 'miss':
                stats.cache_misses += 1

        if self.run is not None:
            obj = self.run.get(full_key)
            if obj is not None:
                if isinstance(obj, (pd.DataFrame, ppl.PathsDataFrame)):
                    obj = obj.copy()
                return CacheResult(True, CacheKind.RUN, obj)

        kind = CacheKind.LOCAL
        record_event(kind, 'req')
        data = self.local.get(full_key)
        if data is not None:
            obj = self.deserialize_object(data)
            record_event(kind, 'hit')
            if self.run is not None:
                self.run.add(full_key, obj)
            return CacheResult(True, kind, obj)

        record_event(kind, 'miss')

        if not self.client:
            return CacheResult(False, kind, None)

        record_event(CacheKind.EXT, 'req')
        resp = self.client.getex(full_key, ex=self.timeout)
        data = cast(bytes | None, resp)
        if data is None:
            record_event(CacheKind.EXT, 'miss')
            return CacheResult(False, CacheKind.EXT, None)
        record_event(CacheKind.EXT, 'hit')
        self.local.put(full_key, data)
        obj = self.deserialize_object(data)
        if self.run is not None:
            self.run.add(full_key, obj)
        return CacheResult(True, CacheKind.EXT, obj)

    def set(self, key: str, obj: object, no_expiry: bool = False):
        full_key = '%s:%s' % (self.prefix, key)
        if self.run:
            self.run.add(full_key, obj)
            if isinstance(obj, pl.DataFrame):
                s = obj.estimated_size()
                if s > self.local.max_size:
                    warnings.warn('Discarding a large object of %d KiB: %s' % (s // 1024, key), stacklevel=3)
                elif s > 128 * 1024:
                    self.log.warning('Attempting to cache a large object of %d KiB: %s' % (s // 1024, key))
            self.run.add_ext(full_key, obj)
        else:
            data = self.serialize_object(obj)
            if self.client:
                if no_expiry:
                    self.client.set(full_key, value=data)
                else:
                    self.client.setex(full_key, time=self.timeout, value=data)
            self.local.put(full_key, data)

    def clear(self):
        self.log.info('Clearing cache')
        if self.client:
            self.client.flushall()
        self.local.clear()
