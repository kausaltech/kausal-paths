from __future__ import annotations

import pickle
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property, wraps
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Concatenate, Literal, Self, cast

import loguru
import polars as pl
import redis
from redis import client as redis_client

from kausal_common.debugging.perf import PerfCounter
from kausal_common.perf.perf_context import PerfStats

from common import base32_crockford, polars as ppl

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from pandas import DataFrame as PandasDataFrame

    from nodes.units import CachingUnitRegistry as UnitRegistry


class PickledPintDataFrame:
    df: PandasDataFrame
    units: dict[str, str]

    def __init__(self, df, units):
        self.df = df
        self.units = units

    def to_df(self, ureg: UnitRegistry) -> PandasDataFrame:
        import pint_pandas

        df = self.df
        for col, unit in self.units.items():
            pt = pint_pandas.PintType(ureg.parse_units(unit))
            df[col] = df[col].astype(pt)
        return df

    @classmethod
    def from_df(cls, df: PandasDataFrame) -> PandasDataFrame | PickledPintDataFrame:
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
    explanation: list[str]

    def __init__(self, df, units, primary_keys, explanation):
        self.df = df
        self.units = units
        self.primary_keys = primary_keys
        self.explanation = explanation

    def to_df(self, ureg: UnitRegistry) -> ppl.PathsDataFrame:
        df = self.df
        units = {}
        for col, unit in self.units.items():
            units[col] = ureg.parse_units(unit)
        meta = ppl.DataFrameMeta(units, self.primary_keys)
        pdf = ppl.to_ppdf(df, meta=meta)
        # Restore explanations that were preserved through caching
        pdf._explanation = self.explanation.copy()
        return pdf

    @classmethod
    def from_df(cls, df: ppl.PathsDataFrame) -> PickledPathsDataFrame:
        meta = df.get_meta()
        pldf = pl.DataFrame._from_pydf(df._df)
        units = {col: str(unit) for col, unit in meta.units.items()}
        # Preserve explanations when serializing for cache
        explanation = df._explanation.copy() if hasattr(df, '_explanation') else []
        return cls(pldf, units, meta.primary_keys, explanation)


type LRUFuncT[**P, R, SC: LocalLRUCache] = Callable[Concatenate[SC, P], R]

class LocalLRUCache:
    """
    A Least Recently Used (LRU) cache implementation with a maximum size limit.

    This cache stores key-value pairs where the values are bytes objects. It maintains
    the order of access, removing the least recently used items when the size limit is exceeded.
    """

    cache: OrderedDict[str, bytes]

    @staticmethod
    def _lock[**P, R, SC: LocalLRUCache](method: LRUFuncT[P, R, SC]) -> LRUFuncT[P, R, SC]:
        """Make a method thread-safe using the instance's lock."""

        @wraps(method)
        def wrapper(self: SC, *args, **kwargs) -> Any:
            _rich_traceback_omit = True
            with self.lock:
                return method(self, *args, **kwargs)
        return cast('LRUFuncT[P, R, SC]', wrapper)

    def __init__(self, max_size: int, log: loguru.Logger):
        """
        Initialize the LocalLRUCache.

        Args:
            max_size (int): The maximum allowed size of the cache in bytes.
            log (loguru.Logger): Logger for outputting messages.

        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.current_size = 0
        self.discarded = 0
        self.log = log
        self.lock = Lock()

    @_lock
    def get(self, key: str) -> bytes | None:
        """
        Retrieve a value from the cache.

        If the key exists, it's moved to the end of the cache to indicate recent use.

        Args:
            key (str): The key to look up in the cache.

        Returns:
            bytes | None: The value associated with the key if found, None otherwise.

        """
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def has(self, key: str) -> bool:
        return key in self.cache

    @_lock
    def put(self, key: str, value: bytes) -> None:
        """
        Add or update a key-value pair in the cache.

        If adding the new item would exceed the maximum size, the least recently used
        items are removed until there's enough space. If the new item itself exceeds
        the maximum size, it's not added and a warning is logged.

        Args:
            key (str): The key to add or update.
            value (bytes): The value to store.

        """
        value_len = len(value)
        if value_len > self.max_size:
            self.log.warning('Discarding object of %d KiB: %s' % (value_len // 1024, key))
            return

        # Remove the old value if the key already exists
        if key in self.cache:
            old_value = self.cache[key]
            self.current_size -= len(old_value)

        self.cache[key] = value
        self.cache.move_to_end(key)
        self.current_size += len(value)
        while self.current_size > self.max_size:
            _, item = self.cache.popitem(last=False)
            nr = len(item)
            self.current_size -= nr
            self.discarded += nr

        assert self.current_size >= 0

    @_lock
    def clear(self):
        """Clear all items from the cache and reset the current size to zero."""
        self.cache = OrderedDict()
        self.current_size = 0
        self.discarded = 0


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


type CacheEvent = Literal['req', 'hit', 'miss']


@dataclass(slots=True)
class CacheResult[ObjT]:
    key: str
    is_hit: bool = False
    kind: CacheKind = CacheKind.RUN
    obj: ObjT | None = None
    key_prefix: str | None = None

    @property
    def full_key(self) -> str:
        if self.key_prefix:
            return '%s:%s' % (self.key_prefix, self.key)
        return self.key

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


@dataclass
class CacheRun:
    cache: Cache
    store: dict[str, Any] = field(init=False, default_factory=dict)
    new_objs: list[tuple[str, int | None, bytes | object]] = field(init=False, default_factory=list)
    nr_calls: int = field(init=False, default=0)
    nr_hits: int = field(init=False, default=0)
    stats: PerfStats = field(init=False, default_factory=PerfStats)
    run_stats: PerfStats = field(init=False, default_factory=PerfStats)
    ext_stats: PerfStats = field(init=False, default_factory=PerfStats)
    local_stats: PerfStats = field(init=False, default_factory=PerfStats)
    nr_new_objs: int = field(init=False, default=0)
    nr_new_bytes: int = field(init=False, default=0)
    flush_time: float = field(init=False, default=0.0)
    obj_id: str = field(init=False)

    def __post_init__(self):
        self.obj_id = base32_crockford.gen_obj_id(self)
        self.cache.log.debug('Start execution run')

    def flush(self):
        nr_new = len(self.new_objs)
        if not nr_new:
            return
        pc = PerfCounter(level=PerfCounter.Level.VERBOSE_DEBUG)
        self.nr_new_objs += len(self.new_objs)
        nr_bytes = self.cache.store_multiple(self.new_objs)
        self.nr_new_bytes += nr_bytes
        self.new_objs = []
        self.flush_time += pc.measure()

    def end(self):
        comp_time = self.cache.pc.measure()
        self.flush()
        if comp_time < 50:
            style = 'green'
        elif comp_time < 200:
            style = 'yellow'
        elif comp_time < 1000:
            style = 'magenta'
        else:
            style = 'bold red'

        self.cache.log.info(
            'End execution run ([%(style)s]{comp_time:.2f}[/] ms, {nr_reqs} reqs ({nr_hits} hits, {nr_misses} misses); '
            'caching {nr_new_objs} new objects ({nr_new_bytes} MiB) took {dump_time:.2f} ms)' % dict(style=style),
            comp_time=comp_time,
            nr_reqs=self.stats.nr_calls,
            nr_hits=self.stats.cache_hits,
            nr_misses=self.stats.cache_misses,
            nr_new_objs=self.nr_new_objs,
            nr_new_bytes=self.nr_new_bytes // (1024 * 1024),
            dump_time=self.flush_time,
        )
        if self.local_stats.nr_calls or self.ext_stats.nr_calls:
            self.cache.log.debug(
                'Cache stats:\nRun cache: {run}\nLocal cache: {local}\nExt cache: {ext}\n'
                'Objs in LRU {lru_count}, KiB in LRU: {lru_size}, discarded KiB: {lru_discarded}',
                run=repr(self.run_stats),
                local=repr(self.local_stats),
                ext=repr(self.ext_stats),
                lru_count=len(self.cache.local.cache),
                lru_size=self.cache.local.current_size // 1024,
                lru_discarded=self.cache.local.discarded // 1024,
            )

    def get(self, key: str) -> object | None:
        obj = self.store.get(key)
        if obj is None:
            return None
        return obj

    def has(self, key: str) -> bool:
        return key in self.store

    def add(self, key: str, obj: object):
        if key in self.store:
            return
        if isinstance(obj, ppl.PathsDataFrame | self.cache.PandasDataFrame):
            obj = obj.copy()
        self.store[key] = obj

    def add_ext(self, key: str, expiry: int | None, obj: object | bytes):
        self.new_objs.append((key, expiry, obj))

    def record_event(self, cache: CacheKind | None, event: CacheEvent):
        if cache == CacheKind.RUN:
            stats = self.ext_stats
        elif cache == CacheKind.LOCAL:
            stats = self.local_stats
        elif cache == CacheKind.EXT:
            stats = self.ext_stats
        else:
            stats = self.stats

        if event == 'req':
            stats.nr_calls += 1
        elif event == 'hit':
            stats.cache_hits += 1
        elif event == 'miss':
            stats.cache_misses += 1


LOCAL = 1
EXT = 2

REQ = 1
HIT = 2
MISS = 3


class ExtCacheBatch(ABC):
    @abstractmethod
    def add(self, key: str, data: bytes, expiry: int | None = None) -> None: ...

    @abstractmethod
    def send(self): ...


class ExtCache(ABC):
    @abstractmethod
    def get(self, key: str, expiry: int | None = None) -> bytes | None: ...

    @abstractmethod
    def get_many(self, keys: Sequence[str], expiry: int | None = None) -> dict[str, bytes | None]: ...

    @abstractmethod
    def set(self, key: str, data: bytes, expiry: int | None = None) -> None: ...

    @abstractmethod
    def clear(self): ...

    @abstractmethod
    def start_batch(self) -> ExtCacheBatch: ...


class RedisCacheBatch(ExtCacheBatch):
    client: redis.Redis
    pipe: redis_client.Pipeline

    def __init__(self, client: redis.Redis):
        self.client = client
        self.pipe = client.pipeline(transaction=False)

    def add(self, key: str, data: bytes, expiry: int | None = None) -> None:
        self.pipe.set(key, data, ex=expiry)

    def send(self):
        self.pipe.execute()


class RedisCache(ExtCache):
    def __init__(self, client: redis.Redis):
        self.client = client
        super().__init__()

    def get(self, key: str, expiry: int | None = None) -> bytes | None:
        resp = self.client.getex(
            key,
            ex=expiry,
            persist=expiry is None,
        )
        if resp is None:
            return None
        return cast('bytes', resp)

    def get_many(self, keys: Sequence[str], expiry: int | None = None) -> dict[str, bytes | None]:
        resp = cast('list[bytes | None]', self.client.mget(keys=keys))
        return dict(zip(keys, resp, strict=True))

    def set(self, key: str, data: bytes, expiry: int | None = None) -> None:
        self.client.set(key, data, ex=expiry)

    def clear(self):
        self.client.flushall()

    def start_batch(self) -> RedisCacheBatch:
        return RedisCacheBatch(self.client)


class Cache(AbstractContextManager):
    ext_cache: ExtCache | None
    prefix: str
    cache_misses: set[str]
    local: LocalLRUCache
    run: CacheRun | None
    allowed_kinds: set[CacheKind]

    def _make_full_key(self, key: str) -> str:
        if self.prefix:
            return '%s:%s' % (self.prefix, key)
        return key

    def __init__(self, ureg: UnitRegistry, redis_url: str | None = None, base_logger: loguru.Logger | None = None):
        self.prefix = 'kausal-paths-model'
        self.ext_cache_ttl = 60 * 60
        self.ureg = ureg
        self.run = None
        self.log = base_logger.bind(markup=True) if base_logger else loguru.logger.bind(markup=True)
        self.obj_id = base32_crockford.gen_obj_id(self)
        self.pc = PerfCounter(f'cache {self.obj_id}')
        self.local = LocalLRUCache(32 * 1024 * 1024, self.log)
        self.allowed_kinds = {CacheKind.RUN, CacheKind.LOCAL, CacheKind.EXT}
        from pandas import DataFrame as PandasDataFrame
        self.PandasDataFrame = PandasDataFrame
        self.cache_misses = set()

        if redis_url:
            client = redis.Redis.from_url(redis_url)
            self.ext_cache = RedisCache(client)
            redis_str = ' using Redis at [repr.url]%s[/]' % redis_url
        else:
            self.ext_cache = None
            redis_str = ', [warning]not using external cache[/]'
            self.allowed_kinds.remove(CacheKind.EXT)

        self.log.debug('Cache initialized%s' % redis_str)
        super().__init__()

    def set_allowed_cache_kinds(self, kinds: set[CacheKind]):
        self.allowed_kinds = set(kinds)

    def set_lru_size(self, nr_bytes: int):
        self.local = LocalLRUCache(nr_bytes, self.log)

    def __del__(self):
        # self.log.debug('<{}> Cache destroyed', self.obj_id)
        pass

    def __enter__(self) -> Self:
        self.start_run()
        super().__enter__()
        return self

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,  # noqa: PYI063
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> None:
        if __exc_type is not None and self.run:
            self.run.new_objs = []
        self.end_run()

    def start_run(self):
        self.pc.measure()
        self.run = CacheRun(self)

    def end_run(self):
        if self.run is None:
            self.log.error('<{}> end_run() called while no run was active', self.obj_id)  # noqa: PLE1205
            return
        self.run.end()
        self.run = None

    def is_run_cached(self, key: str) -> bool:
        if self.run is None:
            return False
        return self.run.has(key)

    def is_local_cached(self, key: str) -> bool:
        return self.local.has(key)

    def store_multiple(self, objs: list[tuple[str, int | None, bytes | object]]) -> int:
        if self.ext_cache and CacheKind.EXT in self.allowed_kinds:
            batch = self.ext_cache.start_batch()
        else:
            batch = None

        nr_bytes = 0
        for key, expiry, data_or_obj in objs:
            if not isinstance(data_or_obj, bytes):
                data = self.serialize_object(data_or_obj)
            else:
                data = data_or_obj

            nr_bytes += len(data)
            if batch is not None:
                batch.add(key, data, expiry=expiry)

            if CacheKind.LOCAL in self.allowed_kinds:
                self.local.put(key, data)

        if batch is not None:
            batch.send()

        return nr_bytes

    def serialize_object(self, obj: object) -> bytes:
        if isinstance(obj, ppl.PathsDataFrame):
            obj = PickledPathsDataFrame.from_df(obj)
        elif isinstance(obj, self.PandasDataFrame) and hasattr(obj, 'pint'):
            obj = PickledPintDataFrame.from_df(obj)
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return data

    def deserialize_object(self, data: bytes) -> ppl.PathsDataFrame | PandasDataFrame | object:
        obj = pickle.loads(data)  # noqa: S301
        if isinstance(obj, PickledPathsDataFrame):
            return obj.to_df(self.ureg)
        if isinstance(obj, PickledPintDataFrame):
            return obj.to_df(self.ureg)
        return obj

    def _record_event(self, kind: CacheKind | None, event: Literal['req', 'hit', 'miss']) -> None:
        run = self.run
        if not run:
            return
        run.record_event(kind, event)

    def _get_from_run_cache(self, res: CacheResult) -> bool:
        kind = res.kind = CacheKind.RUN
        if not self.run:
            return False

        self._record_event(kind, 'req')
        obj = self.run.get(res.full_key)
        if obj is not None:
            self._record_event(kind, 'hit')
            # Make a copy if the object is a DataFrame to avoid modifying the cached version
            if isinstance(obj, self.PandasDataFrame | ppl.PathsDataFrame):
                obj = obj.copy()
            res.obj = obj
            return True

        self._record_event(kind, 'miss')
        return False

    def _get_from_local_cache(self, res: CacheResult) -> bool:
        kind = res.kind = CacheKind.LOCAL
        self._record_event(kind, 'req')
        data = self.local.get(res.full_key)
        if data is None:
            self._record_event(kind, 'miss')
            return False

        obj = self.deserialize_object(data)
        self._record_event(kind, 'hit')
        # If we're in a run, add the object to the run cache for future lookups
        if self.run is not None and CacheKind.RUN in self.allowed_kinds:
            self.run.add(res.full_key, obj)
        res.obj = obj
        return True

    def _get_from_ext_cache(self, res: CacheResult, expiry: int) -> None | bytes:
        if self.ext_cache is None:
            return None

        res.kind = kind = CacheKind.EXT

        self._record_event(kind, 'req')
        data = self.ext_cache.get(res.full_key, expiry=expiry)
        if data is None:
            self._record_event(kind, 'miss')
            return None

        # Object found in external cache
        self._record_event(kind, 'hit')
        return data

    def _get_from_caches(self, res: CacheResult, expiry: int) -> None:  # noqa: C901
        self._record_event(None, 'req')

        allowed = self.allowed_kinds
        run = None
        if self.run is not None and CacheKind.RUN in allowed:
            run = self.run

        if CacheKind.RUN in allowed:
            hit = self._get_from_run_cache(res)
            if hit:
                return

        # If not found in the run cache, check the local LRU cache
        if CacheKind.LOCAL in allowed:
            hit = self._get_from_local_cache(res)
            if hit:
                if run:
                    run.add(res.full_key, res.obj)
                return

        if CacheKind.EXT not in allowed:
            res.is_hit = False
            return

        # Finally, check the external cache
        data = self._get_from_ext_cache(res, expiry)
        if data is None:
            res.is_hit = False
            return

        if CacheKind.LOCAL in allowed:
            # Add the object to the local cache for faster future access
            self.local.put(res.full_key, data)

        obj = self.deserialize_object(data)
        if run is not None:
            # If we're in a run, add the object to the run cache as well
            run.add(res.full_key, obj)
        res.obj = obj

    def get(self, key: str, expiry: int | None = None) -> CacheResult:
        """
        Retrieve an object from the cache hierarchy.

        This method checks for the object in the following order:
        1. Current execution run cache
        2. Local LRU cache
        3. External Redis cache (if configured)

        Args:
            key: The key to lookup in the cache.
            expiry: The expiration time for the key in seconds
            only_kind: If not None, use only specified cache type.

        Returns:
            CacheResult: An object containing the cache hit status, cache kind, and the retrieved object (if found).

        """
        res = CacheResult[ppl.PathsDataFrame](key=key, is_hit=True, key_prefix=self.prefix)
        if expiry is None:
            expiry = self.ext_cache_ttl
        self._record_event(None, 'req')
        if res.full_key in self.cache_misses:
            res.is_hit = False
        else:
            self._get_from_caches(res, expiry=expiry)
        if res.is_hit:
            self._record_event(None, 'hit')
        else:
            self._record_event(None, 'miss')
        return res

    def prefetch_keys(self, keys: set[str], caller_ids: Mapping[str, str] | None = None) -> set[str]:
        """
        Prefetch multiple keys from the cache.

        :param keys: A set of cache keys to prefetch.
        :param caller_ids: A mapping of cache keys to their caller-specific IDs.
            The caller-specific ID can be None if the caller does not have
            a specific ID for the object.

        :returns:
            A set of IDs that were found in the cache. If caller_ids is provided,
            the returned IDs will be the caller-specific IDs. Otherwise, the
            IDs will be the cache keys.
        """
        if not self.ext_cache:
            return set()
        full_keys_to_fetch: list[str] = []
        full_key_to_id: dict[str, str] = {}
        if caller_ids is None:
            caller_ids = {}
        for key in keys:
            full_key = self._make_full_key(key)
            if full_key in self.cache_misses or (self.run and self.run.has(full_key)) or self.local.has(full_key):
                continue
            full_keys_to_fetch.append(full_key)
            full_key_to_id[full_key] = caller_ids.get(key, key)
        if not full_keys_to_fetch:
            return set()
        res = self.ext_cache.get_many(full_keys_to_fetch)
        found_ids = set[str]()
        for full_key, data in res.items():
            if data is None:
                self.cache_misses.add(full_key)
            else:
                self.local.put(full_key, data)
                found_ids.add(full_key_to_id[full_key])
        return found_ids

    def set(self, key: str, obj: object, expiry: int | None = None):  # noqa: C901
        if expiry is None:
            expiry = self.ext_cache_ttl
        elif expiry <= 0:
            expiry = None
        allowed_kinds = self.allowed_kinds
        res = CacheResult(key, obj=obj, key_prefix=self.prefix)

        serialize = CacheKind.LOCAL in allowed_kinds or CacheKind.EXT in allowed_kinds

        run = None
        if self.run and CacheKind.RUN in allowed_kinds:
            run = self.run

        if run:
            if run.has(res.full_key):
                return
            run.add(res.full_key, obj)

        if not serialize:
            return

        obj_size = obj.estimated_size() if isinstance(obj, pl.DataFrame) else obj.__sizeof__()
        if obj_size > 128 * 1024:
            self.log.warning('Attempting to cache a large object of %d KiB: %s' % (obj_size // 1024, key))
        if obj_size > self.local.max_size:
            warnings.warn('Discarding a large object of %d KiB: %s' % (obj_size // 1024, key), stacklevel=3)
            return

        self.cache_misses.discard(res.full_key)

        if run:
            run.add_ext(res.full_key, expiry, obj)
            return

        data = self.serialize_object(obj)
        if CacheKind.EXT in allowed_kinds and self.ext_cache:
            self.ext_cache.set(res.full_key, data, expiry=expiry)
        if CacheKind.LOCAL in allowed_kinds:
            self.local.put(res.full_key, data)

    def clear(self):
        self.log.info('Clearing cache')
        if self.ext_cache:
            self.ext_cache.clear()
        self.local.clear()
