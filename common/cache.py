from __future__ import annotations
from contextlib import AbstractContextManager
from types import TracebackType

from typing import Any, Dict, Optional, Union, Tuple
import pickle

import pandas as pd
import polars as pl
import pint_pandas
from pint import UnitRegistry
import redis

from common import polars as ppl
from common.perf import PerfCounter


class PickledPintDataFrame:
    df: pd.DataFrame
    units: Dict[str, str]

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
    units: Dict[str, str]
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


class Cache(AbstractContextManager):
    client: Optional[redis.Redis]
    prefix: str

    local_cache: Dict[str, bytes]
    run_cache: Dict[str, Any] | None
    run_pipe: list[Tuple[str, Any]] | None

    def __init__(self, ureg: UnitRegistry, redis_url: Optional[str] = None):
        if redis_url is not None:
            self.client = redis.Redis.from_url(redis_url)
        else:
            self.client = None
            self.local_cache = {}
        self.prefix = 'kausal-paths'
        self.timeout = 600
        self.ureg = ureg
        self.run_cache = None
        self.run_pipe = None

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, __exc_type: type[BaseException] | None, __exc_value: BaseException | None, __traceback: TracebackType | None) -> bool | None:
        if __exc_type is not None:
            self.run_pipe = None
        self.end_run()
        return None

    def start_run(self):
        self.run_cache = {}
        if self.client is not None:
            self.run_pipe = []

    def end_run(self):
        self.run_cache = None

        if self.run_pipe:
            pc = PerfCounter('end run')
            if self.client is not None:
                pipe = self.client.pipeline(transaction=False)
            else:
                pipe = None

            pc.display('dumping %d objects' % len(self.run_pipe))
            for key, obj in self.run_pipe:
                data = self.dump_object(obj)
                if pipe is not None:
                    pipe.setex(key, self.timeout, data)
                else:
                    self.local_cache[key] = data
            pc.display('dumped')

            if pipe is not None:
                pipe.execute()
                pc.display('executed')

        self.run_pipe = None

    def dump_object(self, obj: Any) -> bytes:
        if isinstance(obj, ppl.PathsDataFrame):
            obj = PickledPathsDataFrame.from_df(obj)
        elif isinstance(obj, pd.DataFrame) and hasattr(obj, 'pint'):
            obj = PickledPintDataFrame.from_df(obj)
        data = pickle.dumps(obj)
        return data

    def load_object(self, data: bytes) -> Any:
        obj = pickle.loads(data)
        if isinstance(obj, PickledPathsDataFrame):
            return obj.to_df(self.ureg)
        elif isinstance(obj, PickledPintDataFrame):
            return obj.to_df(self.ureg)
        return obj

    def get(self, key: str) -> Any:
        full_key = '%s:%s' % (self.prefix, key)
        if self.run_cache is not None and full_key in self.run_cache:
            obj = self.run_cache[full_key]
            if isinstance(obj, (pd.DataFrame, ppl.PathsDataFrame)):
                return obj.copy()
            return obj

        if self.client:
            data = self.client.get(full_key)
        else:
            data = self.local_cache.get(full_key)
        if data is None:
            return None

        obj = self.load_object(data)
        if self.run_cache is not None:
            self.run_cache[full_key] = obj
        return obj

    def set(self, key: str, obj: Any):
        full_key = '%s:%s' % (self.prefix, key)
        if self.run_pipe is not None:
            self.run_pipe.append((full_key, obj))
        else:
            data = self.dump_object(obj)
            if self.client:
                self.client.setex(full_key, time=self.timeout, value=data)
            else:
                self.local_cache[full_key] = data

        if self.run_cache is not None:
            self.run_cache[full_key] = obj

    def clear(self):
        if self.client:
            self.client.flushall()
        else:
            self.local_cache = {}
