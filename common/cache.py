from __future__ import annotations

from typing import Any, Dict, Optional, Union
import pickle

import pandas as pd
import pint_pandas
from pint import UnitRegistry
import redis


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


class Cache:
    client: Optional[redis.Redis]
    local_cache: Dict[str, bytes]
    run_cache: Dict[str, Any] | None
    prefix: str

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

    def start_run(self):
        self.run_cache = {}

    def end_run(self):
        self.run_cache = None

    def dump_object(self, obj: Any) -> bytes:
        if isinstance(obj, pd.DataFrame) and hasattr(obj, 'pint'):
            obj = PickledPintDataFrame.from_df(obj)
        data = pickle.dumps(obj)
        return data

    def load_object(self, data: bytes) -> Any:
        obj = pickle.loads(data)
        if isinstance(obj, PickledPintDataFrame):
            return obj.to_df(self.ureg)
        return obj

    def get(self, key: str) -> Any:
        full_key = '%s:%s' % (self.prefix, key)
        if self.run_cache is not None and full_key in self.run_cache:
            return self.run_cache[full_key]

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
        data = self.dump_object(obj)
        if self.client:
            self.client.setex(full_key, time=self.timeout, value=data)
        else:
            self.local_cache[full_key] = data
        if self.run_cache is not None:
            self.run_cache[full_key] = obj
