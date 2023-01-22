from __future__ import annotations

import typing
from typing import Any, Iterable, Sequence
from dataclasses import dataclass

import pandas as pd
from pint_pandas import PintType
import polars as pl
from polars import polars
import polars.internals as pli
import numpy as np
from nodes.constants import YEAR_COLUMN

from nodes.units import Unit


if typing.TYPE_CHECKING:
    from .polars_ext import PathsExt, UnitsExt


@dataclass
class DataFrameMeta:
    units: dict[str, Unit]
    primary_keys: list[str]

    @property
    def dim_ids(self) -> list[str]:
        keys = self.primary_keys.copy()
        if YEAR_COLUMN in keys:
            keys.remove(YEAR_COLUMN)
        return keys


class PathsDataFrame(pl.DataFrame):
    _units: dict[str, Unit]
    _primary_keys: typing.List[str]
    paths: 'PathsExt'


    @classmethod
    def _from_pydf(cls, py_df: polars.PyDataFrame, meta: DataFrameMeta | None = None) -> PathsDataFrame:
        df = super()._from_pydf(py_df)
        df._units = {}
        df._primary_keys = []

        if meta is None:
            return df
        for col, unit in meta.units.items():
            if col in df.columns:
                df._units[col] = unit
        for col in meta.primary_keys:
            if col in df.columns:
                df._primary_keys.append(col)

        return df

    def replace_meta(self, meta: DataFrameMeta):
        return self._from_pydf(self._df, meta=meta)

    def filter(self, predicate: pli.Expr | str | pli.Series | list[bool] | np.ndarray[Any, Any]) -> PathsDataFrame:
        meta = self.get_meta()
        df = super().filter(predicate)
        return to_ppdf(df, meta=meta)

    def rename(self, mapping: dict[str, str]) -> PathsDataFrame:
        meta = self.get_meta()
        units = dict(meta.units)
        primary_keys = list(meta.primary_keys)
        for old_col, new_col in mapping.items():
            if old_col in meta.units:
                units[new_col] = meta.units[old_col]
            elif old_col in meta.primary_keys:
                primary_keys[meta.primary_keys.index(old_col)] = new_col
        meta.units = units
        meta.primary_keys = primary_keys
        df = super().rename(mapping)
        return to_ppdf(df, meta=meta)

    def drop(self, columns: str | Sequence[str]) -> PathsDataFrame:
        meta = self.get_meta()
        df = super().drop(columns)
        for col in list(meta.units.keys()):
            if col not in df.columns:
                del meta.units[col]
        for col in list(meta.primary_keys):
            if col not in df.columns:
                meta.primary_keys.remove(col)
        df._units = meta.units
        df._primary_keys = meta.primary_keys
        return df

    def select(self, exprs: (str | pli.Expr | pli.Series | Iterable[str | pli.Expr | pli.Series | pli.WhenThen | pli.WhenThenThen])) -> PathsDataFrame:
        meta = self.get_meta()
        df = super().select(exprs)
        return PathsDataFrame._from_pydf(df._df, meta=meta)

    def get_meta(self) -> DataFrameMeta:
        return DataFrameMeta(units=self._units, primary_keys=self._primary_keys)


class Series(pl.Series):  # type: ignore
    units: 'UnitsExt'


def to_ppdf(df: pl.DataFrame | PathsDataFrame, meta: DataFrameMeta | None = None) -> PathsDataFrame:
    if isinstance(df, PathsDataFrame):
        return df
    pdf = PathsDataFrame._from_pydf(df._df, meta=meta)
    return pdf


def from_pandas(df: 'pd.DataFrame') -> PathsDataFrame:
    dtypes = df.dtypes
    units = {}
    primary_keys: list[str] = []
    for col, dt in dtypes.items():
        assert isinstance(col, str)
        if isinstance(dt, PintType):
            units[col] = dt.units  # type: ignore
            df[col] = df[col].pint.m

    if isinstance(df.index, pd.MultiIndex):
        primary_keys = list(df.index.names)
    else:
        primary_keys = [df.index.name]

    pldf = PathsDataFrame(df.reset_index())
    pldf._units = units
    pldf._primary_keys = primary_keys
    return pldf
