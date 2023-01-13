from __future__ import annotations

import typing
from dataclasses import dataclass

import pandas as pd
from pint_pandas import PintType
from polars import DataFrame as PlDataFrame, Series as PlSeries, from_pandas as from_pandas_pl

from nodes.units import Unit


if typing.TYPE_CHECKING:
    from .polars_ext import PathsExt, UnitsExt


@dataclass
class DataFrameMeta:
    units: dict[str, Unit]
    primary_keys: list[str]


class PathsDataFrame(PlDataFrame):  # type: ignore
    _units: dict[str, Unit]
    _primary_keys: typing.List[str]
    paths: 'PathsExt'

    def get_meta(self) -> DataFrameMeta:
        return DataFrameMeta(units=self._units, primary_keys=self._primary_keys)


class Series(PlSeries):  # type: ignore
    units: 'UnitsExt'


def to_ppdf(df: PlDataFrame) -> PathsDataFrame:
    if isinstance(df, PathsDataFrame):
        return df
    return PathsDataFrame._from_pydf(df._df)


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
