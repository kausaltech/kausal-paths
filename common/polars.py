from __future__ import annotations
from functools import reduce

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

    @property
    def metric_cols(self) -> list[str]:
        return list(self.units.keys())


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

    def _pyexprs_to_meta(self, exprs: list[polars.PyExpr], units: dict[str, Unit]) -> DataFrameMeta:
        meta = self.get_meta()
        for expr in exprs:
            root_cols = expr.meta_roots()
            output_col = expr.meta_output_name()
            if output_col in units:
                meta.units[output_col] = units[output_col]
                continue
            else:
                if len(root_cols) == 1 and root_cols[0] in meta.units:
                    meta.units[output_col] = meta.units[root_cols[0]]
        return meta

    def select(
        self,
        exprs: (str | pli.Expr | pli.Series | Iterable[str | pli.Expr | pli.Series | pli.WhenThen | pli.WhenThenThen]),
        units: dict[str, Unit] | None = None
    ) -> PathsDataFrame:
        pyexprs = pli.selection_to_pyexpr_list(exprs)
        df = super().select(exprs)
        meta = self._pyexprs_to_meta(pyexprs, units or {})
        return PathsDataFrame._from_pydf(df._df, meta=meta)

    def with_columns(
        self,
        exprs: pli.Expr | pli.Series | Sequence[pli.Expr | pli.Series] | None = None,
        units: dict[str, Unit] | None = None,
        **named_exprs: Any
    ) -> PathsDataFrame:
        df = super().with_columns(exprs, **named_exprs)
        if exprs is None:
            exprs = []
        elif isinstance(exprs, pli.Expr):
            exprs = [exprs]
        elif isinstance(exprs, pli.Series):
            exprs = [pli.lit(exprs)]
        else:
            exprs = list(exprs)
        exprs.extend(
            pli.expr_to_lit_or_expr(expr).alias(name)
            for name, expr in named_exprs.items()
        )

        conv_exprs: list[polars.PyExpr] = []
        for e in exprs:
            if isinstance(e, pli.Expr):
                conv_exprs.append(e._pyexpr)
            elif isinstance(e, pli.Series):
                conv_exprs.append(pli.lit(e)._pyexpr)
            else:
                raise ValueError(f"Expected an expression, got {e}")

        meta = self._pyexprs_to_meta(conv_exprs, units or {})
        return PathsDataFrame._from_pydf(df._df, meta=meta)

    def with_column(self, column: pli.Series | pli.Expr, unit: Unit | None = None, is_primary_key: bool = False) -> PathsDataFrame:
        raise NotImplementedError("Use with_columns() instead")

    def drop_nulls(self, subset: str | Sequence[str] | None = None) -> PathsDataFrame:
        df = super().drop_nulls(subset)
        return PathsDataFrame._from_pydf(df._df, meta=self.get_meta())

    def get_meta(self) -> DataFrameMeta:
        return DataFrameMeta(units=self._units.copy(), primary_keys=self._primary_keys.copy())

    def get_unit(self, col: str) -> Unit:
        return self._units[col]

    def has_unit(self, col: str) -> bool:
        return col in self._units

    def set_unit(self, col: str, unit: Unit, warn: bool = True) -> PathsDataFrame:
        assert col in self.columns
        if col in self._units:
            if warn:
                raise Exception("Column %s already has a unit set" % col)
        self._units[col] = unit
        return PathsDataFrame._from_pydf(self._df, meta=self.get_meta())

    def multiply_cols(self, cols: list[str], out_col: str) -> PathsDataFrame:
        res_unit = reduce(lambda x, y: x * y, [self._units[col] for col in cols])
        s = reduce(lambda x, y: x * y, [self[col] for col in cols])
        df = self.with_columns([s.alias(out_col)])
        df._units[out_col] = res_unit
        return df

    def divide_cols(self, cols: list[str], out_col: str) -> PathsDataFrame:
        res_unit = reduce(lambda x, y: x / y, [self._units[col] for col in cols])
        s = reduce(lambda x, y: x / y, [self[col] for col in cols])
        df = self.with_columns([s.alias(out_col)])
        df._units[out_col] = res_unit
        return df

    def ensure_unit(self, col: str, unit: Unit) -> PathsDataFrame:
        col_unit = self._units[col]
        if col_unit == unit:
            return self
        if not col_unit.is_compatible_with(unit):
            raise Exception("Unit '%s' for column %s is not compatible with '%s'" % (col_unit, col, unit))

        vls = self[col].to_numpy(zero_copy_only=True)
        vls = (vls * col_unit).to(unit).m
        df = self.with_columns([pl.Series(name=col, values=vls)], units={col: unit})
        return df

    def to_pandas(self, *args: Any, date_as_object: bool = False, meta: DataFrameMeta | None = None, **kwargs: Any) -> pd.DataFrame:
        if meta is None:
            meta = self.get_meta()
        df = super().to_pandas(*args, date_as_object=date_as_object, **kwargs)
        primary_keys = meta.primary_keys if meta else self._primary_keys
        units = meta.units if meta else self._units
        if primary_keys:
            df = df.set_index(primary_keys)
        for col, unit in units.items():
            pt = PintType(unit)
            df[col] = df[col].astype(pt)
        return df

    def copy(self) -> PathsDataFrame:
        return PathsDataFrame._from_pydf(self._df, meta=self.get_meta())


class Series(pl.Series):  # type: ignore
    units: 'UnitsExt'


def to_ppdf(df: pl.DataFrame | PathsDataFrame, meta: DataFrameMeta | None = None) -> PathsDataFrame:
    if isinstance(df, PathsDataFrame) and meta is None:
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


if not pl.using_string_cache():
    pl.toggle_string_cache(True)

pl.Config.with_columns_kwargs = True
pl.Config.set_fmt_str_lengths(60)
