from __future__ import annotations

import re
import typing
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, cast
from typing_extensions import deprecated

import polars as pl
from polars._utils.parse import parse_into_list_of_expressions

from kausal_common.models.types import copy_signature

from nodes.constants import FORECAST_COLUMN, TIME_INTERVAL, VALUE_COLUMN, YEAR_COLUMN
from nodes.units import Quantity, Unit, unit_registry

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

    import numpy as np
    import pandas as pd
    from dvc_pandas.dataset import Dataset as DVCDataset
    from polars.polars import PyDataFrame, PyExpr
    from polars.type_aliases import ColumnNameOrSelector, IntoExpr, IntoExprColumn

    from .polars_ext import PathsExt


@dataclass
class DataFrameMeta:
    units: dict[str, Unit]
    primary_keys: list[str]
    explanation: list[str] = field(default_factory=list)

    @classmethod
    def get_dim_ids(cls, pks: list[str]) -> list[str]:
        if YEAR_COLUMN in pks:
            pks.remove(YEAR_COLUMN)
        return pks

    @property
    def dim_ids(self) -> list[str]:
        keys = self.primary_keys.copy()
        return self.get_dim_ids(keys)

    @property
    def metric_cols(self) -> list[str]:
        return list(self.units.keys())

    def copy(self) -> DataFrameMeta:
        return DataFrameMeta(
            units=self.units.copy(),
            primary_keys=self.primary_keys.copy(),
            explanation = self.explanation.copy()
        )

    def serialize(self) -> dict[str, Any]:
        return dict(units={key: str(val) for key, val in self.units.items()}, primary_keys=self.primary_keys)

    def is_equal(self, other: DataFrameMeta, ignore_order: bool = False) -> bool:
        if self.units != other.units:
            return False
        if self.primary_keys != other.primary_keys:
            if ignore_order:
                if set(self.primary_keys) != set(other.primary_keys):
                    return False
            else:
                return False
        return True


class PathsDataFrame(pl.DataFrame):
    _units: dict[str, Unit]
    _primary_keys: list[str]
    _explanation:  list[str]
    paths: PathsExt

    @copy_signature(pl.DataFrame.__init__)
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._units = {}
        self._primary_keys = []
        self._explanation = []

    @classmethod
    def _from_pydf(cls, py_df: PyDataFrame, meta: DataFrameMeta | None = None) -> PathsDataFrame:
        df = cls.__new__(cls)
        df._df = py_df
        df._units = {}
        df._primary_keys = []
        df._explanation = []

        if meta is None:
            return df
        for col, unit in meta.units.items():
            if col in df.columns:
                df._units[col] = unit
        for col in meta.primary_keys:
            if col in df.columns:
                df._primary_keys.append(col)

        df._explanation = meta.explanation.copy() if meta.explanation else []
        validate_ppdf(df)

        return df

    @property
    def primary_keys(self) -> list[str]:
        return list(self._primary_keys)

    @property
    def dim_ids(self) -> list[str]:
        return DataFrameMeta.get_dim_ids(list(self._primary_keys))

    @property
    def metric_cols(self) -> list[str]:
        return list(self._units.keys())

    @property
    def explanation(self) -> list:
        """Get the explanation from attribute (for consistency)."""
        if not hasattr(self, '_explanation'):
            self._explanation = []
        return self._explanation

    def with_explanation(self, explanation: list) -> PathsDataFrame:
        """Return a new PathsDataFrame with the updated explanation."""
        meta = self.get_meta()
        meta.explanation = explanation.copy()
        df = self.replace_meta(meta)
        return df

    def replace_meta(self, meta: DataFrameMeta):
        return self._from_pydf(self._df, meta=meta)

    def serialize_meta(self) -> dict[str, Any]:
        meta = self.get_meta().serialize()
        meta['columns'] = list(self.columns)
        meta['dtypes'] = [str(dt) for dt in self.dtypes]
        meta['height'] = self.height
        return meta

    def filter(self, *predicates: (
        IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool] | np.ndarray[Any, Any]
       ), **constraints: Any) -> PathsDataFrame:
        meta = self.get_meta()
        df = super().filter(*predicates, **constraints)
        return to_ppdf(df, meta=meta)

    def rename(self, mapping: Mapping[str, str] | Callable[[str], str], *, strict: bool = True) -> PathsDataFrame:
        meta = self.get_meta()
        units = dict(meta.units)
        primary_keys = list(meta.primary_keys)
        assert not callable(mapping)
        for old_col, new_col in mapping.items():
            if old_col in meta.units:
                units[new_col] = meta.units[old_col]
            elif old_col in meta.primary_keys:
                primary_keys[meta.primary_keys.index(old_col)] = new_col
        meta.units = units
        meta.primary_keys = primary_keys
        df = super().rename(mapping)
        return to_ppdf(df, meta=meta)

    def drop(
        self,
        *columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        strict: bool = True,
    ) -> PathsDataFrame:
        meta = self.get_meta()
        df = super().drop(*columns, strict=strict)
        for col in list(meta.units.keys()):
            if col not in df.columns:
                del meta.units[col]
        for col in list(meta.primary_keys):
            if col not in df.columns:
                meta.primary_keys.remove(col)
        return to_ppdf(df, meta=meta)

    def _pyexprs_to_meta(self, exprs: list[PyExpr], units: dict[str, Unit]) -> DataFrameMeta:
        meta = self.get_meta()
        for expr in exprs:
            if expr.meta_has_multiple_outputs():
                continue
            output_col = expr.meta_output_name()
            root_cols = expr.meta_root_names()
            if output_col in units:
                meta.units[output_col] = units[output_col]
                continue
            if len(root_cols) == 1 and root_cols[0] in meta.units:
                meta.units[output_col] = meta.units[root_cols[0]]
        return meta

    def select(  # type: ignore[override]
        self, *exprs: IntoExpr | Iterable[IntoExpr], units: dict[str, Unit] | None = None, **named_exprs: IntoExpr,
    ) -> PathsDataFrame:
        structify = False
        pyexprs = parse_into_list_of_expressions(
            *exprs, **named_exprs, __structify=structify,
        )
        df = super().select(*exprs, **named_exprs)
        meta = self._pyexprs_to_meta(pyexprs, units or {})
        return PathsDataFrame._from_pydf(df._df, meta=meta)

    def select_metrics(self, metric_cols: list[str] | str, rename: list[str] | str | None = None) -> PathsDataFrame:
        if isinstance(metric_cols, str):
            metric_cols = [metric_cols]
        if rename is not None and isinstance(rename, str):
            rename = [rename]
        for col in metric_cols:
            if col not in self._units:
                raise Exception('No unit for column %s' % col)
        cols = [*self._primary_keys, *metric_cols]
        if FORECAST_COLUMN in self.columns:
            cols.append(FORECAST_COLUMN)
        df = self.select(cols)
        if rename is not None:
            df = df.rename(dict(zip(metric_cols, rename, strict=True)))
        return df

    def with_columns(  # type: ignore[override]
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        units: dict[str, Unit] | None = None,
        **named_exprs: IntoExpr,
    ) -> PathsDataFrame:
        structify = False
        pyexprs = parse_into_list_of_expressions(
            *exprs, **named_exprs, __structify=structify,
        )
        df = super().with_columns(*exprs, **named_exprs)
        meta = self._pyexprs_to_meta(pyexprs, units or {})
        return PathsDataFrame._from_pydf(df._df, meta=meta)

    @deprecated("Use with_columns() instead")
    def with_column(self, column: pl.Series | pl.Expr, unit: Unit | None = None, is_primary_key: bool = False) -> PathsDataFrame:  # pyright: ignore
        raise NotImplementedError("Use with_columns() instead")

    def drop_nulls(self, subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None) -> PathsDataFrame:
        df = super().drop_nulls(subset)
        return PathsDataFrame._from_pydf(df._df, meta=self.get_meta())

    def sort(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] = False,
        multithreaded: bool = True,
        maintain_order: bool = False,
    ) -> PathsDataFrame:
        df = super().sort(
            by,
            *more_by,
            descending=descending,
            nulls_last=nulls_last,
            multithreaded=multithreaded,
            maintain_order=maintain_order,
        )
        return PathsDataFrame._from_pydf(df._df, meta=self.get_meta())

    def join(
        self,
        other: pl.DataFrame | PathsDataFrame,
        on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        how: pl.JoinStrategy = "inner",
        *,
        left_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        right_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        suffix: str = "_right",
        validate: pl.JoinValidation = "m:m",
        nulls_equal: bool = False,
        coalesce: bool | None = None,
        maintain_order: pl.MaintainOrderJoin | None = None,
    ) -> pl.DataFrame:
        plain_df = pl.DataFrame(self._df)
        df = plain_df.join(
            other,
            on,
            how,
            left_on=left_on,
            right_on=right_on,
            suffix=suffix,
            validate=validate,
            nulls_equal=nulls_equal,
            coalesce=coalesce,
            maintain_order=maintain_order,
        )
        return df

    def get_meta(self) -> DataFrameMeta:
        explanation_list = getattr(self, '_explanation', [])
        meta = DataFrameMeta(
            units=self._units.copy(),
            primary_keys=self._primary_keys.copy(),
            explanation=explanation_list.copy()
        )
        return meta

    def get_unit(self, col: str) -> Unit:
        if col not in self._units:
            raise KeyError("Column %s not found" % col)
        return self._units[col]

    def has_unit(self, col: str) -> bool:
        return col in self._units

    def set_unit(self, col: str, unit: Unit | str, force: bool = False) -> PathsDataFrame:
        assert col in self.columns
        if col in self._units and not force:
            raise Exception("Column %s already has a unit set" % col)
        meta = self.get_meta()
        if isinstance(unit, str):
            unit = unit_registry.parse_units(unit)
        meta.units[col] = unit  # type: ignore
        return PathsDataFrame._from_pydf(self._df, meta=meta)

    def clear_unit(self, col: str) -> PathsDataFrame:
        if col not in self._units:
            raise Exception("Column %s does not have a unit" % col)
        meta = self.get_meta()
        del meta.units[col]
        return PathsDataFrame._from_pydf(self._df, meta=meta)

    def multiply_cols(self, cols: list[str], out_col: str, out_unit: Unit | None = None) -> PathsDataFrame:
        res_unit = cast('Unit', reduce(lambda x, y: x * y, [self._units[col] for col in cols]))  # pyright: ignore
        s = reduce(lambda x, y: x * y, [self[col] for col in cols])
        df = self.with_columns([s.alias(out_col)])
        df._units[out_col] = res_unit
        if out_unit:
            df = df.ensure_unit(out_col, out_unit)
        return df

    def multiply_quantity(self, col: str, quantity: Quantity, out_unit: Unit | None = None) -> PathsDataFrame:
        res_unit = self._units[col] * quantity.units
        df = self.with_columns((pl.col(col) * pl.lit(quantity.m)).alias(col))
        df._units[col] = res_unit
        if out_unit:
            df = df.ensure_unit(col, out_unit)
        return df

    def divide_cols(self, cols: list[str], out_col: str, out_unit: Unit | None = None) -> PathsDataFrame:
        res_unit = cast('Unit', reduce(lambda x, y: x / y, [self._units[col] for col in cols]))  # pyright: ignore
        s = reduce(lambda x, y: x / y, [self[col] for col in cols])
        df = self.with_columns([s.alias(out_col)])
        df._units[out_col] = res_unit
        if out_unit:
            df = df.ensure_unit(out_col, out_unit)
        return df

    def divide_quantity(self, col: str, quantity: Quantity, out_unit: Unit | None = None) -> PathsDataFrame:
        res_unit = cast('Unit', quantity.units / self._units[col])
        df = self.with_columns((pl.lit(quantity.m) / pl.col(col)).alias(col))
        df._units[col] = res_unit
        if out_unit:
            df = df.ensure_unit(col, out_unit)
        return df

    def sum_cols(self, cols: list[str], out_col: str, out_unit: Unit | None = None, skip_missing: bool = False) -> PathsDataFrame:
        res_unit = None
        s = None
        for col in cols:
            if col not in self.columns and skip_missing:
                continue
            if res_unit is None:
                res_unit = self._units[col]
                s = self[col]
            else:
                s = s + self.ensure_unit(col, res_unit)[col]

        assert s is not None
        assert res_unit is not None
        df = self.with_columns([s.alias(out_col)])
        df._units[out_col] = res_unit
        if out_unit:
            df = df.ensure_unit(out_col, out_unit)
        return df

    def subtract_cols(self, cols: list[str], out_col: str, out_unit: Unit | None = None) -> PathsDataFrame:
        assert len(cols) > 0
        first_col = cols[0]
        res_unit = self._units[first_col]
        s = self[first_col]
        for col in cols[1:]:
            s = s - self.ensure_unit(col, res_unit)[col]

        assert res_unit is not None
        df = self.with_columns([s.alias(out_col)])
        df._units[out_col] = res_unit
        if out_unit:
            df = df.ensure_unit(out_col, out_unit)
        return df

    def cumulate(self, col: str) -> PathsDataFrame:
        meta = self.get_meta()
        unit = unit_registry(TIME_INTERVAL)
        meta.units[col] *= unit

        df = self.paths.to_wide()
        for df_col in df.columns:
            if col + '@' in df_col or col == df_col:
                df = df.with_columns(pl.col(df_col).cum_sum())
            else:
                continue
        df = df.paths.to_narrow()

        df = to_ppdf(df, meta=meta)
        return df

    def cumprod(self, col: str, complement: bool = False) -> PathsDataFrame:
        meta = self.get_meta()
        unit = unit_registry(TIME_INTERVAL)
        meta.units[col] *= unit
        df = to_ppdf(self, meta=meta)
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')
        if complement:
            df = df.with_columns((pl.lit(1.0) - pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))

        df = df.paths.to_wide()
        for df_col in df.metric_cols:
            if col + '@' in df_col or col == df_col:
                df = df.with_columns(pl.col(df_col).cum_prod().alias(df_col))
            else:
                continue
        df = df.paths.to_narrow()

        return df

    def diff(self, col: str, n: int = 1) -> PathsDataFrame:
        meta = self.get_meta()
        unit = unit_registry(TIME_INTERVAL)
        meta.units[col] = cast('Unit', meta.units[col] / unit)

        df = self.paths.to_wide()
        for df_col in df.columns:
            if col + '@' in df_col or col == df_col:
                df = df.with_columns(pl.col(df_col).diff(n))
            else:
                continue
        df = df.paths.to_narrow().drop_nulls()

        df = to_ppdf(df, meta=meta)
        return df

    def add_to_index(self, cols: str | list[str]) -> PathsDataFrame:
        if isinstance(cols, str):
            cols = [cols]

        df = self.copy()

        for col in cols:
            assert col in self.columns
            assert col not in self._primary_keys
            df._primary_keys.append(col)

        return df

    def ensure_unit(self, col: str, unit: Unit | str | None) -> PathsDataFrame:
        assert unit is not None, 'Unit is missing.'
        if isinstance(unit, str):
            unit_obj = unit_registry.parse_units(unit)
        else:
            unit_obj = unit
        col_unit = self._units[col]
        if col_unit == unit_obj:
            return self
        if not col_unit.is_compatible_with(unit_obj):
            raise Exception("Unit '%s' for column %s is not compatible with '%s'" % (col_unit, col, unit_obj))

        assert isinstance(unit_obj, Unit), f'Unit is not a Unit: {type(unit)}'
        vls = self[col].to_numpy()
        vls = (vls * col_unit).to(unit_obj).m
        df = self.with_columns([pl.Series(name=col, values=vls)], units={col: unit_obj})
        return df

    def to_pandas(
        self, *args: Any, date_as_object: bool = False, meta: DataFrameMeta | None = None, **kwargs: Any,
    ) -> pd.DataFrame:
        from pint_pandas import PintType

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

    def get_last_historical_values(self, year=None):
        meta = self.get_meta()
        df = self.paths.to_wide()

        if year is None:
            last_hist_year = df.filter(pl.col(FORECAST_COLUMN).eq(other=False))[YEAR_COLUMN].max()
        else:
            last_hist_year = year
        df = df.filter(pl.col(YEAR_COLUMN).eq(last_hist_year))
        df = df.paths.to_narrow()
        return to_ppdf(df, meta=meta)

    def __str__(self) -> str:  # noqa: C901
        meta = self.get_meta()
        df = self.copy()
        renames = {}
        print_dimensions = []
        for col, unit in meta.units.items():
            new_col = col
            # Wide format
            if '@' in col:
                new_col = new_col.replace('@', '\n')
            if '/' in col:
                new_col = new_col.replace('/', '\n')

            lines = new_col.splitlines()
            for idx, line in enumerate(list(lines)):
                m = re.match(r'^(.*):(.*)$', line)
                if not m:
                    continue
                dim_id, cat = m.groups()
                if dim_id not in print_dimensions:
                    print_dimensions.append(dim_id)
                lines[idx] = cat
            new_col = '\n'.join(lines)

            new_col = '[%s] %s' % (str(unit), new_col)
            renames[col] = new_col

        for col in meta.primary_keys:
            if col not in df:
                print("WARNING: primary key %s not in columns" % col)
                continue
            renames[col] = '[idx] %s' % col

        if renames:
            df = df.rename(renames)

        out = ''
        if print_dimensions:
            out += 'Dimensions:\n%s\n' % '\n'.join(
                '  %d. %s' % (idx + 1, dim) for idx, dim in enumerate(print_dimensions)
            )
        out += df._df.as_str()
        return out

    def print(self):
        from rich.console import Console
        from rich.table import Table

        table = Table()
        for col in self.columns:
            col_newlines = col.replace('@', '\n').replace(':', ':\n')
            table.add_column(col_newlines)
        for row in self.iter_rows():
            vals = []
            for val in row:
                if isinstance(val, float | int):
                    vals.append(str(val))
                else:
                    vals.append(val)
            table.add_row(*vals)
        console = Console()
        console.print(table)


    def select_category(
            self,
            dimension: str,
            category: str | None = None,
            category_number: int | None = None,
            baseline_year: int | None = None,
            baseline_year_level: Quantity | None = None,
            keep_dimension: bool | None = None) -> PathsDataFrame:
        """
        Select one category of a dimension and further process that.

        The purpose is to choose a hypothesis of scenario among several ones.
        * Give the dimension name (often in the node class).
        * Give the category name (often in a string parameter).
        * Or, give the category as an integer that is used to select from a list of categories.
        * Give the output as such.
        * Or, give the output as a ratio relative to the baseline year value.
        * Or, give the output as a difference to the baseline year level.
        """
        if category_number is not None:
            assert category is None
            category = sorted(self[dimension].unique())[category_number]  # FIXME Improve ordering method
        df = self.filter(pl.col(dimension).eq(category))
        if keep_dimension is not True:
            df = df.drop(dimension)

        if baseline_year is not None:
            df = df.filter(pl.col(YEAR_COLUMN).ge(baseline_year))
            if baseline_year_level is None:  # Assume a relative change
                level = df.filter(pl.col(YEAR_COLUMN).eq(baseline_year))[VALUE_COLUMN][0]
                df = df.with_columns(
                    pl.col(VALUE_COLUMN) / pl.lit(level) - pl.lit(1),
                )
                df = df.clear_unit(VALUE_COLUMN)
                df = df.set_unit(VALUE_COLUMN, 'dimensionless')
            else:
                unit = df.get_unit(VALUE_COLUMN)
                baseline_year_level = cast('Quantity', baseline_year_level.to(unit))
                df = df.with_columns(pl.col(VALUE_COLUMN) - pl.lit(baseline_year_level.m))
        return df

def validate_ppdf(df: PathsDataFrame):
    units = list(df._units.keys())  # pyright: ignore[reportPrivateUsage]
    pks = list(df._primary_keys)  # pyright: ignore[reportPrivateUsage]
    for col in units + pks:
        if col not in df.columns:
            raise Exception('Column %s in metadata not found in DF columns' % col)


def to_ppdf(df: pl.DataFrame | PathsDataFrame, meta: DataFrameMeta | None = None) -> PathsDataFrame:
    if isinstance(df, PathsDataFrame) and meta is None:
        validate_ppdf(df)
        return df

    source_explanation = []
    if isinstance(df, PathsDataFrame) and hasattr(df, '_explanation'):
        source_explanation = df._explanation.copy() if df._explanation else []

    if meta is None:
        meta = DataFrameMeta(
            units={},
            primary_keys=[],
            explanation=source_explanation
        )
    elif source_explanation and not meta.explanation:
        meta.explanation = source_explanation

    pdf = PathsDataFrame._from_pydf(df._df, meta=meta)
    validate_ppdf(pdf)
    return pdf


def from_pandas(df: pd.DataFrame) -> PathsDataFrame:
    import pandas as pd
    from pint_pandas import PintType

    dtypes = df.dtypes
    units: dict[str, Unit] = {}
    primary_keys: list[str] = []
    for col, dt in dtypes.items():
        assert isinstance(col, str)
        if isinstance(dt, PintType):
            units[col] = dt.units  # type: ignore
            df[col] = df[col].pint.m

    if isinstance(df.index, pd.MultiIndex):
        primary_keys = [str(x) for x in df.index.names]
    else:
        name = df.index.name
        assert name is not None
        primary_keys = [str(name)]

    pldf = PathsDataFrame(df.reset_index())
    #for col in primary_keys:
    #    if not isinstance(col, str):
    #        raise Exception("Column name is not a string (it is %s)" % type(col))
    pldf._units = units  # pyright: ignore[reportPrivateUsage]
    pldf._primary_keys = primary_keys  # pyright: ignore[reportPrivateUsage]
    return pldf


def from_dvc_dataset(ds: DVCDataset):
    assert ds.df is not None
    units: dict[str, Unit] = {}
    if ds.units:
        for col, unit in ds.units.items():
            units[col] = unit_registry.parse_units(unit)
    primary_keys = ds.index_columns or []
    pldf = PathsDataFrame._from_pydf(ds.df._df, meta=DataFrameMeta(units, primary_keys, explanation=[]))  # pyright: ignore[reportPrivateUsage]
    return pldf


if not pl.using_string_cache():
    pl.enable_string_cache()


pl.Config.set_fmt_str_lengths(100)
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
