# from __future__ import annotations

# import hashlib
# import os
# import inspect
# from types import FunctionType
from logging import exception
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, TYPE_CHECKING, Union

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import pint
import pint_pandas

import copy

from common.i18n import TranslatedString
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y, YEAR_COLUMN
from params import Parameter

from .context import Context
from .datasets import Dataset
from .exceptions import NodeError
from .node import Node
from .simple import AdditiveNode, SimpleNode

if TYPE_CHECKING:
    from pages.models import NodeContent


class Ovariable(SimpleNode):
    # quantity: what the ovariable measures, e.g. exposure, exposure_response, disease_burden
    quantity: Optional[str]

    def get_input(self, quantity, query: str = None, drop: List = None):
        count = 0
        for node in self.input_nodes:
            if node.quantity == quantity:
                out = node
                count += 1
        if count == 0:
            print(quantity)
        assert count == 1

        of = out.get_output()

        if query is not None:
            of = of.query(query)

        if drop is not None:
            of = of.droplevel(drop)

        return OvariableFrame(of)

    def clean_computing(self, of, drop_columns=None):
        of[VALUE_COLUMN] = self.ensure_output_unit(of[VALUE_COLUMN])
        if drop_columns is not None:
            groupby = list(set(of.index.names) - set(drop_columns))
            of = of.aggregate_by_column(groupby=groupby, fun='sum')

        self.print_outline(of)
        return OvariableFrame(of)

    def print_outline(self, df):
        print(type(self))
        print(self.id)
        if YEAR_COLUMN in df.index.names:
            pick_rows = df.index.get_level_values(YEAR_COLUMN).isin([2017, 2021, 2022])
            self.print_pint_df(df.iloc[pick_rows])
        else:
            self.print_pint_df(df.iloc[[0, 1, 2, -3, -2, -1]])


class OvariableFrame(pd.DataFrame):

    def do_inner_join(self, other):
        assert VALUE_COLUMN in self.columns  # Cannot be in a merged format

        def add_temporary_index(self):
            tst = self.index.to_frame().assign(temporary=1)
            tst = pd.MultiIndex.from_frame(tst)
            return self.set_index(tst)

        if isinstance(other, pd.DataFrame):
            df2 = other
        else:
            df2 = pd.DataFrame([other], columns=[VALUE_COLUMN])

        df1 = add_temporary_index(self)
        df2 = add_temporary_index(df2)

        out = df1.merge(df2, left_index=True, right_index=True)
        out = OvariableFrame(out)
        out.index = out.index.droplevel(['temporary'])

        return out

    def aggregate_by_column(self, groupby, fun):
        self = self.groupby(groupby)
        if fun == 'sum':
            self = self.sum()
        else:
            self = self.mean()
        self[FORECAST_COLUMN] = self[FORECAST_COLUMN].mask(self[FORECAST_COLUMN] > 0, 1).astype('boolean')
        return self

    def clean(self):
        df = self.reset_index()
        if FORECAST_x in df.columns:
            df[FORECAST_COLUMN] = df[FORECAST_x] | df[FORECAST_y]
        keep = set(df.columns) - {0, 'index', VALUE_x, VALUE_y, FORECAST_x, FORECAST_y}
        df = df[list(keep)].set_index(list(keep - {VALUE_COLUMN, FORECAST_COLUMN}))
        return OvariableFrame(df.copy())

    def print_pint_df(self):
        df = self
        pint_cols = [col for col in df.columns if hasattr(df[col], 'pint')]
        if not pint_cols:
            print(df)
            return

        out = df[pint_cols].pint.dequantify()
        for col in df.columns:
            if col in pint_cols:
                continue
            out[col] = df[col]
        print(out)

    def __add__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] + self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] + other
            return OvariableFrame(self.copy())

    def __sub__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] - self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] - other
            return OvariableFrame(self.copy())

    def __mul__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] * self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] * other
            return OvariableFrame(self.copy())

    def __truediv__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] / self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] / other
            return OvariableFrame(self.copy())

    def __mod__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] % self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] % other
            return OvariableFrame(self.copy())

    def __pow__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] ** self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] ** other
            return OvariableFrame(self.copy())

    def __floordiv__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] // self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] // other
            return OvariableFrame(self.copy())

    def __lt__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] < self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] < other
            return OvariableFrame(self.copy())

    def __le__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] <= self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] <= other
            return OvariableFrame(self.copy())

    def __gt__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] > self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] > other
            return OvariableFrame(self.copy())

    def __ge__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] >= self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] >= other
            return OvariableFrame(self.copy())

    def __eq__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] == self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] == other
            return OvariableFrame(self.copy())

    def __ne__(self, other):
        if isinstance(other, pd.DataFrame):
            self = self.do_inner_join(other)
            self[VALUE_COLUMN] = self[VALUE_x] != self[VALUE_y]
            return self.clean()
        else:
            self[VALUE_COLUMN] = self[VALUE_COLUMN] != other
            return OvariableFrame(self.copy())

    def exp(self):
        s = self[VALUE_COLUMN]
        assert s.pint.units.dimensionless
        s = np.exp(s.pint.m)
        s = pd.Series(s, dtype='pint[dimensionless]')
        self[VALUE_COLUMN] = s
        return OvariableFrame(self.copy())

    def log10(self):
        s = self[VALUE_COLUMN]
        assert s.pint.units.dimensionless
        s = np.log10(s.pint.m)
        s = pd.Series(s, dtype='pint[dimensionless]')
        self[VALUE_COLUMN] = s
        return OvariableFrame(self.copy())

    def log(self):
        s = self[VALUE_COLUMN]
        assert s.pint.units.dimensionless
        s = np.log(s.pint.m)
        s = pd.Series(s, dtype='pint[dimensionless]')
        self[VALUE_COLUMN] = s
        return OvariableFrame(self.copy())
