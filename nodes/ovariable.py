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
from nodes.constants import FORECAST_COLUMN, KNOWN_QUANTITIES, VALUE_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y
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

    def prepare_ovariable(self, quantity, query: str = None, drop: List = None):
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
            #    out = copy.copy(out)
            of = of.query(query)

        if drop is not None:
            of = of.droplevel(drop)

        return OvariableFrame(of)

    def clean_computing(self, of):
        of[VALUE_COLUMN] = self.ensure_output_unit(of[VALUE_COLUMN])
        return OvariableFrame(of)


class OvariableFrame(pd.DataFrame):

    def merge(self, other):
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

    def clean(self):
        df = self.reset_index()
        if FORECAST_x in df.columns:
            df[FORECAST_COLUMN] = df[FORECAST_x] | df[FORECAST_y]
        keep = set(df.columns) - {0, VALUE_x, VALUE_y, FORECAST_x, FORECAST_y}
        df = df[list(keep)].set_index(list(keep - {VALUE_COLUMN, FORECAST_COLUMN}))
        return OvariableFrame(df)

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
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] + self[VALUE_y]
        return self.clean()

    def __sub__(self, other):
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] - self[VALUE_y]
        return self.clean()

    def __mul__(self, other):
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] * self[VALUE_y]
        return self.clean()

    def __truediv__(self, other):
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] / self[VALUE_y]
        return self.clean()

    def __mod__(self, other):
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] % self[VALUE_y]
        return self.clean()

    def __pow__(self, other):
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] ** self[VALUE_y]
        return self.clean()

    def __floordiv__(self, other):
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] // self[VALUE_y]
        return self.clean()

    def __lt__(self, other):
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] < self[VALUE_y]
        return self.clean()

    def __le__(self, other):
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] <= self[VALUE_y]
        return self.clean()

    def __gt__(self, other):
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] > self[VALUE_y]
        return self.clean()

    def __ge__(self, other):
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] >= self[VALUE_y]
        return self.clean()

    def __eq__(self, other):
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] == self[VALUE_y]
        return self.clean()

    def __ne__(self, other):
        self = self.merge(other)
        self[VALUE_COLUMN] = self[VALUE_x] != self[VALUE_y]
        return self.clean()

    def log(self):
        self[VALUE_COLUMN] = np.log(self[VALUE_COLUMN])
        return self.clean()

    def log10(self):
        self[VALUE_COLUMN] = np.log10(self[VALUE_COLUMN])
        return self.clean()

    def exp(self):
        self[VALUE_COLUMN] = np.exp(self[VALUE_COLUMN])
        return self.clean()

#    def clean_computing(self, node):
#        self[VALUE_COLUMN] = node.ensure_output_unit(self[VALUE_COLUMN])
#        return OvariableFrame(self)

    # def clean_computing(self, output=None):  # FIXME clean_computing(self, node)

        # df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])
        # FIXME node.ensure_output_unit(self[VALUE_COLUMN])
    #    return pd.DataFrame(self)


"""

class OvaOps():
    content: pd.DataFrame

    def __init__(self, content):
        self.content = content

    def clean(self):
        df = self.content.reset_index()
        if FORECAST_x in df.columns:
            df[FORECAST_COLUMN] = df[FORECAST_x] | df[FORECAST_y]
        keep = set(df.columns) - {0, VALUE_x, VALUE_y, FORECAST_x, FORECAST_y}
        df = df[list(keep)].set_index(list(keep - {VALUE_COLUMN, FORECAST_COLUMN}))
        return Ovariable2(content=df)

    def __add__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] + self.content[VALUE_y]
        return self.clean()

    def __sub__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] - self.content[VALUE_y]
        return self.clean()

    def __mul__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] * self.content[VALUE_y]
        return self.clean()

    def __truediv__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] / self.content[VALUE_y]
        return self.clean()

    def __mod__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] % self.content[VALUE_y]
        return self.clean()

    def __pow__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] ** self.content[VALUE_y]
        return self.clean()

    def __floordiv__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] // self.content[VALUE_y]
        return self.clean()

    def __lt__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] < self.content[VALUE_y]
        return self.clean()

    def __le__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] <= self.content[VALUE_y]
        return self.clean()

    def __gt__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] > self.content[VALUE_y]
        return self.clean()

    def __ge__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] >= self.content[VALUE_y]
        return self.clean()

    def __eq__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] == self.content[VALUE_y]
        return self.clean()

    def __ne__(self):
        self.content[VALUE_COLUMN] = self.content[VALUE_x] != self.content[VALUE_y]
        return self.clean()


class Ovariable(SimpleNode):
    # content is the dataframe with the estimates
    content: Optional[pd.DataFrame]

    # quantity: what the ovariable measures, e.g. exposure, exposure_response, disease_burden
    quantity: Optional[str]

    def prepare_ovariable(self, quantity, query: str = None, drop: List = None):
        count = 0
        for node in self.input_nodes:
            if node.quantity == quantity:
                out = node
                count += 1
        if count == 0:
            print(quantity)
        assert count == 1

        out.content = out.get_output()  # FIXME Use dataframe, not node, as ovariable

        if query is not None:
            out = copy.copy(out)
            out.content = out.content.query(query)

        if drop is not None:
            out.content = out.content.droplevel(drop)

        return out  # FIXME return ovariableFrame

    def clean_computing(self, output):  # FIXME clean_computing(self, node)
        # Make this a function of ovariableFrame
        if isinstance(output, Ovariable):
            df = output.content
        else:
            df = output

        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])
        # FIXME node.ensure_output_unit(self[VALUE_COLUMN])
        return df

    def merge(self, other):  # Self and other must have content calculated.
        # FIXME make this a function for ovariableFrame

        def add_temporary_index(self):
            tst = self.index.to_frame().assign(temporary=1)
            tst = pd.MultiIndex.from_frame(tst)
            return self.set_index(tst)

        df1 = self.content.copy()  # FIXME Make ovariable a dataframe

        if isinstance(other, Ovariable) or isinstance(other, OvaOps):
            df2 = other.content.copy()
        else:
            df2 = pd.DataFrame([other], columns=[VALUE_COLUMN])

        df1 = add_temporary_index(df1)
        df2 = add_temporary_index(df2)

        out = df1.merge(df2, left_index=True, right_index=True)
        out.index = out.index.droplevel(['temporary'])

        return out

    def clean(self):
        #        df = self.content.reset_index()
        #        if FORECAST_x in df.columns:
        #            df[FORECAST_COLUMN] = df[FORECAST_x]  | df[FORECAST_y]
        #        keep = set(df.columns)- {0,VALUE_x,VALUE_y,FORECAST_x,FORECAST_y}
        #        df = df[list(keep)].set_index(list(keep - {VALUE_COLUMN,FORECAST_COLUMN}))
        #        self.content = df
        #        return OvaOps(content=self)
        raise exception('Do not use Ovariable.clean()')

    # FIXME Make all these functions of ovariableFrame
    def __add__(self, other):  # FIXME Assign __add__ function to FUN attribute to simplify OvaOps.
        out = self.merge(other)
        out = OvaOps(out).__add__()
        return out

    def __sub__(self, other):
        out = self.merge(other)
        out = OvaOps(out).__sub__()
        return out

    def __mul__(self, other):
        out = self.merge(other)
        out = OvaOps(out).__mul__()
        return out

    def __truediv__(self, other):
        out = self.merge(other)
        out = OvaOps(out).__truediv__()
        return out

    def __mod__(self, other):
        out = self.merge(other)
        out = OvaOps(out).__mod__()
        return out

    def __pow__(self, other):
        out = self.merge(other)
        out = OvaOps(out).__pow__()
        return out

    def __floordiv__(self, other):
        out = self.merge(other)
        out = OvaOps(out).__floordiv__()
        return out

    def __lt__(self, other):
        out = self.merge(other)
        out = OvaOps(out).__lt__()
        return out

    def __le__(self, other):
        out = self.merge(other)
        out = OvaOps(out).__le__()
        return out

    def __gt__(self, other):
        out = self.merge(other)
        out = OvaOps(out).__gt__()
        return out

    def __ge__(self, other):
        out = self.merge(other)
        out = OvaOps(out).__ge__()
        return out

# FIXME: THIS SHOULD WORK, TOO
#    def __eq__(self, other): # Messes up with node in self.input_nodes
#        out = self.merge(other)
#        out = OvaOps(out).__eq__()
#        return out

    def __ne__(self, other):
        out = self.merge(other)
        out = OvaOps(out).__ne__()
        return out

    def log(self):
        self.content[VALUE_COLUMN] = np.log(self.content[VALUE_COLUMN])
        return self

    def log10(self):
        self.content[VALUE_COLUMN] = np.log10(self.content[VALUE_COLUMN])
        return self

    def exp(self):
        self.content[VALUE_COLUMN] = np.exp(self.content[VALUE_COLUMN])
        return self


class Ovariable2(Ovariable):
    def __init__(self, content):
        self.content = content
        if hasattr(content[VALUE_COLUMN], 'pint'):
            self.unit = content[VALUE_COLUMN].pint.units
        else:
            self.unit = "dimensionless"


class OvariableFrame(DataFrame):
    def __init__(self):
        pass


"""