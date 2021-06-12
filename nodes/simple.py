from typing import List
import pandas as pd

from common.i18n import gettext_lazy as _
from .constants import FORECAST_COLUMN, VALUE_COLUMN
from .node import Node
from .exceptions import NodeError


EMISSION_UNIT = 'kg'


class AdditiveNode(Node):
    """Simple addition of inputs"""

    def add_nodes(self, df: pd.DataFrame, nodes: List[Node]):
        for node in nodes:
            node_df = node.get_output(self)
            if node_df is None:
                continue

            if df is None:
                df = node_df
                continue

            if VALUE_COLUMN not in df.columns:
                raise NodeError(self, "Value column missing in output of %s" % node.id)
            val1 = self.ensure_output_unit(df[VALUE_COLUMN])
            if hasattr(val1, 'pint'):
                val1 = val1.pint.m
            val2 = self.ensure_output_unit(node_df[VALUE_COLUMN], input_node=node)
            if hasattr(val2, 'pint'):
                val2 = val2.pint.m
            val1 = val1.add(val2, fill_value=0)
            df[VALUE_COLUMN] = self.ensure_output_unit(val1)
            df[FORECAST_COLUMN] = df[FORECAST_COLUMN] | node_df[FORECAST_COLUMN]
        return df

    def compute(self):
        df = self.get_input_dataset()

        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise NodeError(self, "Input is not a DataFrame")
            if VALUE_COLUMN not in df.columns:
                raise NodeError(self, "Input dataset doesn't have Value column")

        return self.add_nodes(df, self.input_nodes)


class SectorEmissions(AdditiveNode):
    quantity = 'emissions'
    """Simple addition of subsector emissions"""
    pass


class MultiplicativeNode(AdditiveNode):
    """Multiply nodes together with potentially adding other input nodes.

    Multiplication and addition is determined based on the input node units.
    """

    def compute(self):
        additive_nodes = []
        multiply_nodes = []
        for node in self.input_nodes:
            if self.is_compatible_unit(node.unit, self.unit):
                additive_nodes.append(node)
            else:
                multiply_nodes.append(node)

        if len(multiply_nodes) != 2:
            raise NodeError(self, "Must receive exactly two multiplicative inputs")

        n1, n2 = multiply_nodes
        output_unit = n1.unit * n2.unit
        if not self.is_compatible_unit(output_unit, self.unit):
            raise NodeError(self, "Multiplying inputs must in a unit compatible with '%s'" % self.unit)

        df1 = n1.get_output()
        df2 = n2.get_output()
        df = df1.copy()
        df[VALUE_COLUMN] *= df2[VALUE_COLUMN]
        df[FORECAST_COLUMN] = df1[FORECAST_COLUMN] | df2[FORECAST_COLUMN]

        df[VALUE_COLUMN] = df[VALUE_COLUMN].pint.to(self.unit)

        """
        ds_df = self.get_input_dataset()
        if ds_df is not None:
            index_diff = ds_df.index.difference(df)
            if len(index_diff):
                df = ...
        """

        return self.add_nodes(df, additive_nodes)


class EmissionFactorActivity(MultiplicativeNode):
    """Multiply an activity by an emission factor."""
    quantity = 'emissions'
    unit = EMISSION_UNIT


class PerCapitaActivity(MultiplicativeNode):
    pass


class Activity(AdditiveNode):
    """Add activity amounts together."""
    pass
