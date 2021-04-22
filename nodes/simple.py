import pandas as pd

from common.i18n import gettext_lazy as _
from .constants import FORECAST_COLUMN, VALUE_COLUMN
from .node import Node
from .exceptions import NodeError


EMISSION_UNIT = 'kg'


class AdditiveNode(Node):
    """Simple addition of inputs"""

    def compute(self):
        df = self.get_input_dataset()

        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise NodeError(self, "Input is not a DataFrame")
            if VALUE_COLUMN not in df.columns:
                raise NodeError(self, "Input dataset doesn't have Value column")

        for node in self.input_nodes:
            node_df = node.get_output()
            if node_df is None:
                continue

            if df is None:
                df = node_df
            else:
                val1 = self.ensure_output_unit(df[VALUE_COLUMN])
                if hasattr(val1, 'pint'):
                    val1 = val1.pint.m
                val2 = self.ensure_output_unit(node_df[VALUE_COLUMN])
                if hasattr(val2, 'pint'):
                    val2 = val2.pint.m
                val1 = val1.add(val2, fill_value=0)
                df[VALUE_COLUMN] = self.ensure_output_unit(val1)
                df[FORECAST_COLUMN] = df[FORECAST_COLUMN] | node_df[FORECAST_COLUMN]

        return df


class SectorEmissions(AdditiveNode):
    quantity = 'emissions'
    """Simple addition of subsector emissions"""
    pass


class MultiplicativeNode(Node):
    """Multiply nodes together."""

    def compute(self):
        if len(self.input_nodes) != 2:
            raise NodeError(self, "Must receive exactly two inputs")

        n1, n2 = self.input_nodes
        output_unit = n1.unit * n2.unit
        if not self.is_compatible_unit(output_unit, self.unit):
            raise NodeError(self, "Multiplying inputs must in a unit compatible with '%s'" % self.unit)

        df1 = n1.get_output()
        df2 = n2.get_output()
        df = df1.copy()
        df[VALUE_COLUMN] *= df2[VALUE_COLUMN]
        df[FORECAST_COLUMN] = df1[FORECAST_COLUMN] | df2[FORECAST_COLUMN]

        df[VALUE_COLUMN] = df[VALUE_COLUMN].pint.to(self.unit)

        return df


class EmissionFactorActivity(MultiplicativeNode):
    """Multiply an activity by an emission factor."""
    quantity = 'emissions'
    unit = EMISSION_UNIT


class PerCapita(MultiplicativeNode):
    pass


class Activity(AdditiveNode):
    """Add activity amounts together."""
    pass
