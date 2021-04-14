import pandas as pd
from .constants import FORECAST_COLUMN, VALUE_COLUMN
from .base import Node, _


EMISSION_UNIT = 'kg'


class SectorEmissions(Node):
    """Simple addition of subsector emissions"""

    units = {
        'Emissions': EMISSION_UNIT
    }

    def compute(self):
        df = self.get_input_dataset()

        for node in self.input_nodes:
            node_df = node.get_output()
            if df is None:
                df = node_df
            else:
                df[VALUE_COLUMN] += node_df[VALUE_COLUMN]
                df[FORECAST_COLUMN] = df[FORECAST_COLUMN] | node_df[FORECAST_COLUMN]

        return df
