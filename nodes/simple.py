import pandas as pd

from common.i18n import gettext_lazy as _
from .constants import FORECAST_COLUMN, VALUE_COLUMN
from .node import Node


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
            if node_df is None:
                continue
            if df is None:
                df = node_df
            else:
                val = df[VALUE_COLUMN]
                val = val.add(node_df[VALUE_COLUMN], fill_value=0)
                df[VALUE_COLUMN] = val
                df[FORECAST_COLUMN] = df[FORECAST_COLUMN] | node_df[FORECAST_COLUMN]

        return df
