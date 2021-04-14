import pandas as pd
from .base import Node, _


EMISSION_UNIT = 'kg'


class SectorEmissions(Node):
    """Simple addition of subsector emissions"""

    units = {
        'Emissions': EMISSION_UNIT
    }

    def compute(self):
        print(self)
        df = self.get_input_dataset()
        if df is not None:
            print(df)

        for node in self.input_nodes:
            df = node.compute()

        return pd.DataFrame()

