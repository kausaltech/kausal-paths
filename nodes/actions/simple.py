import pandas as pd
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN
from .base import Action


EMISSION_UNIT = 'kg'


class EmissionReductionAction(Action):
    """Simple emission reduction impact"""

    units = {
        'Emissions': EMISSION_UNIT
    }
    no_effect_value = 0

    def compute(self):
        if not self.enabled:
            return None

        vals = range(20)
        years = [x + 2019 for x in vals]
        s = pd.Series(list(vals), index=years)
        s = 0 - s
        return self.forecast_series(s)
