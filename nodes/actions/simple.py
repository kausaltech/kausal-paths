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

        df = self.get_input_dataset()
        df[VALUE_COLUMN] = 0 - df[VALUE_COLUMN]
        return df
