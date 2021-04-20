import pandas as pd
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN
from .base import Action


class AdditiveAction(Action):
    """Simple action that produces an additive change to a value."""
    no_effect_value = 0

    def compute_effect(self):
        return self.get_input_dataset()


class CumulativeAdditiveAction(Action):
    """Additive action where the effect is cumulative and remains in the future."""

    def compute_effect(self):
        df = self.get_input_dataset()
        target_year = self.get_target_year()

        df = df.reindex(range(df.index.min(), target_year + 1))
        df[FORECAST_COLUMN] = True

        val = df[VALUE_COLUMN]
        if hasattr(val, 'pint'):
            val = val.pint.m
        val = val.fillna(0).cumsum()
        df[VALUE_COLUMN] = self.ensure_output_unit(val)

        return df


class EmissionReductionAction(Action):
    """Simple emission reduction impact"""

    no_effect_value = 0

    def compute_effect(self):
        df = self.get_input_dataset()
        df[VALUE_COLUMN] = 0 - df[VALUE_COLUMN]
        return df
