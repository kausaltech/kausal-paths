from params import NumberParameter
import pandas as pd
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN
from .action import ActionNode


class AdditiveAction(ActionNode):
    """Simple action that produces an additive change to a value."""
    no_effect_value = 0

    def compute_effect(self):
        df = self.get_input_dataset()
        if not self.is_enabled():
            df[VALUE_COLUMN] = 0.0
        return df


class CumulativeAdditiveAction(ActionNode):
    """Additive action where the effect is cumulative and remains in the future."""

    allowed_params = [
        NumberParameter('target_year_ratio', min_value=0, default_value=1)
    ]

    def compute_effect(self):
        if self.is_enabled():
            df = pd.DataFrame
        df = self.get_input_dataset()
        target_year = self.get_target_year()

        df = df.reindex(range(df.index.min(), target_year + 1))
        df[FORECAST_COLUMN] = True

        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue

            val = df[col]
            if hasattr(val, 'pint'):
                val = val.pint.m
            val = val.fillna(0).cumsum()

            target_year_ratio = self.get_param_value('target_year_ratio', local=True, required=False)
            if target_year_ratio is not None:
                val *= target_year_ratio

            df[col] = self.ensure_output_unit(val)
            if not self.is_enabled():
                df[col] = 0.0

        return df


class EmissionReductionAction(ActionNode):
    """Simple emission reduction impact"""

    no_effect_value = 0

    def compute_effect(self):
        df = self.get_input_dataset()
        df[VALUE_COLUMN] = 0 - df[VALUE_COLUMN]
        return df
