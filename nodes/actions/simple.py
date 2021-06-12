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
            df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])
        return df


class CumulativeAdditiveAction(ActionNode):
    """Additive action where the effect is cumulative and remains in the future."""

    allowed_params = [
        NumberParameter('target_year_ratio', min_value=0, default_value=1, unit='%')
    ]

    def add_cumulatively(self, df):
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

            df[col] = val
            if not self.is_enabled():
                df[col] = 0.0
            df[col] = self.ensure_output_unit(df[col])

        return df

    def compute_effect(self):
        df = self.get_input_dataset()
        return self.add_cumulatively(df)


class LinearCumulativeAdditiveAction(CumulativeAdditiveAction):
    """Cumulative additive action where a yearly target is set and the effect is linear."""
    def compute_effect(self):
        df = self.get_input_dataset()
        start_year = df.index.min()
        end_year = df.index.max()
        df = df.reindex(range(start_year, end_year + 1))
        df[FORECAST_COLUMN] = True
        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            dt = df.dtypes[col]
            df[col] = df[col].pint.m.interpolate(method='linear').diff().fillna(0).astype(dt)
        return self.add_cumulatively(df)


class EmissionReductionAction(ActionNode):
    """Simple emission reduction impact"""

    no_effect_value = 0

    def compute_effect(self):
        df = self.get_input_dataset()
        df[VALUE_COLUMN] = 0 - df[VALUE_COLUMN]
        return df
