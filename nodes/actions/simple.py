from cmath import nan
from threading import local
from params.param import Parameter
from params import PercentageParameter, NumberParameter
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.node import NodeError
from .action import ActionNode

import pandas as pd


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

    allowed_parameters: list[Parameter] = [
        PercentageParameter('target_year_ratio', min_value=0),
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

            target_year_ratio = self.get_parameter_value('target_year_ratio', required=False)
            if target_year_ratio is not None:
                val *= target_year_ratio / 100  # FIXME This just multiplies ALL years with a constant

            df[col] = val
            if not self.is_enabled():
                df[col] = 0.0
            df[col] = self.ensure_output_unit(df[col])

        return df

    def compute_effect(self):
        df = self.get_input_dataset()
        return self.add_cumulatively(df)


class LinearCumulativeAdditiveAction(CumulativeAdditiveAction):
    allowed_parameters = CumulativeAdditiveAction.allowed_parameters + [
        NumberParameter('target_year_level'),
        NumberParameter(
            local_id='action_delay',
            label='Years of delay (a)',
        ),
    ]

    """Cumulative additive action where a yearly target is set and the effect is linear."""
    def compute_effect(self):
        df = self.get_input_dataset()
        start_year = df.index.min()
        delay = self.get_parameter_value('action_delay', required=False)
        if delay is not None:
            start_year = start_year + int(delay)
        end_year = df.index.max()
        df = df.reindex(range(start_year, end_year + 1))
        df[FORECAST_COLUMN] = True

        target_year_level = self.get_parameter_value('target_year_level', required=False)
        if target_year_level is not None:
            if set(df.columns) != set([VALUE_COLUMN, FORECAST_COLUMN]):
                raise NodeError(self, "target_year_level parameter can only be used with single-value nodes")
            df.loc[end_year, VALUE_COLUMN] = target_year_level
            if delay is not None:
                df.loc[range(start_year + 1, end_year), VALUE_COLUMN] = nan

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


class ExponentialAction(ActionNode):
    allowed_parameters = [
        NumberParameter(
            local_id='current_value',
            unit='EUR/t',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='annual_change',
            unit='%',
            is_customizable=True,
        ),
    ]

    def compute_effect(self):
        current_value = self.get_parameter_value_w_unit('current_value')
        unit = str(current_value.units)
        current_value = current_value.m
        annual_change = self.get_parameter_value_w_unit('annual_change')
        assert {str(annual_change.units)} <= {'%/a', '%/year', '%'}
        annual_change = (1 + annual_change).m

        target_year = self.context.target_year
        start_year = self.context.instance.minimum_historical_year
        current_time = self.context.instance.maximum_historical_year - start_year
        duration = target_year - start_year + 1
        s = [current_value] * duration
        year = []
        forecast = []
        factor = [1]

        for i in range(duration):
            if i > current_time:
                factor = factor + [factor[-1] * annual_change]
                forecast = forecast + [True]
            else:
                factor = factor + [factor[-1]]
                forecast = forecast + [False]
            year = year + [start_year + i]
        if self.is_enabled():
            s = pd.Series(s) * pd.Series(factor[1:])
        df = pd.DataFrame({
            YEAR_COLUMN: year,
            VALUE_COLUMN: pd.Series(s, dtype='pint[' + unit + ']'),
            FORECAST_COLUMN: forecast}).set_index([YEAR_COLUMN])
        return df
