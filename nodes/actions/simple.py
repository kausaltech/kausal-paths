from cmath import nan
from threading import local
from params.param import Parameter
from params import PercentageParameter, NumberParameter
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.node import NodeError
from .action import ActionNode

import pandas as pd
import pint_pandas


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
                val *= target_year_ratio / 100

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
        current_value = self.get_parameter('current_value')
        pt = pint_pandas.PintType(current_value.unit)
        base_value = self.get_parameter('annual_change')
        base_unit = base_value.unit
        if self.is_enabled():
            current_value = current_value.value
            base_value = base_value.value
        else:
            current_value = current_value.scenario_settings['default']
            base_value = base_value.scenario_settings['default']
        base_value = 1 + (base_value * base_unit).to('dimensionless').m
        start_year = self.context.instance.minimum_historical_year
        target_year = self.get_target_year()
        current_year = self.context.instance.maximum_historical_year

        df = pd.DataFrame(
            {VALUE_COLUMN: range(start_year - current_year, target_year - current_year + 1)},
            index=range(start_year, target_year + 1))
        val = current_value * base_value ** df[VALUE_COLUMN]
        df[VALUE_COLUMN] = val.astype(pt)
        df[FORECAST_COLUMN] = df.index > current_year

        return df
