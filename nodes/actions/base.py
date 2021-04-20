import pandas as pd

from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN
from typing import Any, Dict
from nodes import Node


class Action(Node):
    enabled: bool
    param_defaults: Dict[str, Any]
    params: Dict[str, Any]

    # The value to use for "no effect" years.
    # For additive actions, it probably is 0, and for multiplicative
    # actions, 1.0.
    no_effect_value: float = None

    def __post_init__(self):
        self.param_defaults = {}
        self.params = {}
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def set_params(self, params: Dict[str, Any]):
        self.params.update(params)
        if self.params.get('enabled'):
            self.enable()
        else:
            self.disable()

    def forecast_series(self, series: pd.Series):
        df = pd.DataFrame(index=series.index)
        # Reindex the forecasted series to fill in years that
        # are not defined.
        df[VALUE_COLUMN] = series.values
        new_index = range(df.index.min(), self.get_target_year() + 1)
        df = df.reindex(new_index, fill_value=self.no_effect_value)
        df[FORECAST_COLUMN] = True
        return df
