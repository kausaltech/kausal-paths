
from typing import Optional
import pandas as pd

from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, DecisionLevel
from nodes import Node
from params import BoolParameter


ENABLED_PARAM_ID = 'enabled'


class ActionNode(Node):
    decision_level: DecisionLevel = DecisionLevel.MUNICIPALITY

    # The value to use for "no effect" years.
    # For additive actions, it probably is 0, and for multiplicative
    # actions, 1.0.
    no_effect_value: float = None

    def __post_init__(self):
        self.param_defaults = {}
        self.params = {}

    def is_enabled(self) -> bool:
        return self.get_param_value(ENABLED_PARAM_ID)

    def register_params(self):
        super().register_params()
        self.register_param(BoolParameter(id=ENABLED_PARAM_ID))

    """
    def set_params(self, params: Dict[str, Any]):
        self.params.update(params)
        if self.params.get('enabled'):
            self.enable()
        else:
            self.disable()
    """

    def forecast_series(self, series: pd.Series):
        df = pd.DataFrame(index=series.index)
        # Reindex the forecasted series to fill in years that
        # are not defined.
        df[VALUE_COLUMN] = series.values
        new_index = range(df.index.min(), self.get_target_year() + 1)
        df = df.reindex(new_index, fill_value=self.no_effect_value)
        df[FORECAST_COLUMN] = True
        return df

    def compute_effect(self) -> pd.DataFrame:
        raise Exception("Implement in subclass")

    def compute(self) -> pd.DataFrame:
        return self.compute_effect()

    def compute_impact(self, target_node: Node) -> pd.DataFrame:
        # Determine the impact of this action in the target node
        enabled = self.is_enabled()
        self.set_param_value(ENABLED_PARAM_ID, False)
        disabled_df = target_node.get_output()
        assert disabled_df is not None and VALUE_COLUMN in disabled_df.columns
        self.set_param_value(ENABLED_PARAM_ID, enabled)
        df = target_node.get_output()
        df['ValueWithoutAction'] = disabled_df[VALUE_COLUMN]
        df['Impact'] = df[VALUE_COLUMN] - df['ValueWithoutAction']
        return df

    def print_impact(self, target_node: Node):
        df = self.compute_impact(target_node)
        self.print_pint_df(df)

    def on_scenario_created(self, scenario):
        super().on_scenario_created(scenario)
        param = self.get_param('enabled')
        scenario.params[param.id] = scenario.all_actions_enabled
