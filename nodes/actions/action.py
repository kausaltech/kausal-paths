import pandas as pd
from typing import Optional

from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, DecisionLevel
from nodes import Node, NodeError
from params import BoolParameter


ENABLED_PARAM_ID = 'enabled'


class ActionNode(Node):
    decision_level: DecisionLevel = DecisionLevel.MUNICIPALITY

    # The value to use for "no effect" years.
    # For additive actions, it probably is 0, and for multiplicative
    # actions, 1.0.
    no_effect_value: Optional[float] = None
    enabled_param: BoolParameter

    def __post_init__(self):
        self.enabled_param = BoolParameter(local_id=ENABLED_PARAM_ID)
        self.enabled_param.set(False)
        self.add_parameter(self.enabled_param)

    def is_enabled(self) -> Optional[bool]:
        return self.enabled_param.value

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
        self.enabled_param.set(False)
        disabled_df = target_node.get_output()
        if disabled_df is None:
            raise NodeError(self, 'Output for node %s was null' % target_node.id)
        if VALUE_COLUMN not in disabled_df.columns:
            raise NodeError(self, 'Output for node %s did not contain the Value column' % target_node.id)
        self.enabled_param.set(enabled)
        df = target_node.get_output()
        df['ValueWithoutAction'] = disabled_df[VALUE_COLUMN]
        df['Impact'] = df[VALUE_COLUMN] - df['ValueWithoutAction']
        return df

    def print_impact(self, target_node: Node):
        df = self.compute_impact(target_node)
        self.print_pint_df(df)

    def on_scenario_created(self, scenario):
        super().on_scenario_created(scenario)
        if self.enabled_param.get_scenario_setting(scenario) is None:
            self.enabled_param.add_scenario_setting(scenario.id, scenario.all_actions_enabled)
