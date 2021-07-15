import pandas as pd
from typing import Optional

from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, DecisionLevel
from nodes import Context, Node
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

    def forecast_series(self, context: Context, series: pd.Series):
        df = pd.DataFrame(index=series.index)
        # Reindex the forecasted series to fill in years that
        # are not defined.
        df[VALUE_COLUMN] = series.values
        new_index = range(df.index.min(), self.get_target_year(context) + 1)
        df = df.reindex(new_index, fill_value=self.no_effect_value)
        df[FORECAST_COLUMN] = True
        return df

    def compute_effect(self, context: Context) -> pd.DataFrame:
        raise Exception("Implement in subclass")

    def compute(self, context: Context) -> pd.DataFrame:
        return self.compute_effect(context)

    def compute_impact(self, context: Context, target_node: Node) -> pd.DataFrame:
        # Determine the impact of this action in the target node
        enabled = self.is_enabled()
        self.enabled_param.set(False)
        disabled_df = target_node.get_output(context)
        assert disabled_df is not None and VALUE_COLUMN in disabled_df.columns
        self.enabled_param.set(enabled)
        df = target_node.get_output(context)
        df['ValueWithoutAction'] = disabled_df[VALUE_COLUMN]
        df['Impact'] = df[VALUE_COLUMN] - df['ValueWithoutAction']
        return df

    def print_impact(self, context: Context, target_node: Node):
        df = self.compute_impact(context, target_node)
        self.print_pint_df(df)

    def on_scenario_created(self, scenario):
        super().on_scenario_created(scenario)
        if self.enabled_param.get_scenario_setting(scenario) is None:
            self.enabled_param.add_scenario_setting(scenario.id, scenario.all_actions_enabled)
