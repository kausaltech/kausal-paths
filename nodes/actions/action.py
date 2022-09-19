import pandas as pd
from typing import Optional

from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, DecisionLevel
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

    def compute_action_efficiency(self) -> pd.Series:  # FIXME Allows only one afficiency metric
        def get_discount_factor(base_value):
            target_year = self.context.target_year
            start_year = self.context.instance.minimum_historical_year
            current_time = self.context.instance.maximum_historical_year - start_year
            duration = target_year - start_year + 1
            year = []
            factor = [1]

            for i in range(duration):
                if i > current_time:
                    factor = factor + [factor[-1] * base_value]
                else:
                    factor = factor + [factor[-1]]
                year = year + [start_year + i]

            s = pd.Series(factor[1:]).set_index(year)
            return s

        cost = self.context.get_node(self.context.get_parameter_value('cost_node'))
        impact = self.context.get_node(self.context.get_parameter_value('impact_node'))
        efficiency_unit = self.context.get_parameter_value('efficiency_unit')
        discount_rate = self.context.get_parameter_value_w_unit('discount_rate')
        discount_factor = get_discount_factor(discount_rate)

        cost = self.compute_impact(cost)[VALUE_COLUMN]
        impact = self.compute_impact(impact)[VALUE_COLUMN]
        efficiency = (cost * discount_factor).sum / (impact).sum
        efficiency = efficiency.astype('pint[' + efficiency_unit + ']')

        return [cost.sum, impact.sum, efficiency]
