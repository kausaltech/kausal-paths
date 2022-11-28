from __future__ import annotations

from dataclasses import dataclass
import typing
from typing import Iterable, Iterator, Optional
import numpy as np

import pandas as pd
import pint
import pint_pandas
from common.i18n import TranslatedString

from nodes.constants import FORECAST_COLUMN, IMPACT_COLUMN, VALUE_COLUMN, VALUE_WITHOUT_ACTION_COLUMN, DecisionLevel
from nodes import Node, NodeError
from params import BoolParameter

if typing.TYPE_CHECKING:
    from nodes.context import Context


ENABLED_PARAM_ID = 'enabled'


@dataclass
class ActionGroup:
    id: str
    name: TranslatedString | str
    color: str | None


class ActionNode(Node):
    decision_level: DecisionLevel = DecisionLevel.MUNICIPALITY
    group: ActionGroup | None = None

    # The value to use for "no effect" years.
    # For additive actions, it probably is 0, and for multiplicative
    # actions, 1.0.
    no_effect_value: Optional[float] = None
    enabled_param: BoolParameter
    global_parameters: list[str] = [
        'discount_node_name'
    ]

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
        new_index = range(df.index.min(), self.get_end_year() + 1)
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
        df[VALUE_WITHOUT_ACTION_COLUMN] = disabled_df[VALUE_COLUMN]
        df[IMPACT_COLUMN] = df[VALUE_COLUMN] - df[VALUE_WITHOUT_ACTION_COLUMN]
        return df

    def print_impact(self, target_node: Node):
        df = self.compute_impact(target_node)
        self.print_pint_df(df)

    def on_scenario_created(self, scenario):
        super().on_scenario_created(scenario)
        if self.enabled_param.get_scenario_setting(scenario) is None:
            self.enabled_param.add_scenario_setting(scenario.id, scenario.all_actions_enabled)

    def compute_efficiency(self, cost_node: Node, impact_node: Node, unit: pint.Unit) -> pd.DataFrame:
        discount = self.context.get_parameter_value('discount_node_name')
        discount_factor = self.context.get_node(discount).compute()[VALUE_COLUMN]
        minuscule_limit = self.context.get_parameter_value('minuscule_limit')
        # Consider impact as zero if deviates less than this.

        cost = self.compute_impact(cost_node)[IMPACT_COLUMN]
        cost.name = 'Cost'
        impact = self.compute_impact(impact_node)[IMPACT_COLUMN]
        s = (minuscule_limit < impact.pint.m.abs()).astype(int)
        impact *= s
        df = pd.concat([cost], axis=1)
        df['Cost'] *= discount_factor
        df['Impact'] = impact.replace({0: np.nan})
        pd_pt = pint_pandas.PintType(unit)
        df['Efficiency'] = (df['Cost'] / df['Impact']).astype(pd_pt)
        df = df.dropna()
        return df


class ActionEfficiency(typing.NamedTuple):
    action: ActionNode
    df: pd.DataFrame
    cumulative_efficiency: pint.Quantity
    cumulative_cost: pint.Quantity
    cumulative_impact: pint.Quantity


@dataclass
class ActionEfficiencyPair:
    cost_node: Node
    impact_node: Node
    unit: pint.Unit
    label: TranslatedString | str | None

    @classmethod
    def from_config(
        self, context: 'Context', cost_node_id: str, impact_node_id: str, unit: str,
        label: TranslatedString | str | None = None
    ) -> ActionEfficiencyPair:
        cost_node = context.get_node(cost_node_id)
        impact_node = context.get_node(impact_node_id)
        unit_obj = context.unit_registry(unit).u
        aep = ActionEfficiencyPair(cost_node=cost_node, impact_node=impact_node, unit=unit_obj, label=label)
        aep.validate()
        return aep

    def validate(self):
        # Ensure units are compatible
        div_unit = self.cost_node.unit / self.impact_node.unit
        if not self.unit.is_compatible_with(div_unit):
            raise Exception("Unit %s is not compatible with %s" % (self.unit, div_unit))

    def calculate_iter(
        self, context: 'Context', actions: Iterable[ActionNode] | None = None
    ) -> Iterator[ActionEfficiency]:
        if actions is None:
            actions = list(context.get_actions())
        for action in actions:
            df = action.compute_efficiency(self.cost_node, self.impact_node, self.unit)
            if not len(df):
                # No impact for this action, skip it
                continue
            cost = df['Cost'].sum()
            impact = df['Impact'].sum()
            efficiency = (cost / impact).to(self.unit)
            if impact < 0:
                efficiency *= -1
            ae = ActionEfficiency(
                action=action, df=df,
                cumulative_cost=cost,
                cumulative_impact=impact,
                cumulative_efficiency=efficiency
            )
            yield ae

    def calculate(self, context: 'Context', actions: Iterable[ActionNode] | None = None) -> list[ActionEfficiency]:
        out = list(self.calculate_iter(context, actions))
        return out
