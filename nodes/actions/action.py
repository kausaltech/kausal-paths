from __future__ import annotations

from dataclasses import dataclass
import typing
from typing import Iterable, Iterator, Optional
import numpy as np

import pandas as pd
import pint_pandas
from common.i18n import TranslatedString
from common.perf import PerfCounter

from nodes.constants import (
    FORECAST_COLUMN, IMPACT_GROUP, VALUE_COLUMN, VALUE_WITH_ACTION_GROUP,
    VALUE_WITHOUT_ACTION_GROUP, DecisionLevel
)
from nodes import Node, NodeError
from nodes.units import Unit, Quantity
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
        metrics = target_node.output_metrics

        self.enabled_param.set(False)
        ddf = target_node.get_output()
        self.enabled_param.set(enabled)
        edf = target_node.get_output()

        for metric in metrics.values():
            if metric.column_id not in ddf.columns:
                raise NodeError(self, 'Output for node %s did not contain the %s column' % (target_node.id, metric.column_id))

        fc = edf.pop(FORECAST_COLUMN)
        ddf.pop(FORECAST_COLUMN)

        if isinstance(edf.columns, pd.MultiIndex):
            other_cols = edf.columns.levels
        else:
            other_cols = [edf.columns]
        new_cols = [[VALUE_WITH_ACTION_GROUP, VALUE_WITHOUT_ACTION_GROUP, IMPACT_GROUP]] + other_cols
        df = pd.DataFrame(columns=pd.MultiIndex.from_product(new_cols), index=edf.index)
        df[VALUE_WITH_ACTION_GROUP] = edf
        df[VALUE_WITHOUT_ACTION_GROUP] = ddf[VALUE_COLUMN]
        df[IMPACT_GROUP] = df[VALUE_WITH_ACTION_GROUP] - df[VALUE_WITHOUT_ACTION_GROUP]
        return df

    def print_impact(self, target_node: Node):
        df = self.compute_impact(target_node)
        self.print_pint_df(df)

    def on_scenario_created(self, scenario):
        super().on_scenario_created(scenario)
        if self.enabled_param.get_scenario_setting(scenario) is None:
            self.enabled_param.add_scenario_setting(scenario.id, scenario.all_actions_enabled)

    def compute_efficiency(self, cost_node: Node, impact_node: Node, unit: Unit) -> pd.DataFrame:
        pc = PerfCounter('Impact %s [%s / %s]' % (self.id, cost_node.id, impact_node.id), level=PerfCounter.Level.DEBUG)

        # FIXME: Discount needs to be handled maybe through ActionEfficiencyPairs?
        # discount = self.context.get_parameter_value('discount_node_name')
        # discount_factor = self.context.get_node(discount).compute()[VALUE_COLUMN]

        pc.display('starting')
        cost = self.compute_impact(cost_node)[IMPACT_COLUMN]
        pc.display('cost impact of %s on %s computed' % (self.id, cost_node.id))
        cost.name = 'Cost'
        impact = self.compute_impact(impact_node)[IMPACT_COLUMN]
        pc.display('impact of %s on %s computed' % (self.id, impact_node.id))
        df = pd.concat([cost], axis=1)
        # df['Cost'] *= discount_factor  # FIXME
        df['Impact'] = impact.replace({0: np.nan})
        pd_pt = pint_pandas.PintType(unit)
        df['Efficiency'] = (df['Cost'] / df['Impact']).astype(pd_pt)
        df = df.dropna()
        pc.display('done')
        return df


class ActionEfficiency(typing.NamedTuple):
    action: ActionNode
    df: pd.DataFrame
    cumulative_efficiency: Quantity
    cumulative_cost: Quantity
    cumulative_impact: Quantity
    cumulative_cost_unit: Unit
    cumulative_impact_unit: Unit


@dataclass
class ActionEfficiencyPair:
    cost_node: Node
    impact_node: Node
    unit: Unit
    plot_limit_efficiency: float | None
    invert_cost: bool
    invert_impact: bool
    label: TranslatedString | str | None

    @classmethod
    def from_config(
        cls, context: 'Context', cost_node_id: str, impact_node_id: str, unit: str,
        plot_limit_efficiency: float | None = None,
        invert_cost: bool = False, invert_impact: bool = True,
        label: TranslatedString | str | None = None
    ) -> ActionEfficiencyPair:
        cost_node = context.get_node(cost_node_id)
        impact_node = context.get_node(impact_node_id)
        unit_obj = context.unit_registry.parse_units(unit)
        aep = ActionEfficiencyPair(
            cost_node=cost_node, impact_node=impact_node, unit=unit_obj,
            invert_cost=invert_cost, invert_impact=invert_impact,
            plot_limit_efficiency=plot_limit_efficiency, label=label)
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

        pc = PerfCounter(
            "Action efficiency %s / %s" % (self.cost_node.id, self.impact_node.id),
            level=PerfCounter.Level.DEBUG)
        pc.display('starting')
        for action in actions:
            if not action.is_connected_to(self.cost_node) or not action.is_connected_to(self.impact_node):
                # Action is not connected to either cost or impact nodes, skip it
                continue

            df = action.compute_efficiency(self.cost_node, self.impact_node, self.unit)
            if not len(df):
                # No impact for this action, skip it
                continue
            cost = df['Cost'].sum() * Quantity('1 a')
            if self.invert_cost:
                cost *= -1
            impact = df['Impact'].sum() * Quantity('1 a')
            if self.invert_impact:
                impact *= -1
            efficiency = (cost / impact).to(self.unit)
            if impact < 0:
                efficiency *= -1
            ae = ActionEfficiency(
                action=action, df=df,
                cumulative_cost=cost,
                cumulative_impact=impact,
                cumulative_efficiency=efficiency,
                cumulative_cost_unit=cost.units,
                cumulative_impact_unit=impact.units
            )
            yield ae

        pc.display("done")

    def calculate(self, context: 'Context', actions: Iterable[ActionNode] | None = None) -> list[ActionEfficiency]:
        out = list(self.calculate_iter(context, actions))
        return out
