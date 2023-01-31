from __future__ import annotations

from dataclasses import dataclass
import typing
from typing import Iterable, Iterator, Optional
import numpy as np

import pandas as pd
import pint_pandas
import polars as pl
from common.i18n import TranslatedString
from common.perf import PerfCounter
from common import polars as ppl

from nodes.constants import (
    FORECAST_COLUMN, IMPACT_GROUP, VALUE_COLUMN, VALUE_WITH_ACTION_GROUP,
    VALUE_WITHOUT_ACTION_GROUP, YEAR_COLUMN, DecisionLevel
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

    def compute_effect(self) -> pd.DataFrame | ppl.PathsDataFrame:
        raise Exception("Implement in subclass")

    def compute(self) -> pd.DataFrame | ppl.PathsDataFrame:
        return self.compute_effect()

    def compute_impact(self, target_node: Node) -> ppl.PathsDataFrame:
        # Determine the impact of this action in the target node
        enabled = self.is_enabled()
        metrics = target_node.output_metrics

        self.enabled_param.set(False)
        ddf = target_node.get_output_pl()
        self.enabled_param.set(enabled)
        edf = target_node.get_output_pl()

        for metric in metrics.values():
            if metric.column_id not in ddf.columns:
                raise NodeError(self, 'Output for node %s did not contain the %s column' % (target_node.id, metric.column_id))

        assert len(ddf) == len(edf)
        ddf = ddf.rename({VALUE_COLUMN: VALUE_WITHOUT_ACTION_GROUP})
        df = edf.paths.join_over_index(ddf)
        df = df.with_columns(
            [(pl.col(VALUE_COLUMN) - pl.col(VALUE_WITHOUT_ACTION_GROUP)).alias(IMPACT_GROUP)],
            units={IMPACT_GROUP: df.get_unit(VALUE_COLUMN)}
        )
        return df

    def print_impact(self, target_node: Node):
        df = self.compute_impact(target_node)
        meta = df.get_meta()
        if meta.dim_ids:
            df = df.paths.to_wide()
        self.print(df)

    def on_scenario_created(self, scenario):
        super().on_scenario_created(scenario)
        if self.enabled_param.get_scenario_setting(scenario) is None:
            self.enabled_param.add_scenario_setting(scenario.id, scenario.all_actions_enabled)

    def compute_efficiency(self, cost_node: Node, impact_node: Node, unit: Unit) -> ppl.PathsDataFrame:
        pc = PerfCounter('Impact %s [%s / %s]' % (self.id, cost_node.id, impact_node.id), level=PerfCounter.Level.DEBUG)

        pc.display('starting')
        cost_df = self.compute_impact(cost_node)
        cost_meta = cost_df.get_meta()
        cost_df = cost_df.select([*cost_meta.primary_keys, FORECAST_COLUMN, pl.col(IMPACT_GROUP).alias('Cost')])
        pc.display('cost impact of %s on %s computed' % (self.id, cost_node.id))
        impact_df = self.compute_impact(impact_node)
        impact_meta = impact_df.get_meta()
        # Replace impact values that are very close to zero with null
        zero_to_nan = pl.when(pl.col(IMPACT_GROUP).abs() < pl.lit(1e-9)).then(pl.lit(None)).otherwise(pl.col(IMPACT_GROUP))
        impact_df = impact_df.select([*impact_meta.primary_keys, FORECAST_COLUMN, zero_to_nan.alias(IMPACT_GROUP)])

        pc.display('impact of %s on %s computed' % (self.id, impact_node.id))
        df = cost_df.paths.join_over_index(impact_df, how='left')
        df = df.with_columns(Efficiency=pl.col('Cost') / pl.col('Impact'))

        df = df.drop_nulls()
        df = df.set_unit('Efficiency', df.get_unit('Cost') / df.get_unit('Impact'))
        df = df.ensure_unit('Efficiency', unit)
        return df


class ActionEfficiency(typing.NamedTuple):
    action: ActionNode
    df: ppl.PathsDataFrame
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
        if self.cost_node.unit is None or self.impact_node.unit is None:
            raise Exception("Cost or impact node does not have a unit")
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

            cost: Quantity = df['Cost'].sum() * df.get_unit('Cost') * Quantity('1 a')  # type: ignore
            if self.invert_cost:
                cost *= -1
            impact: Quantity = df['Impact'].sum() * df.get_unit('Impact') * Quantity('1 a')  # type: ignore
            if self.invert_impact:
                impact *= -1
            efficiency: Quantity = (cost / impact).to(self.unit)  # type: ignore
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
