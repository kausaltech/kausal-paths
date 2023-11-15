from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import ClassVar, Iterable, Iterator, Optional

import pandas as pd
import polars as pl

from common import polars as ppl
from common.i18n import TranslatedString, gettext_lazy as _
from common.perf import PerfCounter

from nodes.node import Node, NodeError
from nodes.constants import (
    FORECAST_COLUMN, IMPACT_COLUMN, IMPACT_GROUP, SCENARIO_ACTION_GROUP,
    VALUE_COLUMN, WITHOUT_ACTION_GROUP, YEAR_COLUMN, DecisionLevel
)
from nodes.units import Quantity, Unit
from params import BoolParameter
from params.param import Parameter

if typing.TYPE_CHECKING:
    from nodes.context import Context
    from .parent import ParentActionNode


ENABLED_PARAM_ID = 'enabled'


@dataclass
class ActionGroup:
    id: str
    name: TranslatedString | str
    color: str | None


class EnabledParam(BoolParameter):
    def set_node(self, node: Node):
        if self.context is not None:
            # If the Instance defines a custom label for the 'enabled' parameter,
            # replace the default with it.
            instance = self.context.instance
            if instance.terms.enabled_label:
                self.label = instance.terms.enabled_label
        return super().set_node(node)


ENABLED_PARAM = EnabledParam(
    local_id=ENABLED_PARAM_ID, label=_("Is implemented"), description=_("Is the action included in the scenario"),
    is_customizable=True
)


class ActionNode(Node):
    decision_level: DecisionLevel = DecisionLevel.MUNICIPALITY
    group: ActionGroup | None = None
    parent_action: 'ParentActionNode' | None = None

    # The value to use for "no effect" years.
    # For additive actions, it probably is 0, and for multiplicative
    # actions, 1.0.
    no_effect_value: Optional[float] = None
    enabled_param: BoolParameter
    allowed_parameters: ClassVar[list[Parameter]] = [ENABLED_PARAM]

    def __init_subclass__(cls) -> None:
        """Ensure the 'enabled' parameter is allowed for all action classes."""
        for p in cls.allowed_parameters:
            if p.local_id == ENABLED_PARAM_ID:
                break
        else:
            # No 'enabled' parameter in allowed_parameters – add it here.
            cls.allowed_parameters = [
                ENABLED_PARAM,
                *cls.allowed_parameters,
            ]
        super().__init_subclass__()

    def finalize_init(self):
        if hasattr(self, 'enabled_param'):
            # Init already called
            return

        param = self.get_parameter(ENABLED_PARAM_ID, required=False)
        if param is None:
            for param in self.allowed_parameters:
                if param.local_id == ENABLED_PARAM_ID:
                    break
            else:
                raise NodeError(self, "'enabled' is missing from allowed parameters")
            param = param.copy()
            param.context = self.context
            self.add_parameter(param)
        assert isinstance(param, BoolParameter)
        assert param.node == self
        if param.value is None:
            param.set(False, notify=False)
        self.enabled_param = param

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

        edf = target_node.get_output_pl()
        if enabled:
            self.enabled_param.set(False)
            ddf = target_node.get_output_pl()
            self.enabled_param.set(True)
        else:
            ddf = edf

        assert len(ddf) == len(edf)

        metrics = target_node.output_metrics.values()
        mcols = [m.column_id for m in metrics]
        renames = {}
        for m in mcols:
            if m not in ddf.metric_cols:
                raise NodeError(self, 'Output of %s did not contain the %s metric column' % (target_node.id, m))
            renames[m] = '%s:WithoutAction' % m
        ddf = ddf.rename(renames)
        df = edf.paths.join_over_index(ddf)

        value_vars = []
        for m in mcols:
            df = df.with_columns([
                (pl.col(m) - pl.col('%s:WithoutAction' % m)).alias('%s:Impact' % m)
            ]).set_unit('%s:Impact' % m, df.get_unit(m))
            value_vars += [m, '%s:WithoutAction' % m, '%s:Impact' % m]

        common_cols = [YEAR_COLUMN, *df.dim_ids, FORECAST_COLUMN]
        edf = df.select([*common_cols, pl.lit(SCENARIO_ACTION_GROUP).alias(IMPACT_COLUMN), *mcols])
        ddf = df.select([*common_cols, pl.lit(WITHOUT_ACTION_GROUP).alias(IMPACT_COLUMN), *[pl.col('%s:WithoutAction' % m).alias(m) for m in mcols]])
        idf = df.select([*common_cols, pl.lit(IMPACT_GROUP).alias(IMPACT_COLUMN), *[pl.col('%s:Impact' % m).alias(m) for m in mcols]])

        meta = edf.get_meta()
        zdf = pl.concat([edf, ddf, idf], how='vertical')
        df = ppl.to_ppdf(zdf, meta=meta)
        df = df.add_to_index(IMPACT_COLUMN)
        return df

    def print_impact(self, target_node: Node):
        df = self.compute_impact(target_node)
        if self.context.active_normalization:
            _, df = self.context.active_normalization.normalize_output(metric=target_node.get_default_output_metric(), df=df)
        meta = df.get_meta()
        if meta.dim_ids:
            df = df.paths.to_wide()
        self.print(df)

    def on_scenario_created(self, scenario):
        super().on_scenario_created(scenario)
        if not scenario.has_parameter(self.enabled_param):
            scenario.add_parameter(self.enabled_param, scenario.all_actions_enabled)

    def compute_efficiency(self, cost_node: Node, impact_node: Node, unit: Unit) -> ppl.PathsDataFrame:
        pc = PerfCounter('Impact %s [%s / %s]' % (self.id, cost_node.id, impact_node.id), level=PerfCounter.Level.DEBUG)

        pc.display('starting')
        with self.context.perf_context.exec_node(cost_node):
            cost_df = self.compute_impact(cost_node)
            cost_m = cost_node.get_default_output_metric()
            cost_df = (
                cost_df.filter(pl.col(IMPACT_COLUMN).eq(IMPACT_GROUP)).drop(IMPACT_COLUMN)
            )
            cost_df = cost_df.select([*cost_df.primary_keys, FORECAST_COLUMN, pl.col(cost_m.column_id).alias('Cost')])
        pc.display('cost impact of %s on %s computed' % (self.id, cost_node.id))

        with self.context.perf_context.exec_node(impact_node):
            impact_df = self.compute_impact(impact_node)
            impact_m = impact_node.get_default_output_metric()
            impact_df = (
                impact_df.filter(pl.col(IMPACT_COLUMN).eq(IMPACT_GROUP)).drop(IMPACT_COLUMN)
            )
            # Replace impact values that are very close to zero with null
            zero_to_nan = pl.when(pl.col(impact_m.column_id).abs() < pl.lit(1e-9)).then(pl.lit(None)).otherwise(pl.col(impact_m.column_id))
            impact_df = (
                impact_df.select([*impact_df.primary_keys, FORECAST_COLUMN, zero_to_nan.alias('Impact')])
                .set_unit('Impact', impact_df.get_unit(impact_m.column_id))
            )

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
    efficiency_divisor: float
    cumulative_efficiency: Quantity | None
    cumulative_cost: Quantity
    cumulative_impact: Quantity
    cumulative_cost_unit: Unit
    cumulative_impact_unit: Unit


@dataclass
class ActionEfficiencyPair:
    cost_node: Node
    impact_node: Node
    efficiency_unit: Unit
    cost_unit: Unit
    impact_unit: Unit
#    efficiency_divisor: float
    plot_limit_efficiency: float | None
    invert_cost: bool
    invert_impact: bool
    label: TranslatedString | str | None

    @classmethod
    def from_config(
        cls, context: 'Context', cost_node_id: str, impact_node_id: str,
        efficiency_unit: str, cost_unit: str, impact_unit: str,
#        efficiency_divisor: float,
        plot_limit_efficiency: float | None = None,
        invert_cost: bool = False, invert_impact: bool = True,
        label: TranslatedString | str | None = None
    ) -> ActionEfficiencyPair:
        cost_node = context.get_node(cost_node_id)
        impact_node = context.get_node(impact_node_id)
        efficiency_unit_obj = context.unit_registry.parse_units(efficiency_unit)
        cost_unit_obj = context.unit_registry.parse_units(cost_unit)
        impact_unit_obj = context.unit_registry.parse_units(impact_unit)
        aep = ActionEfficiencyPair(
            cost_node=cost_node, impact_node=impact_node, efficiency_unit=efficiency_unit_obj,
            cost_unit=cost_unit_obj, impact_unit=impact_unit_obj,
#            efficiency_divisor=efficiency_divisor,
            invert_cost=invert_cost, invert_impact=invert_impact,
            plot_limit_efficiency=plot_limit_efficiency, label=label)
        aep.validate()
        return aep

    def validate(self):
        # Ensure units are compatible
        if self.cost_node.unit is None or self.impact_node.unit is None:
            raise Exception("Cost or impact node does not have a unit")
        div_unit = self.cost_node.unit / self.impact_node.unit
        if not self.efficiency_unit.is_compatible_with(div_unit):
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

            with context.perf_context.exec_node(action):
                df = action.compute_efficiency(self.cost_node, self.impact_node, self.efficiency_unit)
            if not len(df):
                # No impact for this action, skip it
                continue

            df = df.set_unit('Cost', df.get_unit('Cost') * Quantity('1 a'), force=True)
            df = df.set_unit('Impact', df.get_unit('Impact') * Quantity('1 a'), force=True)
            df = df.ensure_unit('Cost', self.cost_unit)
            df = df.ensure_unit('Impact', self.impact_unit)

            ccost: Quantity = df['Cost'].sum() * df.get_unit('Cost') * Quantity('1 a')  # type: ignore
            if self.invert_cost:
                ccost *= -1
            cimpact: Quantity = df['Impact'].sum() * df.get_unit('Impact') * Quantity('1 a')  # type: ignore
            if self.invert_impact:
                cimpact *= -1

            efficiency: Quantity | None
            if abs(cimpact.m) < 1e-9:
                cimpact = 0 * cimpact.units  # type: ignore
                efficiency = None
            else:
                efficiency = (ccost / cimpact).to(self.efficiency_unit)  # type: ignore

            efficiency_divisor = 1 * self.efficiency_unit * self.impact_unit / self.cost_unit
            efficiency_divisor = efficiency_divisor.to('dimensionless')

            ae = ActionEfficiency(
                action=action,
                df=df,
                efficiency_divisor=efficiency_divisor,
                cumulative_cost=ccost,
                cumulative_impact=cimpact,
                cumulative_efficiency=efficiency,
                cumulative_cost_unit=ccost.units,
                cumulative_impact_unit=cimpact.units
            )
            yield ae

        pc.display("done")

    def calculate(self, context: 'Context', actions: Iterable[ActionNode] | None = None) -> list[ActionEfficiency]:
        out = list(self.calculate_iter(context, actions))
        return out
