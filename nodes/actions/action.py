from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import ClassVar, cast

import pandas as pd
import polars as pl

from kausal_common.debugging.perf import PerfCounter

from common import polars as ppl
from common.i18n import TranslatedString, gettext_lazy as _
from nodes.constants import (
    FORECAST_COLUMN,
    IMPACT_COLUMN,
    IMPACT_GROUP,
    SCENARIO_ACTION_GROUP,
    STACKABLE_QUANTITIES,
    UNCERTAINTY_COLUMN,
    VALUE_COLUMN,
    WITHOUT_ACTION_GROUP,
    YEAR_COLUMN,
    DecisionLevel,
)
from nodes.node import Node, NodeError
from nodes.units import Quantity, Unit
from params import BoolParameter, NumberParameter

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from nodes.context import Context
    from params.param import Parameter

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
    local_id=ENABLED_PARAM_ID,
    label=_('Is implemented'),
    description=_('Is the action included in the scenario'),
    is_customizable=True,
)


class ActionNode(Node):
    global_parameters = ['action_impact_from_baseline']
    decision_level: DecisionLevel = DecisionLevel.MUNICIPALITY
    group: ActionGroup | None = None
    parent_action: ParentActionNode | None = None

    # The value to use for "no effect" years.
    # For additive actions, it probably is 0, and for multiplicative
    # actions, 1.0.
    no_effect_value: float | None = None
    enabled_param: BoolParameter
    allowed_parameters: ClassVar[Sequence[Parameter]] = [
        ENABLED_PARAM,
        NumberParameter(local_id='multiplier', label='Multiplies the output', is_customizable=True),
    ]

    def __init_subclass__(cls) -> None:
        """Ensure the 'enabled' parameter is allowed for all action classes."""
        for p in cls.allowed_parameters:
            if p.local_id == ENABLED_PARAM_ID:
                break
        else:
            # No 'enabled' parameter in allowed_parameters - add it here.
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

    def is_enabled(self) -> bool:
        return bool(self.enabled_param.value)

    def forecast_series(self, series: pd.Series):
        df = pd.DataFrame(index=series.index)
        # Reindex the forecasted series to fill in years that
        # are not defined.
        df[VALUE_COLUMN] = series.values
        new_index = range(df.index.min(), self.get_end_year() + 1)
        df = df.reindex(new_index, fill_value=self.no_effect_value)
        df[FORECAST_COLUMN] = True
        return df

    # See also sister function in SimpleNode
    def apply_multiplier(self, df: ppl.PathsDataFrame, required, units) -> ppl.PathsDataFrame:
        multiplier = self.get_parameter_value('multiplier', required=required, units=units)
        if multiplier:
            if isinstance(multiplier, Quantity):
                df = df.multiply_quantity(VALUE_COLUMN, multiplier)
            else:
                df = df.with_columns((pl.col(VALUE_COLUMN) * pl.lit(multiplier)).alias(VALUE_COLUMN))
            df = df.ensure_unit(VALUE_COLUMN, self.single_metric_unit)
        return df

    def compute_effect(self) -> pd.DataFrame | ppl.PathsDataFrame:
        raise Exception('Implement in subclass')

    def compute(self) -> pd.DataFrame | ppl.PathsDataFrame:
        return self.compute_effect()

    def compute_impact(self, target_node: Node) -> ppl.PathsDataFrame:
        from_baseline: bool | None = cast(
            bool, self.get_global_parameter_value('action_impact_from_baseline', required=False) or False,
        )

        was_enabled = self.is_enabled()
        if from_baseline:
            # Calculate impact by first activating the baseline scenario
            # and then enabling just this one action.
            baseline = self.context.get_scenario('baseline')
            with baseline.override():
                ddf = target_node.get_output_pl()
                if self.is_enabled() != was_enabled:
                    self.enabled_param.set(was_enabled)
                    edf = target_node.get_output_pl()
                else:
                    edf = ddf
        else:
            # Calculate impact by disabling the action, computing
            # the results, and then do the same after the action
            # is enabled.
            with self.enabled_param.override(value=False):
                # Determine the impact of this action in the target node
                ddf = target_node.get_output_pl(extra_span_desc='action disabled')
            if self.is_enabled():
                edf = target_node.get_output_pl(extra_span_desc='action enabled')
            else:
                edf = ddf

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
        impact_cols = []
        impact_units = {}
        for m in mcols:
            wc = pl.col(m)
            woc = pl.col('%s:WithoutAction' % m)
            # If the values are very close to each other, make them match.
            tol = 1e-6
            woc = pl.when((wc - woc).abs() < (tol * woc).abs()).then(wc).otherwise(woc)
            ic_name = '%s:Impact' % m
            ic = (wc - woc).alias(ic_name)
            impact_cols.append(ic)
            value_vars += [m, '%s:WithoutAction' % m, ic_name]
            impact_units[ic_name] = df.get_unit(m)

        df = df.with_columns(impact_cols)
        for col, unit in impact_units.items():
            df = df.set_unit(col, unit)
        common_cols = [YEAR_COLUMN, *df.dim_ids, FORECAST_COLUMN]
        edf = df.select([*common_cols, pl.lit(SCENARIO_ACTION_GROUP).alias(IMPACT_COLUMN), *mcols])
        ddf = df.select(
            [
                *common_cols,
                pl.lit(WITHOUT_ACTION_GROUP).alias(IMPACT_COLUMN),
                *[pl.col('%s:WithoutAction' % m).alias(m) for m in mcols],
            ],
        )
        idf = df.select(
            [*common_cols, pl.lit(IMPACT_GROUP).alias(IMPACT_COLUMN), *[pl.col('%s:Impact' % m).alias(m) for m in mcols]],
        )

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

    def compute_indicator(self, cost_node: Node, impact_node: Node, match_dims_i: list,
                           match_dims_c: list, graph_type: str) -> ppl.PathsDataFrame:
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

        if graph_type == 'value_of_information':
            match_dims_c += [UNCERTAINTY_COLUMN]
            meta = cost_df.get_meta()
            df = cost_df.filter(pl.col(UNCERTAINTY_COLUMN).ne('median'))
            last_forecast_year = df.filter(pl.col(FORECAST_COLUMN)).select(YEAR_COLUMN).max()
            dfp = df.group_by(pl.col(UNCERTAINTY_COLUMN)).agg(pl.sum('Cost'))

            dfp = dfp.with_columns([
                pl.when(pl.col('Cost') > 0.0).then(pl.col('Cost'))
                .otherwise(pl.lit(0.0)).alias('under_knowledge')
            ])

            dfp = dfp.select(pl.all().mean())
            dfp = pl.concat([dfp, last_forecast_year], how='horizontal')
            dfp = dfp.with_columns([
                (pl.col('under_knowledge') - pl.col('Cost')).alias('Cost'),
                pl.lit(value=True).alias(FORECAST_COLUMN),
                pl.lit(0.0).alias('Impact'),  # Impact is not used with value of information
                pl.lit('expectation').alias(UNCERTAINTY_COLUMN)
            ])
            dfp = dfp.select([YEAR_COLUMN, FORECAST_COLUMN, UNCERTAINTY_COLUMN, 'Cost', 'Impact']) # TODO Do we need this?
            meta.units['Impact'] = meta.units['Cost']
            df = ppl.to_ppdf(df=dfp, meta=meta)

        else:

            with self.context.perf_context.exec_node(impact_node):
                impact_df = self.compute_impact(impact_node)
                impact_m = impact_node.get_default_output_metric()
                impact_df = (
                    impact_df.filter(pl.col(IMPACT_COLUMN).eq(IMPACT_GROUP)).drop(IMPACT_COLUMN)
                )
                # Replace impact values that are very close to zero with null
                zero_to_nan = (
                    pl.when(pl.col(impact_m.column_id).abs() < pl.lit(1e-9))
                    .then(pl.lit(None))
                    .otherwise(pl.col(impact_m.column_id))
                )
                impact_df = (
                    impact_df.select([*impact_df.primary_keys, FORECAST_COLUMN, zero_to_nan.alias('Impact')])
                    .set_unit('Impact', impact_df.get_unit(impact_m.column_id))
                )
            pc.display('impact of %s on %s computed' % (self.id, impact_node.id))

            df = cost_df.paths.join_over_index(impact_df, how='outer', index_from='union')
            df = df.with_columns([
                pl.col('Cost').fill_null(0.0),
                pl.col('Impact').fill_null(0.0)
            ])

            if not set(impact_df.dim_ids) == set(match_dims_i):
                raise NodeError(
                    self, """With impact overview %s, impact node %s dimensions %s
                    do not match with expected: %s.""" %
                    (graph_type, impact_node.id, impact_df.dim_ids, match_dims_i))

        if not set(cost_df.dim_ids) == set(match_dims_c):
            raise NodeError(
                self, """With impact overview %s, cost node %s dimensions %s
                do not match with expected: %s.""" %
                (graph_type, cost_node.id, cost_df.dim_ids, match_dims_c))

        return df


class ActionEfficiency(typing.NamedTuple):
    action: ActionNode
    df: ppl.PathsDataFrame
    efficiency_divisor: float  # FIXME AEP depreciated
    unit_adjustment_multiplier: float


@dataclass
class ActionEfficiencyPair:
    graph_type: str
    cost_node: Node
    impact_node: Node
    efficiency_unit: Unit  # FIXME AEP depreciated
    cost_unit: Unit
    impact_unit: Unit
    indicator_unit: Unit
    plot_limit_efficiency: float | None  # FIXME depreciated, replace by plot_limit_for_indicator
    plot_limit_for_indicator: float | None
    invert_cost: bool
    invert_impact: bool
    indicator_cutpoint: float | None
    cost_cutpoint: float | None
    stakeholder_dimension: str | None
    outcome_dimension: str | None
    label: TranslatedString | str | None

    @classmethod
    def from_config(  # noqa: PLR0913
        cls,
        context: Context,
        graph_type: str,
        cost_node_id: str,
        impact_node_id: str,
        cost_unit: str,
        impact_unit: str,
        invert_cost: bool,
        invert_impact: bool,
        indicator_unit: str | None = None,
        plot_limit_for_indicator: float | None = None,
        indicator_cutpoint: float | None = None,
        cost_cutpoint: float | None = None,
        stakeholder_dimension: str | None = None,
        outcome_dimension: str | None = None,
        label: TranslatedString | str | None = None,
    ) -> ActionEfficiencyPair:
        cost_node = context.get_node(cost_node_id)
        impact_node = context.get_node(impact_node_id)
        indicator_unit_obj = context.unit_registry.parse_units(indicator_unit or '')
        cost_unit_obj = context.unit_registry.parse_units(cost_unit)
        impact_unit_obj = context.unit_registry.parse_units(impact_unit)
        aep = ActionEfficiencyPair(
            graph_type=graph_type,
            cost_node=cost_node,
            impact_node=impact_node,
            efficiency_unit=indicator_unit_obj,  # FIXME depreciated
            cost_unit=cost_unit_obj,
            impact_unit=impact_unit_obj,
            indicator_unit=indicator_unit_obj,
            plot_limit_efficiency=plot_limit_for_indicator,  # FIXME depreciated
            plot_limit_for_indicator=plot_limit_for_indicator,
            invert_cost=invert_cost,
            invert_impact=invert_impact,
            indicator_cutpoint=indicator_cutpoint,
            cost_cutpoint=cost_cutpoint,
            stakeholder_dimension=stakeholder_dimension,
            outcome_dimension=outcome_dimension,
            label=label,
        )
        aep.validate()
        return aep

    def validate(self):  # noqa: C901
        # Ensure that quantities, units and dimensions are compatible
        # TODO Currently, the function assumes that there are no extra dimensions. This should be updated
        # so that the dimensions that are not stakeholder or outcome dimensions are summed up.
        # If the quantity is not stackable, the need to sum up gives an error.
        impact_overview_settings = {  # noqa: F841
            # TODO Validate based on this dict, not based on a stack of if clauses.
            # Applies to calculate_iter() as well.
            'impact': {
                'allow_non_stackable': False,
                'is_cost_node_required': False,
                'is_impact_node_required': True,
                'indicator_unit_type': 'None',
                'allow_stakeholders': False,
                'allow_outcomes': True,
            },
            'cost_efficiency': {
                'allow_non_stackable': False,
                'is_cost_node_required': True,  # If node is required, then its unit is as well.
                'is_impact_node_required': True,
                'indicator_unit_type': 'cost/impact',  # Possible values: None, cost/impact, dimensionless
                'allow_stakeholders': False,
                'allow_outcomes': True,
            },
            'cost_benefit': {
                'allow_non_stackable': False,
                'is_cost_node_required': True,
                'is_impact_node_required': False,
                'indicator_unit_type': 'None',
                'allow_stakeholders': True,
                'allow_outcomes': True,
            },
            'return_on_investment': {
                'allow_non_stackable': False,
                'is_cost_node_required': True,
                'is_impact_node_required': True,
                'indicator_unit_type': 'dimensionless',
                'allow_stakeholders': False,
                'allow_outcomes': False,
            },
            'value_of_information': {
                'allow_non_stackable': True,  # TODO Think whether we actually need this attribute
                'is_cost_node_required': True,
                'is_impact_node_required': False,
                'indicator_unit_type': 'None',
                'allow_stakeholders': False,
                'allow_outcomes': False,
            },
        }

        if not (self.cost_node.quantity in STACKABLE_QUANTITIES and self.impact_node.quantity in STACKABLE_QUANTITIES):
            raise Exception('Cost and impact nodes must have stackable quantities')
        if self.cost_node.unit is None or self.impact_node.unit is None:
            raise Exception('Cost or impact node does not have a unit')
        if self.graph_type == 'cost_effectiveness':
            div_unit = self.cost_node.unit / self.impact_node.unit
            if not self.indicator_unit.is_compatible_with(div_unit):
                raise Exception('Indicator unit %s is not compatible with %s' % (self.indicator_unit, div_unit))
            if self.stakeholder_dimension is not None:
                raise Exception('Stakeholder dimension is not allowed for a cost-effectiveness graph')
        if self.graph_type == 'cost_benefit' and self.cost_unit != self.impact_unit:
                raise Exception('Units must be the same for cost %s and impact %s' % (self.cost_unit, self.impact_unit))
        if self.graph_type == 'return_of_investment':
            if not self.indicator_unit.dimensionless:
                raise Exception('The indicator unit %s must be dimensionless' % (self.indicator_unit))
            if not self.cost_unit == self.impact_unit:
                raise Exception('Units must be the same for cost %s and impact %s' % (self.cost_unit, self.impact_unit))
            if self.stakeholder_dimension is not None:
                raise Exception('Stakeholder dimension is not allowed for a return-of-investment graph')
            if self.outcome_dimension is not None:
                raise Exception('Outcome indicator is not allowed in a return-of-investment graph')

    def calculate_iter(self, context: Context, actions: Iterable[ActionNode] | None = None) -> Iterator[ActionEfficiency]:
        if actions is None:
            actions = list(context.get_actions())

        pc = PerfCounter('Action efficiency %s / %s' % (self.cost_node.id, self.impact_node.id), level=PerfCounter.Level.DEBUG)
        pc.display('starting')
        for action in actions:
            if not action.is_connected_to(self.cost_node) or not action.is_connected_to(self.impact_node):
                # Action is not connected to either cost or impact nodes, skip it
                continue

            # Dimensions that cost and impact nodes can have, and should if they are given:
            # cba i_out i_sta c_out c_sta
            # cea             c_out
            # roi
            # voi i_out                   c_iter
            match_dims_c = match_dims_i = []  # For cost and impact nodes, respectively
            if self.graph_type == 'cost_effectiveness':
                match_dims_c += [self.outcome_dimension]
            if self.graph_type == 'cost_benefit':
                match_dims_c.extend([self.outcome_dimension, self.stakeholder_dimension])
                match_dims_i.extend([self.outcome_dimension, self.stakeholder_dimension])
            if self.graph_type == 'value_of_information':
                match_dims_i.extend([self.outcome_dimension])
                match_dims_c.extend([UNCERTAINTY_COLUMN])
            match_dims_i = list(set(match_dims_i) - {None})
            match_dims_c = list(set(match_dims_c) - {None})

            with context.perf_context.exec_node(action):
                df = action.compute_indicator(
                    self.cost_node, self.impact_node, match_dims_i, match_dims_c, self.graph_type
                )
            if not len(df):
                # No impact for this action, skip it
                continue

            df = df.set_unit('Cost', df.get_unit('Cost') * Quantity('1 a'), force=True)
            df = df.set_unit('Impact', df.get_unit('Impact') * Quantity('1 a'), force=True)
            df = df.ensure_unit('Cost', self.cost_unit)
            df = df.ensure_unit('Impact', self.impact_unit)

            if self.graph_type == 'cost_effectiveness':
                unit_adjustment_multiplier = 1 * self.cost_unit / self.impact_unit / self.indicator_unit
            elif self.graph_type == 'return_of_investment':
                unit_adjustment_multiplier = 1 * self.impact_unit / self.cost_unit / self.indicator_unit
            else:
                assert self.cost_unit == self.impact_unit
                unit_adjustment_multiplier = 1 * self.cost_unit / self.indicator_unit

            unit_adjustment_multiplier = unit_adjustment_multiplier.to('dimensionless') # type: ignore

            ae = ActionEfficiency(
                action=action,
                df=df,
                efficiency_divisor=1 / unit_adjustment_multiplier,  # FIXME depreciated
                unit_adjustment_multiplier=unit_adjustment_multiplier,
            )
            yield ae

        pc.display('done')

    def calculate(self, context: Context, actions: Iterable[ActionNode] | None = None) -> list[ActionEfficiency]:
        out = list(self.calculate_iter(context, actions))
        return out
