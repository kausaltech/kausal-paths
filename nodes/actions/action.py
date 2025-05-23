from __future__ import annotations

import typing

# from collections.abc import Callable
from dataclasses import dataclass
from typing import Callable, ClassVar, cast

import pandas as pd
import polars as pl

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
        BoolParameter(local_id='allow_null_categories', description='Allow null dimension categories', is_customizable=False),
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
            'bool', self.get_global_parameter_value('action_impact_from_baseline', required=False) or False,
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

    def get_value_of_information(self, cost_df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        if UNCERTAINTY_COLUMN not in cost_df.columns:
            return ppl.PathsDataFrame() # FIXME Return zero dataframe
        meta = cost_df.get_meta() # TODO Function not tested yet
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
            pl.lit(0.0).alias('Effect'),  # Impact is not used with value of information
            pl.lit('expectation').alias(UNCERTAINTY_COLUMN)
        ])
        dfp = dfp.select([YEAR_COLUMN, FORECAST_COLUMN, UNCERTAINTY_COLUMN, 'Cost', 'Effect']) # TODO Do we need this?
        meta.units['Effect'] = meta.units['Cost']
        df = ppl.to_ppdf(df=dfp, meta=meta)

        return df

    def compute_node_impact(self, node: Node, colname: str | None,
                            keepcols: list[str | None]) -> ppl.PathsDataFrame:
        df = self.compute_impact(node)
        m = node.get_default_output_metric()
        df = (
            df.filter(pl.col(IMPACT_COLUMN).eq(IMPACT_GROUP)).drop(IMPACT_COLUMN)
        )
        df = df.select([*df.primary_keys, FORECAST_COLUMN, pl.col(m.column_id)])
        if colname is not None:
            df = df.rename({m.column_id: colname})
        if keepcols is not None:
            dropcols = [col for col in df.dim_ids if col not in keepcols]
            df = df.paths.sum_over_dims(dropcols)

        return df

    def compute_indicator(self, io: ImpactOverview) -> ppl.PathsDataFrame:
        # TODO Use indicator_unit in cols

        dims = [io.outcome_dimension, io.stakeholder_dimension]
        if io.cost_node is not None:
            cost_df = self.compute_node_impact(io.cost_node, 'Cost', dims)

        effect_df = self.compute_node_impact(io.effect_node, 'Effect', dims)
        if io.graph_type == 'value_of_information':
            effect_df = self.get_value_of_information(effect_df)

        od = io.outcome_dimension
        sd = io.stakeholder_dimension
        if io.graph_type != 'cost_benefit':
            assert od is None
            assert sd is None

        if io.graph_type == 'cost_benefit':
            assert io.cost_node is not None
            if od is not None:
                assert od in cost_df.dim_ids
                assert od in effect_df.dim_ids
            if sd is not None:
                assert sd in cost_df.dim_ids
                assert sd in effect_df.dim_ids

        if io.cost_node is None:
            df = effect_df.with_columns(pl.lit(0.0).alias('Cost'))
            df = df.set_unit('Cost', df.get_unit('Effect'))
        else:
            df = cost_df.paths.join_over_index(effect_df, how='outer', index_from='union')
        df = df.with_columns([
            pl.col('Cost').fill_null(0.0),
            pl.col('Effect').fill_null(0.0),
            pl.when(pl.col('Effect').abs() < pl.lit(1e-9)) # TODO Do we still need this?
            .then(pl.lit(0.0))
            .otherwise(pl.col('Effect'))
        ])
        if io.invert_cost:
            df = df.with_columns((pl.col('Cost') * pl.lit(-1.0)).alias('Cost'))
        if io.invert_effect:
            df = df.with_columns((pl.col('Effect') * pl.lit(-1.0)).alias('Effect'))

        return df


class ActionImpact(typing.NamedTuple):
    action: ActionNode
    df: ppl.PathsDataFrame
    unit_adjustment_multiplier: float


@dataclass
class ImpactOverview:
    graph_type: str
    effect_node: Node
    indicator_unit: Unit
    cost_node: Node | None = None
    cost_unit: Unit | None = None
    effect_unit: Unit | None = None
    plot_limit_for_indicator: float | None = None
    invert_cost: bool = False
    invert_effect: bool = False
    indicator_cutpoint: float | None = None
    cost_cutpoint: float | None = None
    stakeholder_dimension: str | None = None
    outcome_dimension: str | None = None
    label: TranslatedString | str | None = None

    @classmethod
    def from_config(  # noqa: PLR0913
        cls,
        context: Context,
        graph_type: str,
        effect_node_id: str,
        invert_cost: bool,
        invert_effect: bool,
        cost_node_id: str | None,
        effect_unit: str | None,
        cost_unit: str | None,
        indicator_unit: str,
        plot_limit_for_indicator: float | None,
        indicator_cutpoint: float | None,
        cost_cutpoint: float | None,
        stakeholder_dimension: str | None,
        outcome_dimension: str | None,
        label: TranslatedString | str | None,
    ) -> ImpactOverview:
        if cost_node_id is not None:
            cost_node = context.get_node(cost_node_id)
        else:
            cost_node = None
        effect_node = context.get_node(effect_node_id)
        indicator_unit_obj = context.unit_registry.parse_units(indicator_unit)
        if cost_unit is not None:
            cost_unit_obj = context.unit_registry.parse_units(cost_unit)
        else:
            cost_unit_obj = None
        if effect_unit is not None:
            effect_unit_obj = context.unit_registry.parse_units(effect_unit)
        else:
            effect_unit_obj = None
        aep = ImpactOverview(
            graph_type=graph_type,
            cost_node=cost_node,
            effect_node=effect_node,
            cost_unit=cost_unit_obj,
            effect_unit=effect_unit_obj,
            indicator_unit=indicator_unit_obj,
            plot_limit_for_indicator=plot_limit_for_indicator,
            invert_cost=invert_cost,
            invert_effect=invert_effect,
            indicator_cutpoint=indicator_cutpoint,
            cost_cutpoint=cost_cutpoint,
            stakeholder_dimension=stakeholder_dimension,
            outcome_dimension=outcome_dimension,
            label=label,
        )
        aep.validate()
        return aep

    def validate(self):

        if self.effect_node.quantity not in STACKABLE_QUANTITIES:
            raise Exception('Impact node must have stackable quantities')
        if self.cost_node is not None and self.cost_node.quantity not in STACKABLE_QUANTITIES:
                raise Exception('Cost node must have stackable quantities')
        if self.graph_type in ['cost_efficiency']:
            assert self.cost_node is not None
            assert self.cost_unit is not None
            assert self.effect_unit is not None
            div_unit = self.cost_unit / self.effect_unit
            if not self.indicator_unit.is_compatible_with(div_unit):
                raise Exception('Indicator unit %s is not compatible with %s' % (self.indicator_unit, div_unit))

    def _adjust_graph_units(self, df: ppl.PathsDataFrame, has_cost: bool,
                            is_same_unit: bool) -> ppl.PathsDataFrame:
        if has_cost:
            assert self.cost_node is not None
            if is_same_unit:
                if self.cost_unit is None:
                    cost_unit = self.effect_unit
                else:
                    print(f"For {self.graph_type} graph, give only effect_unit, not cost_unit.") # FIXME Make error
                    cost_unit = self.effect_unit
            else:
                assert self.cost_unit is not None
                cost_unit = self.cost_unit
            df = df.set_unit('Cost', df.get_unit('Cost') * Quantity('1 a'), force=True)
            df = df.ensure_unit('Cost', cost_unit)
        elif self.cost_unit or self.cost_node:
                raise ValueError(f"{self.graph_type} graphs should not have cost information.")

        df = df.set_unit('Effect', df.get_unit('Effect') * Quantity('1 a'), force=True)
        df = df.ensure_unit('Effect', self.effect_unit)

        return df

    def calculate_iter(self, context: Context, actions: Iterable[ActionNode] | None = None) -> Iterator[ActionImpact]:

        def _cea(df: ppl.PathsDataFrame) -> float:
            uam = 1 * df.get_unit('Cost') / df.get_unit('Effect') / self.indicator_unit
            assert isinstance(uam, Quantity)
            return uam.to('dimensionless').m

        def _roi(df: ppl.PathsDataFrame) -> float:
            uam = 1 * df.get_unit('Effect') / df.get_unit('Cost') / self.indicator_unit
            assert isinstance(uam, Quantity)
            return uam.to('dimensionless').m

        def _unity(df: ppl.PathsDataFrame) -> float:
            return 1.0

        unit_adjustments = {
            'cost_efficiency': {'has_cost':True, 'is_same_unit': False, 'fn': _cea},
            'cost_benefit': {'has_cost':True, 'is_same_unit': True, 'fn': _unity},
            'cost_benefit1': {'has_cost': False, 'is_same_unit': False, 'fn': _unity},
            'return_on_investment': {'has_cost':True, 'is_same_unit': True, 'fn': _roi},
            'value_of_information': {'has_cost':False, 'is_same_unit': False, 'fn': _unity},
        }

        if actions is None:
            actions = list(context.get_actions())
        has_cost = unit_adjustments[self.graph_type]['has_cost']
        assert isinstance(has_cost, bool)
        is_same_unit = unit_adjustments[self.graph_type]['is_same_unit']
        assert isinstance(is_same_unit, bool)
        unit_adjustment_function = cast('Callable[[ppl.PathsDataFrame], float]',
                              unit_adjustments[self.graph_type]['fn'])

        for action in actions:
            if not action.is_connected_to(self.effect_node):
                continue
            if not action.is_enabled(): # Inactive actions would give false zeros
                continue

            df = action.compute_indicator(self)
            if not len(df):
                # No impact for this action, skip it
                continue

            df = self._adjust_graph_units(df, has_cost, is_same_unit)
            uam = unit_adjustment_function(df)

            ae = ActionImpact(
                action=action,
                df=df,
                unit_adjustment_multiplier=uam,
            )
            yield ae

    def calculate(self, context: Context, actions: Iterable[ActionNode] | None = None) -> list[ActionImpact]:
        out = list(self.calculate_iter(context, actions))
        return out
