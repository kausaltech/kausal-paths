from __future__ import annotations

import typing

# from collections.abc import Callable
from dataclasses import dataclass
from typing import Callable, ClassVar, cast

import pandas as pd
import pint
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
from nodes.units import Quantity, Unit, unit_registry
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

    def _get_value_of_information(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        if UNCERTAINTY_COLUMN not in df.columns:
            return ppl.PathsDataFrame() # FIXME Return zero dataframe
        meta = df.get_meta() # TODO Function not tested yet
        df = df.filter(pl.col(UNCERTAINTY_COLUMN).ne('median'))
        last_forecast_year = df.filter(pl.col(FORECAST_COLUMN)).select(YEAR_COLUMN).max()
        col = 'Effect'
        dfp = df.group_by(pl.col(UNCERTAINTY_COLUMN)).agg(pl.sum(col))

        dfp = dfp.with_columns([
            pl.when(pl.col(col) > 0.0).then(pl.col(col))
            .otherwise(pl.lit(0.0)).alias('under_knowledge')
        ])

        dfp = dfp.select(pl.all().mean())
        dfp = pl.concat([dfp, last_forecast_year], how='horizontal')
        dfp = dfp.with_columns([
            (pl.col('under_knowledge') - pl.col(col)).alias(col),
            pl.lit(value=True).alias(FORECAST_COLUMN),
            pl.lit('expectation').alias(UNCERTAINTY_COLUMN)
        ])
        dfp = dfp.select([YEAR_COLUMN, FORECAST_COLUMN, UNCERTAINTY_COLUMN, 'Cost', 'Effect']) # TODO Do we need this?
        df = ppl.to_ppdf(df=dfp, meta=meta)

        return df

    def _get_cost_benefit(self, df: ppl.PathsDataFrame, io: ImpactOverview) -> ppl.PathsDataFrame:
            od = io.outcome_dimension
            sd = io.stakeholder_dimension
            if od is not None:
                assert od in df.dim_ids
            if sd is not None:
                assert sd in df.dim_ids
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
            df = df.paths.sum_over_dims(dropcols) # FIXME Dims used later but they are summed over.

        return df

    def compute_indicator(self, io: ImpactOverview) -> ppl.PathsDataFrame:

        dims = [io.outcome_dimension, io.stakeholder_dimension]

        effect_df = self.compute_node_impact(io.effect_node, 'Effect', dims)

        if io.graph_type == 'value_of_information':
            effect_df = self._get_value_of_information(effect_df)

        no_cost_io = ['cost_benefit', 'simple_effect']
        if io.graph_type in no_cost_io:
            df = self._get_cost_benefit(effect_df, io)
        else:
            assert io.cost_node is not None
            cost_df = self.compute_node_impact(io.cost_node, 'Cost', dims)
            df = cost_df.paths.join_over_index(effect_df, how='outer', index_from='union')
            df = df.with_columns([
                pl.col('Cost').fill_null(0.0),
                pl.col('Effect').fill_null(0.0)
            ])
            df = df.with_columns([
                (pl.when(pl.col('Effect').abs() < pl.lit(1e-9)) # TODO Do we still need this?
                .then(pl.lit(0.0))
                .otherwise(pl.col('Effect'))).alias('Effect')
            ])

            if io.invert_cost:
                df = df.with_columns((pl.col('Cost') * pl.lit(-1.0)).alias('Cost'))

        if io.invert_effect:
            df = df.with_columns((pl.col('Effect') * pl.lit(-1.0)).alias('Effect'))

        if io.graph_type in [ # Flip sign for benefit
            'cost_efficiency',
            'return_on_investment',
            'return_on_investment_gross',
            'benefit_cost_ratio']:
            df = df.with_columns((pl.col('Effect') * pl.lit(-1.0)).alias('Effect'))

        if io.graph_type == 'return_on_investment_gross':
            df = df.with_columns((pl.col('Effect') - pl.col('Cost')).alias('Effect'))

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
    cost_category_label: TranslatedString | str | None = None
    effect_category_label: TranslatedString | str | None = None
    cost_label: TranslatedString | str | None = None
    effect_label: TranslatedString | str | None = None
    indicator_label: TranslatedString | str | None = None
    description: TranslatedString | str | None = None

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
        cost_category_label: TranslatedString | str | None,
        effect_category_label: TranslatedString | str | None,
        cost_label: TranslatedString | str | None,
        effect_label: TranslatedString | str | None,
        indicator_label: TranslatedString | str | None,
        description: TranslatedString | str | None,
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
            cost_category_label=cost_category_label,
            effect_category_label=effect_category_label,
            cost_label=cost_label,
            effect_label=effect_label,
            indicator_label=indicator_label,
            description=description,
        )
        aep.validate()
        return aep

    def validate(self):

        if self.effect_node.quantity not in STACKABLE_QUANTITIES:
            raise Exception(f"Effect node {self.effect_node.id} quantity {self.effect_node.quantity} is not stackable.")
        if self.cost_node is not None and self.cost_node.quantity not in STACKABLE_QUANTITIES:
                raise Exception(f"Cost node {self.cost_node.id} quantity {self.cost_node.quantity} is not stackable.")

        rf = ['effect_node', 'cost_node', 'indicator_unit']
        ff = ['outcome_dimension', 'stakeholder_dimension', 'cost_unit', 'effect_unit']
        field_lists = {
            'cost_benefit':
                {'required': ['effect_node', 'indicator_unit'],
                 'forbidden': ['cost_node', 'cost_unit', 'effect_unit']},
            'cost_efficiency':
                {'required': rf,
                 'forbidden': ['outcome_dimension', 'stakeholder_dimension']},
            'return_on_investment':
                {'required': rf,
                 'forbidden': ff},
            'return_on_investment_gross':
                {'required': rf,
                 'forbidden': ff},
            'benefit_cost_ratio':
                {'required': rf,
                 'forbidden': ff},
            'value_of_information':
                {'required': ['effect_node', 'indicator_unit'],
                 'forbidden': [*ff, 'cost_node']},
            'simple_effect':
                {'required': ['effect_node', 'indicator_unit'],
                 'forbidden': [*ff, 'cost_node']},
        }
        required_fields = field_lists[self.graph_type]['required']
        forbidden_fields = field_lists[self.graph_type]['forbidden']

        for field_name, field_value in self.__dict__.items():
            if field_value is not None and field_name in forbidden_fields:
                print(f"Field '{field_name}' must not be used for graph type '{self.graph_type}'") # TODO raise ValueError

            if field_value is None and field_name in required_fields:
                print(f"Field '{field_name}' must be given for graph type '{self.graph_type}'") # TODO raise ValueError

    def _adjust_graph_units(self, df: ppl.PathsDataFrame,
                            is_same_unit: bool) -> ppl.PathsDataFrame:

        has_cost = 'Cost' in df.columns
        if has_cost:
            df = df.set_unit('Cost', df.get_unit('Cost') * unit_registry.a, force=True)
        df = df.set_unit('Effect', df.get_unit('Effect') * unit_registry.a, force=True)
        if is_same_unit:
            cost_unit = effect_unit = self.indicator_unit
        else:
            if has_cost:
                cost_unit = self.cost_unit or df.get_unit('Cost')
            effect_unit = self.effect_unit or df.get_unit('Effect')

        if has_cost:
            df = df.ensure_unit('Cost', cost_unit)
        df = df.ensure_unit('Effect', effect_unit)

        return df

    def _get_unit_adjustment_function(self) -> tuple[Callable[[ppl.PathsDataFrame], float], bool]:

        def _cea(df: ppl.PathsDataFrame) -> float:
            uam = 1 * df.get_unit('Cost') / df.get_unit('Effect') / self.indicator_unit
            assert isinstance(uam, Quantity)
            try:
                return uam.to('dimensionless').m
            except pint.DimensionalityError as e:
                raise Exception(
                    f"Indicator unit {self.indicator_unit} is not compatible with Cost / Effect."
                ) from e

        def _roi(df: ppl.PathsDataFrame) -> float:
            uam = 1 * df.get_unit('Effect') / df.get_unit('Cost') / self.indicator_unit
            assert isinstance(uam, Quantity)
            try:
                return uam.to('dimensionless').m
            except pint.DimensionalityError as e:
                raise Exception(
                    f"Indicator unit {self.indicator_unit} is not compatible with Effect / Cost."
                ) from e

        def _unity(df: ppl.PathsDataFrame) -> float:
            return 1.0

        unit_adjustments = {
            'cost_efficiency': {'is_same_unit': False, 'fn': _cea},
            'cost_benefit': {'is_same_unit': True, 'fn': _unity},
            'return_on_investment': {'is_same_unit': False, 'fn': _roi},
            'return_on_investment_gross': {'is_same_unit': False, 'fn': _roi},
            'benefit_cost_ratio': {'is_same_unit': False, 'fn': _roi},
            'value_of_information': {'is_same_unit': True, 'fn': _unity},
            'simple_effect': {'is_same_unit': True, 'fn': _unity},
        }

        is_same_unit = unit_adjustments[self.graph_type]['is_same_unit']
        assert isinstance(is_same_unit, bool)
        unit_adjustment_function = cast('Callable[[ppl.PathsDataFrame], float]',
                              unit_adjustments[self.graph_type]['fn'])

        return unit_adjustment_function, is_same_unit

    def calculate_iter(self, context: Context, actions: Iterable[ActionNode] | None = None) -> Iterator[ActionImpact]:

        if actions is None:
            actions = list(context.get_actions())

        unit_adjustment_function, is_same_unit = self._get_unit_adjustment_function()

        for action in actions:
            if not action.is_connected_to(self.effect_node):
                continue
            if not action.is_enabled(): # Inactive actions would give false zeros
                continue

            df = action.compute_indicator(self)
            if not len(df):
                # No impact for this action, skip it
                continue

            df = self._adjust_graph_units(df, is_same_unit)
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

"""
A discussion started with Claude.

I have class ImpactOverView, which produces graphs from data. Depending on the graph_type,
different attributes are needed. For example, benefit_cost type needs effect_node but not
effect_unit, while cost_efficiency needs both. So far,
I have defined the attributes in a static yaml file and tested for compliance in code.
However, I want to develop a user interface for administrators based on Wagtail. There
the admin user could create a new ImpactOverview, and, after selecting the graph_type,
would only be asked about the attributes that are relevant. What is a good way to
implement this?

Claude recommended Wagtail StreamFields with different block types for different graph_types.
"""
