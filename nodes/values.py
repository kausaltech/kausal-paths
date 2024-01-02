from nodes.calc import extend_last_historical_value_pl, nafill_all_forecast_years
from params.param import Parameter, BoolParameter, NumberParameter, ParameterWithUnit, StringParameter
from typing import List, ClassVar, Tuple
import polars as pl
import pandas as pd
import pint
import functools

from common.i18n import TranslatedString
from common import polars as ppl
from .constants import VALUE_COLUMN, IMPACT_COLUMN, IMPACT_GROUP
from .node import Node
from .simple import SimpleNode, AdditiveNode, MultiplicativeNode
from .actions.simple import AdditiveAction
from .actions.action import ActionNode
from .units import Quantity
from .exceptions import NodeError


class UtilityNode(AdditiveNode):
    '''
    Utility nodes take in outcome nodes and value weight parameters.
    They produce a dataframe showing the value-weighted sums of outcomes.
    Cost-efficiency can be implemented if a decision criterion is known, e.g. 50 €/t.
    Then, emissions and cost are put to an equal scale by multiplying
    cost by 0.02 1/EUR and emissions by 1 1/t, resulting in a scale
    where cost-effective scenarios show value < 0 when cumulated over the time span.
    Therefore, in this case, the threshold parameter should get value 0.
    '''
    allowed_parameters = SimpleNode.allowed_parameters + [
        NumberParameter(
            local_id='impact_threshold',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='emissions_weight',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='cost_weight',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='health_weight',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='equity_weight',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='biodiversity_weight',
            is_customizable=True,
        )
    ]

    def weighted_impact(self, weight, quantity=None, tag=None):
        w = self.get_parameter_value(weight, required=False, units=True)
        if w is None:
            return None
        if quantity is not None:
            n = self.get_input_node(quantity=quantity)
        else:
            n = self.get_input_node(tag=tag)

        df = n.get_output_pl(target_node=self)  # Calculate output rather than impact.
        df = df.multiply_quantity(VALUE_COLUMN, w)
        
        return df

    def sum_dfs(self, dfs: list[ppl.PathsDataFrame | None]) -> ppl.PathsDataFrame:
        def join_and_sum(df1: ppl.PathsDataFrame, df2: ppl.PathsDataFrame | None):
            if df2 is None:
                return df1
            df = df1.paths.join_over_index(df2, how='outer', index_from='union')
            df = df.sum_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN)
            df = df.drop(VALUE_COLUMN + '_right')
            return df

        dfs = [x for x in dfs if x is not None]
        df = functools.reduce(lambda df1, df2: join_and_sum(df1, df2), [df for df in dfs])
        return df
    
    def compute(self) -> ppl.PathsDataFrame:
        nodes: list[Node] = []

        for node in self.input_nodes:
            if not isinstance(node, ActionNode):
                nodes += [node]
        assert len(nodes) > 0

        dfs = [
            self.weighted_impact('emissions_weight', tag='emissions'),
            self.weighted_impact('cost_weight', tag='currency'),
            self.weighted_impact('health_weight', tag='disease_burden'),
            self.weighted_impact('equity_weight', tag='equity'),
            self.weighted_impact('biodiversity_weight', tag='biodiversity')
        ]
        df = self.sum_dfs(dfs)
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        th = self.get_parameter_value('impact_threshold', required=True, units=True)
        th = th.to(df.get_unit(VALUE_COLUMN))
        df = df.with_columns([(
            pl.col(VALUE_COLUMN) - pl.lit(th.m)).alias(VALUE_COLUMN)
        ])

        return df


class AssociationNode(SimpleNode):  # FIXME Use AdditiveNode for compatible units
    '''
    Association nodes connect to their upstream nodes in a loose way:
    Their values follow the relative changes of the input nodes but
    their quantities and units are not dependent on those of the input nodes.
    The node MUST have exactly one dataset, which is the prior estimate.
    Fractions 1..3 can be used to tell how much the input node should adjust
    the output node. The default relation is "increase", if "decrease" is used,
    that must be explicitly said in the tags.
    '''
    allowed_parameters = MultiplicativeNode.allowed_parameters + [
        NumberParameter(local_id='fraction1', label='Fraction for node 1', is_customizable=False),
        NumberParameter(local_id='fraction2', label='Fraction for node 2', is_customizable=False),
        NumberParameter(local_id='fraction3', label='Fraction for node 3', is_customizable=False)
    ]

    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        df = extend_last_historical_value_pl(df, end_year=self.get_end_year())
        for edge in self.edges:
            if edge.output_node is self:
                node = edge.input_node
                m = node.get_default_output_metric().column_id

                fraction = 0.1  # Default fraction of the output node's output
                for fr in ['fraction1', 'fraction2', 'fraction3']:
                    if fr in edge.tags:
                        fraction = self.get_parameter_value(fr, units=False, required=True)

                default = self.context.get_scenario('default')
                with default.override():
                    mean_in = node.get_output_pl(target_node=self)[m].mean()
                mean_out = df[VALUE_COLUMN].mean()

                if abs(mean_in) > 0.01:  # Relative adjustment makes no sense when too close to zero.
                    multiplier = fraction * mean_out / mean_in
                else:
                    multiplier = 1
                if 'decrease' in edge.tags:
                    multiplier *= -1

                dfn = node.get_output_pl(target_node=self)
                df = df.paths.join_over_index(dfn, how='outer', index_from='union')
                df = df.with_columns(pl.col(m + '_right').fill_null(0))
                df = df.with_columns((
                    pl.col(m) + pl.lit(multiplier) * pl.col(m + '_right').alias(m)
                    )).drop(m + '_right')

        return df


class LogicalNode(MultiplicativeNode):
    '''
    LogicalNode takes in logical values (either 0 or 1) and gives an output 0 or 1
    based on the Boolean operators (and, or) given as tags. The operator of an edge is used
    between this and the previous input node; therefore, the boolean of the first node is ignored.
    Note! The ordering of input nodes is the ordering of the logical operators.
    If the input is not logical, a threshold can be used to check whether the value is equal or above
    the threshold; this becomes the logical value. There are three threshold parameters available:
    threshold1, threshold2, and threshold3. If there is a threshold tag, the respective parameter
    must be found.
    Tag "not" can be used to invert the logical values.
    '''
    allowed_parameters = MultiplicativeNode.allowed_parameters + [
        NumberParameter(local_id='threshold1', label='Threshold for trigger 1', is_customizable=False),
        NumberParameter(local_id='threshold2', label='Threshold for trigger 2', is_customizable=False),
        NumberParameter(local_id='threshold3', label='Threshold for trigger 3', is_customizable=False)
    ]

    def compute(self) -> ppl.PathsDataFrame:
        df = None
        for edge in self.edges:
            if edge.output_node is self:
                node = edge.input_node
                dfn = node.get_output_pl(target_node=self)
                m = VALUE_COLUMN  # FIXME use default metric instead

                for thr in ['threshold1', 'threshold2', 'threshold3']:
                    if thr in edge.tags:
                        threshold = self.get_parameter_value(thr, units=True, required=True)
                        dfn = dfn.ensure_unit(m, threshold.units)
                        dfn = dfn.with_columns((pl.col(m) >= pl.lit(threshold.m)).alias(m))
                dfn = dfn.clear_unit(m).set_unit(m, 'dimensionless')
                if 'not' in edge.tags:
                    dfn = dfn.with_columns(pl.col(m).not_().alias(m))
                if df is None:
                    df = dfn
                else:
                    df = df.paths.join_over_index(dfn, how='outer', index_from='union')
                    if 'and' in edge.tags:
                        df = df.with_columns(pl.all_horizontal(m, m + '_right').alias(m))
                    elif 'or' in edge.tags:
                        df = df.with_columns(pl.any_horizontal(m, m + '_right').alias(m))
                    else:
                        raise NodeError(self, 'You must give a boolean tag (and, or) for all input nodes except the first')
                    df = df.drop(m + '_right')
        df = df.with_columns((pl.col(m) * pl.lit(1)).alias(m))
        return df
