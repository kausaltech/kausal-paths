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
from .simple import SimpleNode, AdditiveNode
from .actions.simple import AdditiveAction
from .actions.action import ActionNode
from .units import Quantity
from .exceptions import NodeError


class ValueProfile(SimpleNode):
    '''
    Value profiles are nodes that take in actions, outcome nodes and parameters.
    They produce a dataframe showing which of the actions should be implemented
    and when based on the impacts of outcome nodes and parameters.
    Cost-efficiency can be implemented if a decision criterion is known, e.g. 50 €/t.
    Then, emissions and cost are put to an equal scale by multiplying
    cost by -0.02 1/EUR and emissions by -1 1/t, resulting in a scale
    where cost-effective scenarios show value > 0 when cumulated over the time span.
    Therefore, in this case, the threshold parameter should get value 0.
    The value profile results in 0 if the criterion is not fulfilled and
    1 for years when the cumulative criterion is fulfilled.
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
        )
    ]

    def weighted_impact(self, action: ActionNode, weight, quantity=None, tag=None):
        w = self.get_parameter_value(weight, required=False, units=True)
        if w is None:
            return None
        if quantity is not None:
            n = self.get_input_node(quantity=quantity)
        else:
            n = self.get_input_node(tag=tag)
        df = action.compute_impact(n)
        df = df.filter(pl.col(IMPACT_COLUMN).eq(pl.lit(IMPACT_GROUP))).drop(IMPACT_COLUMN)
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
    
    def compute(self):
        actions: list[ActionNode] = []
        nodes: list[Node] = []

        for node in self.input_nodes:
            if isinstance(node, ActionNode):
                actions += [node]
            else:
                nodes += [node]
        assert len(actions) > 0
        assert len(nodes) > 0

        df = None
        round = 0
        for action in actions:
            round += 1
            dfs = [
                self.weighted_impact(action, 'emissions_weight', quantity='emissions'),
                self.weighted_impact(action, 'cost_weight', quantity='currency'),
                self.weighted_impact(action, 'health_weight', quantity='disease_burden'),
                self.weighted_impact(action, 'equity_weight', tag='equity')
            ]
            df_action = self.sum_dfs(dfs)
            df_action = df_action.with_columns(
                pl.lit('hypothesis_' + str(round)).alias('hypothesis')).add_to_index('hypothesis')

            if df is None:
                df = df_action
            else:
                meta = df_action.get_meta()
                df = pl.concat([df, df_action])
                df = ppl.to_ppdf(df, meta=meta)

        df = df.cumulate(VALUE_COLUMN)
        df = df.multiply_quantity(VALUE_COLUMN, Quantity('1 a'))
        th = self.get_parameter_value('impact_threshold', required=True, units=True)
        th = th.to(df.get_unit(VALUE_COLUMN))
        df = df.with_columns([
            pl.when(pl.col(VALUE_COLUMN).gt(pl.lit(th.m))).then(pl.lit(1))
            .otherwise(pl.lit(0)).alias(VALUE_COLUMN)
        ])
        df = df.ensure_unit(VALUE_COLUMN, self.unit)

        return df
