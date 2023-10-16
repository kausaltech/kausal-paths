from nodes.calc import extend_last_historical_value_pl, nafill_all_forecast_years
from params.param import Parameter, BoolParameter, NumberParameter, ParameterWithUnit, StringParameter
from typing import List, ClassVar, Tuple
import polars as pl
import pandas as pd
import pint

from common.i18n import TranslatedString
from common import polars as ppl
from .constants import FORECAST_COLUMN, NODE_COLUMN, VALUE_COLUMN, YEAR_COLUMN
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
            local_id='threshold',
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

        def add_pdf(df1: ppl.PathsDataFrame, df2: ppl.PathsDataFrame):
            df = df1.paths.join_over_index(df2, how='outer', index_from='union')
            df = df.sum_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN)
            df = df.drop(VALUE_COLUMN + '_right')
            return df
        
        def weighted_sum(action: ActionNode, weight, df=None, quantity=None, tag=None):
            w = self.get_parameter_value(weight, required=False, units=True)
            if w is None:
                return df
            if quantity is not None:
                n = self.get_input_node(quantity=quantity)
            else:
                n = self.get_input_node(tag=tag)
            dft = action.compute_impact(n)
            dft = dft.filter(pl.col('Impact').eq(pl.lit('Impact'))).drop('Impact')
            dft = dft.multiply_quantity(VALUE_COLUMN, w)
            
            if df is None:
                df = dft
            else:
                df = add_pdf(df, dft)
            return df

        df_out = None
        round = 0
        for action in actions:
            df = None
            round += 1
            df = weighted_sum(action, 'emissions_weight', df, quantity='emissions')
            df = weighted_sum(action, 'cost_weight', df, quantity='currency')
            df = weighted_sum(action, 'health_weight', df, quantity='disease_burden')
            df = weighted_sum(action, 'equity_weight', df, tag='equity')
            df = df.with_columns(pl.lit('hypothesis_' + str(round)).alias('hypothesis')).add_to_index('hypothesis')

            if df_out is None:
                df_out = df
            else:
                meta = df.get_meta()
                df_out = pl.concat([df_out, df])
                df_out = ppl.to_ppdf(df_out, meta=meta)

        th = self.get_parameter_value('threshold', required=True, units=True)
        df_out = df_out.cumulate(VALUE_COLUMN)
        df_out = df_out.multiply_quantity(VALUE_COLUMN, Quantity('1 a'))
        df_out = df_out.with_columns([
            pl.when(pl.col(VALUE_COLUMN).gt(pl.lit(th.m))).then(pl.lit(1))
            .otherwise(pl.lit(0)).alias(VALUE_COLUMN)
        ])

        return df_out
