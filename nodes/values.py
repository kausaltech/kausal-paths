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
from .exceptions import NodeError


class HypothesisNode(SimpleNode):
    allowed_parameters: ClassVar[List[Parameter]] = [
        NumberParameter(
            local_id='min_value',
            label=TranslatedString(en="Smallest estimate this hypothesis can get"),
            is_customizable=True
        ),
        NumberParameter(
            local_id='max_value',
            label=TranslatedString(en="Largest estimate this hypothesis can get"),
            is_customizable=True
        ),
    ] + SimpleNode.allowed_parameters

    def compute(self):
#        df = self.get_input_dataset_pl()
        df = self.add_nodes_pl(df=None, nodes=self.input_nodes)
        min_value = self.get_parameter_value(id='min_value', units=False)  # Must be in the node units
        max_value = self.get_parameter_value(id='max_value', units=False)

        print(min_value)

        cond = pl.col(VALUE_COLUMN)<min_value
        df = df.with_columns([pl.when(cond).then(min_value).otherwise(pl.col(VALUE_COLUMN)).alias('hypo')])
        cond = pl.col('hypo')>max_value
        df = df.with_columns([pl.when(cond).then(max_value).otherwise(pl.col('hypo')).alias(VALUE_COLUMN)])
        df = df.drop('hypo')

        return df


class MultiAttributeUtilityNode(SimpleNode):
    allowed_parameters: ClassVar[List[Parameter]] = [
        NumberParameter(
            local_id='biodiversity_weight',
            label=TranslatedString(en='Utility weight for biodiersity'),
            is_customizable=True
        ),
        NumberParameter(
            local_id='climate_emissions_weight',
            label=TranslatedString(en='Utility weight for climate emissions'),
            is_customizable=True
        ),
        NumberParameter(
            local_id='state_budget_weight',
            label=TranslatedString(en='Utility weight for state budget'),
            is_customizable=True
        )
    ]

    def compute(self):
        bio_w = self.get_parameter_value(id='biodiversity_weight', units=False)  # FIXME Add units
        cli_w = self.get_parameter_value(id='climate_emissions_weight', units=False)
        bud_w = self.get_parameter_value(id='state_budget_weight', units=False)

        bio_node = self.get_input_node(tag='biodiversity')
        cli_node = self.get_input_node(tag='climate_emissions')
        bud_node = self.get_input_node(tag='state_budget')

        df = bio_node.get_output_pl()
        meta = df.get_meta()
        df = df.with_columns([pl.col(VALUE_COLUMN) * bio_w])  # FIXME make units match

        cli_df = cli_node.get_output_pl()
        df = df.join(cli_df, on=YEAR_COLUMN, how='outer')
        df = df.with_columns([(pl.col(VALUE_COLUMN) + pl.col(VALUE_COLUMN + '_right') * cli_w).alias(VALUE_COLUMN)])
        df = df.with_columns([(pl.col(FORECAST_COLUMN) | pl.col(FORECAST_COLUMN + '_right')).alias(FORECAST_COLUMN)])
        df = df.drop([FORECAST_COLUMN + '_right', VALUE_COLUMN + '_right'])

        bud_df = bud_node.get_output_pl()
        df = df.join(bud_df, on=YEAR_COLUMN, how='outer')
        df = df.with_columns([(pl.col(VALUE_COLUMN) + pl.col(VALUE_COLUMN + '_right') * bud_w).alias(VALUE_COLUMN)])
        df = df.with_columns([(pl.col(FORECAST_COLUMN) | pl.col(FORECAST_COLUMN + '_right')).alias(FORECAST_COLUMN)])
        df = df.drop([FORECAST_COLUMN + '_right', VALUE_COLUMN + '_right'])

        return ppl.to_ppdf(df, meta=meta)
