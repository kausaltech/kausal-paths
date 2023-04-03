from nodes.calc import extend_last_historical_value_pl, nafill_all_forecast_years
from params.param import Parameter, BoolParameter, NumberParameter, ParameterWithUnit, StringParameter
from typing import List, ClassVar, Tuple
from nodes.constants import VALUE_COLUMN, FORECAST_COLUMN, YEAR_COLUMN
import polars as pl
import pandas as pd
import pint

from common.i18n import TranslatedString
from common import polars as ppl
from .simple import AdditiveAction


class PriorityNode(AdditiveAction):
    allowed_parameters: ClassVar[List[Parameter]] = [
        StringParameter(
            local_id='endorsers',
            label=TranslatedString(en="Identifiers for people that endorse this priority"),  # FIXME Should be a list
            is_customizable=True
        ),
        StringParameter(
            local_id='action_id',
            label=TranslatedString(en="Action to be prioritized"),
            is_customizable=True
        ),
        BoolParameter(
            local_id='priority',
            label=TranslatedString(en='Should the action be implemented?'),
            is_customizable=True
        )
    ]


class MultiAttributeUtilityAction(AdditiveAction):
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

    def compute_effect(self):
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
