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


class Hypothesis(AdditiveAction):
    '''
    A hypothesis is an action-like node that allows users to define their own belief
    about a node. A hypothesis is defined in a dataset that contains deviations
    from the default value (managed by the admin user). A user can build their
    own beliefs into a user-specific scenario (with the help of admin user).
    Hypothesis 0 is the equal-weight average of all hypotheses.
    If the hypothesis is disabled, the admin user's default values are used.
    '''
    allowed_parameters: ClassVar[List[Parameter]] = [
        NumberParameter(
            local_id='hypothesis_number',
            label=TranslatedString(en='Number of hypothesis'),
            is_customizable=True
        )
    ]

    def compute_effect(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        meta = df.get_meta()
        hp = self.get_parameter('hypothesis_number', required=True)

        assert 'hypothesis' in df.primary_keys
        assert hp.min_value is not None
        assert hp.max_value is not None
        hp = int(hp.value)

        if hp == 0:
            n = df['hypothesis'].unique().len()

            df = df.paths.sum_over_dims('hypothesis')
            df = df.with_columns([
                (pl.col(VALUE_COLUMN) / pl.lit(n)).alias(VALUE_COLUMN),
                pl.lit('equal_weight').alias('hypothesis')
            ])

        else:
            df = df.filter(pl.col('hypothesis').eq(pl.lit('hypothesis_' + str(hp))))

        if not self.is_enabled():
            df = df.with_columns((pl.col(VALUE_COLUMN) * pl.lit(0)).alias(VALUE_COLUMN))

        df = ppl.to_ppdf(df, meta=meta)
        return df


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
