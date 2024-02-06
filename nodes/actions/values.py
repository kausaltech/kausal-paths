from nodes.calc import extend_last_historical_value_pl, nafill_all_forecast_years
from params.param import Parameter, BoolParameter, NumberParameter, ParameterWithUnit, StringParameter
from typing import List, ClassVar, Tuple
from nodes.constants import VALUE_COLUMN, FORECAST_COLUMN, YEAR_COLUMN, DEFAULT_METRIC
from nodes.node import NodeMetric
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


class BudgetingAction(AdditiveAction):
    '''
    A budgeting action is for giving both cost and effect information. It has two parameters for scaling and postponing the action.
    '''
    output_metrics = {
        DEFAULT_METRIC: NodeMetric(unit='pcs', quantity='number', column_id='number'),
        'cost': NodeMetric(unit='kEUR/a', quantity='currency', column_id='currency')
    }
    allowed_parameters: ClassVar[List[Parameter]] = AdditiveAction.allowed_parameters + [
        NumberParameter(
            local_id='scale_by',
            label=TranslatedString(en='Scale the action by this number'),
            is_customizable=True
        ),
        NumberParameter(
            local_id='postpone_by',
            label=TranslatedString(en='Postpone the action by this many years'),
            is_customizable=True
        ),
    ]

    def compute_effect(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        scale = self.get_parameter_value('scale_by',units=False, required=False)
        postpone = self.get_parameter_value('postpone_by', units=False, required=False)

        if not self.is_enabled():
            scale = 0

        if postpone is not None:
            postpone = int(postpone)
            forecast_from = df.filter(pl.col(FORECAST_COLUMN).eq(True))[YEAR_COLUMN].min()
            df = df.with_columns((pl.col(YEAR_COLUMN) + postpone).alias(YEAR_COLUMN))
            df = df.with_columns((pl.col(YEAR_COLUMN) >= forecast_from).alias(FORECAST_COLUMN))

        if scale is not None:
            for m in df.metric_cols:
                df = df.with_columns((pl.col(m) * scale).alias(m))
        return df
    