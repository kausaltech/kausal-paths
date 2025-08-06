from __future__ import annotations

from typing import ClassVar

import polars as pl

from common import polars as ppl
from common.i18n import TranslatedString
from nodes.constants import DEFAULT_METRIC, FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.node import NodeMetric
from params.param import BoolParameter, NumberParameter, Parameter, StringParameter

from .simple import AdditiveAction


class PriorityNode(AdditiveAction):
    allowed_parameters: ClassVar[list[Parameter]] = [
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
    """
    A hypothesis is an action-like node that allows users to define their own belief about a node.

    A hypothesis is defined in a dataset that contains deviations
    from the default value (managed by the admin user). A user can build their
    own beliefs into a user-specific scenario (with the help of admin user).
    Hypothesis 0 is the equal-weight average of all hypotheses.
    If the hypothesis is disabled, the admin user's default values are used.
    """

    allowed_parameters: ClassVar[list[Parameter]] = [
        NumberParameter(
            local_id='hypothesis_number',
            label=TranslatedString(en='Number of hypothesis'),
            is_customizable=True
        )
    ]

    def compute_effect(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        meta = df.get_meta()
        hp_param = self.get_parameter('hypothesis_number', required=True)
        assert isinstance(hp_param, NumberParameter)

        assert 'hypothesis' in df.primary_keys
        assert hp_param.min_value is not None
        assert hp_param.max_value is not None
        assert hp_param.value is not None
        hp = int(hp_param.value)

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
    """
    A budgeting action is for giving both cost and effect information.

    It has two parameters for scaling and postponing the action.
    """

    output_metrics = {
        DEFAULT_METRIC: NodeMetric(unit='pcs', quantity='number', column_id='number'),
        'cost': NodeMetric(unit='kEUR/a', quantity='currency', column_id='currency')
    }
    allowed_parameters: ClassVar[list[Parameter]] = [
        *AdditiveAction.allowed_parameters,
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
        scale = self.get_parameter_value_float('scale_by', units=False, required=False)
        postpone = self.get_parameter_value_int('postpone_by', required=False)

        if not self.is_enabled():
            scale = 0

        if postpone is not None:
            postpone = int(postpone)
            forecast_from = df.filter(pl.col(FORECAST_COLUMN).eq(other=True))[YEAR_COLUMN].min()
            df = df.with_columns((pl.col(YEAR_COLUMN) + postpone).alias(YEAR_COLUMN))
            df = df.with_columns((pl.col(YEAR_COLUMN) >= forecast_from).alias(FORECAST_COLUMN))

        if scale is not None:
            for m in df.metric_cols:
                df = df.with_columns((pl.col(m) * scale).alias(m))
        return df
