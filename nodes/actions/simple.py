from __future__ import annotations

from cmath import nan
from typing import TYPE_CHECKING, ClassVar

from django.utils.translation import gettext_lazy as _

import polars as pl

from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN
from nodes.generic import GenericNode
from nodes.gpc import DatasetNode
from nodes.node import NodeError
from nodes.simple import SimpleNode
from params import BoolParameter, NumberParameter, PercentageParameter, StringParameter

from .action import ActionNode

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from params.param import Parameter


class GenericAction(GenericNode, ActionNode):
    allowed_parameters = [
        *GenericNode.allowed_parameters,
        *ActionNode.allowed_parameters,
    ]

    no_effect_value = 0.0

    def compute_effect(self) -> PathsDataFrame:
        df = super().compute()
        if not self.is_enabled():
            df = df.with_columns(pl.lit(self.no_effect_value).alias(VALUE_COLUMN))
        return df

    def compute(self) -> PathsDataFrame:
        return self.compute_effect()


class AdditiveAction(ActionNode):
    explanation = _("""Simple action that produces an additive change to a value.""")
    no_effect_value = 0.0

    def compute_effect(self):
        df = self.get_input_dataset_pl()

        if self.get_parameter_value('allow_null_categories', required=False):
            self.allow_null_categories = True

        for m in self.output_metrics.values():
            if not self.is_enabled():
                df = df.with_columns(pl.when(pl.col(m.column_id).is_null()).then(None)
                                     .otherwise(self.no_effect_value).alias(m.column_id))
            df = df.ensure_unit(m.column_id, m.unit)

        return df


class AdditiveAction2(AdditiveAction, SimpleNode):  # FIXME Merge with AdditiveAction
    allowed_parameters = AdditiveAction.allowed_parameters + SimpleNode.allowed_parameters

    def compute_effect(self):
        df = super().compute_effect()
        multiplier = self.get_parameter_value('multiplier', required=False, units=True)
        if multiplier is not None:
            df = df.multiply_quantity(VALUE_COLUMN, multiplier)
        return df


# FIXME Update to deal with old-fashioned multi-metric nodes such as Tampere/private_building_energy_renovation
class CumulativeAdditiveAction(ActionNode):
    explanation = _("""Additive action where the effect is cumulative and remains in the future.""")

    allowed_parameters: ClassVar[list[Parameter]] = [
        PercentageParameter('target_year_ratio', min_value=0),
    ]

    def add_cumulatively(self, df):
        end_year = self.get_end_year()
        df = df.reindex(range(df.index.min(), end_year + 1))
        df[FORECAST_COLUMN] = True

        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue

            val = df[col]
            if hasattr(val, 'pint'):
                val = val.pint.m
            val = val.fillna(0).cumsum()

            target_year_ratio = self.get_parameter_value('target_year_ratio', required=False)
            if target_year_ratio is not None:
                val *= target_year_ratio / 100

            df[col] = val
            if not self.is_enabled():
                df[col] = 0.0
            df[col] = self.ensure_output_unit(df[col])

        return df

    def compute_effect(self):
        df = self.get_input_dataset()
        return self.add_cumulatively(df)


class LinearCumulativeAdditiveAction(CumulativeAdditiveAction):
    allowed_parameters = CumulativeAdditiveAction.allowed_parameters + [
        NumberParameter('target_year_level'),
        NumberParameter(
            local_id='action_delay',
            label='Years of delay (a)',
        ),
        NumberParameter('multiplier'),
    ]

    explanation = _("""Cumulative additive action where a yearly target is set and the effect is linear.
    This can be modified with these parameters:
    target_year_level is the value to be reached at the target year.
    action_delay is the year when the implementation of the action starts.
    multiplier scales the size of the impact (useful between scenarios).
    """)
    def compute_effect(self):
        df = self.get_input_dataset()
        start_year = df.index.min()
        delay = self.get_parameter_value('action_delay', required=False)
        if delay is not None:
            start_year = start_year + int(delay)
        target_year = self.get_target_year()
        df = df.reindex(range(start_year, target_year + 1))
        df[FORECAST_COLUMN] = True

        target_year_level = self.get_parameter_value('target_year_level', required=False)
        if target_year_level is not None:
            if set(df.columns) != {VALUE_COLUMN, FORECAST_COLUMN}:
                raise NodeError(self, "target_year_level parameter can only be used with single-value nodes")
            df.loc[target_year, VALUE_COLUMN] = target_year_level
            if delay is not None:
                df.loc[range(start_year + 1, target_year), VALUE_COLUMN] = nan

        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            dt = df.dtypes[col]
            df[col] = df[col].pint.m.interpolate(method='linear').diff().fillna(0).astype(dt)

        df = self.add_cumulatively(df)
        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            multiplier = self.get_parameter_value('multiplier', required=False, units=True)
            if multiplier is not None:
                df[col] *= multiplier
            df[col] = self.ensure_output_unit(df[col])
        return df


class EmissionReductionAction(ActionNode):
    explanation = _("""Simple emission reduction impact""")

    no_effect_value = 0

    def compute_effect(self):
        df = self.get_input_dataset()
        df[VALUE_COLUMN] = 0 - df[VALUE_COLUMN]
        return df


class TrajectoryAction(ActionNode):
    explanation = _("""
    TrajectoryAction uses select_category() to select a category from a dimension
    and then possibly do some relative or absolute conversions.
    """)
    allowed_parameters = [
        *ActionNode.allowed_parameters,
        StringParameter(local_id='dimension'),
        StringParameter(local_id='category'),
        NumberParameter(local_id='category_number'),
        NumberParameter(local_id='baseline_year'),
        NumberParameter(local_id='baseline_year_level'),
        BoolParameter(local_id='keep_dimension'),
    ]
    def compute_effect(self):
        df = self.get_input_dataset_pl()
        dim_id = self.get_parameter_value('dimension', required=True)
        cat_id = self.get_parameter_value('category', required=False)
        cat_no = self.get_parameter_value('category_number', units=False, required=False)
        if cat_no is not None:
            cat_no = int(cat_no)
        year = self.get_parameter_value('baseline_year', units=False, required=False)
        if year is not None:
            year = int(year)
        level = self.get_parameter_value('baseline_year_level', units=True, required=False)
        keep = self.get_parameter_value('keep_dimension', required=False)
        if not self.is_enabled():
            cat_id = 'baseline'  # FIXME Generalize this
            cat_no = None

        df = df.select_category(dim_id, cat_id, cat_no, year, level, keep)
        assert self.unit is not None
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df


class GpcTrajectoryAction(TrajectoryAction, DatasetNode):
    explanation = _("""
    GpcTrajectoryAction is a trajectory action that uses the DatasetNode to fetch the dataset.
    """)
    allowed_parameters = TrajectoryAction.allowed_parameters + DatasetNode.allowed_parameters

    def compute_effect(self):
        df = DatasetNode.compute(self)
        dim_id = self.get_parameter_value('dimension', required=True)
        cat_id = self.get_parameter_value('category', required=False)
        cat_no = self.get_parameter_value('category_number', units=False, required=False)
        if cat_no is not None:
            cat_no = int(cat_no)
        year = self.get_parameter_value('baseline_year', units=False, required=False)
        if year is not None:
            year = int(year)
        level = self.get_parameter_value('baseline_year_level', units=True, required=False)
        keep = self.get_parameter_value('keep_dimension', required=False)
        if not self.is_enabled():
            cat_id = 'baseline'  # FIXME Generalize this
            cat_no = None

        df = df.select_category(dim_id, cat_id, cat_no, year, level, keep)
        assert self.unit is not None
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df
