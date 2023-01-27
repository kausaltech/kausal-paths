from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Union, TypedDict
from dataclasses import dataclass, field

import pandas as pd
import pint
import polars as pl

from common import polars as ppl
from nodes import Node
from nodes.constants import ACTIVITY_QUANTITIES, BASELINE_VALUE_COLUMN, DEFAULT_METRIC, FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN


@dataclass
class YearlyValue:
    year: int
    value: float


class SplitValues(TypedDict):
    historical: list[YearlyValue]
    forecast: list[YearlyValue]
    baseline: list[YearlyValue]
    cumulative_forecast_value: float | None


@dataclass
class Metric:
    id: str
    name: str
    df: ppl.PathsDataFrame
    node: Optional[Node] = None
    unit: Optional[pint.Unit] = None

    split_values: SplitValues | None = field(init=False)

    def __post_init__(self):
        self.split_values = None

    @staticmethod
    def from_node(node: Node):
        df = node.get_output_pl()
        if VALUE_COLUMN not in df.columns:
            return None
        if len(node.output_metrics) > 1 and DEFAULT_METRIC not in node.output_metrics:
            return None

        if len(node.output_metrics) == 1:
            m = list(node.output_metrics.values())[0]
        else:
            m = node.output_metrics[DEFAULT_METRIC]

        assert node.baseline_values is not None

        if m.column_id != VALUE_COLUMN:
            df = df.rename({m.column_id: VALUE_COLUMN})

        bdf = node.baseline_values
        meta = df.get_meta()
        if meta.dim_ids:
            if m.quantity not in ACTIVITY_QUANTITIES:
                return None
            df = df.paths.sum_over_dims()
            bdf = bdf.paths.sum_over_dims()

        meta = df.get_meta()
        bdf = bdf.select([
            YEAR_COLUMN,
            pl.col(m.column_id).alias(BASELINE_VALUE_COLUMN)
        ])
        tdf = df.join(bdf, on=YEAR_COLUMN, how='left').sort(YEAR_COLUMN)
        df = ppl.to_ppdf(tdf, meta=meta)
        return Metric(id=node.id, name=str(node.name), unit=node.unit, node=node, df=df)

    def split_df(self) -> SplitValues | None:
        if self.split_values is not None:
            return self.split_values

        if self.df is None or VALUE_COLUMN not in self.df.columns:
            return None

        df = self.df.drop_nulls()

        hist = []
        forecast = []
        baseline = []
        for row in df.iterrows(named=True):
            is_fc = getattr(row, FORECAST_COLUMN)
            val = getattr(row, VALUE_COLUMN)
            year = getattr(row, YEAR_COLUMN)
            if np.isnan(val):
                raise Exception("Metric %s contains NaN values" % self.id)
            if not is_fc:
                hist.append(YearlyValue(year=year, value=val))
            else:
                bl_val = getattr(row, BASELINE_VALUE_COLUMN, None)
                if bl_val is not None:
                    if np.isnan(bl_val):
                        raise Exception("Metric %s baseline contains NaN values" % self.id)
                    baseline.append(YearlyValue(year=year, value=bl_val))
                forecast.append(YearlyValue(year=year, value=val))

        cum_fc = df.filter(pl.col(FORECAST_COLUMN))[VALUE_COLUMN].sum()

        out = SplitValues(
            historical=hist,
            forecast=forecast,
            cumulative_forecast_value=cum_fc,
            baseline=baseline
        )
        self.split_values = out
        return out

    def get_historical_values(self) -> List[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return []
        return vals['historical']

    def get_forecast_values(self) -> List[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return []
        return vals['forecast']

    def get_baseline_forecast_values(self) -> List[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return []
        return vals['baseline']

    def get_cumulative_forecast_value(self) -> float | None:
        vals = self.split_df()
        if not vals:
            return None
        return vals['cumulative_forecast_value']

    @property
    def yearly_cumulative_unit(self) -> Optional[pint.Unit]:
        if not self.unit:
            return None
        # Check if the unit as a time divisor
        if self.unit.dimensionality.get('[time]') > -1:
            return None
        year_unit = self.unit._REGISTRY('year').units
        return self.unit * year_unit
