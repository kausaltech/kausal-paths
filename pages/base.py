from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
from nodes import Node
from nodes.constants import BASELINE_VALUE_COLUMN, FORECAST_COLUMN, VALUE_COLUMN


@dataclass
class YearlyValue:
    year: int
    value: float


@dataclass
class Metric:
    id: str
    name: str
    df: pd.DataFrame
    node: Optional[Node] = None

    @staticmethod
    def from_node(node, context):
        df = node.get_output(context)
        if df is None:
            return None
        if VALUE_COLUMN not in df.columns:
            return None
        if node.baseline_values is not None:
            df[BASELINE_VALUE_COLUMN] = node.baseline_values[VALUE_COLUMN]
        return Metric(id=node.id, name=str(node.name), node=node, df=df)

    def split_df(self) -> Dict[str, List[YearlyValue]]:
        if hasattr(self, 'split_values'):
            return self.split_values

        if self.df is None or VALUE_COLUMN not in self.df.columns:
            self.split_values = None
            return None

        df = self.df.copy()
        for col in df.columns:
            if hasattr(df[col], 'pint'):
                df[col] = df[col].pint.m

        hist = []
        forecast = []
        baseline = []
        for row in df.itertuples():
            is_fc = getattr(row, FORECAST_COLUMN)
            val = getattr(row, VALUE_COLUMN)
            year = row.Index
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

        out = dict(
            historical=hist,
            forecast=forecast,
            baseline=baseline if baseline else None
        )
        self.split_values = out
        return out

    def get_historical_values(self) -> List[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return None
        return vals['historical']

    def get_forecast_values(self) -> List[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return None
        return vals['forecast']

    def get_baseline_forecast_values(self) -> List[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return None
        return vals['baseline']

    @property
    def unit(self):
        return self.node.unit if self.node else None
