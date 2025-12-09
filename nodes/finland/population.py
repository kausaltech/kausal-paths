from __future__ import annotations

import polars as pl

from common import polars as ppl
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.exceptions import NodeError
from nodes.node import Node

HISTORICAL_DATASET = 'statfi/StatFin/vrm/vaerak/statfin_vaerak_pxt_11re'
FORECAST_DATASET = 'statfi/StatFin/vrm/vaenn/statfin_vaenn_pxt_139f'


class Population(Node): # FIXME Convert functionality to GenericNode
    TOTAL_POPULATION_COLUMN = 'Väestö 31.12.'

    global_parameters = ['municipality_name']
    input_datasets = [
        HISTORICAL_DATASET, FORECAST_DATASET
    ]
    default_unit = 'person'
    quantity = 'population'

    def get_historical_input(self) -> ppl.PathsDataFrame:
        return self.get_input_datasets_pl()[0]

    def get_forecast_input(self) -> ppl.PathsDataFrame | None:
        return self.get_input_datasets_pl()[1]

    def compute(self):
        muni_name = self.get_global_parameter_value('municipality_name')

        df_hist = self.get_historical_input().lazy()
        hist_cols = df_hist.collect_schema().names()
        if 'Alue' in hist_cols:
            df_hist = df_hist.filter(pl.col('Alue') == muni_name)
        df_hist = df_hist.group_by('Vuosi').agg(pl.col(self.TOTAL_POPULATION_COLUMN).sum().alias(VALUE_COLUMN))
        df_hist = df_hist.with_columns([
            pl.col('Vuosi').cast(pl.Int64),
            pl.lit(False).alias(FORECAST_COLUMN)
        ])
        df = df_hist.collect()

        fc_df = self.get_forecast_input()
        if fc_df is not None:
            ldf = fc_df.lazy()
            col_names = ldf.collect_schema().names()
            if 'Alue' in col_names:
                ldf = ldf.filter(pl.col('Alue') == muni_name)
            pop_col = [col for col in col_names if col.startswith(self.TOTAL_POPULATION_COLUMN)]
            if len(pop_col) != 1:
                raise NodeError(self, 'Unable to find population forecast column')
            col = pop_col[0]
            ldf = ldf.group_by('Vuosi').agg(pl.col(col).sum().alias(VALUE_COLUMN))
            ldf = ldf.with_columns([
                pl.col('Vuosi').cast(pl.Utf8).cast(pl.Int64)
            ])
            # remove years that are also in historical df
            ldf = ldf.filter(~pl.col('Vuosi').is_in(df['Vuosi']))
            ldf = ldf.with_columns([
                pl.lit(True).alias(FORECAST_COLUMN)
            ])
            df = pl.concat([df, ldf.collect()])

        df = df.sort('Vuosi').rename(dict(Vuosi=YEAR_COLUMN))
        df = df.with_columns([
            pl.col(VALUE_COLUMN).cast(pl.Float64)
        ])
        assert self.unit is not None
        df = ppl.to_ppdf(df, meta=ppl.DataFrameMeta(units={VALUE_COLUMN: self.unit}, primary_keys=[YEAR_COLUMN]))
        return df


class HistoricalPopulation(Population):
    def get_forecast_input(self) -> ppl.PathsDataFrame | None:
        return None

    def compute(self):
        return super().compute()
