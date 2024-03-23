import pandas as pd
import polars as pl
import numpy as np
from params import StringParameter, BoolParameter
from nodes.calc import extend_last_historical_value
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN, FORECAST_COLUMN
from nodes.dimensions import Dimension
from nodes.node import Node
from nodes.simple import AdditiveNode
from common import polars as ppl


class DatasetNode(AdditiveNode):
    allowed_parameters = [StringParameter('gpc_sector', description = 'GPC Sector', is_customizable = False)]

    qlookup = {'currency': 'Price',
               'emission_factor': 'Emission Factor',
               'emissions': 'Emissions',
               'energy': 'Energy Consumption',
               'fuel_consumption': 'Fuel Consumption',
               'mass': 'Waste Disposal',
               'mileage': 'Mileage',
               'unit_price': 'Unit Price'}

    def makeid(self, name: str):
        return (name.lower().replace('.', '').replace(',', '').replace(':', '').replace('-', '').replace(' ', '_')
                .replace('&', 'and').replace('å', 'a').replace('ä', 'a').replace('ö', 'o'))

    def compute(self) -> pd.DataFrame:
        sector = self.get_parameter_value('gpc_sector')

        df = self.get_input_dataset()
        df = df[(df.index.get_level_values('Sector') == sector) &
                (df.index.get_level_values('Quantity') == self.qlookup[self.quantity])]

        droplist = ['Sector', 'Quantity']
        for i in df.index.names:
            empty = df.index.get_level_values(i).all()
            if empty is np.nan or empty == '.':
                droplist.append(i)

        df.index = df.index.droplevel(droplist)

        unit = df['Unit'].unique()[0]
        df['Value'] = df['Value'].astype('pint[' + unit + ']')
        df = df.drop(columns = ['Unit'])

        dims = []
        for i in list(df.index.names):
            if i == YEAR_COLUMN:
                dims.append(i)
            else:
                dims.append(self.makeid(i))
        df.index = df.index.set_names(dims)
        df = df.reset_index()
        for i in list(set(dims) - {YEAR_COLUMN}):
            if isinstance(df[i][0], str):
                for j in range(len(df)):
                    df[i][j] = self.makeid(df[i][j])
        df = df.set_index(dims)
        if FORECAST_COLUMN not in df.columns:
            df[FORECAST_COLUMN] = False
        df = df[[FORECAST_COLUMN, VALUE_COLUMN]]

        na_nodes = self.get_input_nodes(tag='non_additive')
        input_nodes = [node for node in self.input_nodes if node not in na_nodes]


#        df = ppl.to_ppdf(df)
        df = self.add_nodes(df, input_nodes)

        if len(na_nodes) > 0:
            assert len(na_nodes) == 1 # Only one multiplier allowed
            mult = na_nodes[0].get_output(target_node=self)
 #           df = mult.paths.join_over_index(df)
            df = df.join(mult, how='outer', rsuffix='_right')  # FIXME Use PPDF, not pandas.df

            df[VALUE_COLUMN] *= df[VALUE_COLUMN + '_right']
            df = df[[FORECAST_COLUMN, VALUE_COLUMN]]

#       df = extend_last_historical_value(df, self.get_end_year())

        return df


class WeatherNode(AdditiveNode):
    allowed_parameters = AdditiveNode.allowed_parameters + [
        BoolParameter('weather_normalization', description = 'Is energy normaized for weather?')
    ]
    def compute(self):
        df = super().compute()
        is_corrected = self.get_parameter_value('weather_normalization', required=True)

        if is_corrected:
            df = df.with_columns(pl.col(VALUE_COLUMN) * pl.lit(0) + pl.lit(1))

        return df