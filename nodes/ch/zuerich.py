import pandas as pd
from pint_pandas import PintType

from nodes.node import NodeMetric, NodeError
from nodes.simple import AdditiveNode
from nodes.constants import EMISSION_FACTOR_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN


class BuildingEnergy(AdditiveNode):
    output_metrics = {
        ENERGY_QUANTITY: NodeMetric(unit='GWh/a', quantity=ENERGY_QUANTITY)
    }
    output_dimension_ids = [
        'energy_carrier',
    ]
    input_dimension_ids = [
        'energy_carrier',
    ]

    def compute(self) -> pd.DataFrame:
        df = self.get_input_dataset()

        ec_dim = self.output_dimensions['energy_carrier']
        df[ec_dim.id] = ec_dim.series_to_ids(df['energy_carrier'])
        df[YEAR_COLUMN] = df['year']
        df[ENERGY_QUANTITY] = df['energy'].astype('pint[GWh/a]')
        df = df.set_index([YEAR_COLUMN, ec_dim.id])
        df[FORECAST_COLUMN] = False
        df = df[[ENERGY_QUANTITY, FORECAST_COLUMN]]
        df = df.rename(columns={ENERGY_QUANTITY: VALUE_COLUMN})
        return df


class ElectricityEmissionFactor(AdditiveNode):
    output_metrics = {
        EMISSION_FACTOR_QUANTITY: NodeMetric(unit='g/kWh', quantity=EMISSION_FACTOR_QUANTITY)
    }
    #output_dimension_ids = [
    #    'electricity_source',
    #]
    input_dimension_ids = [
        'electricity_source',
    ]
    default_unit = 'g/kWh'

    def compute(self) -> pd.DataFrame:
        dfs = self.get_input_datasets()
        mix_df = None
        ef_df = None
        for df in dfs:
            if 'share' in df.columns:
                mix_df = df
            elif 'emission_factor' in df.columns:
                ef_df = df
        if mix_df is None:
            raise NodeError(self, "Electricity mix dataset not supplied")
        if ef_df is None:
            raise NodeError(self, "Emission factor dataset not supplied")

        es_dim = self.input_dimensions['electricity_source']

        mix_df[es_dim.id] = es_dim.series_to_ids(mix_df['electricity_source'])
        mix_df[YEAR_COLUMN] = mix_df.pop('year')
        mix_df = mix_df.set_index([YEAR_COLUMN, es_dim.id])

        ef_df[es_dim.id] = es_dim.series_to_ids(ef_df['electricity_source'])
        ef_df[YEAR_COLUMN] = ef_df.pop('year')
        ef_df = ef_df.set_index([YEAR_COLUMN, es_dim.id])

        df = ef_df
        df['Share'] = mix_df['share'].astype('pint[dimensionless]')
        df['EF'] = (df['Share'] * df['emission_factor']).fillna(0)
        s = df['EF'].unstack(es_dim.id).sum(axis=1)
        s = s.astype(PintType(self.unit))
        df = pd.DataFrame(data=s, index=s.index, columns=[VALUE_COLUMN])
        df[FORECAST_COLUMN] = False
        return df
