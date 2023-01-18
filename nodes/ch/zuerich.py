import pandas as pd

from nodes.node import NodeMetric
from nodes.simple import AdditiveNode
from nodes.constants import ENERGY_QUANTITY, FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN


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
