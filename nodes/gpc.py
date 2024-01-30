import numpy as np
import pandas as pd
from params import StringParameter, BoolParameter
from nodes.node import Node
from nodes.constants import (
    VALUE_COLUMN, YEAR_COLUMN, FORECAST_COLUMN, EMISSION_FACTOR_QUANTITY, EMISSION_QUANTITY, ENERGY_QUANTITY
)
from nodes.dimensions import Dimension
from nodes.exceptions import NodeError
from nodes.node import NodeMetric


class EmissionsNode(Node):
    input_datasets = ['gpc/saskatoon']
    # global_parameters = ['municipality_name']
    # output_metrics = {
    #     EMISSION_QUANTITY: NodeMetric(unit='kt/a', quantity=EMISSION_QUANTITY),
    #     ENERGY_QUANTITY: NodeMetric(unit='GWh/a', quantity=ENERGY_QUANTITY),
    #     EMISSION_FACTOR_QUANTITY: NodeMetric(unit='g/kWh', quantity=EMISSION_FACTOR_QUANTITY)
    # }
    # output_dimensions = {
    #     'Sector': Dimension(id='syke_sector', label=dict(en='SYKE emission sector'), is_internal=True)
    # }

    def compute(self) -> pd.DataFrame:
        # muni_name = self.get_global_parameter_value('municipality_name')

        df = self.get_input_dataset()
        df = df[df['Sector'] == 'III.1.1']
        # df = df[df['kunta'] == muni_name].drop(columns=['kunta'])
        # df = df.rename(columns={
        #     'vuosi': YEAR_COLUMN,
        #     'ktCO2e': EMISSION_QUANTITY,
        #     'energiankulutus': ENERGY_QUANTITY,
        # })
        # df[EMISSION_FACTOR_QUANTITY] = df[EMISSION_QUANTITY] / df[ENERGY_QUANTITY].replace(0, np.nan)

        # df['Sector'] = ''
        # for i in range(1, 6):
        #     if i > 1:
        #         df['Sector'] += '|'
        #     df['Sector'] += df['taso_%d' % i].astype(str)
        # df.loc[df['hinku-laskenta'], 'Sector'] += ':HINKU'
        # df.loc[df['päästökauppa'], 'Sector'] += ':ETS'

        # df = df[[YEAR_COLUMN, EMISSION_QUANTITY, ENERGY_QUANTITY, EMISSION_FACTOR_QUANTITY, 'Sector']]
        # df = df.set_index([YEAR_COLUMN, 'Sector']).sort_index()
        # if len(df) == 0:
        #     raise NodeError(self, "Municipality %s not found in data" % muni_name)
        # for metric_id, metric in self.output_metrics.items():
        #     if hasattr(df[metric_id], 'pint'):
        #         df[metric_id] = self.convert_to_unit(df[metric_id], metric.unit)
        #     else:
        #         df[metric_id] = df[metric_id].astype('pint[' + str(metric.unit) + ']')

        # df[FORECAST_COLUMN] = False

        return df

