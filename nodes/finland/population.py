import pandas as pd
from nodes import Node
from nodes.constants import VALUE_COLUMN
from nodes.exceptions import NodeError


class Population(Node):
    TOTAL_POPULATION_COLUMN = 'Väestö 31.12.'

    global_parameters = ['municipality_name']
    input_datasets = [
        'statfi/StatFin/vrm/vaerak/statfin_vaerak_pxt_11re',
        'statfi/StatFin/vrm/vaenn/statfin_vaenn_pxt_139f'
    ]
    unit = 'person'
    quantity = 'population'

    def compute(self):
        muni_name = self.get_global_parameter_value('municipality_name')

        df_hist, df_forecast = self.get_input_datasets()
        df_hist = df_hist.xs(muni_name, level='Alue')
        df_forecast = df_forecast.xs(muni_name, level='Alue')

        pop_col = [col for col in df_forecast.columns if col.startswith(self.TOTAL_POPULATION_COLUMN)]
        if len(pop_col) != 1:
            raise NodeError(self, 'Unable to find population forecast column')

        sf = df_forecast.groupby('Vuosi', axis=0)[pop_col[0]].sum()
        sf.index = sf.index.astype(int)
        sh = df_hist.groupby('Vuosi', axis=0)[self.TOTAL_POPULATION_COLUMN].sum()
        sh.index = sh.index.astype(int)

        # Drop forecast rows that are in the historical series
        sf = sf[~sf.index.isin(sh.index)]
        s = pd.concat([sh, sf])
        df = pd.DataFrame(s.values, index=s.index, columns=[VALUE_COLUMN])
        df.index = df.index.astype(int)
        df['Forecast'] = False
        df.loc[df.index >= sf.index.min(), 'Forecast'] = True
        df[VALUE_COLUMN] = df[VALUE_COLUMN].astype(float).astype('pint[person]')
        return df
