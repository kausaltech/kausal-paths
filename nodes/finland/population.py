import pandas as pd
from nodes import Node
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN
from nodes.exceptions import NodeError


HISTORICAL_DATASET = 'statfi/StatFin/vrm/vaerak/statfin_vaerak_pxt_11re'
FORECAST_DATASET = 'statfi/StatFin/vrm/vaenn/statfin_vaenn_pxt_139f'


class Population(Node):
    TOTAL_POPULATION_COLUMN = 'Väestö 31.12.'

    global_parameters = ['municipality_name']
    input_datasets = [
        HISTORICAL_DATASET, FORECAST_DATASET
    ]
    default_unit = 'person'
    quantity = 'population'

    def get_historical_input(self) -> pd.DataFrame:
        df = self.get_input_datasets()[0]
        assert isinstance(df, pd.DataFrame)
        return df

    def get_forecast_input(self) -> pd.DataFrame | None:
        df = self.get_input_datasets()[1]
        assert isinstance(df, pd.DataFrame)
        return df

    def compute(self):
        muni_name = self.get_global_parameter_value('municipality_name')

        df_hist = self.get_historical_input()
        df_hist = df_hist.xs(muni_name, level='Alue')

        sh = df_hist.groupby('Vuosi', axis=0)[self.TOTAL_POPULATION_COLUMN].sum()
        sh.index = sh.index.astype(int)

        df_forecast = self.get_forecast_input()
        if df_forecast is not None:
            df_forecast = df_forecast.xs(muni_name, level='Alue')
            pop_col = [col for col in df_forecast.columns if col.startswith(self.TOTAL_POPULATION_COLUMN)]
            if len(pop_col) != 1:
                raise NodeError(self, 'Unable to find population forecast column')

            sf = df_forecast.groupby('Vuosi', axis=0)[pop_col[0]].sum()
            sf.index = sf.index.astype(int)

            # Drop forecast rows that are in the historical series
            sf = sf[~sf.index.isin(sh.index)]
            s = pd.concat([sh, sf])
            first_forecast_year = sf.index.min()
        else:
            s = sh
            first_forecast_year = None

        df = pd.DataFrame(s.values, index=s.index, columns=[VALUE_COLUMN])
        df.index = df.index.astype(int)
        df.index.name = YEAR_COLUMN
        df['Forecast'] = False
        if first_forecast_year is not None:
            df.loc[df.index >= first_forecast_year, 'Forecast'] = True
        df[VALUE_COLUMN] = df[VALUE_COLUMN].astype(float).astype('pint[person]')
        return df


class HistoricalPopulation(Population):
    def get_forecast_input(self) -> pd.DataFrame | None:
        return None

    def compute(self):
        df = super().compute()
        return df
