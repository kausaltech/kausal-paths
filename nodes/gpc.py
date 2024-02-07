import pandas as pd
from params import StringParameter
from nodes.calc import extend_last_historical_value
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN, FORECAST_COLUMN
from nodes.dimensions import Dimension
from nodes.node import Node


class EmissionsNode(Node):
    allowed_parameters = [StringParameter('gpc_sector', description = 'GPC Sector', is_customizable = False)]

    qlookup = {'emissions': 'Emissions'}

    def compute(self) -> pd.DataFrame:
        sector = self.get_parameter_value('gpc_sector')

        df = self.get_input_dataset()
        df = df[(df.index.get_level_values('Sector') == sector) &
                (df.index.get_level_values('Quantity') == self.qlookup[self.quantity])]

        droplist = ['Quantity']
        for i in df.index.names:
            if df.index.get_level_values(i).all() == '.':
                droplist.append(i)

        df.index = df.index.droplevel(droplist)

        unit = df['Unit'].unique()[0]
        df['Value'] = df['Value'].astype('pint[' + unit + ']')
        df = df.drop(columns = ['Unit'])

        for i in list(set(df.index.names) - set(['Year'])):
            self.output_dimensions[i] = Dimension(id = i.lower().replace(' ', '_'),
                                                  label = {'en': i}, help_text = {'en': ''},
                                                  is_internal = True)

        df[FORECAST_COLUMN] = False
#       df = extend_last_historical_value(df, self.get_end_year())

        return df

