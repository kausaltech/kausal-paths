import pandas as pd
import numpy as np
from params import StringParameter
from nodes.calc import extend_last_historical_value
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN, FORECAST_COLUMN
from nodes.dimensions import Dimension
from nodes.actions import ActionNode


class DatasetAction(ActionNode):
    allowed_parameters = [StringParameter('gpc_sector', description = 'GPC Sector', is_customizable = False)]

    no_effect_value = 0.0

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

    def compute_effect(self) -> pd.DataFrame:
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
        df[FORECAST_COLUMN] = False
        df = df[[FORECAST_COLUMN, VALUE_COLUMN]]

        if not self.is_enabled():  # FIXME DataFrame is calculated correctly but still does not affect the model outcome?!?
            df[VALUE_COLUMN] *= self.no_effect_value

#       df = extend_last_historical_value(df, self.get_end_year())

        return df

class DatasetActionMFM(ActionNode):
    allowed_parameters = [StringParameter('action', description = 'Action Name', is_customizable = False)]

    no_effect_value = 0.0

    qlookup = {'Emission Factor': 'emission_factor',
               'Emissions': 'emissions',
               'Energy Consumption': 'energy',
               'Fuel Consumption': 'fuel_consumption',
               'Mileage': 'mileage',
               'Price': 'currency',
               'Unit Price': 'unit_price',
               'Waste Disposal': 'mass'}

    # -----------------------------------------------------------------------------------
    def makeid(self, label: str):
        # Supported languages: Czech, Danish, English, Finnish, German, Latvian, Polish, Swedish
        idlookup = {'': ['.', ',', ':', '-'],
                    '_': [' '],
                    'and': ['&'],
                    'a': ['ä', 'å', 'ą', 'á', 'ā'],
                    'c': ['ć', 'č'],
                    'd': ['ď'],
                    'e': ['ę', 'é', 'ě', 'ē'],
                    'g': ['ģ'],
                    'i': ['í', 'ī'],
                    'k': ['ķ'],
                    'l': ['ł', 'ļ'],
                    'n': ['ń', 'ň', 'ņ'],
                    'o': ['ö', 'ø', 'ó'],
                    'r': ['ř'],
                    's': ['ś', 'š'],
                    't': ['ť'],
                    'u': ['ü', 'ú', 'ů', 'ū'],
                    'y': ['ý'],
                    'z': ['ź', 'ż', 'ž'],
                    'ae': ['æ'],
                    'ss': ['ß']}

        idtext = label.lower()
        if idtext[:5] == 'scope':
            idtext = idtext.replace(' ', '')

        for tochar in idlookup:
            for fromchar in idlookup[tochar]:
                idtext = idtext.replace(fromchar, tochar)

        return idtext

    # -----------------------------------------------------------------------------------
    def compute_effect(self) -> pd.DataFrame:
        df = self.get_input_dataset()
        df = df[df.index.get_level_values('Action') == self.get_parameter_value('action')]

        if not self.is_enabled():
            df['Value'] = self.no_effect_value

        df.index = df.index.droplevel(['Sector', 'Action'])
        df.index = df.index.set_names([self.makeid(i) for i in df.index.names])
        df.index = df.index.set_names({'year': 'Year'})

        dfi = df.index.to_frame(index = False)
        for col in list(set(df.index.names) - set(['quantity', 'Year'])):
            for cat in dfi[col].unique():
                dfi[col] = dfi[col].replace(cat, self.makeid(cat))

        df.index = pd.MultiIndex.from_frame(dfi)
        qdfs = []
        for quantity in df.index.get_level_values('quantity').unique().to_list():
            qdf = df[df.index.get_level_values('quantity') == quantity].copy()
            qdf.index = qdf.index.droplevel('quantity')

            qdf['Value'] = qdf['Value'].astype('pint[' + qdf['Unit'].unique()[0] + ']')
            qdf = qdf.drop(columns = ['Unit'])

            qdf = qdf.rename(columns = {'Value': self.qlookup[quantity]})
            qdfs.append(qdf)

        jdf = qdfs[0]
        for qdf in qdfs[1:]:
            jdf = jdf.join(qdf, how = 'outer')

        for dim in self.output_dimensions:
            self.output_dimensions[dim].is_internal = True

        jdf[FORECAST_COLUMN] = False
        return(jdf)
