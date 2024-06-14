import pandas as pd
import polars as pl
import numpy as np
from params import StringParameter
from nodes.calc import extend_last_historical_value
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN, FORECAST_COLUMN
from nodes.dimensions import Dimension
from nodes.actions import ActionNode
from common import polars as ppl


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
               'unit_price': 'Unit Price',
               'occupancy_factor': 'Occupancy Factor',
               'energy_factor': 'Energy Factor',
               'fraction': 'Fraction'}

    # -----------------------------------------------------------------------------------
    def makeid(self, label: str):
        # Supported languages: Czech, Danish, English, Finnish, German, Latvian, Polish, Swedish
        idlookup = {'': ['.', ',', ':', '-', '(', ')'],
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

        # Convert index level names from labels to IDs.
        dims = []
        for i in df.index.names:
            if i == YEAR_COLUMN:
                dims.append(i)
            else:
                dims.append(self.makeid(i))
        df.index = df.index.set_names(dims)

        # Convert levels within each index level from labels to IDs.
        dfi = df.index.to_frame(index = False)
        for col in list(set(dims) - {YEAR_COLUMN}):
            for cat in dfi[col].unique():
                dfi[col] = dfi[col].replace(cat, self.makeid(cat))

        df.index = pd.MultiIndex.from_frame(dfi)

        # Add forecast column if needed.
        if 'Forecast' not in df.columns:
            df['Forecast'] = False

        # Add missing years and interpolate missing values.
        df = ppl.from_pandas(df)
        df = df.paths.to_wide()

        yeardf = pd.DataFrame({'Year': range(dfi['Year'].min(), dfi['Year'].max() + 1)})
        yeardf = yeardf.set_index(['Year'])
        yeardf = ppl.from_pandas(yeardf)

        df = df.paths.join_over_index(yeardf, how = 'outer')
        for col in list(set(df.columns) - set(['Year', 'Forecast'])):
            df = df.with_columns(pl.col(col).interpolate())

        df = df.with_columns(pl.col('Forecast').fill_null(strategy = 'forward'))

        df = df.paths.to_narrow()
        df = df.to_pandas()

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
        idlookup = {'': ['.', ',', ':', '-', '(', ')'],
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
        # Perform initial filtering of GPC dataset.
        df = self.get_input_dataset()
        df = df[df['Value'].notnull()]
        df = df[df.index.get_level_values('Action') == self.get_parameter_value('action')]

        # Drop filter levels and empty dimension levels.
        droplist = ['Action']
        for col in df.index.names:
            vals = df.index.get_level_values(col).unique().to_list()
            if vals == ['.']:
                droplist.append(col)
        df.index = df.index.droplevel(droplist)

        # Convert index level names from labels to IDs.
        df.index = df.index.set_names([self.makeid(i) for i in df.index.names])
        df.index = df.index.set_names({'year': 'Year'})

        # Convert levels within each index level from labels to IDs.
        dfi = df.index.to_frame(index = False)
        for col in list(set(df.index.names) - set(['quantity', 'Year'])):
            for cat in dfi[col].unique():
                dfi[col] = dfi[col].replace(cat, self.makeid(cat))

        df.index = pd.MultiIndex.from_frame(dfi)

        # Create DF with all years and forecast true/false values.
        yeardf = pd.DataFrame({'Year': range(dfi['Year'].min(), dfi['Year'].max() + 1)})
        yeardf = yeardf.set_index(['Year'])

        if 'Forecast' in df.columns:
            fc = df.reset_index()
            fc = pd.DataFrame(fc.groupby('Year')['Forecast'].max())
            fc = yeardf.join(fc)
            fc = fc['Forecast'].ffill()

            df = df.drop(columns = ['Forecast'])
        else:
            fc = yeardf.copy()
            fc['Forecast'] = False

        yeardf = ppl.from_pandas(yeardf)

        # Set value to 'no effect' if action is not enabled.
        if not self.is_enabled():
            df['Value'] = self.no_effect_value

        # Create a DF for each sector/quantity pair...
        dfi = dfi[['sector', 'quantity']].drop_duplicates()
        qdfs = []
        for pair in list(zip(dfi['sector'], dfi['quantity'])):
            qdf = df[(df.index.get_level_values('sector') == pair[0]) &
                     (df.index.get_level_values('quantity') == pair[1])].copy()
            qdf.index = qdf.index.droplevel(['sector', 'quantity'])

            qdf['Value'] = qdf['Value'].astype('pint[' + qdf['Unit'].unique()[0] + ']')
            qdf = qdf.drop(columns = ['Unit'])

            # ...add missing years and interpolate missing values.
            qdf = ppl.from_pandas(qdf)
            qdf = qdf.paths.to_wide()

            qdf = yeardf.paths.join_over_index(qdf)
            for col in list(set(qdf.columns) - set(['Year'])):
                qdf = qdf.with_columns(pl.col(col).interpolate())

            qdf = qdf.paths.to_narrow()
            qdf = qdf.to_pandas()

            # ...rename value column.
            qdf = qdf.rename(columns = {'Value': '%s_%s' % (pair[0], self.qlookup[pair[1]])})
            qdfs.append(qdf)

        # Join sector/quantity DFs into a single multi-metric DF.
        jdf = qdfs[0]
        for qdf in qdfs[1:]:
            jdf = jdf.join(qdf, how = 'outer')

        jdf = jdf.join(fc)

        for dim in self.output_dimensions:
            self.output_dimensions[dim].is_internal = True

        return(jdf)
