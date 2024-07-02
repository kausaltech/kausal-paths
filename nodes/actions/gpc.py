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
    allowed_parameters = ActionNode.allowed_parameters + [
        StringParameter('gpc_sector', description = 'GPC Sector', is_customizable = False)
    ]

    no_effect_value = 0.0

    qlookup = {'currency': 'Price',  # FIXME Make a generic qlookup to constants.py and use that.
               'emission_factor': 'Emission Factor',
               'emissions': 'Emissions',
               'energy': 'Energy Consumption',
               'fuel_consumption': 'Fuel Consumption',
               'mass': 'Waste Disposal',
               'mileage': 'Mileage',
               'unit_price': 'Unit Price',
               'occupancy_factor': 'Occupancy Factor',
               'energy_factor': 'Energy Factor',
               'fraction': 'Fraction',
               'number': 'Amount'}

    # -----------------------------------------------------------------------------------
    def makeid(self, label: str):  # FIXME Move all makeid() functions to .calc.py.
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
    def compute_effect(self) -> pd.DataFrame:  # FIXME use sub-functions like DatasetNode
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
        df = self.apply_multiplier(df, required=False, units=True)
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        df = df.to_pandas()

        if not self.is_enabled():
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

class StockReplacementAction(ActionNode):
    allowed_parameters = [StringParameter('sector', description = 'Sector', is_customizable = False),
                          StringParameter('action', description = 'Detailed action module', is_customizable = False)]

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

    def convert_labels_to_ids(self, df: pd.DataFrame) -> pd.DataFrame:
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
        return df

    def drop_unnecessary_levels(self, df: pd.DataFrame, droplist: list) -> pd.DataFrame:
        # Drop filter levels and empty dimension levels.
        for col in df.index.names:
            vals = df.index.get_level_values(col).unique().to_list()
            if vals == ['']:
                droplist.append(col)
        df.index = df.index.droplevel(droplist)
        return df

    def stock_delta(self, stock, cat, delta, delta_type, repcat):
        deltalookup = {'base': 0, 'new': 1, 'rep': 1}

        if delta > 0:
            if delta_type != 'rep':
                stock['%s > %s' % (cat, cat)][-1] += delta
            else:
                stockkey = '%s > %s' % (repcat, cat)
                if stockkey not in stock:
                    stock[stockkey] = [0.0] * len(stock['%s > %s' % (cat, cat)])
                stock[stockkey][-1] += delta

        elif delta < 0:
            keylist = []
            total = 0.0
            for key in list(stock.keys()):
                if key.split(' > ')[deltalookup[delta_type]] == cat:
                    keylist.append(key)
                    total += stock[key][-1]
            for key in keylist:
                stock[key][-1] += delta * (stock[key][-1] / total)

        return stock

    def category_filter(self, df, year, cat):
        df = df.loc[df.index.get_level_values(YEAR_COLUMN) == year]
        for subcat in cat.split('/'):
            subcat = subcat.split(':')
            df = df.loc[df.index.get_level_values(subcat[0]) == subcat[1]]

        return df[VALUE_COLUMN].item()

    def compute(self) -> ppl.PathsDataFrame:
        # Perform initial filtering of dataset. ---------------------------------------------------
        df = self.get_input_dataset()

        df = df[df[VALUE_COLUMN].notnull()]
        df = df[(df.index.get_level_values('Sector') == self.get_parameter_value('sector')) &
                (df.index.get_level_values('Action') == self.get_parameter_value('action'))]

        df.index = df.index.droplevel(['Sector', 'Action'])
        df = self.convert_labels_to_ids(df)

        # -----------------------------------------------------------------------------------------
        yearlist = df.index.get_level_values(YEAR_COLUMN).unique().to_list()
        yearlist.sort()

        investment = df[df.index.get_level_values('node_name') == 'annual_investment']
        investment = self.drop_unnecessary_levels(investment, ['node_name'])

        repcost = df[df.index.get_level_values('node_name') == 'replacement_unit_costs']
        repcost = self.drop_unnecessary_levels(repcost, ['node_name'])

        repscheme = df[df.index.get_level_values('node_name').str.endswith('replacement_scheme')]
        repscheme = self.drop_unnecessary_levels(repscheme, ['node_name'])

        target = df[df.index.get_level_values('node_name').str.startswith('target')]
        target = self.drop_unnecessary_levels(target, ['node_name'])

        catindex = target.index.droplevel(YEAR_COLUMN)
        catlist = []
        for catcombo in catindex.unique().to_list():
            if catindex.nlevels == 1:
                catcombo = [catcombo]

            cattext = ""
            for i in range(len(catcombo)):
                cattext += '%s:%s/' % (catindex.names[i], catcombo[i])
            catlist.append(cattext.lower().replace(' ', '_').strip('/'))
        catlist.sort()

        # -----------------------------------------------------------------------------------------
        base_node = self.get_input_node(tag = 'baseline')
        base = base_node.get_output_pl(target_node = self).rename({VALUE_COLUMN:'Value_Base'})

        new_node = self.get_input_node(tag = 'new')
        new = new_node.get_output_pl(target_node = self).rename({VALUE_COLUMN:'Value_New'})
        new = new.paths.join_over_index(new.cumulate('Value_New').rename({'Value_New':'Value_Cum'}))

        base = base.paths.join_over_index(base.diff('Value_Base').rename({'Value_Base':'Value_Diff'}))
        base = base.paths.join_over_index(new)
        basew = base.paths.to_wide().fill_null(0)

        # -----------------------------------------------------------------------------------------
        stock = {}
        for cat in catlist:
            key = '%s > %s' % (cat, cat)
            stock[key] = [basew.filter(pl.col(YEAR_COLUMN) == yearlist[0]).select('Value_Base@%s' % cat).item()]

        for i in range(len(yearlist)):
            stats = {}
            targets = []
            scheme = {}
            for cat in catlist:
                base_delta = basew.filter(pl.col(YEAR_COLUMN) == yearlist[i]).select('Value_Diff@%s' % cat).item()
                stock = self.stock_delta(stock, cat, base_delta, 'base', '.')

                new_delta = basew.filter(pl.col(YEAR_COLUMN) == yearlist[i]).select('Value_New@%s' % cat).item()
                stock = self.stock_delta(stock, cat, new_delta, 'new', '.')

                stats[cat] = [0.0]
                targets.append([self.category_filter(target, yearlist[i], cat), cat])
                scheme[cat] = self.category_filter(repscheme, yearlist[i], cat)

            # -------------------------------------------------------------------------------------
            total = 0.0
            for key in list(stock.keys()):
                cat = key.split(' > ')[1]
                stats[cat][0] += stock[key][-1]
                total += stock[key][-1]

            for cat in catlist:
                stats[cat].append((stats[cat][0] * 100) / total)

            targets.sort()
            targets.reverse()
            targetcat = targets[0][1]

            # -------------------------------------------------------------------------------------
            iinvestment = investment.loc[investment.index.get_level_values(YEAR_COLUMN) == yearlist[i]][VALUE_COLUMN].item()
            irepcost = repcost.loc[repcost.index.get_level_values(YEAR_COLUMN) == yearlist[i]][VALUE_COLUMN].item()

            repneed = ((targets[0][0] - stats[targetcat][1]) * total) / scheme[targetcat]
            repneed_cost = repneed * irepcost
            repfunded = iinvestment / irepcost

            repcount = min([repneed, repfunded])
            reppool = []
            for cat in catlist:
                if scheme[cat] < 0:
                    reppool.append([(scheme[cat] * repcount) * -1, cat])
                    stock = self.stock_delta(stock, cat, (scheme[cat] * repcount), 'rep', '.')
            
            for cat in catlist:
                if scheme[cat] > 0:
                    catcount = scheme[cat] * repcount
                    while catcount > 0:
                        if catcount <= reppool[0][0]:
                            stock = self.stock_delta(stock, cat, catcount, 'rep', reppool[0][1])
                            catcount = 0
                            if catcount < reppool[0][0]:
                                reppool[0][0] -= catcount
                            else:
                                del reppool[0]
                        else:
                            stock = self.stock_delta(stock, cat, reppool[0][0], 'rep', reppool[0][1])
                            catcount -= reppool[0][0]
                            del reppool[0]

            if yearlist[i] < yearlist[-1]:
                for key in list(stock.keys()):
                    stock[key].append(stock[key][-1])

        # -----------------------------------------------------------------------------------------
        base = base.with_columns(pl.col('Value_Cum').fill_null(0).alias('Value_Cum'))
        base = base.sum_cols(['Value_Base', 'Value_Cum'], 'Value')
        base = base.drop(['Value_Base', 'Value_Diff', 'Value_New', 'Value_Cum'])
        base = base.paths.to_wide()

        for cat in catlist:
            catstock = [0.0] * len(yearlist)
            for key in list(stock.keys()):
                if key.split(' > ')[1] == cat:
                    for i in range(len(yearlist)):
                        catstock[i] += stock[key][i]

            catcol = 'Value@%s' % cat
            if self.is_enabled():
                base = base.with_columns((pl.Series(catstock) - pl.col(catcol)).alias(catcol))
            else:
                base = base.with_columns(pl.lit(0.0).alias(catcol))

        base = base.paths.to_narrow()
        return base


#                cattarget = target.loc[target.index.get_level_values(YEAR_COLUMN) == yearlist[i]]
#                for subcat in cat.split('/'):
#                    subcat = subcat.split(':')
#                    cattarget = cattarget.loc[cattarget.index.get_level_values(subcat[0]) == subcat[1]]
#                cattargets.append([cattarget[VALUE_COLUMN].item(), cat])


        # For each unique combination of dimension categories...
#        base_index = base_frame.index.droplevel(YEAR_COLUMN).unique().to_frame(index = False)

#        baseline = {}
#        for row in base_index.iterrows():
#            type_frame = base_frame.xs(tuple(row[1].values), level = list(row[1].index), drop_level = False).copy()

#            # ...calculate the annual difference.
#            type_frame['Value_Diff'] = type_frame[VALUE_COLUMN].diff().fillna(0)
#            type_frame = type_frame.reset_index().to_dict(orient = 'list')

#            if len(baseline):
#                for key in list(baseline.keys()):
#                    baseline[key].extend(type_frame[key])
#            else:
#                baseline = type_frame
