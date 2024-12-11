from __future__ import annotations

from django.utils.translation import gettext_lazy as _

import pandas as pd
import polars as pl

from common import polars as ppl
from nodes.actions import ActionNode
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.gpc import DatasetNode
from nodes.units import unit_registry
from params import NumberParameter, StringParameter


class DatasetAction(ActionNode, DatasetNode):
    allowed_parameters = [
        *ActionNode.allowed_parameters,
        *DatasetNode.allowed_parameters,
    ]
    no_effect_value = 0.0

    def compute_effect(self) -> ppl.PathsDataFrame:
        df = DatasetNode.compute(self)

        if not self.is_enabled():
            df = df.with_columns(pl.lit(self.no_effect_value).alias(VALUE_COLUMN))

        return df

    def compute(self) -> ppl.PathsDataFrame:
        return self.compute_effect()

class DatasetAction2(DatasetAction):
    pass

class DatasetActionMFM(ActionNode, DatasetNode):
    allowed_parameters = [StringParameter('action', description='Action name in GPC dataset', is_customizable=False),
                          NumberParameter('target_value', description='Target action impact value', is_customizable=True),
                          StringParameter('target_metric', description='Target action metric id', is_customizable=False)]

    allow_null_categories = True
    no_effect_value = 0.0

    def compute_effect(self) -> ppl.PathsDataFrame:
        # Perform initial filtering of GPC dataset.
        df = self.get_input_dataset_pl()
        df = df.filter((pl.col(VALUE_COLUMN).is_not_null()) &
                       (pl.col('Action') == self.get_parameter_value('action')))

        # Drop filter level and empty dimension levels, convert names to IDs.
        df = self.drop_unnecessary_levels(df, droplist=['Action'])
        df = self.convert_names_to_ids(df)
        df = df.with_columns(df['quantity'].replace(self.quantitylookup))

        # Set value to 'no effect' if action is not enabled, get target parameters.
        if not self.is_enabled():
            df = df.with_columns(pl.lit(self.no_effect_value).alias(VALUE_COLUMN))

        tvalue = self.get_parameter_value('target_value', required=False)
        tmetric = self.get_parameter_value('target_metric', required=False)

        # Create DF with all years.
        yearrange = range(df[YEAR_COLUMN].min(), (df[YEAR_COLUMN].max() + 1))
        yeardf = ppl.PathsDataFrame({YEAR_COLUMN: yearrange})
        yeardf._units = {}
        yeardf._primary_keys = [YEAR_COLUMN]

        # Aggregate and interpolate, or add, forecast.
        if FORECAST_COLUMN in df.columns:
            fcdf = ppl.PathsDataFrame(df.group_by(YEAR_COLUMN).agg(pl.col(FORECAST_COLUMN).max()))
            fcdf._units = {}
            fcdf._primary_keys = [YEAR_COLUMN]

            fcdf = fcdf.paths.join_over_index(yeardf, how='outer')
            fcdf = fcdf.with_columns(pl.col(FORECAST_COLUMN).fill_null(strategy='backward'))

            df = df.drop(FORECAST_COLUMN)
        else:
            fcdf = yeardf.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))

        # For each metric...
        mdfs = []
        mlist = df.select(['sector', 'quantity']).unique().rows()
        for m in mlist:
            # ...perform initial filtering.
            mdf = df.filter((pl.col('sector') == m[0]) & (pl.col('quantity') == m[1]))
            mdf = mdf.drop(['sector', 'quantity'])

            # ...define the metric ID, implement metric-specific units.
            if len(mlist) == 1:
                mid = 'Value'
            else:
                mid = '%s_%s' % m

            mdf = self.implement_unit_col(mdf)

            # ...add missing years, substitute target value, and interpolate missing values.
            mdf = mdf.paths.to_wide()
            mdf = mdf.paths.join_over_index(yeardf, how='outer')

            for col in list(set(mdf.columns) - {YEAR_COLUMN}):
                if self.is_enabled() and isinstance(tvalue, float) and tmetric == mid:
                    mdf = mdf.with_columns(pl.when(pl.col(YEAR_COLUMN) == yearrange[-1])
                                             .then(pl.lit(tvalue))
                                             .otherwise(pl.col(col)).alias(col))

                mdf = mdf.with_columns(pl.col(col).interpolate())

            # ...perform final formatting.
            mdf = mdf.paths.to_narrow()
            mdf = mdf.rename({'Value': mid})
            mdfs.append(mdf)

        # Join metric DFs into a single multi-metric DF.
        jdf = mdfs[0]
        for mdf in mdfs[1:]:
            jdf = jdf.paths.join_over_index(mdf, how='outer')

        # Add forecast.
        jdf = jdf.paths.join_over_index(fcdf, how='left')
        return(jdf)

    def compute(self) -> ppl.PathsDataFrame:
        return self.compute_effect()

class StockReplacementAction(ActionNode):
    allowed_parameters = [StringParameter('sector', description = 'Sector', is_customizable = False),
                          StringParameter('action', description = 'Detailed action module', is_customizable = False)]

    def makeid(self, label: str):
        # Supported languages: Czech, Danish, English, Finnish, German, Latvian, Polish, Swedish
        idlookup = {'': ['.', ',', ':', '-', '(', ')'],
                    '_': [' ', '/'],
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

        df = df[df[VALUE_COLUMN].notna()]
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

class SCurveAction(DatasetAction):
    explanation = _(
        """
        This is S Curve Action. It calculates non-linear effect with two parameters,
        max_impact = A and max_year (year when most of the impact has occurred).
        The parameters come from Dataset. In addition, there
        must be one input node for background data. Function for
        S-curve = A/(1+exp(-k*(x-x0)). A is the maximum value, k is the steepness
        of the curve (always 0.5), and x0 is the midpoint.
        """)
    allowed_parameters = DatasetAction2.allowed_parameters

    no_effect_value = 0.0

    def compute_effect(self) -> ppl.PathsDataFrame:
        baseline_year = self.context.instance.reference_year

        df = self.get_input_node().get_output_pl(target_node=self)

        params = self.get_gpc_dataset()
        params = self.drop_unnecessary_levels(params, droplist=['Description'])
        params = self.rename_dimensions(params)
        params = self.convert_names_to_ids(params)
        params = self.implement_unit_col(params)
        params = self.apply_multiplier(params, required=False, units=True)
        params = params.paths.join_over_index(df, how='inner')

        drops = [FORECAST_COLUMN, VALUE_COLUMN + '_right', YEAR_COLUMN, 'parameter']
        ymax = params.filter(pl.col('parameter') == 'max_year').drop(drops)
        ymax = ymax.rename({VALUE_COLUMN: 'ymax'})
        df = df.paths.join_over_index(ymax, how='inner')

        params = params.ensure_unit(VALUE_COLUMN, df.get_unit(VALUE_COLUMN))
        amax = params.filter(pl.col('parameter') == 'max_impact').drop(drops)
        amax = amax.rename({VALUE_COLUMN: 'amax'})
        df = df.paths.join_over_index(amax, how='inner')

        df = df.with_columns((
            pl.col(YEAR_COLUMN) - (pl.lit(baseline_year) +
            (pl.col('ymax') - pl.lit(baseline_year)) / 2)).alias('x'))
        df = df.with_columns((pl.col('amax') / (
            pl.lit(1.0) + (pl.lit(-0.5) * pl.col('x')).exp())
            ).alias('out'))
        df = df.set_unit('out', df.get_unit(VALUE_COLUMN))
        df = df.with_columns((
            pl.when(pl.col(FORECAST_COLUMN))
            .then(pl.col('out'))
            .otherwise(pl.col(VALUE_COLUMN))).alias('out'))
        df = df.subtract_cols(['out', VALUE_COLUMN], VALUE_COLUMN)
        df = df.drop(['x', 'out', 'amax', 'ymax'])

        if not self.is_enabled():
            df = df.with_columns(pl.lit(self.no_effect_value).alias(VALUE_COLUMN))
        return df
