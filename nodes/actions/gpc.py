from __future__ import annotations

from typing import ClassVar

from django.utils.translation import gettext_lazy as _

import numpy as np
import polars as pl

from common import polars as ppl
from nodes.actions import ActionNode
from nodes.calc import extend_last_historical_value_pl
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.exceptions import NodeError
from nodes.generic import GenericNode
from nodes.gpc import DatasetNode
from nodes.units import unit_registry
from params import BoolParameter, NumberParameter, Parameter, StringParameter


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

        assert self.unit is not None
        df = df.ensure_unit(VALUE_COLUMN, self.unit) # TODO Use get_unit() instead
        return df

    def compute(self) -> ppl.PathsDataFrame:
        return self.compute_effect()

class DatasetAction2(DatasetAction):
    pass

class DatasetActionMFM(DatasetAction):
    allowed_parameters = [
        StringParameter('action', description='Action name in GPC dataset', is_customizable=False),
        NumberParameter('target_value', description='Target action impact value', is_customizable=True),
        StringParameter('target_metric', description='Target action metric id', is_customizable=False)
    ]
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
        ymin = df[YEAR_COLUMN].min()
        assert isinstance(ymin, int)
        ymax = df[YEAR_COLUMN].max()
        assert isinstance(ymax, int)
        yearrange = range(ymin, ymax + 1)
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

class StockReplacementAction(DatasetAction):
    allowed_parameters = [
        StringParameter('sector', description='GPC sector', is_customizable=False),
        StringParameter('action', description='Detailed action module', is_customizable=False),
        NumberParameter('investment_value', description='Maximum annual investment', is_customizable=True),
        StringParameter('investment_units', description='Investment units', is_customizable=False)
    ]
    allow_null_categories = True

    # ---------------------------------------------------------------------------------------------
    def drop_unnecessary_levels(self, df: ppl.PathsDataFrame, droplist: list) -> ppl.PathsDataFrame:
        drops = [d for d in droplist if d in df.columns]

        for col in list(set(df.columns) - set(drops)):
            vals = df[col].unique().to_list()
            if vals in [['.'], [None], ['']]:
                drops.append(col)

        df = df.drop(drops)
        return df

    # ---------------------------------------------------------------------------------------------
    def stock_delta(self, stock, cat, delta, delta_type, repcat):
        deltalookup = {'base': 0, 'new': 1, 'rep': 1}

        # If the delta is positive, and baseline or new, add entirely new 'cat > cat' stock units;
        # if replacement, add new 'repcat > cat' stock units.
        if delta > 0:
            if delta_type != 'rep':
                stock['%s > %s' % (cat, cat)][-1] += delta
            else:
                stockkey = '%s > %s' % (repcat, cat)
                if stockkey not in stock:
                    stock[stockkey] = [0.0] * len(stock['%s > %s' % (cat, cat)])
                stock[stockkey][-1] += delta

        # If the delta is negative, find all relevant stock unit keys, and subtract proportionally
        # from each. For baseline deltas, relevant keys begin with cat; for new and replacement
        # deltas, relevant keys end with cat.
        elif delta < 0:
            keylist = []
            total = 0.0
            for key in list(stock.keys()):
                if key.split(' > ')[deltalookup[delta_type]] == cat:
                    keylist.append(key)
                    total += stock[key][-1]
            for key in keylist:
                stock[key][-1] += delta * (stock[key][-1] / total)

                # Warn about negative stock units, but assume this occurs only due to floating-
                # point errors, and correct to zero.
                if stock[key][-1] < 0.0:
                    print("Warning: Negative stock units with key '%s' in module year %i: %0.10f" %
                          (key, len(stock[key]), stock[key][-1]))
                    stock[key][-1] = 0.0

        return stock

    # ---------------------------------------------------------------------------------------------
    def category_filter(self, df, year, cat):
        fdf = df.filter(pl.col(YEAR_COLUMN) == year)
        for subcat in cat.split('/'):
            subc = subcat.split(':')
            fdf = fdf.filter(pl.col(subc[0]) == subc[1])

        return fdf.select(VALUE_COLUMN).item()

    # ---------------------------------------------------------------------------------------------
    def sync_dimensions(self, df, refdf):
        for dim in refdf._primary_keys:
            if dim not in df._primary_keys:
                df = df.with_columns(pl.lit('').alias(dim))
                df._primary_keys.append(dim)

        return df

    def compute_effect(self) -> ppl.PathsDataFrame:  # noqa: C901, PLR0912, PLR0915
        # Perform initial filtering & processing of dataset. --------------------------------------
        df = self.get_input_dataset_pl()
        sector = str(self.get_parameter_value('sector'))

        df = df.filter((pl.col(VALUE_COLUMN).is_not_null()) &
                       (pl.col('Sector') == sector) &
                       (pl.col('Action') == self.get_parameter_value('action')))

        df = self.drop_unnecessary_levels(df, droplist=['Sector', 'Action'])
        df = self.convert_names_to_ids(df)

        yearlist = df[YEAR_COLUMN].unique().to_list()
        yearlist.sort()

        # Create parameter DFs, list of stock (multi)categories. ----------------------------------
        p = {}
        for parameter in ['investment', 'target', 'replacement_unit_cost', 'replacement_scheme']:
            p[parameter] = df.filter(pl.col('node_name').str.contains(parameter))
            p[parameter] = self.drop_unnecessary_levels(p[parameter], droplist=['node_name'])

        catlist = []
        catcols = list(set(p['target']._primary_keys) - {YEAR_COLUMN})
        for catcombo in p['target'].select(catcols).unique().rows():
            cattext = ''
            for i in range(len(catcombo)):
                cattext += '%s:%s/' % (catcols[i], catcombo[i])

            catlist.append(cattext.strip('/'))
        catlist.sort()

        # Adjust investment DF per investment parameters, action status. --------------------------
        if self.is_enabled():
            investval = self.get_parameter_value('investment_value', required=False)

            if isinstance(investval, float):
                investunits = self.get_parameter_value('investment_units', required=True)
                p['investment'] = p['investment'].with_columns(
                    pl.when(pl.col(FORECAST_COLUMN)).then(pl.lit(investval)).otherwise(pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN),
                    pl.when(pl.col(FORECAST_COLUMN)).then(pl.lit(investunits)).otherwise(pl.col('Unit')).alias('Unit')
                )
        else:
            p['investment'] = p['investment'].with_columns(pl.lit(0.0).alias(VALUE_COLUMN))

        # Create baseline (non-action) stock DF. --------------------------------------------------
        base = self.get_input_node(tag='baseline').get_output_pl(target_node=self).rename({VALUE_COLUMN: 'Value_Base'})
        base = base.paths.join_over_index(base.diff('Value_Base').rename({'Value_Base': 'Value_Diff'}))

        new = self.get_input_node(tag='new').get_output_pl(target_node=self).rename({VALUE_COLUMN: 'Value_New'})
        new = new.paths.join_over_index(new.cumulate('Value_New').rename({'Value_New': 'Value_Cum'}))

        base = base.paths.join_over_index(new)
        basew = base.paths.to_wide().fill_null(0)

        # Initialize stock dictionary with first-year baseline values. ----------------------------
        stock = {}
        for cat in catlist:
            key = '%s > %s' % (cat, cat)
            stock[key] = [basew.filter(pl.col(YEAR_COLUMN) == yearlist[0]).select('Value_Base@%s' % cat).item()]

        # For each year... ------------------------------------------------------------------------
        for i in range(len(yearlist)):
            stats = {}
            targets = []
            scheme = {}
            for cat in catlist:
                # ...for each (multi)category, find and apply the baseline delta, then the new
                # delta. This calculates the non-action change in stock.
                base_delta = basew.filter(pl.col(YEAR_COLUMN) == yearlist[i]).select('Value_Diff@%s' % cat).item()
                stock = self.stock_delta(stock, cat, base_delta, 'base', '.')

                new_delta = basew.filter(pl.col(YEAR_COLUMN) == yearlist[i]).select('Value_New@%s' % cat).item()
                stock = self.stock_delta(stock, cat, new_delta, 'new', '.')

                stats[cat] = [0.0]
                targets.append([self.category_filter(p['target'], yearlist[i], cat), cat])
                scheme[cat] = self.category_filter(p['replacement_scheme'], yearlist[i], cat)

            # -------------------------------------------------------------------------------------
            # ...for each (multi)category, track the number of ending stock units, and the
            # percentage of the total stock.
            total = 0.0
            for key in list(stock.keys()):
                cat = key.split(' > ')[1]
                stats[cat][0] += stock[key][-1]
                total += stock[key][-1]

            for cat in catlist:
                stats[cat].append((stats[cat][0] * 100) / total)

            # ...the (multi)category with the highest target percentage serves as the target.
            targets.sort()
            targets.reverse()
            targetcat = targets[0][1]

            # -------------------------------------------------------------------------------------
            # ...find the annual investment and replacement cost.
            iinvestment = p['investment'].filter(pl.col(YEAR_COLUMN) == yearlist[i]).select(VALUE_COLUMN).item()
            irepcost = p['replacement_unit_cost'].filter(pl.col(YEAR_COLUMN) == yearlist[i]).select(VALUE_COLUMN).item()

            # ...find the number of replacements needed to reach the target, and the number of
            # replacements funded. Use the minimum of these two numbers. If target is overshot, do
            # not allow negative replacements needed.
            repneed = max(0.0, (((targets[0][0] - stats[targetcat][1]) / 100) * total) / scheme[targetcat])
            repfunded = iinvestment / irepcost

            repcount = min([repneed, repfunded])
            reppool = []
            for cat in catlist:
                # If the replacement scheme removes units in this (multi)category, remove units
                # from the stock. Track number of units removed per category in the 'reppool'.
                if scheme[cat] < 0:
                    reppool.append([(scheme[cat] * repcount) * -1, cat])
                    stock = self.stock_delta(stock, cat, (scheme[cat] * repcount), 'rep', '.')

            # -------------------------------------------------------------------------------------
            for cat in catlist:
                # If the replacement scheme adds units in this (multi)category...
                if scheme[cat] > 0:
                    catcount = scheme[cat] * repcount
                    while catcount > 0:
                        # ...if the number of units to add is less than or equal to the first
                        # category in the reppool...
                        if catcount <= reppool[0][0]:
                            stock = self.stock_delta(stock, cat, catcount, 'rep', reppool[0][1])
                            if catcount < reppool[0][0]:
                                reppool[0][0] -= catcount
                            else:
                                del reppool[0]
                            catcount = 0
                        # ...else the number of units to add is greater than the first category in
                        # the reppool. In this case, loop to the next category.
                        else:
                            stock = self.stock_delta(stock, cat, reppool[0][0], 'rep', reppool[0][1])
                            catcount -= reppool[0][0]
                            del reppool[0]

            # -------------------------------------------------------------------------------------
            # ...write the actual value invested to the investment DF.
            p['investment'] = p['investment'].with_columns(pl.when(pl.col(YEAR_COLUMN) == yearlist[i])
                                                             .then(pl.lit(repcount * irepcost))
                                                             .otherwise(pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))

            if yearlist[i] < yearlist[-1]:
                for key in list(stock.keys()):
                    stock[key].append(stock[key][-1])

        # Prepare final DF for return. ------------------------------------------------------------
        base = base.with_columns(pl.col('Value_Cum').fill_null(0).alias('Value_Cum'))
        base = base.sum_cols(['Value_Base', 'Value_Cum'], VALUE_COLUMN)
        base = base.drop(['Value_Base', 'Value_Diff', 'Value_New', 'Value_Cum'])
        base = base.paths.to_wide()

        # For each (multi)category...
        for cat in catlist:
            # ...sum the ending stock units annually.
            catstock = [0.0] * len(yearlist)
            for key in list(stock.keys()):
                if key.split(' > ')[1] == cat:
                    for i in range(len(yearlist)):
                        catstock[i] += stock[key][i]

            # ...if the action is enabled, subtract the non-action stock units from the calculated
            # stock units, to find the action's impact.
            catcol = 'Value@%s' % cat
            if self.is_enabled():
                base = base.with_columns((pl.Series(catstock) - pl.col(catcol)).alias(catcol))
            else:
                base = base.with_columns(pl.lit(0.0).alias(catcol))

        base = base.paths.to_narrow()

        # Add currency metric for investment costs. -----------------------------------------------
        sector = sector.lower().replace('.', '')
        base = base.rename({VALUE_COLUMN: '%s_number' % sector})
        base = self.sync_dimensions(base, p['investment'])

        p['investment'] = self.sync_dimensions(p['investment'], base)
        p['investment']._units[VALUE_COLUMN] = unit_registry(p['investment']['Unit'].to_list()[0]).units
        p['investment'] = p['investment'].drop('Unit').rename({VALUE_COLUMN: '%s_currency' % sector})

        base = base.paths.join_over_index(p['investment'], how='outer')
        return base

class SCurveAction(DatasetAction):
    explanation = _(
        """
        This is S Curve Action. It calculates non-linear effect with two parameters,
        max_impact = A and max_year (year when 98 per cent of the impact has occurred).
        The parameters come from Dataset. In addition, there
        must be one input node for background data. Function for
        S-curve y = A/(1+exp(-k*(x-x0)). A is the maximum value, k is the steepness
        of the curve, and x0 is the midpoint year.
        Newton-Raphson method is used to numerically estimate slope and medeian year.
        """)
    allowed_parameters = DatasetAction2.allowed_parameters

    no_effect_value = 0.0

    def newton_raphson_estimator(self, y1, y2, x1, x2, a, max_iter=100, tol=1e-6):
        # Compute terms dependent on observations
        z1 = np.log(y1 / (a - y1))
        z2 = np.log(y2 / (a - y2))

        # Initial guesses (adjust based on domain knowledge)
        k = 0.1
        x0 = (x1 + x2) / 2  # Midpoint between x1 and x2

        for __ in range(max_iter):
            # Residual vector F
            f = np.array([
                k * (x1 - x0) - z1,
                k * (x2 - x0) - z2
            ])

            # Jacobian matrix J
            j = np.array([
                [x1 - x0, -k],
                [x2 - x0, -k]
            ])

            # Check for singularity
            det = np.linalg.det(j)
            if np.abs(det) < 1e-10:
                print('y1, y2, x1, x2, a:', y1, y2, x1, x2, a)
                raise ValueError("Jacobian is singular; adjust initial guesses.")

            # Newton-Raphson update
            delta = np.linalg.solve(j, -f)
            k_new = k + delta[0]
            x0_new = x0 + delta[1]

            # Check convergence
            if np.abs(k_new - k) < tol and np.abs(x0_new - x0) < tol:
                return k_new, x0_new

            k, x0 = k_new, x0_new

        print("Warning: Did not converge within max iterations.")
        return k, x0

    def apply_scurve_parameters(self, df: ppl.PathsDataFrame, params: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        index_columns = [col for col in params.primary_keys if col in df.primary_keys and col != YEAR_COLUMN]
        assert len(index_columns) > 0, f'There must be at least one primary key in node {self.id} for SCurveAction.'
        out = df.copy()
        out = out.with_columns([
            pl.lit(None).alias('slope'),
            pl.lit(None).alias('x0'),
            pl.lit(None).alias('ymax')
        ])
        indices = params.select(index_columns).unique()
        for row in indices.rows():
            filter_dict = {col: row[indices.columns.index(col)] for col in index_columns}

            filtered_df = df
            filtered_param = params
            for col, value in filter_dict.items():
                filtered_df = filtered_df.filter(pl.col(col) == value)
                filtered_param = filtered_param.filter(pl.col(col) == value)

            if len(filtered_df) == 0:
                continue
            x2 = filtered_param.filter(pl.col('parameter') == 'max_year').select(VALUE_COLUMN).item()
            filtered_param = filtered_param.ensure_unit(VALUE_COLUMN, df.get_unit(VALUE_COLUMN))
            a = filtered_param.filter(pl.col('parameter') == 'max_impact').select(VALUE_COLUMN).item()
            dfnow = filtered_df.filter(~pl.col(FORECAST_COLUMN))
            x1 = dfnow.select(YEAR_COLUMN).max().item()
            y1 = filtered_df.filter(pl.col(YEAR_COLUMN) == x1).select(VALUE_COLUMN).item()
            if y1 < a:
                y1 = min(max(y1 / a, 0.02), 0.98) * a
                y2 = 0.98 * a
                slope, x0 = self.newton_raphson_estimator(y1, y2, x1, x2, a)
            elif y1 == a:
                slope, x0 = (100.0, 100.0)
            else:
                anew = y1
                y2new = min(max(a / anew, 0.02), 0.98) * anew
                y1new = 0.98 * anew
                slope, x0 = self.newton_raphson_estimator(y1new, y2new, x1, x2, anew)
                a = anew

            # Update the main DataFrame with the estimated values for the matching rows
            mask = pl.lit(True)  # noqa: FBT003
            for col, value in filter_dict.items():
                mask &= pl.col(col) == value
            out = out.with_columns(pl.when(mask).then(slope).otherwise(pl.col('slope')).alias('slope'))
            out = out.with_columns(pl.when(mask).then(x0).otherwise(pl.col('x0')).alias('x0'))
            out = out.with_columns(pl.when(mask).then(a).otherwise(pl.col('ymax')).alias('ymax'))

        return out

    def compute_effect(self) -> ppl.PathsDataFrame:
        df = self.get_input_node().get_output_pl(target_node=self)
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')

        params = self.get_gpc_dataset()
        params = self.drop_unnecessary_levels(params, droplist=['Description'])
        params = self.rename_dimensions(params)
        params = self.convert_names_to_ids(params)
        params = self.implement_unit_col(params)
        params = self.apply_multiplier(params, required=False, units=True)

        df = self.apply_scurve_parameters(df, params)

        df = df.with_columns((
                (pl.col('ymax'))
                / (pl.lit(1.0) + (-pl.col('slope') * (pl.col(YEAR_COLUMN) - pl.col('x0'))).exp())
                )
            .alias('out'))

        df = df.set_unit('out', df.get_unit(VALUE_COLUMN))
        df = df.with_columns((
            pl.when(pl.col(FORECAST_COLUMN))
            .then(pl.col('out'))
            .otherwise(pl.col(VALUE_COLUMN))).alias('out'))
        df = df.subtract_cols(['out', VALUE_COLUMN], VALUE_COLUMN)
        assert self.unit is not None, 'Node {self.id} must have unit defined.'
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        df = df.drop(['out', 'ymax', 'slope', 'x0'])

        if not self.is_enabled():
            df = df.with_columns(pl.lit(self.no_effect_value).alias(VALUE_COLUMN))
        return df


class DatasetDifferenceAction2(DatasetAction):
    explanation = _("""
    Receive goal input from a dataset or node and cause an effect.

    The output will be a time series with the difference to the
    predicted baseline value of the input node.

    The goal input can also be relative (for e.g. percentage
    reductions), in which case the input will be treated as
    a multiplier.
    """)

    allowed_parameters: ClassVar[list[Parameter]] = [
        *DatasetAction.allowed_parameters,
        BoolParameter(local_id='relative_goal'),
    ]

    def filter_categories(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        filter_categories = self.get_parameter_value('filter_categories', required=False)
        keep_dimension = self.get_parameter_value('keep_dimension', required=False)
        if filter_categories:
            assert isinstance(filter_categories, str), "filter_categories must be a string"
            dim = filter_categories.split(':')[0]
            cats = filter_categories.split(':')[1].split(',')
            if dim in df.dim_ids:
                df = df.filter(pl.col(dim).is_in(cats))
                if keep_dimension is not True:
                    df = df.drop(dim)
        return df

    def get_input_object(self, tag: str) -> ppl.PathsDataFrame:
        """
        Get the input object either from a node or a dataset.

        The priority is:
        * node
        * gpc dataset, if sector is given
        * dataset.
        """

        n = self.get_input_node(tag=tag, required=False)
        df: ppl.PathsDataFrame | None
        if n is None:
            if self.get_parameter_value('sector', required=False):
                df = self.get_gpc_dataset(tag=tag)
                df = self.drop_unnecessary_levels(df, droplist=['Description'])
                df = self.rename_dimensions(df)
                df = self.convert_names_to_ids(df)
                df = self.implement_unit_col(df)
                df = self.add_missing_years(df)
            else:
                df = self.get_input_dataset_pl(tag=tag, required=False)
        else:
            df = n.get_output_pl(target_node=self)
        assert df is not None
        df = self.filter_categories(df)
        return df

    def compute_effect(self) -> ppl.PathsDataFrame:
        df = self.get_input_object(tag='baseline')
        if FORECAST_COLUMN not in df.columns:
            df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))

        assert len(df.metric_cols) == 1
        df = df.rename({df.metric_cols[0]: VALUE_COLUMN})

        df = df.filter(~pl.col(FORECAST_COLUMN))  # FIXME FOR DIFF

        max_hist_year = df[YEAR_COLUMN].max()
        df = df.filter(pl.col(YEAR_COLUMN) == max_hist_year)

        gdf = self.get_input_object(tag='goal')

        gdf = gdf.paths.cast_index_to_str()
        df = df.paths.cast_index_to_str()

        if not set(gdf.dim_ids).issubset(set(self.input_dimensions.keys())):
            raise NodeError(self, "Dimension mismatch to input nodes")

        # Filter historical data with only the categories that are
        # specified in the goal dataset.

        # if len(gdf.dim_ids) > 0:
        #     exprs = [pl.col(dim_id).is_in(gdf[dim_id].unique()) for dim_id in gdf.dim_ids]
        #     print(exprs)
        #     df = df.filter(pl.all_horizontal(exprs))

        is_mult = self.get_parameter_value('relative_goal', required=False)
        if is_mult:
            # If the goal series is relative (i.e. a multiplier), transform
            # it into absolute values by multiplying with the last historical values.
            # FIXME Make parameter. For a complement multiplier, the no-effect value is 0 rather than 1.
            inverse_complement = False
            if inverse_complement:
                gdf = gdf.with_columns((pl.lit(1.0) - pl.lit(-1.0) * pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))
            gdf = gdf.rename({VALUE_COLUMN: 'Multiplier'})
            hdf = df.drop(YEAR_COLUMN)
            metric_cols = [m.column_id for m in self.output_metrics.values()]
            hdf = hdf.rename({m: 'Historical%s' % m for m in metric_cols})
            gdf = gdf.paths.join_over_index(hdf, how='outer', index_from='union')
            gdf = gdf.filter(~pl.all_horizontal([pl.col('Historical%s' % col).is_null() for col in metric_cols]))
            for m in self.output_metrics.values():
                col = m.column_id
                gdf = gdf.multiply_cols(['Multiplier', 'Historical%s' % col], col, out_unit=m.unit)
                gdf = gdf.with_columns(pl.col(col).fill_nan(None))
            gdf = gdf.select_metrics(metric_cols)

        bdf = df.paths.to_wide().filter(~pl.col(FORECAST_COLUMN))
        gdf = gdf.paths.to_wide()

        meta = bdf.get_meta()
        gdf = gdf.filter(pl.col(YEAR_COLUMN) > max_hist_year)
        df = ppl.to_ppdf(pl.concat([bdf, gdf], how='diagonal'), meta=meta)
        df = df.paths.make_forecast_rows(end_year=self.get_end_year())
        df = df.with_columns([pl.col(m).interpolate() for m in df.metric_cols])

        # Change the time series to be a difference to the last historical
        # year.
        exprs = [pl.col(m) - pl.first(m) for m in df.metric_cols]
        df = df.select([YEAR_COLUMN, FORECAST_COLUMN, *exprs])

        # df = df.filter(pl.col(FORECAST_COLUMN))
        # end_year = self.get_end_year()
        # df = df.filter(pl.col(YEAR_COLUMN).lt(end_year + 1))
        df = df.paths.to_narrow()

        # Change the time series to be a difference to the baseline
        # gdf = gdf.paths.to_narrow()

        # df = df.paths.join_over_index(gdf)
        # df = df.subtract_cols([VALUE_COLUMN + '_right', VALUE_COLUMN], VALUE_COLUMN)
        # df = df.drop(VALUE_COLUMN + '_right')

        for m in self.output_metrics.values():
            if m.column_id not in df.metric_cols:
                raise NodeError(self, "Metric column '%s' not found in output")
            if not self.is_enabled():
                # Replace non-null columns with 0 when action is not enabled
                df = df.with_columns(
                    pl.when(pl.col(m.column_id).is_null()).then(None).otherwise(0.0).alias(m.column_id)
                )
            df = df.ensure_unit(m.column_id, m.unit)

        return df


class DatasetRelationAction(DatasetAction, GenericNode):
    explanation = _(
        """
        ActionRelationshipNode enforces a logical relationship with another action node.

        This node monitors an upstream action node (A) and automatically sets its own
        enabled state (B) according to the relationship specified in the edge tags.
        """
    )
    allowed_parameters = [
        *DatasetAction.allowed_parameters,
        *GenericNode.allowed_parameters,
    ]
    DEFAULT_OPERATIONS = 'apply_relationship'

    # Map of all possible relationship types and their behavior
    RELATIONSHIP_BEHAVIOR = {
        # Format: 'name': (if_a_true, if_a_false, description)
        # if_a_true/if_a_false can be: True, False, or None (meaning "no change")

        # Core logical relationships
        'copy': (True, False, "B equals A (B copies A's state)"),
        'not': (False, True, "B equals NOT A (B is the opposite of A)"),
        'always_true': (True, True, "B is always enabled regardless of A"),
        'always_false': (False, False, "B is always disabled regardless of A"),

        # Named logical relationships (common in digital logic)
        'and': (True, False, "If A is enabled, B is enabled; otherwise B is disabled"),
        'or': (None, True, "If A is disabled, B is enabled; otherwise no change"),
        'nand': (False, None, "If A is enabled, B is disabled; otherwise no change"),
        'nor': (False, False, "B is disabled regardless of A's state"),
        'xor': (False, True, "B is enabled if and only if A is disabled"),
        'xnor': (True, False, "B is enabled if and only if A is enabled"),

        # Implication relationships
        'implication': (True, None, "If A is enabled, B must be enabled; otherwise no change"),
        'reverse_implication': (None, False, "If A is disabled, B must be disabled; otherwise no change"),

        # Inhibition relationships
        'inhibit': (False, None, "If A is enabled, B is disabled; otherwise no change"),
        'enable': (True, None, "If A is enabled, B is enabled; otherwise no change"),
        'block': (None, False, "If A is disabled, B is disabled; otherwise no change"),
        'allow': (None, True, "If A is disabled, B is enabled; otherwise no change"),
    }

    def _find_relationship_from_tags(self, node: ActionNode) -> str:
        """
        Find the relationship type from tags on the input edge or node.

        Returns the first valid relationship tag found.
        """
        # First check edge tags
        for edge in self.edges:
            if edge.input_node == node and edge.output_node == self:
                for tag in edge.tags:
                    if tag in self.RELATIONSHIP_BEHAVIOR:
                        return tag

        # Then check node tags
        for tag in node.tags:
            if tag in self.RELATIONSHIP_BEHAVIOR:
                return tag

        return None

    def _operation_apply_relationship(self, df: ppl.PathsDataFrame | None, baskets: dict, **kwargs) -> tuple:
        """
        Set this node's enabled state.

        It is based on the upstream action node and the
        relationship defined in tags.
        """

        # Validate there's exactly one input node in 'other' basket
        if len(baskets['other']) != 1:
            raise NodeError(self, "Relationship node requires exactly one upstream action node.")

        # Get the upstream action node
        action_a: ActionNode = baskets['other'][0]

        # Get relationship type from tags
        relationship = self._find_relationship_from_tags(action_a)

        if not relationship:
            valid_relationships = ", ".join(self.RELATIONSHIP_BEHAVIOR.keys())
            raise NodeError(self,
                f"No relationship tag found for action {action_a}. Valid options are: {valid_relationships}")

        # Get the behavior rules for this relationship
        if_a_true, if_a_false, _ = self.RELATIONSHIP_BEHAVIOR[relationship]

        # Get the state of input action node
        a_enabled: bool = action_a.is_enabled()

        # Apply the relationship rule
        if a_enabled and if_a_true is not None:
            # A is enabled - apply if_a_true rule
            self.enabled_param.set(if_a_true)

        elif not a_enabled and if_a_false is not None:
            # A is disabled - apply if_a_false rule
            self.enabled_param.set(if_a_false)

        # If the rule returns None, we don't change the enabled state

        baskets['other'] = []  # Mark nodes as processed
        return df, baskets

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register the relationship enforcement operation
        self.OPERATIONS['apply_relationship'] = self._operation_apply_relationship

    def _compute(self) -> ppl.PathsDataFrame:
        df = self.get_gpc_dataset()

        if self.get_global_parameter_value('measure_data_baseline_year_only', required=False):
            filt = (pl.col(YEAR_COLUMN) == self.context.instance.reference_year) | (
                pl.col(YEAR_COLUMN) > self.context.instance.maximum_historical_year
            )
            if FORECAST_COLUMN in df.columns:
                filt |= pl.col(FORECAST_COLUMN)
            df = df.filter(filt)

        df = self.drop_unnecessary_levels(df, droplist=['Description'])
        df = self.rename_dimensions(df)
        df = self.convert_names_to_ids(df)
        df = self.select_variant(df)
        df = self.implement_unit_col(df)
        df = self.add_missing_years(df)
        df = self.crop_to_model_range(df)

        df = extend_last_historical_value_pl(df, end_year=self.get_end_year())
        # First extend, then truncate because there may be measure observations beyond
        # the last historical year in the dataset.
        if self.get_parameter_value('inventory_only', required=False):
            df = df.filter(~pl.col(FORECAST_COLUMN))

        df = self.apply_multiplier(df, required=False, units=True)
        # df = self.add_and_multiply_input_nodes(df)
        df = self.maybe_drop_nulls(df)
        df = df.ensure_unit(VALUE_COLUMN, self.unit)  # type: ignore
        return df

    def compute_effect(self) -> ppl.PathsDataFrame:
        print('in the beginning:', self.is_enabled())
        df = self._compute()
        baskets = self._get_input_baskets(self.input_nodes)
        df, _ = self._operation_apply_relationship(df, baskets)

        if not self.is_enabled():
            df = df.with_columns(pl.lit(self.no_effect_value).alias(VALUE_COLUMN))

        assert self.unit is not None
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df

    def compute(self) -> ppl.PathsDataFrame:
        return self.compute_effect()
