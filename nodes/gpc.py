import pandas as pd
import polars as pl
import numpy as np
from params import StringParameter, BoolParameter
from nodes.calc import extend_last_historical_value_pl
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN, FORECAST_COLUMN, KNOWN_QUANTITIES
from nodes.dimensions import Dimension
from nodes.node import Node
from nodes.simple import AdditiveNode
from common import polars as ppl


class DatasetNode(AdditiveNode):
    allowed_parameters = AdditiveNode.allowed_parameters + [
        StringParameter('gpc_sector', description = 'GPC Sector', is_customizable = False),
        StringParameter('rename_dimensions', description='Rename incompatible dimensions', is_customizable=False)
    ]
        # FIXME Could the parameter be called just sector?

    qlookup = {'price': 'currency',
               'energy_consumption': 'energy',
               'waste_disposal': 'mass',
               'amount': 'number',
               'exposureresponse': 'exposure_response',
               'case_cost': 'unit_price'
               }

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

    def get_gpc_dataset(self) -> pd.DataFrame:
        sector = self.get_parameter_value('gpc_sector', required=True)

        # Perform initial filtering of GPC dataset.
        df = self.get_input_dataset()
        df = df[df[VALUE_COLUMN].notnull()]
        quans = []
        for quan in df.index.get_level_values('Quantity'):
            quan = self.makeid(quan)
            if not quan in KNOWN_QUANTITIES:
                quan = self.qlookup[quan]
            quans.append(quan)
        df = df[(df.index.get_level_values('Sector') == sector) &
                [q == self.quantity for q in quans]]

        return df
    
    def implement_unit_col(self, df: pd.DataFrame) -> pd.DataFrame:
        unit = df['Unit'].unique()[0]
        df[VALUE_COLUMN] = df[VALUE_COLUMN].astype('pint[' + unit + ']')
        df = df.drop(columns = ['Unit'])
        return df

    def convert_names_to_ids(self, df: pd.DataFrame) -> pd.DataFrame:
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
        if 'Description' in df.index.names:
            droplist.append('Description')
        for col in df.index.names:
            vals = df.index.get_level_values(col).unique().to_list()
            if vals == ['.']:
                droplist.append(col)
        df.index = df.index.droplevel(droplist)
        return df

    def add_missing_years(self, df: pd.DataFrame) -> ppl.PathsDataFrame:
        # Add forecast column if needed.
        if FORECAST_COLUMN not in df.columns:
            df[FORECAST_COLUMN] = False

        # Add missing years and interpolate missing values.
        df = ppl.from_pandas(df)
        df = df.paths.to_wide()

        yeardf = pd.DataFrame({YEAR_COLUMN: range(df[YEAR_COLUMN].min(), df[YEAR_COLUMN].max() + 1)})
        yeardf = yeardf.set_index([YEAR_COLUMN])
        yeardf = ppl.from_pandas(yeardf)

        df = df.paths.join_over_index(yeardf, how = 'outer')
        for col in list(set(df.columns) - set([YEAR_COLUMN, FORECAST_COLUMN])):
            df = df.with_columns(pl.col(col).interpolate())

        df = df.with_columns(pl.col(FORECAST_COLUMN).fill_null(strategy = 'forward'))

        df = df.paths.to_narrow()
        return df

    def rename_dimensions(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        renams = self.get_parameter_value('rename_dimensions', required=False)
        if renams:
            for renam in renams.split(','):
                dimfrom, dimto = renam.split(':')
                df = df.rename({dimfrom: dimto})
        return df

    def add_and_multiply_input_nodes(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        # Add and multiply input nodes as tagged.
        na_nodes = self.get_input_nodes(tag = 'non_additive')
        input_nodes = [node for node in self.input_nodes if node not in na_nodes]

        df = self.add_nodes_pl(df, input_nodes)

        if len(na_nodes) > 0:  # FIXME Instead, develop a generic single dimensionless multiplier
            assert len(na_nodes) == 1 # Only one multiplier allowed.
            mult = na_nodes[0].get_output_pl(target_node = self)
            df = df.paths.join_over_index(mult, how = 'outer', index_from='union')

            df = df.multiply_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN)
        return df

    # -----------------------------------------------------------------------------------
    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_gpc_dataset()
        df = self.drop_unnecessary_levels(df, droplist=['Sector', 'Quantity'])
        df = self.convert_names_to_ids(df)
        df = self.implement_unit_col(df)
        df = self.add_missing_years(df)
        df = self.rename_dimensions(df)
        df = extend_last_historical_value_pl(df, end_year=self.get_end_year())
        df = self.add_and_multiply_input_nodes(df)

        return df


class CorrectionNode(DatasetNode):
    allowed_parameters = DatasetNode.allowed_parameters + [
        BoolParameter('do_correction', description = 'Should the values be corrected?')
    ]
    def compute(self):
        df = super().compute()
        df = ppl.from_pandas(df)  # FIXME Shouldn't this be done in DatasetNode?
        do_correction = self.get_parameter_value('do_correction', required=True)

        if not do_correction:
            df = df.with_columns(pl.col(VALUE_COLUMN) * pl.lit(0) + pl.lit(1.0))

        # FIXME Should this code be in the DatasetNode instead?
        inventory_only = self.get_parameter_value('inventory_only', required=False)
        if inventory_only is not None:
            if inventory_only:
                df = df.filter(pl.col(FORECAST_COLUMN) == False)
            else:
                df = extend_last_historical_value_pl(df, self.get_end_year())

        return df