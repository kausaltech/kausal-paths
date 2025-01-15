from __future__ import annotations

from typing import cast

from django.utils.translation import gettext_lazy as _

import polars as pl

from common import polars as ppl
from nodes.calc import extend_last_historical_value_pl
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, UNCERTAINTY_COLUMN
from nodes.exceptions import NodeError
from nodes.simple import AdditiveNode
from params import BoolParameter, StringParameter


class DatasetNode(AdditiveNode):
    explanation = _(
        """This is a DatasetNode. It takes in a specifically formatted dataset and converts the relevant part into a node output.""",  # noqa: E501
    )

    global_parameters = ['measure_data_override', 'measure_data_baseline_year_only']

    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
        StringParameter(
            'gpc_sector',
            description='GPC Sector',
            is_customizable=False,
        ),  # FIXME To be removed, replaced by 'sector' below.
        StringParameter('sector', description='Sector', is_customizable=False),
        StringParameter('rename_dimensions', description='Rename incompatible dimensions', is_customizable=False),
    ]

    quantitylookup = {
        'price': 'currency',
        'energy_consumption': 'energy',
        'waste_disposal': 'mass',
        'amount': 'number',
        'exposureresponse': 'exposure_response',
        'case_cost': 'unit_price',
        'unit_cost': 'unit_price',
        'share': 'fraction',
        'freight_quantity': 'freight_mileage',
    }

    # Supported languages: Czech, Danish, English, Finnish, German, Latvian, Polish, Swedish
    characterlookup = str.maketrans(
        {
            '.': '',
            ',': '',
            ':': '',
            '-': '',
            '(': '',
            ')': '',
            ' ': '_',
            '/': '_',
            '&': 'and',
            'ä': 'a',
            'å': 'a',
            'ą': 'a',
            'á': 'a',
            'ā': 'a',
            'ć': 'c',
            'č': 'c',
            'ď': 'd',
            'ę': 'e',
            'é': 'e',
            'ě': 'e',
            'ē': 'e',
            'ģ': 'g',
            'í': 'i',
            'ī': 'i',
            'ķ': 'k',
            'ł': 'l',
            'ļ': 'l',
            'ń': 'n',
            'ň': 'n',
            'ņ': 'n',
            'ö': 'o',
            'ø': 'o',
            'ó': 'o',
            'ř': 'r',
            'ś': 's',
            'š': 's',
            'ť': 't',
            'ü': 'u',
            'ú': 'u',
            'ů': 'u',
            'ū': 'u',
            'ý': 'y',
            'ź': 'z',
            'ż': 'z',
            'ž': 'z',
            'æ': 'ae',
            'ß': 'ss',
        },
    )

    # -----------------------------------------------------------------------------------
    def get_gpc_dataset(self, tag: str | None = None) -> ppl.PathsDataFrame:
        sector = self.get_parameter_value('gpc_sector', required=False)
        if not sector:
            sector = self.get_parameter_value('sector', required=False)
        if not sector:
            raise NodeError(self, 'You must give either gpc_sector or sector parameter.')
        # Perform initial filtering of GPC dataset.
        df = self.get_input_dataset_pl(tag=tag)
        df = df.filter((pl.col(VALUE_COLUMN).is_not_null()) & (pl.col('Sector') == sector))

        qlookup = {}
        for quantity in df['Quantity'].unique():
            qlookup[quantity] = quantity.lower().translate(self.characterlookup)

        df = df.with_columns(df['Quantity'].replace(qlookup).replace(self.quantitylookup))
        df = df.filter(pl.col('Quantity') == self.quantity)

        dropcols = ['Sector', 'Quantity']
        if 'UUID' in df.columns:
            dropcols += ['UUID']
        df = df.drop(dropcols)

        return df

    # -----------------------------------------------------------------------------------
    def implement_unit_col(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = df.set_unit(VALUE_COLUMN, df['Unit'].unique()[0])
        df = df.drop('Unit')
        return df

    # -----------------------------------------------------------------------------------
    def rename_dimensions(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        renames = self.get_parameter_value_str('rename_dimensions', required=False)
        if renames:
            dlookup = {}
            for rename in renames.split(','):
                dimfrom, dimto = rename.split(':')
                dlookup[dimfrom] = dimto

            df = df.rename(dlookup)
        return df

    # -----------------------------------------------------------------------------------
    def convert_names_to_ids(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        exset = {YEAR_COLUMN, VALUE_COLUMN, FORECAST_COLUMN, UNCERTAINTY_COLUMN, 'Unit'}

        # Convert index level names from labels to IDs.
        collookup = {}
        for col in list(set(df.columns) - exset):
            collookup[col] = col.lower().translate(self.characterlookup)
        df = df.rename(collookup)

        # Convert levels within each index level from labels to IDs.
        if 'scope' in df.columns:
            catlookup = {}
            for cat in df['scope'].unique():
                catlookup[cat] = cat.lower().replace(' ', '').replace('.', '')
            df = df.with_columns(df['scope'].replace(catlookup))
            exset.add('scope')

        for col in list(set(df.columns) - exset):
            catlookup = {}
            for cat in df[col].unique():
                catlookup[cat] = cat.lower().translate(self.characterlookup)
            df = df.with_columns(df[col].replace(catlookup))

            if col in self.context.dimensions:
                for cat in self.context.dimensions[col].categories:
                    if cat.aliases:
                        df = df.with_columns(self.context.dimensions[col].series_to_ids_pl(df[col]))
                        break
        return df

    # -----------------------------------------------------------------------------------
    def drop_unnecessary_levels(self, df: ppl.PathsDataFrame, droplist: list) -> ppl.PathsDataFrame:
        # Drop filter levels and empty dimension levels.
        drops = [d for d in droplist if d in df.columns]

        for col in list(set(df.columns) - set(drops)):
            vals = df[col].unique().to_list()
            if vals in [['.'], [None]]:
                drops.append(col)

        df = df.drop(drops)
        return df

    # -----------------------------------------------------------------------------------
    def add_missing_years(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        # Add forecast column if needed.
        if FORECAST_COLUMN not in df.columns:
            df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))

        # Add missing years and interpolate missing values.
        df = df.paths.to_wide()
        yearrange = range(df[YEAR_COLUMN].min(), (df[YEAR_COLUMN].max() + 1))  # type: ignore
        nullcount = df.null_count().sum_horizontal()[0]

        if (len(df[YEAR_COLUMN].unique()) < len(yearrange)) | (nullcount > 0):
            yeardf = ppl.PathsDataFrame({YEAR_COLUMN: yearrange})
            yeardf._units = {}
            yeardf._primary_keys = [YEAR_COLUMN]

            df = df.paths.join_over_index(yeardf, how='outer')
            for col in list(set(df.columns) - {YEAR_COLUMN, FORECAST_COLUMN}):
                df = df.with_columns(pl.col(col).interpolate())

            df = df.with_columns(pl.col(FORECAST_COLUMN).fill_null(strategy='backward'))

        df = df.paths.to_narrow()
        return df

    # -----------------------------------------------------------------------------------
    def add_and_multiply_input_nodes(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        # Add and multiply input nodes as tagged.
        na_nodes = self.get_input_nodes(tag='non_additive')
        input_nodes = [node for node in self.input_nodes if node not in na_nodes]

        measure_data_override = self.get_global_parameter_value('measure_data_override', required=False)
        start_from_year: int | None = None
        if measure_data_override:
            historical_years = df.filter(~pl.col(FORECAST_COLUMN))
            if len(historical_years) > 0:
                start_from_year = cast(int, historical_years[YEAR_COLUMN].max()) + 1
        df = self.add_nodes_pl(df, input_nodes, start_from_year=start_from_year)
        if len(na_nodes) > 0:  # FIXME Instead, develop a generic single dimensionless multiplier
            assert len(na_nodes) == 1  # Only one multiplier allowed.
            mult = na_nodes[0].get_output_pl(target_node=self)
            if start_from_year is not None:
                # force multiplier to 1 below max year
                mult = mult.ensure_unit(VALUE_COLUMN, 'dimensionless')
                mult = mult.with_columns(
                    pl.when(pl.col(YEAR_COLUMN) <= start_from_year)
                    .then(pl.lit(1.0))
                    .otherwise(pl.col(VALUE_COLUMN))
                    .alias(VALUE_COLUMN),
                )
            df = df.paths.join_over_index(mult, how='outer', index_from='union')

            df = df.multiply_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN).drop(
                VALUE_COLUMN + '_right',
            )  # FIXME Does not treat missing categories well
        return df

    # -----------------------------------------------------------------------------------
    def compute(self) -> ppl.PathsDataFrame:
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
        df = self.implement_unit_col(df)
        df = self.add_missing_years(df)

        if not self.get_parameter_value('inventory_only', required=False):
            df = extend_last_historical_value_pl(df, end_year=self.get_end_year())

        df = self.apply_multiplier(df, required=False, units=True)
        df = self.add_and_multiply_input_nodes(df)
        df = df.ensure_unit(VALUE_COLUMN, self.unit)  # type: ignore
        # out = df.filter(pl.col(YEAR_COLUMN) <= 2020)
        # out = out.select([UNCERTAINTY_COLUMN, VALUE_COLUMN])
        # # out = out.pivot(columns=YEAR_COLUMN, values=VALUE_COLUMN)
        # out.write_csv('/Users/jouni/devel/kausal-paths/notebooks/iterations.csv', separator='\t')
        return df


class DetailedDatasetNode(DatasetNode):
    allowed_parameters = [
        *DatasetNode.allowed_parameters,
        StringParameter('action', description='Detailed action module', is_customizable=False),
    ]

    def compute(self) -> ppl.PathsDataFrame:
        # Perform initial filtering of GPC dataset.
        df = self.get_input_dataset_pl()

        sector = self.get_parameter_value('sector')
        nodename = str(self.name).replace('%s ' % sector, '')

        df = df.filter(
            (pl.col(VALUE_COLUMN).is_not_null())
            & (pl.col('Sector') == sector)
            & (pl.col('Action') == self.get_parameter_value('action'))
            & (pl.col('Node Name') == nodename),
        )

        df = self.drop_unnecessary_levels(df, droplist=['Sector', 'Action', 'Node Name'])
        df = self.convert_names_to_ids(df)
        df = self.implement_unit_col(df)
        df = self.add_missing_years(df)
        df = self.rename_dimensions(df)  # FIXME Should we add add_and_multiply_input_nodes() and ensure_unit()?
        df = extend_last_historical_value_pl(df, end_year=self.get_end_year())

        return df


class CorrectionNode(DatasetNode):  # FIXME Separate correction into another node?
    allowed_parameters = [
        *DatasetNode.allowed_parameters,
        BoolParameter('do_correction', description='Should the values be corrected?'),
    ]

    def compute(self):
        df = super().compute()

        do_correction = self.get_parameter_value('do_correction', required=True)

        if not do_correction:
            df = df.with_columns(pl.col(VALUE_COLUMN) * pl.lit(0) + pl.lit(1.0))

        # FIXME Should this code be in the DatasetNode instead?
        inventory_only = self.get_parameter_value('inventory_only', required=False)
        if inventory_only is not None:
            if inventory_only:
                df = df.filter(~pl.col(FORECAST_COLUMN))
            else:
                df = extend_last_historical_value_pl(df, self.get_end_year())

        return df


class CorrectionNode2(AdditiveNode):
    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
        BoolParameter('do_correction', description='Should the values be corrected?'),
    ]

    def compute(self):
        df = AdditiveNode.compute(self)

        do_correction = self.get_parameter_value('do_correction', required=True)

        if not do_correction:
            df = df.with_columns(pl.col(VALUE_COLUMN) * pl.lit(0) + pl.lit(1.0))

        # FIXME Should this code be in the DatasetNode instead?
        inventory_only = self.get_parameter_value('inventory_only', required=False)
        if inventory_only is not None:
            if inventory_only:
                df = df.filter(~pl.col(FORECAST_COLUMN))
            else:
                df = extend_last_historical_value_pl(df, self.get_end_year())

        return df
