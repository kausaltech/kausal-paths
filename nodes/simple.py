from params.param import BoolParameter, NumberParameter
from typing import List
import pandas as pd
import pint
from .context import unit_registry
import numpy as np

from common.i18n import TranslatedString
from .constants import FORECAST_COLUMN, VALUE_COLUMN
from .node import Context, Node
from .exceptions import NodeError


EMISSION_UNIT = 'kg'


class SimpleNode(Node):
    allowed_parameters = [
        BoolParameter(
            local_id='fill_gaps_using_input_dataset',
            label=TranslatedString("Fill in gaps in computation using input dataset"),
            is_customizable=False
        ),
        BoolParameter(
            local_id='replace_output_using_input_dataset',
            label=TranslatedString("Replace output using input dataset"),
            is_customizable=False
        )
    ]

    def replace_output_using_input_dataset(self, context: Context, df: pd.DataFrame) -> pd.DataFrame:
        # If we have also data from an input dataset, we only fill in the gaps from the
        # calculated data.
        df = df.dropna()

        data_df = self.get_input_dataset(context)
        if data_df is None:
            return df

        latest_year = data_df.index.max()
        if latest_year < df.index.max():
            merge_df = df[df.index > latest_year]
            data_df = data_df.reindex(data_df.index.append(merge_df.index))
            if not set(merge_df.columns).issubset(set(data_df.columns)):
                missing_cols = [col for col in merge_df.columns if col not in data_df.coluns]
                raise Exception('Columns missing from input dataset: %s' % ', '.join(missing_cols))
            for col in data_df.columns:
                if col == FORECAST_COLUMN:
                    continue
                data_df[col] = self.ensure_output_unit(data_df[col])
            data_df.loc[data_df.index > latest_year] = merge_df

        data_df[FORECAST_COLUMN] = data_df[FORECAST_COLUMN].astype(bool)
        return data_df

    def fill_gaps_using_input_dataset(self, context: Context, df: pd.DataFrame) -> pd.DataFrame:
        data_df = self.get_input_dataset(context)
        if data_df is None:
            return df

        index_diff = data_df.index.difference(df.index)
        if not len(index_diff):
            return df
        df = df.reindex(index_diff.append(df.index))
        df.loc[df.index.isin(index_diff)] = data_df
        return df


class AdditiveNode(SimpleNode):
    """Simple addition of inputs"""

    def add_nodes(self, context: Context, df: pd.DataFrame, nodes: List[Node]):
        if self.debug:
            print('%s: input dataset:' % self.id)
            if df is not None:
                print(self.print_pint_df(df))
            else:
                print('\tNo input dataset')
        for node in nodes:
            node_df = node.get_output(context, self)
            if node_df is None:
                continue

            if self.debug:
                print('%s: adding output from node %s' % (self.id, node.id))
                self.print_pint_df(node_df)

            if df is None:
                df = node_df
                continue

            if VALUE_COLUMN not in df.columns:
                raise NodeError(self, "Value column missing in output of %s" % node.id)
            val1 = self.ensure_output_unit(df[VALUE_COLUMN])
            if hasattr(val1, 'pint'):
                val1 = val1.pint.m
            val2 = self.ensure_output_unit(node_df[VALUE_COLUMN], input_node=node)
            if hasattr(val2, 'pint'):
                val2 = val2.pint.m
            val1 = val1.add(val2, fill_value=0)
            df[VALUE_COLUMN] = self.ensure_output_unit(val1)
            df[FORECAST_COLUMN] = df[FORECAST_COLUMN] | node_df[FORECAST_COLUMN]

        return df

    def compute(self, context: Context):
        df = self.get_input_dataset(context)

        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise NodeError(self, "Input is not a DataFrame")
            if VALUE_COLUMN not in df.columns:
                raise NodeError(self, "Input dataset doesn't have Value column")

            df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])

            if df.index.max() < self.get_target_year(context):
                last_year = df.index.max()
                last_val = df.loc[last_year]
                new_index = df.index.append(pd.RangeIndex(last_year + 1, self.get_target_year(context) + 1))
                df = df.reindex(new_index)
                df.iloc[-1] = last_val
                dt = df.dtypes[VALUE_COLUMN]
                df[VALUE_COLUMN] = df[VALUE_COLUMN].pint.m
                df = df.fillna(method='bfill')
                df[VALUE_COLUMN] = df[VALUE_COLUMN].astype(dt)
                df.loc[df.index > last_year, FORECAST_COLUMN] = True

        if self.get_parameter_value('fill_gaps_using_input_dataset', required=False):
            df = self.add_nodes(context, None, self.input_nodes)
            df = self.fill_gaps_using_input_dataset(context, df)
        else:
            df = self.add_nodes(context, df, self.input_nodes)

        df[FORECAST_COLUMN] = df[FORECAST_COLUMN].astype(bool)
        return df


class SectorEmissions(AdditiveNode):
    quantity = 'emissions'
    """Simple addition of subsector emissions"""
    pass


class MultiplicativeNode(AdditiveNode):
    """Multiply nodes together with potentially adding other input nodes.

    Multiplication and addition is determined based on the input node units.
    """

    def compute(self, context: Context):
        additive_nodes = []
        multiply_nodes = []
        for node in self.input_nodes:
            if self.is_compatible_unit(context, node.unit, self.unit):
                additive_nodes.append(node)
            else:
                multiply_nodes.append(node)

        if len(multiply_nodes) != 2:
            raise NodeError(self, "Must receive exactly two multiplicative inputs")

        n1, n2 = multiply_nodes
        output_unit = n1.unit * n2.unit
        if not self.is_compatible_unit(context, output_unit, self.unit):
            raise NodeError(self, "Multiplying inputs must in a unit compatible with '%s'" % self.unit)

        df1 = n1.get_output(context)
        df2 = n2.get_output(context)
        df = df1.copy()

        if self.debug:
            print('%s: Multiply input from node 1 (%s):' % (self.id, n1.id))
            self.print_pint_df(df1)
            print('%s: Multiply input from node 2 (%s):' % (self.id, n2.id))
            self.print_pint_df(df2)

        df[VALUE_COLUMN] *= df2[VALUE_COLUMN]
        df[FORECAST_COLUMN] = df1[FORECAST_COLUMN] | df2[FORECAST_COLUMN]

        df[VALUE_COLUMN] = df[VALUE_COLUMN].pint.to(self.unit)

        df = self.add_nodes(context, df, additive_nodes)
        fill_gaps = self.get_parameter_value('fill_gaps_using_input_dataset', required=False)
        if fill_gaps:
            df = self.fill_gaps_using_input_dataset(context, df)
        replace_output = self.get_parameter_value('replace_output_using_input_dataset', required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset(context, df)
        if self.debug:
            print('%s: Output:' % self.id)
            self.print_pint_df(df)

        df[FORECAST_COLUMN] = df[FORECAST_COLUMN].astype(bool)

        return df


class EmissionFactorActivity(MultiplicativeNode):
    """Multiply an activity by an emission factor."""
    quantity = 'emissions'
    unit = EMISSION_UNIT


class PerCapitaActivity(MultiplicativeNode):
    pass


class Activity(AdditiveNode):
    """Add activity amounts together."""
    pass


class FixedMultiplierNode(SimpleNode):
    allowed_parameters = [
        NumberParameter(local_id='multiplier'),
    ] + SimpleNode.allowed_parameters

    def compute(self, context: Context):
        if len(self.input_nodes) != 1:
            raise NodeError(self, 'FixedMultiplier needs exactly one input node')

        node = self.input_nodes[0]

        df = node.get_output(context)
        multiplier = self.get_parameter_value('multiplier')
        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            df[col] *= multiplier

        replace_output = self.get_parameter_value('replace_output_using_input_dataset', required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset(context, df)

        return df

class ExposureNode(AdditiveNode):
    quantity = 'exposure'
    """Simple addition of exposure"""
    pass

class PopulationAttributableFractionNode(AdditiveNode):
    """Calculate population attributable fraction (PAF) from relative risk (RR)
    and fraction of population exposed.
    """

    def compute(self):

        if len(self.input_nodes) != 2:
            raise NodeError(self, "Must receive exactly two inputs in this order: relative risk and fraction exposed")

        n1, n2 = self.input_nodes
        output_unit = n1.unit * n2.unit
        if not self.is_compatible_unit(output_unit, self.unit):
            raise NodeError(self, "Inputs must in a unit compatible with '%s'" % self.unit)

        df1 = n1.get_output()
        df2 = n2.get_output()
        df = df1.copy()

        if self.debug:
            print('%s: Multiply input from node 1 (%s):' % (self.id, n1.id))
            self.print_pint_df(df1)
            print('%s: Multiply input from node 2 (%s):' % (self.id, n2.id))
            self.print_pint_df(df2)
        
        r = df2[VALUE_COLUMN] * (df1[VALUE_COLUMN] - 1 * unit_registry('meter')) # FIXIT should be unitless, not meters
        df[VALUE_COLUMN] = np.where(r>0, r/(r + 1 * unit_registry('meter ** 2')),r)

        df[FORECAST_COLUMN] = df1[FORECAST_COLUMN] | df2[FORECAST_COLUMN]

#        df[VALUE_COLUMN] = df[VALUE_COLUMN].pint.to(self.unit)

        fill_gaps = self.get_param_value('fill_gaps_using_input_dataset', local=True, required=False)
        if fill_gaps:
            df = self.fill_gaps_using_input_dataset(df)
        replace_output = self.get_param_value('replace_output_using_input_dataset', local=True, required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset(df)
        if self.debug:
            print('%s: Output:' % self.id)
            self.print_pint_df(df)

        df[FORECAST_COLUMN] = df[FORECAST_COLUMN].astype(bool)

        return df


