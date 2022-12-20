from common.perf import PerfCounter
from nodes.calc import nafill_all_forecast_years
from params.param import Parameter, BoolParameter, NumberParameter, ParameterWithUnit, StringParameter
from typing import List, ClassVar, Tuple
import pandas as pd
import pint

from common.i18n import TranslatedString
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from .node import Node
from .exceptions import NodeError


EMISSION_UNIT = 'kg'


class SimpleNode(Node):
    allowed_parameters: ClassVar[List[Parameter]] = [
        BoolParameter(
            local_id='fill_gaps_using_input_dataset',
            label=TranslatedString(en="Fill in gaps in computation using input dataset"),
            is_customizable=False
        ),
        BoolParameter(
            local_id='replace_output_using_input_dataset',
            label=TranslatedString(en="Replace output using input dataset"),
            is_customizable=False
        )
    ]

    def replace_output_using_input_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # If we have also data from an input dataset, we only fill in the gaps from the
        # calculated data.
        df = df.dropna()

        data_df = self.get_input_dataset(required=False)
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

    def fill_gaps_using_input_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        data_df = self.get_input_dataset(required=False)
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
    allowed_parameters: ClassVar[list[Parameter]] = [
        StringParameter(local_id='metric', is_customizable=False),
    ] + SimpleNode.allowed_parameters

    def add_nodes(self, df: pd.DataFrame | None, nodes: List[Node], metric=None) -> pd.DataFrame:
        if self.debug:
            print('%s: input dataset:' % self.id)
            if df is not None:
                print(self.print_pint_df(df))
            else:
                print('\tNo input dataset')

        node_outputs: List[Tuple[Node, pd.DataFrame]] = []
        for node in nodes:
            node_df = node.get_output(self, metric=metric)
            node_outputs.append((node, node_df))

        if df is None:
            df = node_outputs.pop(0)[1]

        cols = df.columns.values
        if VALUE_COLUMN not in cols:
            raise NodeError(self, "Value column missing in data")
        if FORECAST_COLUMN not in cols:
            raise NodeError(self, "Forecast column missing in data")

        val_s = self.ensure_output_unit(df[VALUE_COLUMN])
        pt = val_s.dtype
        if hasattr(val_s, 'pint'):
            val_s = val_s.pint.m
        forecast_s = df[FORECAST_COLUMN]

        for node, node_df in node_outputs:
            if self.debug:
                print('%s: adding output from node %s' % (self.id, node.id))
                self.print_pint_df(node_df)

            if VALUE_COLUMN not in node_df.columns.values:
                raise NodeError(self, "Value column missing in output of %s" % node.id)

            val2 = self.ensure_output_unit(node_df[VALUE_COLUMN], input_node=node)
            if hasattr(val2, 'pint'):
                val2 = val2.pint.m
            val_s = val_s.add(val2, fill_value=0)
            forecast_s |= node_df[FORECAST_COLUMN]

        val_s = val_s.astype(pt)
        df = pd.DataFrame({VALUE_COLUMN: val_s, FORECAST_COLUMN: forecast_s})
        return df

    def compute(self):
        df = self.get_input_dataset(required=False)
        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise NodeError(self, "Input is not a DataFrame")
            if VALUE_COLUMN not in df.columns:
                raise NodeError(self, "Input dataset doesn't have Value column")

            df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])

            if df.index.max() < self.get_end_year():
                last_year = df.index.max()
                last_val = df.loc[last_year]
                new_index = df.index.append(pd.RangeIndex(last_year + 1, self.get_target_year() + 1))
                assert df.index.name == YEAR_COLUMN
                new_index.name = YEAR_COLUMN
                df = df.reindex(new_index)
                df.iloc[-1] = last_val
                dt = df.dtypes[VALUE_COLUMN]
                df[VALUE_COLUMN] = df[VALUE_COLUMN].pint.m
                df = df.fillna(method='bfill')
                df[VALUE_COLUMN] = df[VALUE_COLUMN].astype(dt)
                df.loc[df.index > last_year, FORECAST_COLUMN] = True

        metric = self.get_parameter_value('metric', required=False)
        if self.get_parameter_value('fill_gaps_using_input_dataset', required=False):
            df = self.add_nodes(None, self.input_nodes, metric)
            df = self.fill_gaps_using_input_dataset(df)
        else:
            df = self.add_nodes(df, self.input_nodes, metric)

        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])
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

    operation_label = 'multiplication'

    def perform_operation(self, n1: Node, n2: Node, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        output_unit = n1.unit * n2.unit
        if not self.is_compatible_unit(output_unit, self.unit):
            raise NodeError(
                self,
                "Multiplying inputs must in a unit compatible with '%s' (%s [%s] * %s [%s])" % (self.unit, n1.id, n1.unit, n2.id, n2.unit))

        df1[VALUE_COLUMN] *= df2[VALUE_COLUMN]
        return df1

    def compute(self):
        additive_nodes = []
        operation_nodes = []
        for node in self.input_nodes:
            if self.is_compatible_unit(node.unit, self.unit):
                additive_nodes.append(node)
            else:
                operation_nodes.append(node)

        if len(operation_nodes) != 2:
            raise NodeError(self, "Must receive exactly two inputs to operate %s on" % self.operation_label)

        n1, n2 = operation_nodes
        df1 = n1.get_output()
        df2 = n2.get_output()

        if self.debug:
            print('%s: %s input from node 1 (%s):' % (self.operation_label, self.id, n1.id))
            self.print_pint_df(df1)
            print('%s: %s input from node 2 (%s):' % (self.operation_label, self.id, n2.id))
            self.print_pint_df(df2)

        df = self.perform_operation(n1, n2, df1.copy(), df2)

        df[FORECAST_COLUMN] = df1[FORECAST_COLUMN] | df2[FORECAST_COLUMN]
        df[VALUE_COLUMN] = df[VALUE_COLUMN].pint.to(self.unit)

        df = self.add_nodes(df, additive_nodes)
        fill_gaps = self.get_parameter_value('fill_gaps_using_input_dataset', required=False)
        if fill_gaps:
            df = self.fill_gaps_using_input_dataset(df)
        replace_output = self.get_parameter_value('replace_output_using_input_dataset', required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset(df)
        if self.debug:
            print('%s: Output:' % self.id)
            self.print_pint_df(df)

        df[FORECAST_COLUMN] = df[FORECAST_COLUMN].astype(bool)

        return df


class DivisiveNode(MultiplicativeNode):
    """Divide two nodes together with potentially adding other input nodes.

    Division and addition is determined based on the input node units.
    """

    operation_label = 'division'

    def perform_operation(self, n1: Node, n2: Node, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        output_unit = n1.unit / n2.unit
        if not self.is_compatible_unit(output_unit, self.unit):
            raise NodeError(
                self,
                "Division inputs must in a unit compatible with '%s' (%s [%s] * %s [%s])" % (self.unit, n1.id, n1.unit, n2.id, n2.unit))

        df1[VALUE_COLUMN] /= df2[VALUE_COLUMN]
        return df1


class EmissionFactorActivity(MultiplicativeNode):
    """Multiply an activity by an emission factor."""
    quantity = 'emissions'
    default_unit = '%s/a' % EMISSION_UNIT


class PerCapitaActivity(MultiplicativeNode):
    pass


class Activity(AdditiveNode):
    """Add activity amounts together."""
    pass


class FixedMultiplierNode(SimpleNode):
    allowed_parameters = [
        NumberParameter(local_id='multiplier'),
        StringParameter(local_id='global_multiplier'),
    ] + SimpleNode.allowed_parameters

    def compute(self):
        if len(self.input_nodes) != 1:
            raise NodeError(self, 'FixedMultiplier needs exactly one input node')

        node = self.input_nodes[0]

        df = node.get_output()
        multiplier_param = self.get_parameter('multiplier', required=False)
        if multiplier_param is None:
            global_multiplier = self.get_parameter_value('global_multiplier', required=True)
            assert isinstance(global_multiplier, str)
            # This is a bit of a hack. We need to ensure the dynamically defined (by "global_multiplier") multiplier
            # parameter is in our list of global_parameters.
            if global_multiplier not in self.global_parameters:
                self.global_parameters = list(self.global_parameters) + [global_multiplier]
            multiplier_param = self.context.get_parameter(global_multiplier)

        multiplier = multiplier_param.value
        if isinstance(multiplier_param, ParameterWithUnit):
            multiplier = pint.Quantity(multiplier, multiplier_param.unit)
        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            df[col] *= multiplier

        replace_output = self.get_parameter_value('replace_output_using_input_dataset', required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset(df)

        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])

        return df


class YearlyPercentageChangeNode(SimpleNode):
    allowed_parameters = [
        NumberParameter(local_id='yearly_change', unit_str='%'),
    ] + SimpleNode.allowed_parameters

    def compute(self):
        df = self.get_input_dataset()
        if len(self.input_nodes) != 0:
            raise NodeError(self, "YearlyPercentageChange can't have input nodes")

        df = nafill_all_forecast_years(df, self.get_end_year())
        mult = self.get_parameter_value('yearly_change') / 100 + 1
        df['Multiplier'] = 1
        df.loc[df[FORECAST_COLUMN], 'Multiplier'] = mult
        df['Multiplier'] = df['Multiplier'].cumprod()
        for col in df.columns:
            if col in (FORECAST_COLUMN, 'Multiplier'):
                continue
            dt = df.dtypes[col]
            df[col] = df[col].pint.m.fillna(method='pad').astype(dt)
            df.loc[df[FORECAST_COLUMN], col] *= df['Multiplier']

        replace_output = self.get_parameter_value('replace_output_using_input_dataset', required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset(df)

        df = df.drop(columns=['Multiplier'])

        return df
