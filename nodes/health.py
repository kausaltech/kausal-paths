from logging import log
from types import FunctionType
from params.param import BoolParameter, NumberParameter
from typing import Dict, List
import pandas as pd
import pint
from .context import unit_registry
import numpy as np
import math

from common.i18n import TranslatedString
from .constants import FORECAST_COLUMN, VALUE_COLUMN
from .node import Context, Node
from .exceptions import NodeError

from .simple import AdditiveNode, SimpleNode

#############################33
# Health-related constants and classes

class RelativeRiskNode(AdditiveNode):
    """Applies a function with one input node and parameters.
    """
    allowed_parameters = [
        NumberParameter(local_id='exposure_response_param1'),
        NumberParameter(local_id='exposure_response_param2'),
    ] + SimpleNode.allowed_parameters

    def compute(self):

        if len(self.input_nodes) != 1:
            raise NodeError(self, "Must receive exactly one input")

        input_node = self.input_nodes[0]
        beta = unit_registry(self.get_parameter_value('exposure_response_param1'))
        threshold = unit_registry(self.get_parameter_value('exposure_response_param2'))

#        output_unit = input_node.unit #* beta.unit

#        if not self.is_compatible_unit(context, output_unit, self.unit):
#            raise NodeError(self, "Multiplying inputs must in a unit compatible with '%s'" % self.unit)

        df = input_node.get_output()

        if self.debug:
            print('%s: Parameter input from node 1 (%s):' % (self.id, n1.id))
            self.print_pint_df(df1)

        for col in df.columns: # Why use for loop if we only want to mutate VALUE_COLUMN?
            if col == FORECAST_COLUMN:
                continue
            df[col] = ((beta * (df[col] - threshold))) #.astype(float)) # Should be exp() but fails.
            df[col] = np.exp(df[col].astype(float)) #).apply(lambda x: unit_registry.Quantity(x))

#        df[FORECAST_COLUMN] = df1[FORECAST_COLUMN] | df2[FORECAST_COLUMN]

#        df[VALUE_COLUMN] = df[VALUE_COLUMN].pint.to(self.unit)

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
        
        r = df2[VALUE_COLUMN] * (df1[VALUE_COLUMN] - 1)

        if min(r)>=0: # NOTE! PROBLEMS OCCUR WITH HORMESIS
            df[VALUE_COLUMN] = r/(r + 1)
        else: 
            df[VALUE_COLUMN] = r

        df[FORECAST_COLUMN] = df1[FORECAST_COLUMN] | df2[FORECAST_COLUMN]

        df[VALUE_COLUMN] = df[VALUE_COLUMN].pint.to(self.unit)

        fill_gaps = False # self.get_param_value('fill_gaps_using_input_dataset', local=True, required=False)
        if fill_gaps:
            df = self.fill_gaps_using_input_dataset(df)
        replace_output = False # self.get_param_value('replace_output_using_input_dataset', local=True, required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset(df)
        if self.debug:
            print('%s: Output:' % self.id)
            self.print_pint_df(df)

        df[FORECAST_COLUMN] = df[FORECAST_COLUMN].astype(bool)

        return df

