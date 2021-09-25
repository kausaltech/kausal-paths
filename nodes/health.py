from logging import log
from types import FunctionType
from params.param import BoolParameter, NumberParameter
from typing import Dict, List
import pandas as pd
import pint
from .context import unit_registry
import numpy as np
import math
import copy

from common.i18n import TranslatedString
from .constants import FORECAST_COLUMN, VALUE_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y
from .node import Context, Node
from .exceptions import NodeError

from .simple import AdditiveNode, FixedMultiplierNode, SimpleNode
from .ovariable import Ovariable

#############################33
# Health-related constants and classes

# Ovariable with no input nodes, just data

class DataOvariable(Ovariable, AdditiveNode):
    pass

# Exposure is the intensity of contact with the environment by the target population.

class Exposure(Ovariable):
    
    quantity = 'exposure'
    scaled = False

    def compute(self, context: Context):
        for node in self.input_nodes:
            if node.quantity == 'consumption':
                consumption = node
            if node.quantity == 'concentration':
                concentration = node

        output_unit = consumption.unit * concentration.unit
        if not self.is_compatible_unit(context, output_unit, self.unit):
            raise NodeError(self, "Multiplying inputs must in a unit compatible with '%s'" % self.unit)
        consumption.content = consumption.get_output(context)
        concentration.content = concentration.get_output(context)

        exposure = consumption * concentration
        exposure[VALUE_COLUMN] = exposure[VALUE_COLUMN].pint.to(self.unit)

        return exposure
    
    # scale_exposure() scales the exposure by logarithmic function or body weight.
    # The information about how to scale comes from exposure-response function.
    # Thus, er-function and body weight must be provided.

    def scale_exposure(self, erf, bw):
        if self.scaled == True:
            return self

        exposure = self
        
        out = exposure + erf * 0
        out.content = out.content.copy().query("observation == 'param1'").droplevel('observation')

        out = out.merge(bw).content.reset_index()
        out[VALUE_COLUMN] = np.where(
            out['scaling'] == 'BW',
            out[VALUE_x] / out[VALUE_y],
            out[VALUE_x])

        out[VALUE_COLUMN] = np.where(
            out['scaling'] == 'Log10',
            np.log10(out[VALUE_COLUMN]),
            out[VALUE_COLUMN])

        keep = set(out.columns)- {0,VALUE_x,VALUE_y}
        out = out[list(keep)].set_index(list(keep - {VALUE_COLUMN}))

        self.content_orig = self.content
        self.content = out
        self.scaled = True

        return self
        
class FixedMultiplierHealthImpactNode(FixedMultiplierNode):
    allowed_parameters = [
        NumberParameter(local_id='health_factor'),
    ] + FixedMultiplierNode.allowed_parameters

    quantity = 'disease_burden'
    unit = 'DALY/a'

    def compute(self, context: Context):
        if len(self.input_nodes) != 1:
            raise NodeError(self, 'FixedMultiplier needs exactly one input node')

        node = self.input_nodes[0]

        df = node.get_output(context)
        multiplier = self.get_parameter_value('health_factor')
        multiplier = multiplier * unit_registry('DALY/kt').units

        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            df[col] *= multiplier

        replace_output = self.get_parameter_value('replace_output_using_input_dataset', required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset(context, df)

        return df


# Relative risk (RR) is the risk of an exposed individual compared with a counterfactual
# unexposed individual using the modelled exposures. 

class Rr(Ovariable):
    quantity = 'RR'

    def compute(self):
        for node in self.input_nodes:
            if node.quantity == 'ERF':
                erf = node
            if node.quantity == 'body_weight':
                bw = node
            if node.quantity == 'exposure':
                exposure = node
                
        dose = exposure.scale_exposure(erf, bw)

        out = pd.DataFrame()

        relative_functions = ['RR','Relative Hill']

        for func in relative_functions:
            param1 = copy.deepcopy(erf)
            param1.content = param1.content.loc[(func,'param1')] # The er_function must be the first and observation the second level
            param2 = copy.deepcopy(erf)
            param2.content = param2.content.loc[(func,'param2')]

            if func == 'RR':
                rr = param1
                threshold = param2

                dose2 = (dose - threshold)#.dropna()
                
                dose2.content = np.clip(dose2.content, 0, None) # Smallest allowed value is 0

                out1 = (rr.log() * dose2).exp() #.dropna()
                out = out.append(out1.content.reset_index())

            if func == 'Relative Hill':
                Imax = param1
                ed50 = param2

                out2 = (dose * Imax) / (dose + ed50) + 1

                out = out.append(out2.content.reset_index())

        keep = set(out.columns) - {0}
        out = out[list(keep)].set_index(list(keep - {VALUE_COLUMN}))
        
        self.content = out

        return self
    