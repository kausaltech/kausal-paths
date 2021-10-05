from logging import exception, log
from types import FunctionType

import dvc_pandas
from pint.registry import ContextCacheOverlay
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
from .ovariable import Ovariable, Ovariable2

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
        df = exposure.content

        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])

        return df
    
    # scale_exposure() scales the exposure by logarithmic function or body weight.
    # The information about how to scale comes from exposure-response function.
    # Thus, er-function and body weight must be provided.
    # All ovariables must have contents calculated.

    def scale_exposure(self, erf, bw):
        if 'er_function' in self.content.index.names:
            return self

        exposure = self

        out = copy.deepcopy(erf)
        out.content[VALUE_COLUMN] = out.content[VALUE_COLUMN].pint.m
        out = out * 0

        out = out + exposure

        out = out.merge(bw).reset_index()

        out[VALUE_COLUMN] = np.where(
            out['scaling'] == 'BW',
            out[VALUE_x] / out[VALUE_y],
            out[VALUE_x])

# FIXME log10 not defined for all inputs
#        out[VALUE_COLUMN] = np.where(
#            out['scaling'] == 'Log10',
#            np.log10(out[VALUE_COLUMN]),
#            out[VALUE_COLUMN])

        keep = set(out.columns)- {0,VALUE_x,VALUE_y, FORECAST_x, FORECAST_y}
        out = out[list(keep)].set_index(list(keep - {VALUE_COLUMN, FORECAST_COLUMN}))

        self.content_orig = self.content
        self.content = out

        return self
        
class FixedMultiplierHealthImpactNode(FixedMultiplierNode): # Needed for pre-ovariable nodes
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


class ExposureResponseFunction(Ovariable):
    quantity = 'exposure-response'

    def compute(self, context: Context):
        df = pd.DataFrame([
            ['None','cardiovascular disease','RR','param1',False, 200],
            ['None','cardiovascular disease','RR','param2',False, 0],
            ['None','cerebrovascular disease','Relative Hill','param1',False, 2],
            ['None','cerebrovascular disease','Relative Hill','param2',False, 0.2],
            ['None','dioxin','Step','param1',False, 20],
            ['None','dioxin','Step','param2',False, 0],
            ['None','cancer','UR','param1',False, 2000],
            ['None','cancer','UR','param2', False, 0.2]],
            columns=['scaling','Response','er_function','observation', FORECAST_COLUMN,VALUE_COLUMN]
        ).set_index(['er_function','observation','scaling','Response'])

        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])

        return df

# Relative risk (RR) is the risk of an exposed individual compared with a counterfactual
# unexposed individual using the modelled exposures. 

class RelativeRisk(Ovariable):
    quantity = 'ratio'
    unit = 'dimensionless'

    def compute(self, context: Context):
                
        param1 = self.prepare_ovariable(context, quantity='exposure-response',
            query="observation == 'param1'", drop=['observation'])

        param2 = self.prepare_ovariable(context, quantity='exposure-response',
            query="observation == 'param2'", drop = ['observation']) 
        
        bw = self.prepare_ovariable(context, 'body_weight')
        
        exposure = self.prepare_ovariable(context, 'exposure')
        exposure = exposure.scale_exposure(param1, bw)
        
        df = pd.DataFrame()
        relative_functions = exposure.content.reset_index().er_function
        relative_functions = list(set(relative_functions) & {'RR','Relative Hill'})

        for func in relative_functions:

            if func == 'RR':
                beta = param1 ** -1 #FIXME This is stupid parameterization
                beta.content = beta.content.query("er_function == 'RR'")

                threshold = param2

                dose2 = exposure - threshold
                
                dose2.content = np.clip(dose2.content, 0, None) # Smallest allowed value is 0 FIXME: Not if scaling: Log10

                out1 = beta * dose2
#                print(beta.content)
#                print(out1.content)
                out1.content[VALUE_COLUMN] = out1.content[VALUE_COLUMN].pint.m_as('')

                out1 = out1.exp()
                df = df.append(out1.content.reset_index())

            if func == 'Relative Hill':
                Imax = param1
                Imax.content = Imax.content.query("er_function == 'Relative Hill'")
                Imax.content[VALUE_COLUMN] = Imax.content[VALUE_COLUMN].pint.m

                ed50 = param2

                out2 = (exposure * Imax) / (exposure + ed50) + 1

                df = df.append(out2.content.reset_index())

        keep = set(df.columns) - {'er_function','scaling'}
        df = df[list(keep)].set_index(list(keep - {VALUE_COLUMN, FORECAST_COLUMN}))
        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])

        return df
    
## Population attributable fraction PAF

class PopulationAttributableFraction(Ovariable):
    quantity = 'fraction'
    unit = 'dimensionless'
    
    def compute(self, context):
        param1 = self.prepare_ovariable(context, 'exposure-response',
            query="observation == 'param1'", drop='observation')
        param2 = self.prepare_ovariable(context, 'exposure-response',
            query="observation == 'param2'", drop='observation')
        exposure = self.prepare_ovariable(context, 'exposure')
        frexposed = self.prepare_ovariable(context, 'fraction')
        incidence = self.prepare_ovariable(context, 'incidence')
        rr = self.prepare_ovariable(context, 'ratio')
        p_illness = self.prepare_ovariable(context, 'probability')
        bw = self.prepare_ovariable(context, 'body_weight')

        exposure = exposure.scale_exposure(param1, bw)
        er_function_list = list(set(exposure.content.reset_index()['er_function']))
        if 'RR' in er_function_list:
            er_function_list = list(set(er_function_list) - {'Relative Hill'}) # You don't want to do twice

        out = pd.DataFrame()

        for func in er_function_list:

            if func == 'UR':
                k = copy.deepcopy(param1)
                k.content = k.content.query("er_function == 'UR'")
                k = k ** -1

                threshold = param2

                dose2 = (exposure - threshold)
                dose2.content = np.clip(dose2.content, 0, None) # Smallest allowed value is 0
                out3 = (k * dose2 * frexposed / incidence)
                out = out.append(out3.content.reset_index())

            if func == 'Step':
                upper = copy.deepcopy(param1)
                upper.content = upper.content.query("er_function == 'Step'")

                lower = param2
                out2 = (exposure >= lower) * (exposure <= upper) * -1 + 1
                out2 = out2 * frexposed / incidence
                out = out.append(out2.content.reset_index())

            if func == 'RR' or func == 'Relative Hill':
                r = frexposed * (rr - 1)

                out3 = (r/(r + 1)) # AF=r/(r+1) if r >= 0; AF=r if r<0. Therefore, if the result  
                # is smaller than 0, we should use r instead. It can be converted from the result:
                # r/(r+1)=a <=> r=a/(1-a)
                out3.content[VALUE_COLUMN] = np.where(out3.content[VALUE_COLUMN]<0,
                    out3.content[VALUE_COLUMN] / (1 - out3.content[VALUE_COLUMN]),
                    out3.content[VALUE_COLUMN])

                out = out.append(out3.content.reset_index())

            if func == 'beta poisson approximation':
                out4 = ((exposure/param2 + 1)**(param1 * -1) * -1 + 1) * frexposed
                out4 = (out4 / incidence * p_illness)#.dropna() # dropna is needed before an index with NaN is used for merging
                out = out.append(out4.content.reset_index())

            if func == 'exact beta poisson':
                out5 = ((param1/(param1 + param2) * exposure * -1).exp() * -1 + 1) * frexposed
                out5 = out5 / incidence * p_illness
                out = out.append(out5.content.reset_index())

            if func == 'exponential':
                k = param1
                out6 = ((k * exposure * -1).exp() * -1 + 1) * frexposed
                out6 = out6 / incidence * p_illness
                out = out.append(out6.content.reset_index())

        keep = set(out.columns)- {'scaling','matrix','exposure','exposure_unit','er_function',0}
        out = out[list(keep)].set_index(list(keep - {VALUE_COLUMN}))
#        print(out) # Why does print() break the instance?
#        self.print_pint_df(out)

        return out

# BoD is the current (observed) burden of disease (measured in disability-adjusted life years or DALYs).

class DiseaseBurden(Ovariable):
    quantity = 'disease_burden'
    
    def compute(self, context):
        incidence = self.prepare_ovariable(context, 'incidence')
        population = self.prepare_ovariable(context, 'population')
        case_burden = self.prepare_ovariable(context, 'disease_burden')
        
        out = incidence * population * case_burden

        return out.content

# bod_attr is the burden of disease that can be attributed to the exposure of interest.

class AttributableDiseaseBurden(Ovariable):
    quantity = 'disease_burden'
    
    def compute(self, context):
        bod = self.prepare_ovariable(context, 'disease_burden')
        paf = self.prepare_ovariable(context, 'fraction')

        out = bod * paf
    
        return out.content
    