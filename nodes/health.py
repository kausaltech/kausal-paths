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
        if self.scaled == True:
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
        self.scaled = True

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


# Relative risk (RR) is the risk of an exposed individual compared with a counterfactual
# unexposed individual using the modelled exposures. 

class ExposureResponseFunction(Ovariable):
    quantity = 'exposure-response'

    def compute(self, context: Context):
        df = pd.DataFrame([
            ['None','cardiovascular disease','RR','param1',False, 20],
            ['None','cardiovascular disease','RR','param2', False, 0]],
            columns=['scaling','Response','er_function','observation', FORECAST_COLUMN,VALUE_COLUMN]
        ).set_index(['er_function','observation','scaling','Response'])

        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])

        return df

class RelativeRisk(Ovariable):
    quantity = 'ratio'

    def compute(self, context: Context):
#        for node in self.input_nodes:
#            if node.quantity == 'exposure-response':
#                erf = node
#            if node.quantity == 'body_weight':
#                bw = node
#            if node.quantity == 'exposure':
#                exposure = node
                
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
#            param1 = copy.deepcopy(erf)
#            param1.content = param1.content.loc[(func,'param1')] # The er_function must be the first and observation the second level
#            param2 = copy.deepcopy(erf)
#            param2.content = param2.content.loc[(func,'param2')]

            if func == 'RR':
                beta = param1 ** -1 #FIXME This is stupid parameterization

                threshold = param2

                dose2 = exposure - threshold
                
                dose2.content = np.clip(dose2.content, 0, None) # Smallest allowed value is 0 FIXME: Not if scaling: Log10

                out1 = beta * dose2
                out1.content[VALUE_COLUMN] = out1.content[VALUE_COLUMN].pint.m_as('')

                out1 = out1.exp()
                df = df.append(out1.content.reset_index())

            if func == 'Relative Hill':
                Imax = param1
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
    
    def compute(self, context):
        for node in self.input_nodes:
            if node.quantity == 'exposure-response':
                erf = node # Is there a risk of assigning to a global variable?
            if node.quantity == 'exposure':
                exposure = node
            if node.quantity == 'fraction':
                frexposed = node
            if node.quantity == 'incidence':
                incidence = node
            if node.quantity == 'ratio':
                rr = node
            if node.quantity == 'probability':
                p_illness = node
            if node.quantity == 'body_weight':
                bw = node

        erf.content = erf.get_output(context) 

        dose = exposure.scale_exposure(erf, bw)
        
        er_function_list = list(set(exposure.content.reset_index().er_function))

        out = pd.DataFrame()

        for func in er_function_list:
            param1 = copy.deepcopy(erf) # FIXME Do we actually need deepcopy here?
            param1.content = param1.content.loc[(func,'ERF')]
            param2 = copy.deepcopy(erf)
            param2.content = param2.content.loc[(func,'Threshold')]

            if func == 'UR':
                k = param1
                threshold = param2
                dose2 = (dose - threshold)#.dropna()
                dose2.content = np.clip(dose2.content, 0, None) # Smallest allowed value is 0
                out1 = (k * dose2 * frexposed / incidence)#.dropna()
                out = out.append(out1.content.reset_index())

            if func == 'Step':
                upper = param1
                lower = param2
                out2 = (dose >= lower) * (dose <= upper) * -1 + 1
                out2 = out2 * frexposed / incidence
                out = out.append(out2.content.reset_index())

            if func == 'RR' or func == 'Relative Hill':
                r = frexposed * (rr - 1)
                out3 = (r > 0) * (r/(r + 1)) + (r <= 0) * r
                out = out.append(out3.content.reset_index())

            if func == 'beta poisson approximation':
                out4 = ((dose/param2 + 1)**(param1 * -1) * -1 + 1) * frexposed
                out4 = (out4 / incidence * p_illness)#.dropna() # dropna is needed before an index with NaN is used for merging
                out = out.append(out4.content.reset_index())

            if func == 'exact beta poisson':
                out5 = ((param1/(param1 + param2) * dose * -1).exp() * -1 + 1) * frexposed
                out5 = out5 / incidence * p_illness
                out = out.append(out5.content.reset_index())

            if func == 'exponential':
                k = param1
                out6 = ((k * dose * -1).exp() * -1 + 1) * frexposed
                out6 = out6 / incidence * p_illness
                out = out.append(out6.content.reset_index())

        #keep = set(out.columns[out.notna().any()]) # remove indices that are empty
        #fill = set(out.columns[out.isna().any()]) # fill indices that have some empty locations
        #out = fillna(out, list(fill.intersection(keep) - {VALUE_COLUMN}))

        keep = set(out.columns)- {'scaling','matrix','exposure','exposure_unit','er_function',0}
        out = out[list(keep)].set_index(list(keep - {VALUE_COLUMN}))

        return out

# BoD is the current (observed) burden of disease (measured in disability-adjusted life years or DALYs).

class DiseaseBurden(Ovariable):
    quantity = 'disease_burden'
    
    def compute(self, context):
        for node in self.input_nodes:
            if node.quantity == 'incidence':
                incidence = node
            if node.quantity == 'population':
                population = node
            if node.quantity == 'disease_burden':
                case_burden = node
        print(incidence.content)
        out = incidence * population # * case_burden

        return out

# bod_attr is the burden of disease that can be attributed to the exposure of interest.

class AttributableDiseaseBurden(Ovariable):
    quantity = 'disease_burden'
    
    def compute(self, context):
        for node in self.input_nodes:
            if node.quantity == 'disease_burden':
                bod = node
            if node.quantity == 'fraction':
                paf = node

        out = bod * paf
    
        return out
    
#bod_attr = Bod_attr(input_nodes = [bod, paf], name = 'bod_attr').compute()
#bod_attr.content
