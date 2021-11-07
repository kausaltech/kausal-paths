from logging import exception, log
from types import FunctionType

import dvc_pandas
from graphene.types.scalars import Float, Int
from pint.registry import ContextCacheOverlay
from pint_pandas.pint_array import PintArray
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
from .ovariable import Ovariable, OvariableFrame

# ############################33
# Health-related constants and classes

# Ovariable with no input nodes, just data


class DataOvariable(Ovariable, AdditiveNode):
    pass

# Exposure is the intensity of contact with the environment by the target population.


class Exposure(Ovariable):

    quantity = 'ingestion'
    scaled = False

    def compute(self):
        consumption = self.prepare_ovariable('ingestion')
        concentration = self.prepare_ovariable('mass_concentration')
        bw = self.prepare_ovariable('body_weight')
        er_function = self.prepare_ovariable('exposure-response',
            query='observation=="param1"', drop=['observation', 'Response'])

        exposure = concentration * consumption

        # scale_exposure leads to dtype object, which causes problems in ensure_pint_unit

        # Somehow it is due to er_function, maybe pint.m but not clear. Also, this error may relate to the cause:
        # pint_array.py:648: UnitStrippedWarning: The unit of the quantity is stripped when downcasting to ndarray.
        # exposure = self.scale_exposure(exposure, er_function, bw)
        # This was because er_function was defined by rows and not by pd.Series. This was fixed.

        return self.clean_computing(exposure)

    # scale_exposure() scales the exposure by logarithmic function or body weight.
    # The information about how to scale comes from exposure-response function.
    # Thus, er-function and body weight must be provided.
    # FIXME Is this a sensible thing to do, as then the units and dimensions may differ within node?

    def scale_exposure(self, exposure, er_function, bw):
        out = OvariableFrame(er_function.copy())
#        out[VALUE_COLUMN] = out[VALUE_COLUMN].pint.m
        out = out * unit_registry.Quantity('0 * cap * d / mg') + 1
        tst = out.merge(exposure)[0:4]
        #out = out * exposure
        # FIXME Does not work atm

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

        return OvariableFrame(out).clean()


class FixedMultiplierHealthImpactNode(FixedMultiplierNode):  # Needed for pre-ovariable nodes
    allowed_parameters = [
        NumberParameter(local_id='health_factor'),
    ] + FixedMultiplierNode.allowed_parameters

    quantity = 'disease_burden'
    unit = 'DALY/a'

    def compute(self):
        if len(self.input_nodes) != 1:
            raise NodeError(self, 'FixedMultiplier needs exactly one input node')

        node = self.input_nodes[0]

        df = node.get_output()
        multiplier = self.get_parameter_value('health_factor')
        multiplier = multiplier * unit_registry('DALY/kt').units

        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            df[col] *= multiplier

        replace_output = self.get_parameter_value('replace_output_using_input_dataset', required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset(df)

        return df


class ExposureResponseFunction(Ovariable):
    quantity = 'exposure-response'

    def compute(self):
        df = pd.DataFrame({
            'scaling': pd.Series(['None']*4),
            'Response': pd.Series(['CVD']*2 + ['Cancer']*2),
            'er_function': pd.Series(['RR']*2 + ['UR']*2),
            'observation': pd.Series(['param1', 'param2']*2),
            FORECAST_COLUMN: pd.Series([False]*4),
            VALUE_COLUMN: pd.Series([200., 0., 2000.1, 0.2], dtype='pint[mg/person/d]')
        }).set_index(['er_function', 'Response', 'observation', 'scaling'])

        return self.clean_computing(df)

# Relative risk (RR) is the risk of an exposed individual compared with a counterfactual
# unexposed individual using the modelled exposures


class RelativeRisk(Ovariable):
    quantity = 'ratio'
    unit = 'dimensionless'

    def compute(self):

        param1 = self.prepare_ovariable(
            quantity='exposure-response',
            query="observation == 'param1'", drop=['observation'])

        """
        A possible atlernative for parameterization
        er_function = er_function.reset_index()
        param_names = er_function.observation.unique()

        parameters = {elem: pd.DataFrame for elem in param_names}

        for key in parameters.keys():
            of = er_function[:][er_function.observation == key]
            index_list = list(set(of.columns) - {VALUE_COLUMN, FORECAST_COLUMN})
            of = of.set_index(index_list).droplevel('observation')
            parameters[key] = OvariableFrame(of)
        """

        param2 = self.prepare_ovariable(
            quantity='exposure-response',
            query="observation == 'param2'", drop=['observation'])

        exposure = self.prepare_ovariable('ingestion')

        df = pd.DataFrame()
#        relative_functions = set(self.get_output().reset_index().er_function)   # Contains also unused functions
        relative_functions = {'RR'}  # sorted(relative_functions & {'RR', 'Relative Hill'})

        for func in relative_functions:

            if func == 'RR':
                beta = OvariableFrame(param1.query("er_function == 'RR'"))
                beta = beta ** -1  # FIXME This is stupid parameterization

                threshold = param2

                dose2 = exposure - threshold

                #  Smallest allowed value is 0 FIXME: Not if scaling: Log10
                dose2[VALUE_COLUMN] = np.clip(dose2[VALUE_COLUMN], 0, None)

                out1 = beta * dose2

                out1 = out1.exp()
                df = df.append(out1.reset_index())

            elif func == 'Relative Hill':
                Imax = param1
                Imax = Imax.query("er_function == 'Relative Hill'")
                Imax[VALUE_COLUMN] = Imax[VALUE_COLUMN].pint.m

                ed50 = param2

                out2 = (exposure * Imax) / (exposure + ed50) + 1

                df = df.append(out2.reset_index())

        keep = set(df.columns) - {'er_function', 'scaling'}
        df = df[list(keep)].set_index(list(keep - {VALUE_COLUMN, FORECAST_COLUMN}))

        return self.clean_computing(df)

# Population attributable fraction PAF


class PopulationAttributableFraction(Ovariable):
    quantity = 'fraction'
    unit = 'dimensionless'

    def compute(self):
        param1 = self.prepare_ovariable(
            'exposure-response',
            query="observation == 'param1'", drop='observation')
        param2 = self.prepare_ovariable(
            'exposure-response',
            query="observation == 'param2'", drop='observation')
        exposure = self.prepare_ovariable('ingestion')
        frexposed = self.prepare_ovariable('fraction')
        incidence = self.prepare_ovariable('incidence')
        rr = self.prepare_ovariable('ratio')
        p_illness = self.prepare_ovariable('probability')
        bw = self.prepare_ovariable('body_weight')

#        param1 = param1 * 1.00001 # Integers cause trouble with ** -1

        # er_function_list = list(sorted(set(exposure.reset_index()['er_function'])))
        er_function_list = ['RR', 'UR']

        if 'RR' in er_function_list:
            er_function_list = list(sorted(set(er_function_list) - {'Relative Hill'}))
            # You don't want to do twice

        out = pd.DataFrame()

        for func in er_function_list:

            if func == 'UR':
                k = OvariableFrame(param1.query("er_function == 'UR'"))
                k = k ** -1

                threshold = param2

                dose2 = (exposure - threshold)
                # FIXME clip removes the pint unit. Figure out something else.
                # dose2 = np.clip(dose2, 0, None)  # Smallest allowed value is 0
                out1 = (k * dose2 * frexposed / incidence)
                out = out.append(out1.reset_index())

            elif func == 'Step':
                upper = copy.copy(param1)
                upper = upper.copy().query("er_function == 'Step'")

                lower = param2
                out2 = (exposure >= lower) * (exposure <= upper) * -1 + 1  # FIXME
                out2 = out2 * frexposed / incidence
                out = out.append(out2.reset_index())

            elif func == 'RR' or func == 'Relative Hill':
                r = frexposed * (rr - 1)

                out3 = (r / (r + 1))  # AF=r/(r+1) if r >= 0; AF=r if r<0. Therefore, if the result
                # is smaller than 0, we should use r instead. It can be converted from the result:
                # r/(r+1)=a <=> r=a/(1-a)
                out3[VALUE_COLUMN] = np.where(
                    out3[VALUE_COLUMN] < 0,
                    out3[VALUE_COLUMN] / (1 - out3[VALUE_COLUMN]),
                    out3[VALUE_COLUMN])

                out = out.append(out3.reset_index())

            elif func == 'beta poisson approximation':
                out4 = ((exposure / param2 + 1) ** (param1 * -1) * -1 + 1) * frexposed
                out4 = (out4 / incidence * p_illness)
                out = out.append(out4.reset_index())

            elif func == 'exact beta poisson':
                out5 = ((param1 / (param1 + param2) * exposure * -1).exp() * -1 + 1) * frexposed
                out5 = out5 / incidence * p_illness
                out = out.append(out5.reset_index())

            elif func == 'exponential':
                k = param1
                out6 = ((k * exposure * -1).exp() * -1 + 1) * frexposed
                out6 = out6 / incidence * p_illness
                out = out.append(out6.reset_index())

        keep = set(out.columns) - {'scaling', 'matrix', 'exposure', 'exposure_unit', 'er_function', 0}
        out = out[list(keep)].set_index(list(keep - {VALUE_COLUMN, FORECAST_COLUMN}))

        return self.clean_computing(out)

# BoD is the current (observed) burden of disease (measured in disability-adjusted life years or DALYs).


class DiseaseBurden(Ovariable):
    quantity = 'disease_burden'

    def compute(self):
        incidence = self.prepare_ovariable('incidence')
        population = self.prepare_ovariable('population')
        case_burden = self.prepare_ovariable('disease_burden')

        out = population * incidence * case_burden

        return self.clean_computing(out)

# bod_attr is the burden of disease that can be attributed to the exposure of interest.


class AttributableDiseaseBurden(Ovariable):
    quantity = 'disease_burden'

    def compute(self):
        bod = self.prepare_ovariable('disease_burden')
        paf = self.prepare_ovariable('fraction')

        out = bod * paf

        return self.clean_computing(out)
