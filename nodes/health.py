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


# Possible units for exposure:
# mg/person       acute dose per person
# mg/person/d     mass rate per person
# mg/kg           acute dose per body weight
# mg/kg/d         mass rate per body weight = fraction of body mass rate
# ug/m**3         concentration in breathing air
# ppm             mass concentration in breathing air


class Exposure(Ovariable):
    # Exposure is the intensity of contact with the environment by the target population.

    quantity = 'ingestion'
    scaled = False  # This is probably not needed any more

    def compute(self):
        consumption = self.get_input('ingestion')  # This can be bolus (g/person) or rate (g/person/d)
        concentration = self.get_input('mass_concentration')

        exposure = concentration * consumption

        return self.clean_computing(exposure)


class ExposureResponseFunction(Ovariable):
    quantity = 'exposure-response'

    def compute(self):
        df = pd.DataFrame({
            'scaling': pd.Series(['None'] * 4),
            'Response': pd.Series(['CVD'] * 2 + ['Cancer'] * 2),
            'er_function': pd.Series(['RR'] * 2 + ['UR'] * 2),
            'observation': pd.Series(['param1', 'param2'] * 2),
            FORECAST_COLUMN: pd.Series([False] * 4),
            VALUE_COLUMN: pd.Series([200., 0., 2000.1, 0.2], dtype='pint[mg/person/d]')
        }).set_index(['er_function', 'Response', 'observation', 'scaling'])

        return self.clean_computing(df)


class PopulationAttributableFraction(Ovariable):
    # Population attributable fraction PAF

    quantity = 'fraction'
    unit = 'dimensionless'

    # scale_exposure() scales the exposure by logarithmic function or body weight.
    # The information about how to scale comes from exposure-response function.
    # Thus, er-function and body weight must be provided.
    # FIXME Is this a sensible thing to do, as then the units and dimensions may differ within node?
    # scale_exposure leads to dtype object, which causes problems in ensure_pint_unit
    # Somehow it is due to er_function, maybe pint.m but not clear.

    def scale_exposure(self, exposure, er_function, bw):
        out = OvariableFrame(er_function.copy())
#        out[VALUE_COLUMN] = out[VALUE_COLUMN].pint.m
        out = out * unit_registry.Quantity('0 * cap * d / mg') + 1
        tst = out.merge(exposure)[0:4]
        # out = out * exposure
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

    def compute(self):
        exposure = self.get_input('ingestion')
        exposure[VALUE_COLUMN] = exposure[VALUE_COLUMN].pint.to('mg/kg/d', 'exposure_generic')
        frexposed = self.get_input('fraction')
        erf_context = 'dioxin_cancer'

        erf_type = unit_registry('er_function').to('dimensionless', erf_context)
        p_illness = unit_registry('p_illness').to('dimensionless', erf_context)
        incidence = unit_registry('incidence').to('case/personyear', erf_context)
        period = unit_registry('period').to('d/incident', erf_context)

        if erf_type == 1:  # unit risk
            slope = unit_registry('erf_param_invexposure').to('kg d/mg', erf_context)
            threshold = unit_registry('erf_param_exposure').to('mg/kg/d', erf_context)
            target_population = unit_registry('1 person')
            out = (exposure - threshold) * slope * frexposed * p_illness
            out = (out / target_population / period) / incidence

            self.print_pint_df(out)
            print(p_illness)
            print(incidence)

        elif erf_type != 1:
            out = exposure / exposure

        def postprocess_relative(rr, frexposed):
            r = frexposed * (rr - 1)

            out3 = (r / (r + 1))  # AF=r/(r+1) if r >= 0; AF=r if r<0. Therefore, if the result
            # is smaller than 0, we should use r instead. It can be converted from the result:
            # r/(r+1)=a <=> r=a/(1-a)
            out3[VALUE_COLUMN] = np.where(
                out3[VALUE_COLUMN] < 0,
                out3[VALUE_COLUMN] / (1 - out3[VALUE_COLUMN]),
                out3[VALUE_COLUMN])

        return self.clean_computing(out)

    def compute_old(self):
        param1 = 1  # self.get_input(
        #    'exposure-response',
        #    query="observation =='param1'", drop='observation')
        param2 = 1  # self.get_input(
        #    'exposure-response',
        #    query="observation == 'param2'", drop='observation')
        exposure = self.get_input('ingestion')
        frexposed = self.get_input('fraction')
        incidence = self.get_input('incidence')
        p_illness = self.get_input('probability')

        def postprocess_relative(rr, frexposed):
            r = frexposed * (rr - 1)

            out3 = (r / (r + 1))  # AF=r/(r+1) if r >= 0; AF=r if r<0. Therefore, if the result
            # is smaller than 0, we should use r instead. It can be converted from the result:
            # r/(r+1)=a <=> r=a/(1-a)
            out3[VALUE_COLUMN] = np.where(
                out3[VALUE_COLUMN] < 0,
                out3[VALUE_COLUMN] / (1 - out3[VALUE_COLUMN]),
                out3[VALUE_COLUMN])

            return out3

        er_function_list = exposure.do_inner_join(param2).reset_index()
        er_function_list = list(sorted(set(er_function_list['er_function'])))

        out = pd.DataFrame()

        for func in er_function_list:

            # FIXME Tähän scale_exposure

            # FIXME Tähän yksikön tarkistus. ERF-dataan annetaan altistuksen yksikkö ilman log-muunnosta.

            if func == 'UR':
                k = OvariableFrame(param1.query("er_function == 'UR'"))
                k = k ** -1
                threshold = param2
                dose2 = (exposure - threshold)
                # FIXME clip removes the pint unit. Figure out something else.
                # dose2 = np.clip(dose2, 0, None)  # Smallest allowed value is 0
                out1 = k * dose2 * frexposed  # / incidence
                out = out.append(out1.reset_index())

            if func == 'Step':
                upper = OvariableFrame(param1.query("er_function == 'Step"))
                lower = param2
                out2 = (exposure >= lower) * (exposure <= upper) * -1 + 1  # FIXME
                out2 = out2 * frexposed / incidence
                out = out.append(out2.reset_index())

            if func == 'RR':
                beta = OvariableFrame(param1.query("er_function == 'RR'"))
                beta = beta ** -1  # FIXME This is stupid parameterization
                threshold = param2
                dose2 = exposure - threshold
                #  Smallest allowed value is 0 FIXME: Not if scaling: Log10
                #  dose2[VALUE_COLUMN] = np.clip(dose2[VALUE_COLUMN], 0, None)
                out1 = beta * dose2
                out1 = out1.exp()
                out1 = postprocess_relative(rr=out1, frexposed=frexposed)
                out = out.append(out1.reset_index())

            if func == 'Relative Hill':
                Imax = OvariableFrame(param1.query("er_function == 'Relative Hill'"))
                Imax[VALUE_COLUMN] = Imax[VALUE_COLUMN].pint.m
                ed50 = param2
                out2 = (exposure * Imax) / (exposure + ed50) + 1
                out = out.append(out2.reset_index())

            if func == 'beta poisson approximation':
                p1 = OvariableFrame(param1.query("er_function == 'beta poisson approximation"))
                out4 = ((exposure / param2 + 1) ** (p1 * -1) * -1 + 1) * frexposed
                out4 = (out4 / incidence * p_illness)
                out = out.append(out4.reset_index())

            if func == 'exact beta poisson':
                p1 = OvariableFrame(param1.query("er_function == 'exact beta poisson'"))
                out5 = ((p1 / (p1 + param2) * exposure * -1).exp() * -1 + 1) * frexposed
                out5 = out5 / incidence * p_illness
                out = out.append(out5.reset_index())

            if func == 'exponential':
                k = OvariableFrame(param1.query("er_function == 'exponential'"))
                out6 = ((k * exposure * -1).exp() * -1 + 1) * frexposed
                out6 = out6 / incidence * p_illness
                out = out.append(out6.reset_index())

        keep = set(out.columns) - {'scaling', 'matrix', 'exposure', 'exposure_unit', 'er_function', 0}
        out = out[list(keep)].set_index(list(keep - {VALUE_COLUMN, FORECAST_COLUMN}))

        return self.clean_computing(out)


class DiseaseBurden(Ovariable):
    # BoD is the current (observed) burden of disease (measured in disability-adjusted life years or DALYs).

    quantity = 'disease_burden'

    def compute(self):
        erf_context = 'dioxin_cancer'
        incidence = unit_registry('incidence').to('case/personyear', erf_context)
        case_burden = unit_registry('case_burden').to('DALY/case', erf_context)
        population = self.get_input('population')

        out = population * incidence * case_burden

        return self.clean_computing(out)


class AttributableDiseaseBurden(Ovariable):
    # bod_attr is the burden of disease that can be attributed to the exposure of interest.

    quantity = 'disease_burden'

    def compute(self):
        bod = self.get_input('disease_burden')
        paf = self.get_input('fraction')

        out = bod * paf
        out = out.aggregate_by_column(groupby='Year', fun='sum')

        return self.clean_computing(out)
