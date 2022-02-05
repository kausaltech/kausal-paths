# from logging import exception, log
# from types import FunctionType

# import dvc_pandas
# from graphene.types.scalars import Float, Int
# from pint.registry import ContextCacheOverlay
# from pint_pandas.pint_array import PintArray
from params.param import NumberParameter, PercentageParameter, StringParameter
# from typing import Dict, List
import pandas as pd
# import pint
from .context import unit_registry
import numpy as np
# import math
# import copy

# from common.i18n import TranslatedString
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y
# from .node import Context, Node
# from .exceptions import NodeError

from .simple import AdditiveNode, FixedMultiplierNode, SimpleNode
from .ovariable import Ovariable, OvariableFrame

# ############################33
# Health-related constants and classes

# Ovariable with no input nodes, just data


class DataOvariable(Ovariable, AdditiveNode):
    pass


class MileageDataOvariable(Ovariable):
    allowed_parameters = [
        StringParameter(local_id='population_densities'),
        StringParameter(local_id='emission_heights'),
    ] + Ovariable.allowed_parameters

    quantity = 'mileage'

    def compute(self):
        em_heights = self.get_parameter_value('emission_heights')
        pop_densities = self.get_parameter_value('population_densities')
        mileage = self.get_input_dataset()
        df = pd.DataFrame()
        for em_height in em_heights:
            for pop_density in pop_densities:
                df = df.append(mileage.assign(
                    Emission_height=em_height,
                    Population_density=pop_density
                ))
        df = df.reset_index().set_index(['Emission_height', 'Population_density', YEAR_COLUMN])
        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])
        print('MileageDataOvariable: ' + self.id)
        self.print_pint_df(df)

        return df


class EmissionByFactor(Ovariable):
    allowed_parameters = [
        StringParameter(local_id='emission_factor_contexts'),
        StringParameter(local_id='pollutants'),
    ] + Ovariable.allowed_parameters

    quantity = 'emissions'

    def compute(self):
        ef_contexts = self.get_parameter_value('emission_factor_contexts')
        pollutants = self.get_parameter_value('pollutants')
        mileage = OvariableFrame(self.get_input('mileage'))

        emission_factor = pd.DataFrame()
        for ef_context in ef_contexts:
            for pollutant in pollutants:
                ef = unit_registry('emission_factor_km_' + pollutant).to('g/km', ef_context)
                emission_factor = emission_factor.append(pd.DataFrame({
                    'Vehicle': [ef_context],
                    'Pollutant': [pollutant],
                    FORECAST_COLUMN: [False],
                    VALUE_COLUMN: [ef]
                }))
        emission_factor = OvariableFrame(emission_factor.set_index(['Vehicle', 'Pollutant']))

        emission = mileage * emission_factor
        grouping = ['Emission_height', 'Population_density', 'Pollutant', YEAR_COLUMN]
        emission = emission.aggregate_by_column(grouping, 'sum')
        emission[VALUE_COLUMN] = self.ensure_output_unit(emission[VALUE_COLUMN])
        print('EmissionByFactor: ' + self.id)
        self.print_pint_df(emission)

        return emission


class ExposureInhalation(Ovariable):
    allowed_parameters = [
        StringParameter(local_id='emission_heights'),
        StringParameter(local_id='population_densities'),
        StringParameter(local_id='pollutants'),
    ] + Ovariable.allowed_parameters

    quantity = 'exposure'

    def compute(self):
        em_heights = self.get_parameter_value('emission_heights')
        pop_densities = self.get_parameter_value('population_densities')
        pollutants = self.get_parameter_value('pollutants')
        inhalation_rate = unit_registry('inhalation_rate')
        emission = OvariableFrame(self.get_input('emissions'))
        population = OvariableFrame(self.get_input('population'))
        intake_fraction = pd.DataFrame()

        for em_height in em_heights:
            for pop_density in pop_densities:
                contxt = 'if_' + em_height + '_' + pop_density
                for pollutant in pollutants:
                    value = unit_registry('intake_fraction_' + pollutant).to('ppm', contxt)
                    intake_fraction = intake_fraction.append(pd.DataFrame({
                        'Pollutant': [pollutant],
                        'Emission_height': [em_height],
                        'Population_density': [pop_density],
                        FORECAST_COLUMN: [False],
                        VALUE_COLUMN: [value]
                    }))
        indices = ['Pollutant', 'Emission_height', 'Population_density']
        intake_fraction = OvariableFrame(intake_fraction.set_index(indices))

        exposure = emission * intake_fraction * unit_registry('1 person/year')
        exposure = exposure / (population * inhalation_rate)
        grouping = ['Pollutant', YEAR_COLUMN]
        exposure = exposure.aggregate_by_column(grouping, 'sum')
        exposure[VALUE_COLUMN] = self.ensure_output_unit(exposure[VALUE_COLUMN])
        print('ExposureInhalation: ' + self.id)
        self.print_pint_df(exposure)

        return exposure


class Exposure(Ovariable):
    # Exposure is the intensity of contact with the environment by the target population.

    quantity = 'exposure'
    scaled = False  # This is probably not needed any more

    def compute(self):
        consumption = self.get_input('ingestion')  # This can be bolus (g/person) or rate (g/person/d)
        concentration = self.get_input('mass_concentration')

        exposure = concentration * consumption

        return self.clean_computing(exposure)


class PopulationAttributableFraction(Ovariable):
    # Population attributable fraction PAF
    allowed_parameters = [
        StringParameter(local_id='erf_contexts'),
    ] + Ovariable.allowed_parameters

    quantity = 'fraction'
    unit = 'dimensionless'

    def compute(self):
        routes = ['exposure', 'ingestion', 'inhalation']
        exposure = self.get_input('exposure')
#        self.print_pint_df(exposure)
        exposure[VALUE_COLUMN] = exposure[VALUE_COLUMN].pint.to('mg/kg/d', 'exposure_generic')
        frexposed = self.get_input('fraction')
        erf_contexts = self.get_parameter_value('erf_contexts')

        def postprocess_relative(rr, frexposed):
            r = frexposed * (rr - 1)
            tmp = OvariableFrame(r.copy())  # OvariableFrame objecct cannot be used twice because of inplace
            out3 = (tmp / (r + 1))  # AF=r/(r+1) if r >= 0; AF=r if r<0. Therefore, if the result
            # is smaller than 0, we should use r instead. It can be converted from the result:
            # r/(r+1)=a <=> r=a/(1-a)
            out3[VALUE_COLUMN] = np.where(
                out3[VALUE_COLUMN] < 0,
                out3[VALUE_COLUMN] / (1 - out3[VALUE_COLUMN]),
                out3[VALUE_COLUMN])

            return out3

        output = pd.DataFrame()

        for erf_context in erf_contexts:

            route = routes[unit_registry('route').to('dimensionless', erf_context).m]
            erf_type = unit_registry('er_function').to('dimensionless', erf_context)
            p_illness = unit_registry('p_illness').to('dimensionless', erf_context)
            incidence = unit_registry('incidence').to('case/personyear', erf_context)
            period = unit_registry('period').to('d/incident', erf_context)
            exposure2 = OvariableFrame(exposure.copy())

            if erf_type == 1:  # unit risk
                slope = unit_registry('erf_param_inv_' + route).to('kg d/mg', erf_context)
                threshold = unit_registry('erf_param_' + route).to('mg/kg/d', erf_context)
                target_population = unit_registry('1 person')
                out = (exposure2 - threshold) * slope * frexposed * p_illness
                out = (out / target_population / period) / incidence

            elif erf_type == 2:  # step function
                lower = unit_registry('erf_param_' + route).to('mg/kg/d', erf_context)
                upper = unit_registry('erf_param2_' + route).to('mg/kg/d', erf_context)
                target_population = unit_registry('1 person')
                tmp = OvariableFrame(exposure.copy())
                out = (tmp >= lower) * 1
                out = out * (exposure2 <= upper) * frexposed * p_illness
                out = (out / target_population / period) / incidence

            elif erf_type == 3:  # relative risk
                beta = unit_registry('erf_param_inv_' + route).to('kg d /mg', erf_context)
                threshold = unit_registry('erf_param_' + route).to('mg/kg/d', erf_context)
                out = exposure2 - threshold
                # out[VALUE_COLUMN] = np.where(  # FIXME
                #     out[VALUE_COLUMN] < unit_registry('0 mg/kg/d'),
                #     out[VALUE_COLUMN] * 0,
                #     out[VALUE_COLUMN])
                out = (out * beta).exp()
                out = postprocess_relative(rr=out, frexposed=frexposed)

            elif erf_type == 4:  # Relative Hill
                Imax = unit_registry('erf_param_scale').to('dimensionless', erf_context)
                ed50 = unit_registry('erf_param_ingestion').to('mg/kg/d', erf_context)
                tmp = OvariableFrame(exposure.copy())
                out = (tmp * Imax) / (exposure2 + ed50) + 1
                out = postprocess_relative(rr=out, frexposed=frexposed)

            else:
                out = exposure2 / exposure

            out = out.reset_index().assign(Erf_context=erf_context)
            output = output.append(out)

        indices = list(set(output.columns) - {VALUE_COLUMN, FORECAST_COLUMN})
        output = output.set_index(indices)
        output = self.clean_computing(output)
        print('PopulationAttributableFraction: ' + self.id)
        self.print_pint_df(output)
        return output

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
    allowed_parameters = [
        StringParameter(local_id='erf_contexts'),
    ] + Ovariable.allowed_parameters

    quantity = 'disease_burden'

    def compute(self):
        population = self.get_input('population')
        erf_contexts = self.get_parameter_value('erf_contexts')

        out = pd.DataFrame()

        for erf_context in erf_contexts:
            incidence = unit_registry('incidence').to('case/personyear', erf_context)
            case_burden = unit_registry('case_burden').to('DALY/case', erf_context)

            tmp = OvariableFrame(population.copy())
            tmp = tmp * incidence * case_burden
            tmp = tmp.reset_index().assign(Erf_context=erf_context)

            out = out.append(tmp)

        indices = list(set(out.columns) - {VALUE_COLUMN, FORECAST_COLUMN})
        out = out.set_index(indices)
        out = OvariableFrame(out).aggregate_by_column(groupby='Year', fun='sum')  # FIXME
        out = self.clean_computing(out)
        print('DiseaseBurden: ' + self.id)
        self.print_pint_df(out)
        return out


class AttributableDiseaseBurden(Ovariable):
    # bod_attr is the burden of disease that can be attributed to the exposure of interest.

    quantity = 'disease_burden'

    def compute(self):
        bod = self.get_input('disease_burden')
        paf = self.get_input('fraction')

        out = bod * paf
        out = out.aggregate_by_column(groupby='Year', fun='sum')

        out = self.clean_computing(out)
        print('AttributableDiseaseBurden: ' + self.id)
        self.print_pint_df(out)
        return out
