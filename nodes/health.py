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
        self.print_outline(df)

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
        self.print_outline(emission)

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
        self.print_outline(exposure)

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
        routes = ['exposure', 'ingestion', 'inhalation', '']
        exposure = self.get_input('exposure')
        if YEAR_COLUMN not in exposure.index.names:  # FIXME not the right place for this fix
            assert len(exposure.index.names) == 1
            exposure.index.names = [YEAR_COLUMN]
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

            if 'Pollutant' in exposure.index.names:
                exposure_agent = erf_context.split('_x_')[0]
                exposure_agent = exposure.index.get_level_values('Pollutant') == exposure_agent
                exposure2 = OvariableFrame(exposure.loc[exposure_agent].copy())
            else:
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
                tmp = OvariableFrame(exposure2.copy())
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
                tmp = OvariableFrame(exposure2.copy())
                out = (tmp * Imax) / (exposure2 + ed50) + 1
                out = postprocess_relative(rr=out, frexposed=frexposed)

            elif erf_type == 5:  # beta poisson approximation
                p1 = unit_registry('erf_param_ingestion').to('mg/kg/d', erf_context)
                p2 = unit_registry('erf_param_scale').to('dimensionless', erf_context)
                out = ((exposure2 / p2 + 1) ** (p1 * -1) * -1 + 1) * frexposed
                out = (out / incidence * p_illness)

            elif erf_type == 6:  # exact beta poisson # FIXME Logical error with units!
                p1 = unit_registry('erf_param_ingestion').to('mg/kg/d', erf_context)
                p2 = unit_registry('erf_param_ingestion2').to('mg/kg/d', erf_context)
                out = ((p1 / (p1 + p2) * exposure2 * -1).exp() * -1 + 1) * frexposed
                out = out / incidence * p_illness

            elif erf_type == 7:  # exponential
                k = unit_registry('erf_param_inv_ingestion').to('kg d/mg', erf_context)
                out = ((k * exposure2 * -1).exp() * -1 + 1) * frexposed
                out = out / incidence * p_illness

            elif erf_type == 8:  # polynomial
                print(erf_context)
                p0 = unit_registry('erf_param_poly0').to('dimensionless', erf_context)
                p1 = unit_registry('erf_param_poly1').to('(kg d / mg)', erf_context)
                p2 = unit_registry('erf_param_poly2').to('(kg d / mg)^2', erf_context)
                p3 = unit_registry('erf_param_poly3').to('(kg d / mg)^3', erf_context)
                threshold = unit_registry('erf_param_exposure').to('mg/kg/d', erf_context)
                out = exposure2 - threshold
                tmp = OvariableFrame(out.copy())
                tmp1 = OvariableFrame(out.copy())
                self.print_outline(out)
                self.print_outline(tmp)
                self.print_outline(tmp1)
                print(p0, p1, p2, p3, threshold)
                out = out ** 3 * p3 + tmp ** 2 * p2 + tmp1 * p1 + p0
                self.print_outline(out)

            else:
                out = exposure2 / exposure

            out = out.reset_index().assign(Erf_context=erf_context)
            output = output.append(out)

        indices = list(set(output.columns) - {VALUE_COLUMN, FORECAST_COLUMN})
        output = output.set_index(indices)
        output = self.clean_computing(output)

        self.print_outline(output)
        return output


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

        self.print_outline(out)
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

        self.print_outline(out)
        return out
