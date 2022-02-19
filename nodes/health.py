import pandas as pd
import numpy as np

from params.param import NumberParameter, PercentageParameter, StringParameter
from .context import unit_registry
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y
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

    quantity = 'exposure'

    def compute(self):
        inhalation_rate = unit_registry('inhalation_rate')
        emission = OvariableFrame(self.get_input('emissions'))
        population = OvariableFrame(self.get_input('population'))
        df = self.get_input_dataset()

        inde = list(set(df.columns) - {VALUE_COLUMN})
        intake_fraction = OvariableFrame(df.set_index(inde))

        exposure = emission * intake_fraction * unit_registry('1 person/year')
        exposure = exposure / (population * inhalation_rate)
        grouping = ['Pollutant', YEAR_COLUMN]
        exposure = exposure.aggregate_by_column(grouping, 'sum')
        exposure[VALUE_COLUMN] = self.ensure_output_unit(exposure[VALUE_COLUMN])

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
    ''' Population attributable fraction PAF
    The node must have exactly two input datasets in this order:
    * One with exposure-response functions and parameters
    * One with incidence data and Incidence column
    '''
    allowed_parameters = [
        StringParameter(local_id='erf_contexts'),
    ] + Ovariable.allowed_parameters

    quantity = 'fraction'
    unit = 'dimensionless'

    def compute(self):
        exposure = self.get_input('exposure')
        if YEAR_COLUMN not in exposure.index.names:  # FIXME not the right place for this fix
            assert len(exposure.index.names) == 1
            exposure.index.names = [YEAR_COLUMN]
        frexposed = self.get_input('fraction')
        erf_contexts = self.get_parameter_value('erf_contexts')
        erfs, incidences = self.get_input_datasets()

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

            incidence = incidences.loc[incidences.Erf_context == erf_context].reset_index()
            incidence = incidence.Incidence[0]

            erf = erfs.loc[erfs.Erf_context == erf_context].reset_index()
            assert len(erf) == 1
            route = erf.Route[0]
            period = erf.Period[0]
            if 'P_illness' in erf.columns:
                p_illness = erf.P_illness[0]
            else:
                p_illness = unit_registry('p_illness')

            erf_type = erf.Er_function[0]
            if 'Exposure_agent' in erf.columns:
                exposure_agent = erf.Exposure_agent[0]
            elif 'Pollutant' in erf.columns:
                exposure_agent = erf.Pollutant[0]
            else:
                exposure_agent = erf_context.split(' ')[0]

            indices = list(set(exposure.index.names + ['Exposure_agent']) - {'Pollutant'})
            if 'Pollutant' in exposure.index.names:
                pick_rows = exposure.index.get_level_values('Pollutant') == exposure_agent
                exposure2 = exposure.copy().loc[pick_rows].reset_index()
                exposure2 = exposure2.rename(columns={'Pollutant': 'Exposure_agent'})
            else:
                exposure2 = exposure.copy().assign(Exposure_agent=exposure_agent).reset_index()
            exposure2 = OvariableFrame(exposure2.set_index(indices))

            # If erf and exposure nodes are not in compatible units, converts both to exposure units
            def check_erf_units(param):
                powers = {
                    'p1': 'mg/kg/d',
                    'p0': 'dimensionless',
                    'm1': 'kg d / mg',
                    'm2': '(kg d / mg)**2',
                    'm3': '(kg d / mg)**3'
                }
                power = param.split('_')[1]
                out = erf[param][0]
                if not hasattr(erf[param], 'pint') or power == 'p0':
                    return out

                _exposure_unit = unit_registry(route + '_p1')
                is_erf_compatible = exposure2.Value.pint.units.is_compatible_with(_exposure_unit)

                if not is_erf_compatible:
                    exposure2[VALUE_COLUMN] = exposure2[VALUE_COLUMN].pint.to('mg/kg/d', 'exposure_generic')
                    out = out.to(powers[power], 'exposure_generic')
                return out

            if erf_type == 'unit risk':
                slope = check_erf_units(route + '_m1')
                threshold = check_erf_units(route + '_p1')
                target_population = unit_registry('1 person')
                out = (exposure2 - threshold) * slope * frexposed * p_illness
                out = (out / target_population / period) / incidence

            elif erf_type == 'step function':
                lower = check_erf_units(route + '_p1')
                upper = check_erf_units(route + '_p1_2')
                target_population = unit_registry('1 person')
                tmp = OvariableFrame(exposure2.copy())
                out = (tmp >= lower) * 1
                out = out * (exposure2 <= upper) * frexposed * p_illness
                out = (out / target_population / period) / incidence

            elif erf_type == 'relative risk':
                beta = check_erf_units(route + '_m1')
                threshold = check_erf_units(route + '_p1')
                out = exposure2 - threshold
                # out[VALUE_COLUMN] = np.where(  # FIXME
                #     out[VALUE_COLUMN] < unit_registry('0 mg/kg/d'),
                #     out[VALUE_COLUMN] * 0,
                #     out[VALUE_COLUMN])
                out = (out * beta).exp()
                out = postprocess_relative(rr=out, frexposed=frexposed)

            elif erf_type == 'relative Hill':
                Imax = check_erf_units(route + '_p0')
                ed50 = check_erf_units(route + '_p1')
                tmp = OvariableFrame(exposure2.copy())
                out = (tmp * Imax) / (exposure2 + ed50) + 1
                out = postprocess_relative(rr=out, frexposed=frexposed)

            elif erf_type == 'beta poisson approximation':
                p1 = check_erf_units(route + '_p1')
                p2 = check_erf_units(route + '_p0')
                out = ((exposure2 / p2 + 1) ** (p1 * -1) * -1 + 1) * frexposed
                out = (out / incidence * p_illness)

            elif erf_type == 'exact beta poisson':
                p1 = check_erf_units(route + '_p1')
                p2 = check_erf_units(route + '_p1_2')
                out = ((p1 / (p1 + p2) * exposure2 * -1).exp() * -1 + 1) * frexposed
                out = out / incidence * p_illness

            elif erf_type == 'exponential':
                k = check_erf_units(route + '_m1')
                out = ((k * exposure2 * -1).exp() * -1 + 1) * frexposed
                out = out / incidence * p_illness

            elif erf_type == 'polynomial':
                threshold = check_erf_units(route + '_p1')
                p0 = check_erf_units(route + '_p0')
                p1 = check_erf_units(route + '_m1')
                p2 = check_erf_units(route + '_m2')
                p3 = check_erf_units(route + '_m3')
                out = exposure2 - threshold
                tmp = OvariableFrame(out.copy())
                tmp1 = OvariableFrame(out.copy())
                out = out ** 3 * p3 + tmp ** 2 * p2 + tmp1 * p1 + p0

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
    '''BoD is the current (observed) burden of disease (measured in disability-adjusted life years or DALYs).
    The node must always contain exactly two datasets in this order:
    * One with incidence data and Incidence column
    * One with case burden and Case_burden column
    '''
    allowed_parameters = [
        StringParameter(local_id='erf_contexts'),
    ] + Ovariable.allowed_parameters

    quantity = 'disease_burden'

    def compute(self):
        population = self.get_input('population')
        erf_contexts = self.get_parameter_value('erf_contexts')
        incidences, case_burdens = self.get_input_datasets()

        out = pd.DataFrame()

        for erf_context in erf_contexts:
            incidence = incidences.loc[incidences.Erf_context == erf_context].reset_index()
            incidence = incidence.Incidence[0]
            case_burden = case_burdens.loc[case_burdens.Erf_context == erf_context].reset_index()
            case_burden = case_burden.Case_burden[0]

            tmp = OvariableFrame(population.copy())
            tmp = tmp * incidence * case_burden
            tmp = tmp.reset_index().assign(Erf_context=erf_context)

            out = out.append(tmp)

        indices = list(set(out.columns) - {VALUE_COLUMN, FORECAST_COLUMN})
        out = out.set_index(indices)
        out = OvariableFrame(out).aggregate_by_column(groupby='Year', fun='sum')  # FIXME
        out = self.clean_computing(out)
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
        return out
