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
        frexposed = self.get_input('fraction')
        erf_contexts = self.get_parameter_value('erf_contexts')
        df = self.get_input_dataset()

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

        if False:
            def check_erf_units(param):
                out = unit_registry(param)
                is_erf_compatible = exposure.Value.pint.units.is_compatible_with(out)
                if not is_erf_compatible:
                    print(df6.Value)
            #        df6.Value = df6.Value.pint.to('kg/a')
                    out = out.to('kLden')
                return out

            check_erf_units('Lden')

        output = pd.DataFrame()

        for erf_context in erf_contexts:

            p_illness = unit_registry('p_illness')
            incidence = unit_registry('incidence')
            period = unit_registry('period')
            erf = df.loc[df.Erf_context == erf_context].reset_index()
            route = erf.Route[0]

            assert len(erf) == 1

            if 'Er_function' in erf.columns:
                erf_type = erf.Er_function[0]
                print(type(erf_type))
                print(erf_type)

            if 'Pollutant' in exposure.index.names:
                exposure_agent = erf_context.split(' ')[0]
                exposure_agent = exposure.index.get_level_values('Pollutant') == exposure_agent
                exposure2 = OvariableFrame(exposure.loc[exposure_agent].copy())
            else:
                exposure2 = OvariableFrame(exposure.copy())

            if erf_context.split(' ')[0] == 'dioxin':  # FIXME Not a good place for this adjustment
                exposure2[VALUE_COLUMN] = exposure2[VALUE_COLUMN].pint.to('mg/kg/d', 'exposure_generic')
                self.print_pint_df(exposure2)

            if erf_type == 'unit risk':
                slope = erf['param_inv_' + route][0]
                threshold = erf['param_' + route][0]
                target_population = unit_registry('1 person')
                out = (exposure2 - threshold) * slope * frexposed * p_illness
                out = (out / target_population / period) / incidence

            elif erf_type == 'step function':
                lower = erf['param_' + route][0]
                upper = erf['param2_' + route][0]
                target_population = unit_registry('1 person')
                tmp = OvariableFrame(exposure2.copy())
                out = (tmp >= lower) * 1
                out = out * (exposure2 <= upper) * frexposed * p_illness
                out = (out / target_population / period) / incidence

            elif erf_type == 'relative risk':
                beta = erf['param_inv_' + route][0]
                threshold = erf['param_' + route][0]
                out = exposure2 - threshold
                # out[VALUE_COLUMN] = np.where(  # FIXME
                #     out[VALUE_COLUMN] < unit_registry('0 mg/kg/d'),
                #     out[VALUE_COLUMN] * 0,
                #     out[VALUE_COLUMN])
                out = (out * beta).exp()
                out = postprocess_relative(rr=out, frexposed=frexposed)

            elif erf_type == 'relative Hill':
                Imax = erf['param_scale'][0]
                ed50 = erf['param_' + route][0]
                tmp = OvariableFrame(exposure2.copy())
                out = (tmp * Imax) / (exposure2 + ed50) + 1
                out = postprocess_relative(rr=out, frexposed=frexposed)

            elif erf_type == 'beta poisson approximation':
                p1 = erf['param_' + route][0]
                p2 = erf['param_scale'][0]
                out = ((exposure2 / p2 + 1) ** (p1 * -1) * -1 + 1) * frexposed
                out = (out / incidence * p_illness)

            elif erf_type == 'exact beta poisson':
                p1 = erf['param_' + route][0]
                p2 = erf['param2_' + route][0]
                out = ((p1 / (p1 + p2) * exposure2 * -1).exp() * -1 + 1) * frexposed
                out = out / incidence * p_illness

            elif erf_type == 'exponential':
                k = erf['param_inv_' + route][0]
                out = ((k * exposure2 * -1).exp() * -1 + 1) * frexposed
                out = out / incidence * p_illness

            elif erf_type == 'polynomial':
                p0 = erf['param_poly0_' + route][0]
                p1 = erf['param_poly1_' + route][0]
                p2 = erf['param_poly2_' + route][0]
                p3 = erf['param_poly3_' + route][0]
                threshold = erf['param_' + route][0]
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
