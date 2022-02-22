import pandas as pd
import numpy as np

from params.param import NumberParameter, PercentageParameter, StringParameter
from .context import unit_registry
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y
from .simple import AdditiveNode, FixedMultiplierNode, SimpleNode
from .ovariable import Ovariable, OvariableFrame
from .exceptions import NodeError

# ############################33
# Health-related constants and classes

# Ovariable with no input nodes, just data


class DataOvariable(Ovariable):
    def compute(self):
        df = self.get_input_dataset(required=True)

        if not isinstance(df, pd.DataFrame):
            raise NodeError(self, "Input is not a DataFrame")
        if VALUE_COLUMN not in df.columns:
            raise NodeError(self, "Input dataset doesn't have Value column")

        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])

        if df.index.max() < self.get_target_year():
            last_year = df.index.max()
            last_val = df.loc[last_year]
            new_index = df.index.append(pd.RangeIndex(last_year + 1, self.get_target_year() + 1))
            assert df.index.name == YEAR_COLUMN
            new_index.name = YEAR_COLUMN
            df = df.reindex(new_index)
            df.iloc[-1] = last_val
            dt = df.dtypes[VALUE_COLUMN]
            df[VALUE_COLUMN] = df[VALUE_COLUMN].pint.m
            df = df.fillna(method='bfill')
            df[VALUE_COLUMN] = df[VALUE_COLUMN].astype(dt)
            df.loc[df.index > last_year, FORECAST_COLUMN] = True

        df[FORECAST_COLUMN] = df[FORECAST_COLUMN].astype(bool)
        return self.clean_computing(df)


class DataColumnOvariable(Ovariable):
    allowed_parameters = [
        StringParameter(local_id='index_columns'),
        StringParameter(local_id='value_columns'),
        StringParameter(local_id='var_name')
    ] + Ovariable.allowed_parameters

    def compute(self):
        index_columns = self.get_parameter_value('index_columns')
        value_columns = self.get_parameter_value('value_columns')
        df = self.get_input_dataset(required=True)
        df = df[index_columns + value_columns]
        if len(value_columns) > 1:
            var_name = self.get_parameter_value('var_name')
            df = df.melt(id_vars=index_columns, var_name=var_name, value_name=VALUE_COLUMN)
            index_columns = index_columns + [var_name]
        df = df.set_index(index_columns)
        self.print_outline(df)
        if YEAR_COLUMN not in df.columns:
            years = OvariableFrame(pd.DataFrame({
                YEAR_COLUMN: pd.Series(range(2010, self.context.target_year + 1)),
                VALUE_COLUMN: pd.Series([1] * (self.context.target_year - 2009)),  # FIXME Lower boundary
                FORECAST_COLUMN: pd.Series([False] * (self.context.target_year - 2009)),
            }).set_index([YEAR_COLUMN]))
            df[FORECAST_COLUMN] = [False] * len(df)
            df = OvariableFrame(df) * years
        self.print_pint_df(df)

        return self.clean_computing(df)


class MileageDataOvariable(Ovariable):
    allowed_parameters = [
        StringParameter(local_id='population_densities'),
        StringParameter(local_id='emission_heights'),
    ] + Ovariable.allowed_parameters

    quantity = 'mileage'

    def compute(self):
        em_heights = self.get_parameter_value('emission_heights')
        pop_densities = self.get_parameter_value('population_densities')
        df = self.get_input_dataset()
        df = df.loc[df.Municipality == 'Tampere'].loc[df.Counting_method == 'Käyttöperusteinen']
        df = df[['Year', 'Level_4', 'Level_5', 'Mileage']]
        df = df.rename(columns={'Mileage': VALUE_COLUMN})
        df = df.assign(Emission_height=em_heights[0], Population_density=pop_densities[0])
        df[FORECAST_COLUMN] = False
        df = df.set_index(['Year', 'Level_4', 'Level_5', 'Emission_height', 'Population_density'])

        return self.clean_computing(df)


class EmissionByFactor(Ovariable):
    allowed_parameters = [
        StringParameter(local_id='pollutants'),
    ] + Ovariable.allowed_parameters

    quantity = 'emissions'

    def compute(self):
        pollutants = self.get_parameter_value('pollutants')
        mileage = OvariableFrame(self.get_input('mileage'))

        emission_factor = self.get_input_dataset()
        emission_factor = emission_factor.loc[emission_factor.Pollutant.isin(pollutants)]
        emission_factor = emission_factor[['Abatement', 'Pollutant', 'Value']].set_index(['Abatement', 'Pollutant'])
        assert VALUE_COLUMN == 'Value'

        emission_classes = OvariableFrame(pd.DataFrame({
            'Abatement': pd.Series([
                'Petrol Medium - Euro 5 – EC 715/2007',
                'Diesel Medium - Euro 5 – EC 715/2007',
                'Urban Buses Standard - Euro V - 2008',
                '4-stroke 250 - 750 cm³ - Conventional',
                '4-stroke 250 - 750 cm³ - Conventional',
                '2-stroke  - Mop - Euro 3 and on',
                'Diesel 16 - 32 t - Euro IV - 2005',
                'Diesel - Euro 5 – EC 715/2007'
            ]),
            'Level_5': pd.Series([
                'Henkilöautot', 'Henkilöautot', 'Linja-autot', 'Moottoripyörät', 'Mopoautot',
                'Mopot', 'Kuorma-autot', 'Pakettiautot'
            ]),
            VALUE_COLUMN: pd.Series([0.6, 0.4, 1, 1, 1, 1, 1, 1]),  # Gasoline:diesel 60%:40%
        })).set_index(['Abatement', 'Level_5'])

        pick_rows = emission_classes.index.get_level_values('Abatement')
        emission_factor = emission_factor.loc[emission_factor.index.get_level_values('Abatement').isin(pick_rows)]
        emission = emission_factor * emission_classes * mileage
        grouping = ['Emission_height', 'Population_density', 'Pollutant', YEAR_COLUMN]
        emission = OvariableFrame(emission).aggregate_by_column(grouping, 'sum')
        # FIXME Why does OvariableFrame disappear?

        return self.clean_computing(emission)


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

        return self.clean_computing(exposure)


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
        frexposed = self.get_input('fraction')
        erf_contexts = self.get_parameter_value('erf_contexts')
        erfs, incidences = self.get_input_datasets()

        def postprocess_relative(rr, frexposed):  # FIXME Function inside a function works but is not elegant?
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
            assert len(incidence) == 1
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
            assert len(exposure2) > 0

            # If erf and exposure nodes are not in compatible units, converts both to exposure units
            def check_erf_units(param):  # FIXME Function inside a loop works but is not elegant?
                powers = {
                    'p1': 'mg/kg/d',
                    'p0': 'dimensionless',
                    'm1': 'kg d / mg',
                    'm2': '(kg d / mg)**2',
                    'm3': '(kg d / mg)**3'
                }
                power = param.split('_')[1]
                out = erf[param][0]
                if power == 'p0' or not hasattr(erf[param], 'pint'):
                    return out

                _exposure_unit = unit_registry(route + '_p1')
                is_erf_compatible = exposure2.Value.pint.units.is_compatible_with(_exposure_unit)

                if not is_erf_compatible:
                    exposure2[VALUE_COLUMN] = exposure2[VALUE_COLUMN].pint.to('mg/kg/d', 'exposure_generic')
                    out = out.to(powers[power], 'exposure_generic')
                return out

            # FIXME Check that the periods, incidences and target_populations in all erf_types are consistent
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

            elif erf_type == 'linear relative':
                k = check_erf_units(route + '_m1')
                threshold = check_erf_units(route + '_p1')
                out = exposure2 - threshold
                # out[VALUE_COLUMN] = np.where(  # FIXME
                #     out[VALUE_COLUMN] < unit_registry('0 mg/kg/d'),
                #     out[VALUE_COLUMN] * 0,
                #     out[VALUE_COLUMN])
                out = out * k
                out = postprocess_relative(rr=out + 1, frexposed=frexposed)

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
                out = out * frexposed * p_illness

            else:
                out = exposure2 / exposure

            out = out.reset_index().assign(Erf_context=erf_context)
            output = output.append(out)

        indices = list(set(output.columns) - {VALUE_COLUMN, FORECAST_COLUMN})
        output = output.set_index(indices)

        return self.clean_computing(output)


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
#        out = OvariableFrame(out).aggregate_by_column(groupby='Year', fun='sum')  # FIXME

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
