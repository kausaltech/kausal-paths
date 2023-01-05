import pandas as pd
import numpy as np

from params.param import NumberParameter, PercentageParameter, StringParameter
from .context import unit_registry
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y
from .simple import AdditiveNode, FixedMultiplierNode, SimpleNode
from .ovariable import Ovariable, OvariableFrame
from .exceptions import NodeError

''' QUESTIONS TO ASK:
Q1: If exposure-response has restrictions for e.g. pollutant or population subgroup that have no parameters,
how can that be dealt with if exposure node has those additional rows of data?
A: Put all additional index columns in the classification and thus make sure that right data is filtered.
Q2: How can parameters be created without having a dummy line in yaml?
A: Don't. Just have the extra lines in yaml.
Q3: Can there be e.g. a string parameter containing a filter query?
A: I think answer to Q1 solves this.
Q4: Can parameters have lists? That would make it possible to aggregate related things into one node.
A: No. Put e.g. the dimensions to be summed into the child node, not in parameter.
Q5: Is it a good idea to have these context-insensitive ExposureResponse nodes?
A: Yes, maybe. We'll compare the approaches when we have actual customers.
(ExposureResponse vs PopulationAttributableFraction)
'''
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

    def compute(self):  # FIXME Replace node parameters with dataset parameters
        # Maybe adjust parameter groupby for this purpose?
        index_columns = self.get_parameter_value('index_columns')
        value_columns = self.get_parameter_value('value_columns')
        df = self.get_input_dataset(required=True)
        if len(value_columns) > 1:
            var_name = self.get_parameter_value('var_name')
            df = df.melt(id_vars=index_columns, var_name=var_name, value_name=VALUE_COLUMN)
            index_columns = index_columns + [var_name]
        elif value_columns[0] == 'dummy':
            df[VALUE_COLUMN] = 1
        else:
            df = df.rename(columns={value_columns[0]: VALUE_COLUMN})
        df = df[index_columns + [VALUE_COLUMN]]
        df = df.set_index(index_columns)
        df = self.add_years(df)

        return self.clean_computing(df)


class PhysicalActivity(Ovariable):

    def compute(self):
        nodes = self.input_nodes
        pa_distance = nodes[0]
        pa_met = nodes[1]
        pa_equivalent = nodes[2]
        pa_rate = nodes[3]

        out = OvariableFrame(pa_distance.get_output())
        out = out / OvariableFrame(pa_met.get_output())
        out = out * OvariableFrame(pa_equivalent.get_output())
        out = out * OvariableFrame(pa_rate.get_output())

        return self.clean_computing(out)


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
        df = df.loc[df.Year == 2017].drop('Year', axis=1)  # FIXME
        print(df)
        df = df.set_index(['Level_4', 'Level_5', 'Emission_height', 'Population_density'])
        df = self.add_years(df)

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


class ExposureResponse(Ovariable):
    '''ExposureResponses contain all necessary information for calculating disease burdens
    for a particular exposure and response.
    '''

    allowed_parameters = [
        StringParameter(local_id='erf_contexts'),
        StringParameter(local_id='route'),
        StringParameter(local_id='erf_type'),
        NumberParameter(local_id='period'),
        NumberParameter(local_id='default_incidence'),
        NumberParameter(local_id='default_frexposed'),
        NumberParameter(local_id='p_illness'),
        StringParameter(local_id='exposure_agent'),
        StringParameter(local_id='response'),
        NumberParameter(local_id='target_population_size'),
        NumberParameter(local_id='exposure_unit'),
        NumberParameter(local_id='case_burden'),
        NumberParameter(local_id='p1'),
        NumberParameter(local_id='p1_2'),
        NumberParameter(local_id='p0'),
        NumberParameter(local_id='p0_2'),
        NumberParameter(local_id='m1'),
        NumberParameter(local_id='m2'),
        NumberParameter(local_id='m3'),
    ]

    quantity = 'exposure-response'

    def extract_erf_parameters(self, erf):
        for power in ['p1', 'p1_2', 'p0', 'p0_2', 'm1', 'm2', 'm3']:
            param = self.get_parameter_value('route') + '_' + power
            if param in erf.columns:
                value = erf[param][0]
                self.set_parameter_value(power, value)

    def compute(self):
        classification = self.get_input('ratio')
        erf_contexts = self.get_parameter_value('erf_contexts')
        erf_context = erf_contexts[0]  # FIXME List is meaningful only with multiple erf_contexts in one node.
        datasets = self.get_input_datasets()
        erfs = datasets[0]
        if len(datasets) > 1:
            incidences = datasets[1]
            incidence = incidences.loc[incidences.Erf_context == erf_context].reset_index()
            assert len(incidence) == 1
            incidence = incidence.Incidence[0]
            self.set_parameter_value('default_incidence', incidence)
        else:
            incidence = self.get_parameter_value('default_incidences', units=True)

        erf = erfs.loc[erfs.Erf_context == erf_context].reset_index()
        assert len(erf) == 1
        route = erf.Route[0]
        self.set_parameter_value('route', route, True)

        exposure_unit = unit_registry(route + '_p1')
        self.set_parameter_value('exposure_unit', exposure_unit)

        period = erf.Period[0]
        self.set_parameter_value('period', period)

        if 'P_illness' in erf.columns:
            p_illness = erf.P_illness[0]
        else:
            p_illness = unit_registry('p_illness')
        self.set_parameter_value('p_illness', p_illness)

        erf_type = erf.Er_function[0]
        self.set_parameter_value('erf_type', erf_type)

        if 'Exposure_agent' in erf.columns:
            exposure_agent = erf.Exposure_agent[0]
        elif 'Pollutant' in erf.columns:
            exposure_agent = erf.Pollutant[0]
        else:
            exposure_agent = erf_context.split(' ')[0]
        self.set_parameter_value('exposure_agent', exposure_agent)

        if 'Response' in erf.columns:
            response = erf.Response[0]
        else:
            response = erf_context.split(' ')[1]
        self.set_parameter_value('response', response)

        if 'Target_population_size' in erf.columns:
            target_population_size = erf.Target_population_size[0]
        else:
            target_population_size = unit_registry('1 person')
        self.set_parameter_value('target_population_size', target_population_size)

        self.extract_erf_parameters(erfs)

        case_burden = erf.Case_burden[0]
        self.set_parameter_value('case_burden', case_burden)

        print('erf_contexts:', self.get_parameter_value('erf_contexts'))
        print('route:', self.get_parameter_value('route'))
        print('erf_type:', self.get_parameter_value('erf_type'))
        print('period:', self.get_parameter_value('period', units=True))
        print('default_incidence:', self.get_parameter_value('default_incidence', units=True))
        print('default_frexposed:', self.get_parameter_value('default_frexposed', units=True))
        print('p_illness:', self.get_parameter_value('p_illness', units=True))
        print('exposure_agent:', self.get_parameter_value('exposure_agent'))
        print('response:', self.get_parameter_value('response'))
        print('target_population_size:', self.get_parameter_value('target_population_size', units=True))
        print('exposure_unit:', self.get_parameter_value('exposure_unit', units=True))
        print('case_burden:', self.get_parameter_value('case_burden', units=True))
        print('p1:', self.get_parameter_value('p1', units=True))
#        print(self.get_parameter_value_w_unit('p1_2'))  # FIXME If not exists, give warning not error
        print('p0:', self.get_parameter_value('p0', units=True))
#        print(self.get_parameter_value_w_unit('p0_2'))
        print('m1:', self.get_parameter_value('m1', units=True))
        print('m2:', self.get_parameter_value('m2', units=True))
        print('m3:', self.get_parameter_value('m3', units=True))

        classification = self.add_years(classification)

        return self.clean_computing(classification)


class AttributableFraction(Ovariable):
    quantity = 'fraction'
    default_unit = 'dimensionless'

    def postprocess_relative(self, rr, frexposed):
        '''AF=r/(r+1) if r >= 0; AF=r if r<0. Therefore, if the result
        is smaller than 0, we should use r instead. It can be converted from the result:
        r/(r+1)=s <=> r=s/(1-s)
        '''
        r = frexposed * (rr - 1)
        of = (r / (r + 1))
        s = of[VALUE_COLUMN]
        of[VALUE_COLUMN] = np.where(s < 0, s / (1 - s), s)
        return of

    def compute(self):
        for node in self.input_nodes:
            if node.quantity == 'exposure-response':
                erf = node  # Full node with parameters needed
        exposures = self.get_input('exposure')
        # FIXME If erf has restrictions about e.g. pollutant or subgroup, these are NOT taken into account in exposures.
        frexposed = self.get_input('fraction', required=False)
        if frexposed is None:
            frexposed = erf.get_parameter_value('default_frexposed', units=True)
        incidence = self.get_input('incidence', required=False)
        if incidence is None:
            incidence = erf.get_parameter_value('default_incidence', units=True)
        erf_type = erf.get_parameter_value('erf_type')
        period = erf.get_parameter_value('period', units=True)
        exposure_unit = erf.get_parameter_value('exposure_unit', units=True)
        is_erf_compatible = exposures.Value.pint.units.is_compatible_with(exposure_unit)

        # If erf and exposures nodes are not in compatible units, converts both to exposure units
        powers = {
            'p1': 'mg/kg/d',
            'p0': 'dimensionless',
            'm1': 'kg d / mg',
            'm2': '(kg d / mg)**2',
            'm3': '(kg d / mg)**3'
        }
        exposure = exposures * erf.get_output()

        if not is_erf_compatible:
            exposure[VALUE_COLUMN] = exposure[VALUE_COLUMN].pint.to('mg/kg/d', 'exposure_generic')

        def pick_parameter(local_id):
            value = erf.get_parameter_value(local_id, units=True)
            if not is_erf_compatible:
                power = local_id.split(' ')[0]
                if power != 'p0':
                    value = value.to(powers[power], 'exposure_generic')
            return value

        if erf_type == 'unit risk':
            slope = pick_parameter('m1')
            threshold = pick_parameter('p1')
            target_population_size = pick_parameter('target_population_size')
            out = (exposure - threshold) * slope * frexposed * p_illness
            out = (out / target_population_size / period) / incidence

        elif erf_type == 'step function':
            lower = pick_parameter('p1')
            upper = pick_parameter('p1_2')
            target_population = unit_registry('1 person')
            out = (exposure >= lower) * 1
            out = (out * (exposure <= upper) * -1 + 1) * frexposed * p_illness
            out = (out / target_population / period) / incidence

        elif erf_type == 'relative risk':
            beta = pick_parameter('m1')
            threshold = pick_parameter('p1')
            out = exposure - threshold
            rrmin = pick_parameter('p0')
            out = (out * beta).exp()
            s = out[VALUE_COLUMN]
            out[VALUE_COLUMN] = np.where(s < rrmin, rrmin, s)
            out = self.postprocess_relative(rr=out, frexposed=frexposed)

        elif erf_type == 'linear relative':
            k = pick_parameter('m1')
            threshold = pick_parameter('p1')
            rrmin = pick_parameter('p0')
            out = exposure - threshold
            out = out * k
            s = out[VALUE_COLUMN]
            out[VALUE_COLUMN] = np.where(s < rrmin, rrmin, s)
            out = self.postprocess_relative(rr=out + 1, frexposed=frexposed)

        elif erf_type == 'relative Hill':
            Imax = pick_parameter('p0')
            ed50 = pick_parameter('p1')
            out = (exposure * Imax) / (exposure + ed50) + 1
            out = self.postprocess_relative(rr=out, frexposed=frexposed)

        elif erf_type == 'beta poisson approximation':
            p1 = pick_parameter('p0')
            p2 = pick_parameter('p1')
            out = (exposure / p2 + 1) ** (p1 * -1) * -1 + 1
            out = out * frexposed * p_illness

        elif erf_type == 'exact beta poisson':
            p1 = pick_parameter('p0_2')
            p2 = pick_parameter('p0')
            # Remove unit: exposure is an absolute number of microbes ingested
            s = exposure[VALUE_COLUMN].pint.to('cfu/d')
            s = s / unit_registry('cfu/d')
            exposure[VALUE_COLUMN] = s
            out = (exposure * p1 / (p1 + p2) * -1).exp() * -1 + 1
            out = out * frexposed * p_illness

        elif erf_type == 'exponential':
            k = pick_parameter('m1')
            out = (exposure * k * -1).exp() * -1 + 1
            out = out * frexposed * p_illness

        elif erf_type == 'polynomial':
            threshold = pick_parameter('p1')
            p0 = pick_parameter('p0')
            p1 = pick_parameter('m1')
            p2 = pick_parameter('m2')
            p3 = pick_parameter('m3')
            p_illness = pick_parameter('p_illness')
            x = exposure - threshold
            out = x ** 3 * p3 + x ** 2 * p2 + x * p1 + p0
            out = out * frexposed * p_illness

        else:
            out = exposure / exposures
        return self.clean_computing(out)


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
    default_unit = 'dimensionless'

    def compute(self):
        exposures = self.get_input('exposure')
        frexposed = self.get_input('fraction')
        erf_contexts = self.get_parameter_value('erf_contexts')
        erfs, incidences = self.get_input_datasets()

        def postprocess_relative(rr, frexposed):  # FIXME Function inside a function works but is not elegant?
            r = frexposed * (rr - 1)
            of = (r / (r + 1))  # AF=r/(r+1) if r >= 0; AF=r if r<0. Therefore, if the result
            # is smaller than 0, we should use r instead. It can be converted from the result:
            # r/(r+1)=a <=> r=a/(1-a)
            of[VALUE_COLUMN] = np.where(
                of[VALUE_COLUMN] < 0,
                of[VALUE_COLUMN] / (1 - of[VALUE_COLUMN]),
                of[VALUE_COLUMN])

            return of

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

            indices = list(set(exposures.index.names + ['Exposure_agent']) - {'Pollutant'})
            if 'Pollutant' in exposures.index.names:
                pick_rows = exposures.index.get_level_values('Pollutant') == exposure_agent
                exposure = exposures.copy().loc[pick_rows].reset_index()
                exposure = exposure.rename(columns={'Pollutant': 'Exposure_agent'})
            else:
                exposure = exposures.copy().assign(Exposure_agent=exposure_agent).reset_index()
            exposure = OvariableFrame(exposure.set_index(indices))
            assert len(exposure) > 0

            # If erf and exposures nodes are not in compatible units, converts both to exposure units
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
                is_erf_compatible = exposure.Value.pint.units.is_compatible_with(_exposure_unit)

                if not is_erf_compatible:
                    exposure[VALUE_COLUMN] = exposure[VALUE_COLUMN].pint.to('mg/kg/d', 'exposure_generic')
                    out = out.to(powers[power], 'exposure_generic')
                return out

            if erf_type == 'unit risk':
                slope = check_erf_units(route + '_m1')
                threshold = check_erf_units(route + '_p1')
                target_population = unit_registry('1 person')
                out = (exposure - threshold) * slope * frexposed * p_illness
                out = (out / target_population / period) / incidence

            elif erf_type == 'step function':
                lower = check_erf_units(route + '_p1')
                upper = check_erf_units(route + '_p1_2')
                target_population = unit_registry('1 person')
                out = (exposure >= lower) * 1
                out = (out * (exposure <= upper) * -1 + 1) * frexposed * p_illness
                out = (out / target_population / period) / incidence

            elif erf_type == 'relative risk':
                beta = check_erf_units(route + '_m1')
                threshold = check_erf_units(route + '_p1')
                out = exposure - threshold
                # out[VALUE_COLUMN] = np.where(  # FIXME
                #     out[VALUE_COLUMN] < unit_registry('0 mg/kg/d'),
                #     out[VALUE_COLUMN] * 0,
                #     out[VALUE_COLUMN])
                out = (out * beta).exp()
                out = postprocess_relative(rr=out, frexposed=frexposed)

            elif erf_type == 'linear relative':
                k = check_erf_units(route + '_m1')
                threshold = check_erf_units(route + '_p1')
                out = exposure - threshold
                # out[VALUE_COLUMN] = np.where(  # FIXME
                #     out[VALUE_COLUMN] < unit_registry('0 mg/kg/d'),
                #     out[VALUE_COLUMN] * 0,
                #     out[VALUE_COLUMN])
                out = out * k
                out = postprocess_relative(rr=out + 1, frexposed=frexposed)

            elif erf_type == 'relative Hill':
                Imax = check_erf_units(route + '_p0')
                ed50 = check_erf_units(route + '_p1')
                out = (exposure * Imax) / (exposure + ed50) + 1
                out = postprocess_relative(rr=out, frexposed=frexposed)

            elif erf_type == 'beta poisson approximation':
                p1 = check_erf_units(route + '_p0')
                p2 = check_erf_units(route + '_p1')
                out = (exposure / p2 + 1) ** (p1 * -1) * -1 + 1
                out = out * frexposed * p_illness

            elif erf_type == 'exact beta poisson':
                p1 = check_erf_units(route + '_p0_2')
                p2 = check_erf_units(route + '_p0')
                # Remove unit: exposure is an absolute number of microbes ingested
                s = exposure[VALUE_COLUMN].pint.to('cfu/d')
                s = s / unit_registry('cfu/d')
                exposure[VALUE_COLUMN] = s
                out = (exposure * p1 / (p1 + p2) * -1).exp() * -1 + 1
                out = out * frexposed * p_illness

            elif erf_type == 'exponential':
                k = check_erf_units(route + '_m1')
                out = (exposure * k * -1).exp() * -1 + 1
                out = out * frexposed * p_illness

            elif erf_type == 'polynomial':
                threshold = check_erf_units(route + '_p1')
                p0 = check_erf_units(route + '_p0')
                p1 = check_erf_units(route + '_m1')
                p2 = check_erf_units(route + '_m2')
                p3 = check_erf_units(route + '_m3')
                x = exposure - threshold
                out = x ** 3 * p3 + x ** 2 * p2 + x * p1 + p0
                out = out * frexposed * p_illness

            else:
                out = exposure / exposures

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

            tmp = population * incidence * case_burden
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
        drop_columns = [
            'Erf_context', 'Exposure_agent', 'Response', 'Pollutant', 'Exposure level',
            'Unit', 'Source', 'Vehicle', 'Age group']
        bod = self.get_input('disease_burden')
        paf = self.get_input('fraction')

        out = bod * paf

        return self.clean_computing(out, drop_columns)
