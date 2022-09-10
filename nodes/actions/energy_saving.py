import pandas as pd
import numpy as np
from numba import njit, int32
from pint_pandas import PintType
from nodes.context import unit_registry

from common.i18n import gettext_lazy as _
from nodes import NodeDimension
from nodes.constants import ENERGY_QUANTITY, CURRENCY_QUANTITY, FORECAST_COLUMN, VALUE_COLUMN, UNIT_PRICE_QUANTITY, YEAR_COLUMN
from nodes.calc import nafill_all_forecast_years
from params import NumberParameter

from .action import ActionNode


@njit(cache=True)
def simulate_led_retrofit(
    nr_trad: int, nr_led: int, nr_changed_per_year: int, nr_yearly_increase: int, nr_years: int
):
    trad = np.empty(nr_years, int32)
    led = np.empty(nr_years, int32)
    nr_new_led = np.empty(nr_years, int32)
    for year in range(nr_years):
        change = min(nr_changed_per_year, nr_trad)
        nr_trad -= change
        nr_led += change
        # Assume that the increase in total number of luminaires is
        # only LEDs.
        nr_led += nr_yearly_increase
        nr_new_led[year] = change + nr_yearly_increase
        trad[year] = nr_trad
        led[year] = nr_led
    return nr_new_led, trad, led


class LEDRetrofitAction(ActionNode):
    dimensions = {
        ENERGY_QUANTITY: NodeDimension('MWh/a', ENERGY_QUANTITY),
        CURRENCY_QUANTITY: NodeDimension('EUR/a', CURRENCY_QUANTITY),
    }
    allowed_parameters = [
        NumberParameter(
            local_id='yearly_retrofit_number_baseline',
            label=_('Number of LED bulbs changed per year (baseline)'),
            unit='pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='yearly_retrofit_number',
            label=_('Number of additional LED bulbs changed per year'),
            unit='pcs/a',
        ),
        NumberParameter(
            local_id='yearly_demand_increase',
            label=_('Yearly increase in total number of luminaires'),
            unit='pcs/a',
        ),
        NumberParameter(
            local_id='traditional_luminaire_maintenance_cost',
            label=_('Yearly maintenance cost of traditional luminaires'),
            unit='EUR/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_maintenance_cost',
            label=_('Yearly maintenance cost of LED luminaires'),
            unit='EUR/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='traditional_luminaire_power',
            label=_('Traditional luminaire power consumption'),
            unit='W',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_power',
            label=_('LED luminaire power consumption'),
            unit='W',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='traditional_luminaire_active_time',
            label=_('Traditional luminaire yearly active time'),
            unit='h/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_active_time',
            label=_('LED luminaire yearly active time'),
            unit='h/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_maintenance_cost',
            label=_('Yearly maintenance cost of LED luminaires'),
            unit='EUR/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_investment_cost',
            label=_('Investment cost of one LED retrofit'),
            unit='EUR/pcs',
            is_customizable=False,
        ),
    ]

    def compute_effect(self) -> pd.DataFrame:
        # Input time series are:
        #  - number of luminaires with traditional (high-power) bulbs, historical
        #  - number of luminaires with LED bulbs, historical
        #  - price of electricity, historical + forecast

        trad_df = self.get_input_node(tag='traditional').get_output()
        led_df = self.get_input_node(tag='led').get_output()
        el_price = self.get_input_node(tag='price_of_electricity').get_output()
        target_year = self.get_target_year()

        df = nafill_all_forecast_years(trad_df, target_year)
        df['NrLED'] = nafill_all_forecast_years(led_df, target_year)[VALUE_COLUMN]
        df = df.rename(columns={VALUE_COLUMN: 'NrTraditional'})

        last_hist_year = df.loc[~df[FORECAST_COLUMN]].index.max()
        yearly_baseline_change = self.get_parameter_value('yearly_retrofit_number_baseline', required=False)
        if yearly_baseline_change is None:
            yearly_baseline_change = 0
        yearly_change = self.get_parameter_value('yearly_retrofit_number')
        if not self.is_enabled():
            # If the action is disabled, we assume that only the baseline amount
            # of retrofits are done.
            yearly_change = yearly_baseline_change

        nr_trad = int(self.strip_units(df['NrTraditional']).loc[last_hist_year])
        nr_led = int(self.strip_units(df['NrLED']).loc[last_hist_year])
        el_price = el_price[VALUE_COLUMN]

        # Predict the number of:
        #  - new LED luminaires installed (retrofits + yearly increase)
        #  - traditional luminaires left
        #  - LED luminaires
        nr_new_led, trad, led = simulate_led_retrofit(
            nr_trad=nr_trad,
            nr_led=nr_led,
            nr_changed_per_year=yearly_baseline_change + yearly_change,
            nr_yearly_increase=self.get_parameter_value('yearly_demand_increase'),
            nr_years=target_year - last_hist_year,
        )
        df.loc[df.index > last_hist_year, 'NrTraditional'] = trad
        df.loc[df.index > last_hist_year, 'NrLED'] = led
        df.loc[df.index > last_hist_year, 'NrNewLED'] = nr_new_led
        df['NrNewLED'] = df['NrNewLED'].astype('pint[pcs/a]')

        # Calculate energy consumption, energy cost and maintenance cost
        # for traditional luminaires
        active_time = self.get_parameter_value_w_unit('traditional_luminaire_active_time')
        power = self.get_parameter_value_w_unit('traditional_luminaire_power')
        maint_cost = self.get_parameter_value_w_unit('traditional_luminaire_maintenance_cost')
        df['TraditionalEnergy'] = df['NrTraditional'] * active_time * power
        df['TraditionalEnergy'] = df['NrTraditional'] * active_time * power
        df['TraditionalEnergyCost'] = (df['TraditionalEnergy'] * el_price).astype('pint[EUR/a]')
        df['TraditionalMaintenanceCost'] = (df['NrTraditional'] * maint_cost).astype('pint[EUR/a]')

        # Ditto for LEDs, but include yearly investment costs
        active_time = self.get_parameter_value_w_unit('led_luminaire_active_time')
        power = self.get_parameter_value_w_unit('led_luminaire_power')
        maint_cost = self.get_parameter_value_w_unit('led_luminaire_maintenance_cost')
        inv_cost = self.get_parameter_value_w_unit('led_luminaire_investment_cost')
        df['LEDEnergy'] = df['NrLED'] * active_time * power
        df['LEDEnergyCost'] = (df['LEDEnergy'] * el_price).astype('pint[EUR/a]')
        df['LEDMaintenanceCost'] = (df['NrLED'] * maint_cost).astype('pint[EUR/a]')
        df['LEDInvestmentCost'] = (df['NrNewLED'] * inv_cost)

        total_cost = (
            df['TraditionalEnergyCost'] + df['TraditionalMaintenanceCost']
            + df['LEDEnergyCost'] + df['LEDMaintenanceCost'] + df['LEDInvestmentCost']
        )
        energy_consumption = df['TraditionalEnergy'] + df['LEDEnergy']
        df[CURRENCY_QUANTITY] = total_cost.astype(PintType(self.dimensions[CURRENCY_QUANTITY].unit))
        df[ENERGY_QUANTITY] = energy_consumption.astype(PintType(self.dimensions[ENERGY_QUANTITY].unit))
        df = df[[CURRENCY_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN]]
        return df

############################################


class BuildingEnergySavingAction(ActionNode):
    '''NOTE! The output values are given per TOTAL building floor area,
    not per RENOVATEABLE building floor area. This is necessary because
    the costs from renovations at different timepoints are summed up and
    therefore the renovation rate and cost at a particular year refer to different things.'''
    dimensions = {
        VALUE_COLUMN: NodeDimension('%', 'fraction'),
        'RenovCost': NodeDimension('EUR/a/m**2', 'currency'),
        'HeSaving': NodeDimension('kWh/a/m**2', 'energy_per_area'),
        'ElSaving': NodeDimension('kWh/a/m**2', 'energy_per_area')
    }
    allowed_parameters = [
        NumberParameter(
            local_id='investment_lifetime',
            label=_('Investment lifetime (a)'),
            unit='a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='investment_cost',
            label=_('Investment cost (EUR/m2)'),
            unit='EUR/m**2',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='maintenance_cost',
            label=_('Maintenance cost (EUR/m2/a)'),
            unit='EUR/m**2/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='heat_saving',
            label=_('Heat saving (kWh/m2/a'),
            unit='kWh/m**2/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='electricity_saving',
            label=_('Electricity saving (kWh/m2/a)'),
            unit='kWh/m**2/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='renovation_potential',
            label=_('Renovation potential (% of floor area)'),
            unit='%',
            is_customizable=False,
        ),
    ]

    def compute_effect(self) -> pd.DataFrame:

        def add_with_timeshift(s, activity):  # Activity must be indexed by YEAR_COLUMN
            s_unit = s[0].units
            activity_unit = activity.loc[min(activity.index)].units
            out_unit = str(s_unit * activity_unit)

            index = range(min(activity.index), min(activity.index) + len(activity.index) + len(s))
            out = pd.Series([0.] * len(index), index=index)
            for shift in activity.index:
                new_index = range(shift, shift + len(s))
                increment = [i.m for i in s]
                increment = pd.Series(increment, index=new_index)
                increment = increment * activity.loc[shift].m  # FIXME The treatment of activity units is not foolproof
                out = out.add(increment, fill_value=0)
            out = pd.Series(out, dtype='pint[' + out_unit + ']')
            return out

        # Global parameters
        renovation_rate_baseline = self.context.get_parameter_value_w_unit('renovation_rate_baseline')
        target_year = self.context.target_year
        current_year = self.context.instance.maximum_historical_year

        # Local parameters
        lifetime = self.get_parameter_value_w_unit('investment_lifetime')
        renovation_potential = self.get_parameter_value_w_unit('renovation_potential')
        investment_cost = self.get_parameter_value_w_unit('investment_cost')
        maint_cost = self.get_parameter_value_w_unit('maintenance_cost')
        he_saving = self.get_parameter_value_w_unit('heat_saving')
        el_saving = self.get_parameter_value_w_unit('electricity_saving')

        # Calculations
        year = range(current_year, target_year)
        reno = 1 / lifetime
        if not self.is_enabled():
            reno = renovation_rate_baseline * unit_registry('0.01/%')  # dtype is unable to handle %**2
        reno *= renovation_potential
        reno_unit = str(reno.units)
        df = pd.DataFrame({
            YEAR_COLUMN: year,
            FORECAST_COLUMN: [False] + [True] * (len(year) - 1),
            VALUE_COLUMN: pd.Series([reno.m] * len(year), dtype='pint[' + reno_unit + ']'),
        }).set_index(['Year'])
        df.loc[~df[FORECAST_COLUMN], VALUE_COLUMN] *= 0
        renovcost = [maint_cost * unit_registry('0.01 a/%')] * lifetime.m
        for i in range(lifetime.m):
            if i % lifetime.m == 0:
                renovcost[i] += investment_cost
        df['RenovCost'] = add_with_timeshift(renovcost, df[VALUE_COLUMN])

        el_saving = [el_saving * unit_registry('0.01 a/%')] * lifetime.m
        df['ElSaving'] = add_with_timeshift(el_saving, df[VALUE_COLUMN])
        he_saving = [he_saving * unit_registry('0.01 a/%')] * lifetime.m
        df['HeSaving'] = add_with_timeshift(he_saving, df[VALUE_COLUMN])
        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])
        return df
