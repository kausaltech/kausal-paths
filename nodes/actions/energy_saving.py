import pandas as pd
import numpy as np
from numba import njit, int32
from pint_pandas import PintType

from common.i18n import gettext_lazy as _
from nodes import NodeDimension
from nodes.constants import ENERGY_QUANTITY, CURRENCY_QUANTITY, FORECAST_COLUMN, VALUE_COLUMN
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
