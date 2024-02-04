import typing

import pandas as pd
import numpy as np
import pint_pandas
import polars as pl

from numba import njit, int32, types as nbt
import numba as nb
from pint_pandas import PintType

from common.i18n import gettext_lazy as _
from nodes.metric import NodeMetric
from nodes.constants import ENERGY_QUANTITY, CURRENCY_QUANTITY, FORECAST_COLUMN, VALUE_COLUMN, UNIT_PRICE_QUANTITY, YEAR_COLUMN, DEFAULT_METRIC
from nodes.calc import nafill_all_forecast_years
from params import Parameter, NumberParameter, StringParameter
from params.utils import sep_unit_pt
from common import polars as ppl

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
    output_metrics = {
        ENERGY_QUANTITY: NodeMetric('MWh/a', ENERGY_QUANTITY),
        CURRENCY_QUANTITY: NodeMetric('EUR/a', CURRENCY_QUANTITY),
    }
    allowed_parameters = [
        NumberParameter(
            local_id='yearly_retrofit_number_baseline',
            label=_('Number of LED bulbs changed per year (baseline)'),
            unit_str='pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='yearly_retrofit_number',
            label=_('Number of additional LED bulbs changed per year'),
            unit_str='pcs/a',
        ),
        NumberParameter(
            local_id='yearly_demand_increase',
            label=_('Yearly increase in total number of luminaires'),
            unit_str='pcs/a',
        ),
        NumberParameter(
            local_id='traditional_luminaire_maintenance_cost',
            label=_('Yearly maintenance cost of traditional luminaires'),
            unit_str='EUR/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_maintenance_cost',
            label=_('Yearly maintenance cost of LED luminaires'),
            unit_str='EUR/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='traditional_luminaire_power',
            label=_('Traditional luminaire power consumption'),
            unit_str='W',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_power',
            label=_('LED luminaire power consumption'),
            unit_str='W',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='traditional_luminaire_active_time',
            label=_('Traditional luminaire yearly active time'),
            unit_str='h/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_active_time',
            label=_('LED luminaire yearly active time'),
            unit_str='h/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_maintenance_cost',
            label=_('Yearly maintenance cost of LED luminaires'),
            unit_str='EUR/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_investment_cost',
            label=_('Investment cost of one LED retrofit'),
            unit_str='EUR/pcs',
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
        model_end_year = self.get_model_end_year()

        df = nafill_all_forecast_years(trad_df, model_end_year)
        df['NrLED'] = nafill_all_forecast_years(led_df, model_end_year)[VALUE_COLUMN]
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
            nr_years=model_end_year - last_hist_year,
        )
        df.loc[df.index > last_hist_year, 'NrTraditional'] = trad
        df.loc[df.index > last_hist_year, 'NrLED'] = led
        df.loc[df.index > last_hist_year, 'NrNewLED'] = nr_new_led
        df['NrNewLED'] = df['NrNewLED'].astype('pint[pcs/a]')

        # Calculate energy consumption, energy cost and maintenance cost
        # for traditional luminaires
        active_time = self.get_parameter_value('traditional_luminaire_active_time', units=True)
        power = self.get_parameter_value('traditional_luminaire_power', units=True)
        maint_cost = self.get_parameter_value('traditional_luminaire_maintenance_cost', units=True)
        df['TraditionalEnergy'] = df['NrTraditional'] * active_time * power
        df['TraditionalEnergy'] = df['NrTraditional'] * active_time * power
        df['TraditionalEnergyCost'] = (df['TraditionalEnergy'] * el_price).astype('pint[EUR/a]')
        df['TraditionalMaintenanceCost'] = (df['NrTraditional'] * maint_cost).astype('pint[EUR/a]')  # type: ignore

        # Ditto for LEDs, but include yearly investment costs
        active_time = self.get_parameter_value('led_luminaire_active_time', units=True)
        power = self.get_parameter_value('led_luminaire_power', units=True)
        maint_cost = self.get_parameter_value('led_luminaire_maintenance_cost', units=True)
        inv_cost = self.get_parameter_value('led_luminaire_investment_cost', units=True)
        df['LEDEnergy'] = df['NrLED'] * active_time * power
        df['LEDEnergyCost'] = (df['LEDEnergy'] * el_price).astype('pint[EUR/a]')
        df['LEDMaintenanceCost'] = (df['NrLED'] * maint_cost).astype('pint[EUR/a]')  # type: ignore
        df['LEDInvestmentCost'] = (df['NrNewLED'] * inv_cost)

        total_cost = (
            df['TraditionalEnergyCost'] + df['TraditionalMaintenanceCost']
            + df['LEDEnergyCost'] + df['LEDMaintenanceCost'] + df['LEDInvestmentCost']
        )
        energy_consumption = df['TraditionalEnergy'] + df['LEDEnergy']
        df[CURRENCY_QUANTITY] = total_cost.astype(PintType(self.output_metrics[CURRENCY_QUANTITY].unit))
        df[ENERGY_QUANTITY] = energy_consumption.astype(PintType(self.output_metrics[ENERGY_QUANTITY].unit))
        df = df[[CURRENCY_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN]]
        return df

############################################


class BuildingEnergyParams(typing.NamedTuple):
    start_year: int
    nr_years: int
    lifetime: int
    renovation_rate: float
    renovation_potential: float
    investment_cost: float
    maint_cost: float
    he_saving: float
    el_saving: float
    all_in_investment: bool


class BuildingEnergyRet(typing.NamedTuple):
    year: np.ndarray
    forecast: np.ndarray
    total_renovated: np.ndarray
    cost: np.ndarray
    he_saving: np.ndarray
    el_saving: np.ndarray


def named_tuple_to_nb(cls: typing.Type[typing.NamedTuple]):
    nb_types = [nb.typeof(x()) for x in typing.get_type_hints(cls).values()]
    nb_param = nbt.NamedTuple(nb_types, cls)
    return nb_param


@njit((named_tuple_to_nb(BuildingEnergyParams),), cache=True)
def simulate_building_energy_saving(params: BuildingEnergyParams):
    years = np.arange(params.start_year, params.start_year + params.nr_years)
    total_renovated = np.zeros(params.nr_years, dtype=float)
    renovated_per_year = np.zeros(params.nr_years, dtype=float)
    cost = np.zeros(params.nr_years, dtype=float)
    he_saving = np.zeros(params.nr_years, dtype=float)
    el_saving = np.zeros(params.nr_years, dtype=float)
    forecast = np.zeros(params.nr_years, dtype='int')
    new_renovations = params.renovation_rate

    for i in range(params.nr_years):
        if i:
            val = total_renovated[i - 1]
            reinvestment_round = i // params.lifetime
            if reinvestment_round:
                val -= renovated_per_year[i - params.lifetime]
                if params.all_in_investment:
                    new_renovations *= 0
            val += new_renovations
            if val > params.renovation_potential:
                val = params.renovation_potential

            total_renovated[i] = val
            renovated_per_year[i] = val - total_renovated[i - 1]
            if val < total_renovated[i - 1]:
                renovated_per_year[i] *= 0
            forecast[i] = 1
        else:
            total_renovated[i] = i * params.renovation_rate
            renovated_per_year[i] = total_renovated[i]
            forecast[i] = 0

        cost[i] = renovated_per_year[i] * params.investment_cost
        cost[i] += total_renovated[i] * params.maint_cost
        he_saving[i] = -total_renovated[i] * params.he_saving
        el_saving[i] = -total_renovated[i] * params.el_saving

    return BuildingEnergyRet(
        year=years, forecast=forecast, total_renovated=total_renovated, cost=cost,
        he_saving=he_saving, el_saving=el_saving
    )


class BuildingEnergySavingAction(ActionNode):
    """
    Action that has an energy saving effect on building stock (per floor area).

    The output values are given per TOTAL building floor area,
    not per RENOVATEABLE building floor area. This is useful because
    the costs and savings from total renovations sum up to a meaningful
    impact on nodes that are given per floor area.
    """

    output_metrics = {
        DEFAULT_METRIC: NodeMetric('%', 'fraction', column_id=VALUE_COLUMN),
        'RenovCost': NodeMetric('SEK/a/m**2', 'currency'),
        'Heat': NodeMetric('kWh/a/m**2', 'energy_per_area'),
        'Electricity': NodeMetric('kWh/a/m**2', 'energy_per_area')
    }
    allowed_parameters: typing.ClassVar[list[Parameter]] = [
        NumberParameter(
            local_id='investment_lifetime',
            label=_('Investment lifetime (a)'),
            unit_str='a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='investment_cost',
            label=_('Investment cost (SEK/m2)'),
            unit_str='SEK/m**2',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='maintenance_cost',
            label=_('Maintenance cost (SEK/m2/a)'),
            unit_str='SEK/m**2/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='heat_saving',
            label=_('Heat saving (kWh/m2/a)'),
            unit_str='kWh/m**2/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='electricity_saving',
            label=_('Electricity saving (kWh/m2/a)'),
            unit_str='kWh/m**2/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='renovation_potential',
            label=_('Renovation potential (% of floor area)'),
            unit_str='%',
            is_customizable=False,
        ),
    ]
    global_parameters: list[str] = ActionNode.global_parameters + [
        'renovation_rate_baseline', 'all_in_investment',
    ]

    def compute_effect(self) -> pd.DataFrame:
        # Global parameters
        renovation_rate_param = self.get_global_parameter_value('renovation_rate_baseline', units=True)
        renovation_rate_baseline = renovation_rate_param.to('1/a').m
        model_end_year = self.context.model_end_year
        all_in_investment = self.get_global_parameter_value('all_in_investment')
        current_year = self.context.instance.maximum_historical_year
        assert current_year is not None

        # Local parameters
        lifetime = self.get_parameter_value('investment_lifetime', units=True).to('a').m
        renovation_potential_param = self.get_parameter_value('renovation_potential', units=True)
        renovation_potential: float = renovation_potential_param.to('dimensionless').m  # type: ignore
        investment_cost = self.get_parameter_value('investment_cost', units=True)
        maint_cost = self.get_parameter_value('maintenance_cost', units=True)
        he_saving = self.get_parameter_value('heat_saving', units=True)
        el_saving = self.get_parameter_value('electricity_saving', units=True)

        cost_pt = pint_pandas.PintType(maint_cost.units)
        he_pt = pint_pandas.PintType(he_saving.units)
        el_pt = pint_pandas.PintType(el_saving.units)

        # Calculations
        if self.is_enabled():
            if all_in_investment:  # Renovate everything in one year
                renovation_rate = 1.0
            else:
                renovation_rate = 1 / lifetime
        else:
            renovation_rate = renovation_rate_baseline

        params = BuildingEnergyParams(
            start_year=current_year,
            nr_years=model_end_year - current_year + 1,
            lifetime=int(lifetime),
            renovation_rate=renovation_rate,
            renovation_potential=renovation_potential,
            investment_cost=investment_cost.m,
            maint_cost=maint_cost.m,
            he_saving=he_saving.m,
            el_saving=el_saving.m,
            all_in_investment=all_in_investment,
        )

        ret = simulate_building_energy_saving(params)

        cols = {
            VALUE_COLUMN: pint_pandas.PintArray(ret.total_renovated * 100, '%'),
            'RenovCost': pint_pandas.PintArray(ret.cost, cost_pt),
            'Heat': pint_pandas.PintArray(ret.he_saving, he_pt),
            'Electricity': pint_pandas.PintArray(ret.el_saving, el_pt),
            'Forecast': ret.forecast.astype(bool),
        }
        df = pd.DataFrame(cols, index=ret.year)
        df.index.name = YEAR_COLUMN
        return df

    def compute_effect_old(self) -> pd.DataFrame:
        # Global parameters
        renovation_rate_baseline = self.context.get_parameter_value('renovation_rate_baseline', units=True)
        renovation_rate_baseline = renovation_rate_baseline.to('1/a').m
        model_end_year = self.context.model_end_year
        current_year = self.context.instance.maximum_historical_year

        # Local parameters
        lifetime = self.get_parameter_value('investment_lifetime', units=True).to('a').m
        renovation_potential = self.get_parameter_value('renovation_potential', units=True)
        renovation_potential = renovation_potential.to('dimensionless').m
        investment_cost = self.get_parameter_value('investment_cost', units=True)
        maint_cost = self.get_parameter_value('maintenance_cost', units=True)
        cost_pt = pint_pandas.PintType(maint_cost.units)
        he_saving = self.get_parameter_value('heat_saving', units=True)
        he_pt = pint_pandas.PintType(he_saving.units)
        el_saving = self.get_parameter_value('electricity_saving', units=True)
        el_pt = pint_pandas.PintType(el_saving.units)

        # Calculations
        reno = 1 / lifetime
        if not self.is_enabled():
            reno = renovation_rate_baseline

        df = pd.DataFrame({
            VALUE_COLUMN: range(model_end_year - current_year + 1),
        }, index=range(current_year, model_end_year + 1))
        df.index.name = YEAR_COLUMN
        df[FORECAST_COLUMN] = df.index > current_year
        df[VALUE_COLUMN] = (df[VALUE_COLUMN] * reno).clip(None, renovation_potential)
        cost = df[VALUE_COLUMN].copy()

        # Reinvestments after renovation potential reached
        for round in range(1, len(df.index) // lifetime):
            s = df[VALUE_COLUMN].shift(lifetime * round, fill_value=0)
            cost += s

        cost = cost.diff().fillna(0)
        cost = cost * investment_cost.m
        cost += df[VALUE_COLUMN] * maint_cost.m
        df['RenovCost'] = cost.astype(cost_pt)
        df['Heat'] = (df[VALUE_COLUMN] * he_saving.m * -1).astype(he_pt)
        df['Electricity'] = (df[VALUE_COLUMN] * el_saving.m * -1).astype(el_pt)
        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN] * 100)

        return df


class BuildingEnergySavingActionUs(BuildingEnergySavingAction):
    """
    BuildingEnergySavingAction with U.S. units and natural gas instead of heat.
    """

    output_metrics = {
        VALUE_COLUMN: NodeMetric('%', 'fraction'),
        'RenovCost': NodeMetric('USD/a/ft**2', 'currency'),
        'natural_gas': NodeMetric('thm/a/ft**2', 'energy_per_area'),
        'electricity': NodeMetric('kWh/a/ft**2', 'energy_per_area')
    }
    allowed_parameters: typing.ClassVar[list[Parameter]] = [
        NumberParameter(
            local_id='investment_lifetime',
            label=_('Investment lifetime (a)'),
            unit_str='a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='investment_cost',
            label=_('Investment cost (USD/square foot)'),
            unit_str='USD/ft**2',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='maintenance_cost',
            label=_('Maintenance cost (USD/square foot/a)'),
            unit_str='USD/ft**2/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='natural_gas_saving',
            label=_('Natural gas saving (thm/square foot/a)'),
            unit_str='thm/ft**2/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='electricity_saving',
            label=_('Electricity saving (kWh/square foot/a)'),
            unit_str='kWh/ft**2/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='renovation_potential',
            label=_('Renovation potential (% of floor area)'),
            unit_str='%',
            is_customizable=False,
        ),
    ]
    global_parameters: list[str] = ActionNode.global_parameters + [
        'renovation_rate_baseline', 'all_in_investment',
    ]

    def compute_effect(self) -> pd.DataFrame:
        # Global parameters
        renovation_rate_param = self.get_global_parameter_value('renovation_rate_baseline', units=True)
        renovation_rate_baseline = renovation_rate_param.to('1/a').m
        model_end_year = self.context.model_end_year
        all_in_investment = self.get_global_parameter_value('all_in_investment')
        current_year = self.context.instance.maximum_historical_year
        assert current_year is not None

        # Local parameters
        lifetime = self.get_parameter_value('investment_lifetime', units=True).to('a').m
        renovation_potential_param = self.get_parameter_value('renovation_potential', units=True)
        renovation_potential: float = renovation_potential_param.to('dimensionless').m  # type: ignore
        investment_cost = self.get_parameter_value('investment_cost', units=True)
        maint_cost = self.get_parameter_value('maintenance_cost', units=True)
        he_saving = self.get_parameter_value('natural_gas_saving', units=True)
        el_saving = self.get_parameter_value('electricity_saving', units=True)

        cost_pt = pint_pandas.PintType(maint_cost.units)
        he_pt = pint_pandas.PintType(he_saving.units)
        el_pt = pint_pandas.PintType(el_saving.units)

        # Calculations
        if self.is_enabled():
            if all_in_investment:  # Renovate everything in one year
                renovation_rate = 1.0
            else:
                renovation_rate = 1 / lifetime
        else:
            renovation_rate = renovation_rate_baseline

        params = BuildingEnergyParams(
            start_year=current_year,
            nr_years=model_end_year - current_year + 1,
            lifetime=int(lifetime),
            renovation_rate=renovation_rate,
            renovation_potential=renovation_potential,
            investment_cost=investment_cost.m,
            maint_cost=maint_cost.m,
            he_saving=he_saving.m,
            el_saving=el_saving.m,
            all_in_investment=all_in_investment,
        )

        ret = simulate_building_energy_saving(params)

        cols = {
            VALUE_COLUMN: pint_pandas.PintArray(ret.total_renovated * 100, '%'),
            'RenovCost': pint_pandas.PintArray(ret.cost, cost_pt),
            'natural_gas': pint_pandas.PintArray(ret.he_saving, he_pt),
            'electricity': pint_pandas.PintArray(ret.el_saving, el_pt),
            'Forecast': ret.forecast.astype(bool),
        }
        df = pd.DataFrame(cols, index=ret.year)
        df.index.name = YEAR_COLUMN
        return df


class CfFloorAreaAction(BuildingEnergySavingAction):
    """
    BuildingEnergySavingAction with cumulative energy savings.
    # fraction of existing buildings triggering code updates
    # compliance of new buildings to the more active regulations
    # improvement in energy consumption factor
    """

    output_metrics = {
        'triggered': NodeMetric('%', 'fraction', column_id='triggered'),
        'compliant': NodeMetric('%', 'fraction', column_id='compliant'),
        'improvement': NodeMetric('%', 'fraction', column_id='improvement'),
#        'electricity': NodeMetric('kWh/m**2/a', 'consumption_factor', column_id='electricity'),
#        'natural_gas': NodeMetric('thm/m**2/a', 'consumption_factor', column_id='natural_gas')
    }
    allowed_parameters = BuildingEnergySavingAction.allowed_parameters + [
        StringParameter(
            local_id='electricity_unit',
            label='Electricity unit',
            is_customizable=False
            ),
        StringParameter(
            local_id='natural_gas_unit',
            label='Natural gas unit',
            is_customizable=False
        )
    ]
    
    def compute_effect(self) -> pd.DataFrame:

        df = self.get_input_dataset_pl(tag='floor', required=True)
        assert 'action_change' in df.primary_keys
        triggered = df.filter(pl.col('action_change').eq('triggered'))
        triggered = triggered.rename({'fraction': 'triggered'}).drop('action_change')
        compliant = df.filter(pl.col('action_change').eq('compliant'))
        compliant = compliant.rename({'fraction': 'compliant'}).drop('action_change')
        df = triggered.paths.join_over_index(compliant)

        df2 = self.get_input_dataset_pl(tag='improvement', required=True)
        df2 = df2.rename({'fraction': 'improvement'})

        unit_el = self.get_parameter_value('electricity_unit', units=False, required=False)
        if unit_el is not None:
            self.output_metrics['electricity'].default_unit = unit_el
            self.output_metrics['electricity'].populate_unit(context=self.context)

        unit_gas = self.get_parameter_value('natural_gas_unit', units=False, required=False)
        if unit_gas is not None:
            self.output_metrics['natural_gas'].default_unit = unit_gas
            self.output_metrics['natural_gas'].populate_unit(context=self.context)

        df = df2.paths.join_over_index(df, index_from='union')

        if not self.is_enabled():
            df = df.with_columns(pl.lit(0.0).alias('compliant'))
            if 'electricity' in df.columns:  # If absolute units
                df = df.with_columns(pl.lit(0.0).alias('electricity'))
                df = df.with_columns(pl.lit(0.0).alias('natural_gas'))
            else:
                df = df.with_columns(pl.lit(0.0).alias('improvement'))

        return df


class EnergyAction(ActionNode):
    """Simple action with several energy metrics."""

    output_metrics = {
        'electricity': NodeMetric('kWh/ft**2/a', 'fraction', column_id='electricity'),
        'natural_gas': NodeMetric('thm/ft**2/a', 'fraction', column_id='natural_gas'),
    }

    def compute_effect(self):
        df = self.get_input_dataset_pl(required=True)
        df = df.with_columns([
            (pl.lit(-1) * pl.col('electricity')).alias('electricity'),
            (pl.lit(-1) * pl.col('natural_gas')).alias('natural_gas')
        ])
        if not self.is_enabled():
            df = df.with_columns([
                pl.lit(0.0).alias('electricity'),
                pl.lit(0.0).alias('natural_gas')
            ])

        return df
