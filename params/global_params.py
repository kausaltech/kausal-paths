from __future__ import annotations

from common.i18n import gettext_lazy as _

from .param import BoolParameter, NumberParameter, StringParameter


class MunicipalityName(StringParameter):
    name = _('Municipality name')
    id = 'municipality_name'


class DiscountRate(NumberParameter):
    name = _('Discount rate')
    id = 'discount_rate'


class PopulationGrowthRate(NumberParameter):
    name = _('Population growth rate')
    id = 'population_growth_rate'


class AvoidedElectricityCapacityPrice(NumberParameter):
    name = _('Avoided electricity capacity price')
    id = 'avoided_electricity_capacity_price'


class HealthImpactsPerKwh(NumberParameter):
    name = _('Health impacts per kWh')
    id = 'health_impacts_per_kwh'


class HeatCo2Ef(NumberParameter):
    name = _('Heat CO2 emission factor')
    id = 'heat_co2_ef'


class ElectricityCo2Ef(NumberParameter):
    name = _('Electricity CO2 emission factor')
    id = 'electricity_co2_ef'


class RenovationRateBaseline(NumberParameter):
    name = _('Renovation rate baseline')
    id = 'renovation_rate_baseline'


class IncludeSocial(BoolParameter):
    name = _('Include energy taxes in calculations?')
    id = 'include_energy_taxes'


class IncludeCO2(BoolParameter):
    name = _('Include CO2 cost variable in calculations?')
    id = 'include_co2'


class IncludeHealth(BoolParameter):
    name = _('Include health impact variable in calculations?')
    id = 'include_health'


class IncludeElAvoided(BoolParameter):
    name = _('Include avoided electricity capacity variable in calculations?')
    id = 'include_el_avoided'


class PriceOfCo2(NumberParameter):
    name = _('Price of CO2')
    id = 'price_of_co2'


class PriceOfCo2AnnualChange(NumberParameter):
    name = _('Price of CO2 annual change')
    id = 'price_of_co2_annual_change'


class PriceOfElectricity(NumberParameter):
    name = _('Price of electricity')
    id = 'price_of_electricity'


class PriceOfElectricityAnnualChange(NumberParameter):
    name = _('Price of electricity annual change')
    id = 'price_of_electricity_annual_change'


class PriceOfHeat(NumberParameter):
    name = _('Price of Heat')
    id = 'price_of_heat'


class PriceOfHeatAnnualChange(NumberParameter):
    name = _('Price of heat annual change')
    id = 'price_of_heat_annual_change'


class AllInInvestment(BoolParameter):
    name = _('Invest all on the first year (in contrast to continuous investing)?')
    id = 'all_in_investment'


class Placeholder(BoolParameter):
    name = _('Placeholder for updated_building_code_residential')
    id = 'placeholder'


class ActionImpactFromBaseline(BoolParameter):
    name = _('Action impact based on baseline')
    description = _('Compute action impact based on the baseline scenario instead of the default one')
    id = 'action_impact_from_baseline'
    value = False


class ScenarioName(StringParameter):
    name = _('Scanario name')
    id = 'scenario_name'


class EmissionsWeight(NumberParameter):
    name = _('Weight for emission impacts in value profiles')
    id = 'emissions_weight'


class EconomicWeight(NumberParameter):
    name = _('Weight for economic impacts in value profiles')
    id = 'economic_weight'


class ProsperityWeight(NumberParameter):
    name = _('Weight for prosperity impacts (e.g. jobs) in value profiles')
    id = 'prosperity_weight'


class PurityWeight(NumberParameter):
    name = _('Weight for purity impacts (e.g. lack of pollution) in value profiles')
    id = 'purity_weight'


class HealthWeight(NumberParameter):
    name = _('Weight for health impacts in value profiles')
    id = 'health_weight'


class EquityWeight(NumberParameter):
    name = _('Weight for equity impacts in value profiles')
    id = 'equity_weight'


class BiodiversityWeight(NumberParameter):
    name = _('Weight for biodiversity impacts in value profiles')
    id = 'biodiversity_weight'


class LegalityWeight(NumberParameter):
    name = _('Weight for actions to be legal in value profiles')
    id = 'legality_weight'


class ImpactThreshold(NumberParameter):
    name = _('Threshold for sum of weighted impacts in value profiles')
    id = 'impact_threshold'


class EffectOfEV(NumberParameter):
    name = _('Effect of electric vehicles on car kilometers')
    id = 'effect_of_ev'


class SelectedMunicipalities(StringParameter):
    name = _('List of selected municipalities')
    id = 'selected_municipalities'


class SelectedNodes(StringParameter):
    name = _('List of selected nodes')
    id = 'selected_nodes'


class SelectedFramework(StringParameter):
    name = _('Selected computational framework')
    id = 'selected_framework'


class WeatherNormalization(BoolParameter):
    name = _('Annual weather normalization for energy consumption')
    id = 'weather_correction'


class BiogenicInclusion(BoolParameter):
    name = _('Are biogenic emissions included in total emissions?')
    id = 'biogenic_inclusion'


class UseMileage(BoolParameter):
    name = _('Use mileage rather than energy for transport')
    id = 'use_mileage'


class UseNationalValues(BoolParameter):
    name = _('Use national values rather than local values')
    id = 'use_national_values'


class StatisticalNormalization(BoolParameter):
    name = _('Normalization for matching statistical data')
    id = 'statistical_correction'


class ExtendHistoricalValues(BoolParameter):
    name = _('Is the last historical value extended into the future?')
    id = 'extend_historical_values'


class ShowScenarioImpacts(BoolParameter):
    name = _('On node graphs, show scenario impacts instead of outputs?')
    id = 'show_scenario_impacts'


class MeasureDataOverride(BoolParameter):
    name = _('Override input node values with framework measure data')
    id = 'measure_data_override'


class MeasureDataBaselineYearOnly(BoolParameter):
    name = _('Use only baseline year from measure data')
    id = 'measure_data_baseline_year_only'


class ElectrificationScenario(NumberParameter):
    name = _('Transportation electrification scenario')
    id = 'electrification_scenario'


class PopulationSize(NumberParameter):
    name = _('Population size')
    id = 'population_size'


class SelectedNumber(NumberParameter):
    name = _('Number of the selected item')
    id = 'selected_number'
