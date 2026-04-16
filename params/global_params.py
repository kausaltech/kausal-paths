from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from django.utils.functional import Promise

from kausal_common.i18n.pydantic import TranslatedString, gettext_lazy as _

from params.param import BoolParameter, NumberParameter, StringParameter

if TYPE_CHECKING:
    from django_stubs_ext import StrOrPromise

    from params.base import Parameter


class GlobalParameter:
    name: ClassVar[StrOrPromise]
    id: ClassVar[str]
    description: ClassVar[StrOrPromise | None] = None
    param_class: ClassVar[type[Parameter]]

    @classmethod
    def create_parameter(cls) -> Parameter:
        name = cls.name
        label: TranslatedString | str
        if isinstance(name, Promise):
            label = TranslatedString.from_lazy_string(name)
        else:
            label = name

        return cls.param_class(
            local_id=cls.id,
            label=label,
            description=cls.description,
        )


class StringGlobalParameter(GlobalParameter):
    param_class = StringParameter


class NumberGlobalParameter(GlobalParameter):
    param_class = NumberParameter


class BoolGlobalParameter(GlobalParameter):
    param_class = BoolParameter


class MunicipalityName(StringGlobalParameter):
    name = _('Municipality name')
    id = 'municipality_name'


class DiscountRate(NumberGlobalParameter):
    name = _('Discount rate')
    id = 'discount_rate'


class PopulationGrowthRate(NumberGlobalParameter):
    name = _('Population growth rate')
    id = 'population_growth_rate'


class AvoidedElectricityCapacityPrice(NumberGlobalParameter):
    name = _('Avoided electricity capacity price')
    id = 'avoided_electricity_capacity_price'


class HealthImpactsPerKwh(NumberGlobalParameter):
    name = _('Health impacts per kWh')
    id = 'health_impacts_per_kwh'


class HeatCo2Ef(NumberGlobalParameter):
    name = _('Heat CO2 emission factor')
    id = 'heat_co2_ef'


class ElectricityCo2Ef(NumberGlobalParameter):
    name = _('Electricity CO2 emission factor')
    id = 'electricity_co2_ef'


class RenovationRateBaseline(NumberGlobalParameter):
    name = _('Renovation rate baseline')
    id = 'renovation_rate_baseline'


class IncludeSocial(BoolGlobalParameter):
    name = _('Include energy taxes in calculations?')
    id = 'include_energy_taxes'


class IncludeCO2(BoolGlobalParameter):
    name = _('Include CO2 cost variable in calculations?')
    id = 'include_co2'


class IncludeHealth(BoolGlobalParameter):
    name = _('Include health impact variable in calculations?')
    id = 'include_health'


class IncludeElAvoided(BoolGlobalParameter):
    name = _('Include avoided electricity capacity variable in calculations?')
    id = 'include_el_avoided'


class PriceOfCo2(NumberGlobalParameter):
    name = _('Price of CO2')
    id = 'price_of_co2'


class PriceOfCo2AnnualChange(NumberGlobalParameter):
    name = _('Price of CO2 annual change')
    id = 'price_of_co2_annual_change'


class PriceOfElectricity(NumberGlobalParameter):
    name = _('Price of electricity')
    id = 'price_of_electricity'


class PriceOfElectricityAnnualChange(NumberGlobalParameter):
    name = _('Price of electricity annual change')
    id = 'price_of_electricity_annual_change'


class PriceOfHeat(NumberGlobalParameter):
    name = _('Price of Heat')
    id = 'price_of_heat'


class PriceOfHeatAnnualChange(NumberGlobalParameter):
    name = _('Price of heat annual change')
    id = 'price_of_heat_annual_change'


class AllInInvestment(BoolGlobalParameter):
    name = _('Invest all on the first year (in contrast to continuous investing)?')
    id = 'all_in_investment'


class Placeholder(BoolGlobalParameter):
    name = _('Placeholder for updated_building_code_residential')
    id = 'placeholder'


class ActionImpactFromBaseline(BoolGlobalParameter):
    name = _('Action impact based on baseline')
    description = _('Compute action impact based on the baseline scenario instead of the default one')
    id = 'action_impact_from_baseline'
    value = False


class ScenarioName(StringGlobalParameter):
    name = _('Scanario name')
    id = 'scenario_name'


class EmissionsWeight(NumberGlobalParameter):
    name = _('Weight for emission impacts in value profiles')
    id = 'emissions_weight'


class EconomicWeight(NumberGlobalParameter):
    name = _('Weight for economic impacts in value profiles')
    id = 'economic_weight'


class ProsperityWeight(NumberGlobalParameter):
    name = _('Weight for prosperity impacts (e.g. jobs) in value profiles')
    id = 'prosperity_weight'


class PurityWeight(NumberGlobalParameter):
    name = _('Weight for purity impacts (e.g. lack of pollution) in value profiles')
    id = 'purity_weight'


class HealthWeight(NumberGlobalParameter):
    name = _('Weight for health impacts in value profiles')
    id = 'health_weight'


class EquityWeight(NumberGlobalParameter):
    name = _('Weight for equity impacts in value profiles')
    id = 'equity_weight'


class BiodiversityWeight(NumberGlobalParameter):
    name = _('Weight for biodiversity impacts in value profiles')
    id = 'biodiversity_weight'


class LegalityWeight(NumberGlobalParameter):
    name = _('Weight for actions to be legal in value profiles')
    id = 'legality_weight'


class ImpactThreshold(NumberGlobalParameter):
    name = _('Threshold for sum of weighted impacts in value profiles')
    id = 'impact_threshold'


class EffectOfEV(NumberGlobalParameter):
    name = _('Effect of electric vehicles on car kilometers')
    id = 'effect_of_ev'


class SelectedMunicipalities(StringGlobalParameter):
    name = _('List of selected municipalities')
    id = 'selected_municipalities'


class SelectedNodes(StringGlobalParameter):
    name = _('List of selected nodes')
    id = 'selected_nodes'


class SelectedFramework(StringGlobalParameter):
    name = _('Selected computational framework')
    id = 'selected_framework'


class WeatherNormalization(BoolGlobalParameter):
    name = _('Annual weather normalization for energy consumption')
    id = 'weather_correction'


class BiogenicInclusion(BoolGlobalParameter):
    name = _('Are biogenic emissions included in total emissions?')
    id = 'biogenic_inclusion'


class UseMileage(BoolGlobalParameter):
    name = _('Use mileage rather than energy for transport')
    id = 'use_mileage'


class UseNationalValues(BoolGlobalParameter):
    name = _('Use national values rather than local values')
    id = 'use_national_values'


class UseLocalValues(BoolGlobalParameter):
    name = _('Use local values rather than national values')
    id = 'use_local_values'


class StatisticalNormalization(BoolGlobalParameter):
    name = _('Normalization for matching statistical data')
    id = 'statistical_correction'


class ExtendHistoricalValues(BoolGlobalParameter):
    name = _('Is the last historical value extended into the future?')
    id = 'extend_historical_values'


class ShowScenarioImpacts(BoolGlobalParameter):
    name = _('On node graphs, show scenario impacts instead of outputs?')
    id = 'show_scenario_impacts'


class MeasureDataOverride(BoolGlobalParameter):
    name = _('Override input node values with framework measure data')
    id = 'measure_data_override'


class MeasureDataBaselineYearOnly(BoolGlobalParameter):
    name = _('Use only baseline year from measure data')
    id = 'measure_data_baseline_year_only'


class ElectrificationScenario(NumberGlobalParameter):
    name = _('Transportation electrification scenario')
    id = 'electrification_scenario'


class PopulationSize(NumberGlobalParameter):
    name = _('Population size')
    id = 'population_size'


class SelectedNumber(NumberGlobalParameter):
    name = _('Number of the selected item')
    id = 'selected_number'


class AGSNumber(StringGlobalParameter):
    name = _('AGS Number')
    id = 'ags_number'


class Flatline(BoolGlobalParameter):
    name = _('Use flatline rather than forecast')
    id = 'flatline'
