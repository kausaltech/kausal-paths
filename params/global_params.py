from .param import BoolParameter, NumberParameter, StringParameter
from common.i18n import gettext_lazy as _


class MunicipalityName(StringParameter):
    name = _('Municipality name')
    id = 'municipality_name'


class DiscountRate(NumberParameter):
    name = _('Discount rate')
    id = 'discount_rate'


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


class CostCo2(NumberParameter):
    name = _('Cost of CO2')
    id = 'cost_co2'


class RenovationRateBaseline(NumberParameter):
    name = _('Renovation rate baseline')
    id = 'renovation_rate_baseline'


class CarbonPriceChange(NumberParameter):
    name = _('Change in carbon emission price')
    id = 'carbon_price_change'


class IncludeCO2(BoolParameter):
    name = _('Include CO2 cost variable in calculations?')
    id = 'include_co2'


class IncludeHealth(BoolParameter):
    name = _('Include health impact variable in calculations?')
    id = 'include_health'


class IncludeElAvoided(BoolParameter):
    name = _('Include avoided electricity capacity variable in calculations?')
    id = 'include_el_avoided'
