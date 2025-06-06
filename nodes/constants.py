from __future__ import annotations

from enum import Enum

BASELINE_SCENARIO = 'baseline'

FORECAST_COLUMN = 'Forecast'
YEAR_COLUMN = 'Year'
VALUE_COLUMN = 'Value'
NODE_COLUMN = 'Node'
SCENARIO_COLUMN = 'ScenarioName'

# Impact constants
IMPACT_COLUMN = 'Impact'
# Probability iterations for Monte Carlo
UNCERTAINTY_COLUMN = 'iteration'

# Action as it is in the active scenario
SCENARIO_ACTION_GROUP = 'Scenario'
# With action disabled
WITHOUT_ACTION_GROUP = 'WithoutAction'
# Impact of action
IMPACT_GROUP = 'Impact'


# Dimension flow constants
FLOW_ID_COLUMN = 'Flow'
FLOW_ROLE_COLUMN = 'FlowRole'
FLOW_ROLE_SOURCE = 'source'
FLOW_ROLE_TARGET = 'target'

EMISSION_UNIT = 'kg'
BASELINE_VALUE_COLUMN = 'BaselineValue'
TIME_INTERVAL = 'a'

#
# Quantities
#
EMISSION_QUANTITY = 'emissions'
ENERGY_QUANTITY = 'energy'
MILEAGE_QUANTITY = 'mileage'
VEHICLE_MILEAGE_QUANTITY = 'vehicle_mileage'
PASSENGER_MILEAGE_QUANTITY = 'passenger_mileage'
FREIGHT_MILEAGE_QUANTITY = 'freight_mileage'
EMISSION_FACTOR_QUANTITY = 'emission_factor'
ENERGY_FACTOR_QUANTITY = 'energy_factor'
CONSUMPTION_FACTOR_QUANTITY = 'consumption_factor'
CURRENCY_QUANTITY = 'currency'
UNIT_PRICE_QUANTITY = 'unit_price'
FLOOR_AREA_QUANTITY = 'floor_area'
NUMBER_QUANTITY = 'number'
PER_CAPITA_QUANTITY = 'per_capita'
POPULATION_QUANTITY = 'population'
MIX_QUANTITY = 'mix'
GROUPED_MIX_QUANTITY = 'grouped_mix'
UTILITY_QUANTITY = 'utility'
FRACTION_QUANTITY = 'fraction'

ACTIVITY_QUANTITIES = {
    EMISSION_QUANTITY, ENERGY_QUANTITY, MILEAGE_QUANTITY, VEHICLE_MILEAGE_QUANTITY,
    PASSENGER_MILEAGE_QUANTITY, FREIGHT_MILEAGE_QUANTITY, UTILITY_QUANTITY, 'fuel_consumption', 'consumption',
    'mass', 'volume', 'area', 'employment', 'activity',
}

ACTIVITY_FACTOR_QUANTITIES = {
    ENERGY_FACTOR_QUANTITY, CONSUMPTION_FACTOR_QUANTITY, 'energy_per_area', 'occupancy_factor', 'fuel_factor',
    'demand_factor', 'time_factor', 'factor',
}

UNIT_PRICE_QUANTITIES = {
    UNIT_PRICE_QUANTITY, 'energy_unit_price', 'floor_area_unit_price', 'fuel_unit_price', 'mileage_unit_price',
}

STACKABLE_QUANTITIES = ACTIVITY_QUANTITIES | {
    MIX_QUANTITY, GROUPED_MIX_QUANTITY, POPULATION_QUANTITY, FLOOR_AREA_QUANTITY, CURRENCY_QUANTITY,
    NUMBER_QUANTITY, 'area', 'disease_burden', 'health_effect', 'length', 'volume',
}

KNOWN_QUANTITIES = ACTIVITY_QUANTITIES | ACTIVITY_FACTOR_QUANTITIES | UNIT_PRICE_QUANTITIES | STACKABLE_QUANTITIES | {
    EMISSION_FACTOR_QUANTITY, CURRENCY_QUANTITY, NUMBER_QUANTITY, PER_CAPITA_QUANTITY, FLOOR_AREA_QUANTITY,
    MIX_QUANTITY, GROUPED_MIX_QUANTITY, POPULATION_QUANTITY,
    'ratio', 'exposure', 'exposure_response', 'disease_burden', 'case_burden', 'mass_concentration', 'concentration',
    'body_weight', 'incidence', 'fraction', 'probability', 'ingestion', 'area', 'effect', 'health_effect', 'rate',
    'speed', 'argument',
}


DEFAULT_METRIC = 'default'


def ensure_known_quantity(quantity: str):
    if quantity not in KNOWN_QUANTITIES:
        raise Exception(f"Quantity {quantity} is unknown")


class DecisionLevel(Enum):
    MUNICIPALITY = 1
    NATION = 2
    EU = 3


QUANTITY_ICONS = {
    EMISSION_QUANTITY: '💨',
    ENERGY_QUANTITY: '⚡',
    MILEAGE_QUANTITY: '🚗',
    VEHICLE_MILEAGE_QUANTITY: '🚗',
    PASSENGER_MILEAGE_QUANTITY: '🚗',
    FREIGHT_MILEAGE_QUANTITY: '🚗',
    EMISSION_FACTOR_QUANTITY: '✖',
    ENERGY_FACTOR_QUANTITY: '✖',
    CONSUMPTION_FACTOR_QUANTITY: '✖',
    POPULATION_QUANTITY: '👪',
    MIX_QUANTITY: '💯',
    GROUPED_MIX_QUANTITY: '💯',
    CURRENCY_QUANTITY: '💰',
    FRACTION_QUANTITY: '➗',
}


def get_quantity_icon(quantity: str) -> str | None:
    return QUANTITY_ICONS.get(quantity)
