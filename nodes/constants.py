from enum import Enum

FORECAST_COLUMN = 'Forecast'
YEAR_COLUMN = 'Year'
VALUE_COLUMN = 'Value'
NODE_COLUMN = 'Node'


# Impact constants
IMPACT_COLUMN = 'Impact'

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

#
# Quantities
#
EMISSION_QUANTITY = 'emissions'
ENERGY_QUANTITY = 'energy'
MILEAGE_QUANTITY = 'mileage'
VEHICLE_MILEAGE_QUANTITY = 'vehicle_mileage'
PASSENGER_MILEAGE_QUANTITY = 'passenger_mileage'
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
VALUE_PROFILE_QUANTITY = 'value_profile'
ACTIVITY_QUANTITIES = set([
    EMISSION_QUANTITY, ENERGY_QUANTITY, MILEAGE_QUANTITY, VEHICLE_MILEAGE_QUANTITY,
    PASSENGER_MILEAGE_QUANTITY, 'mass'
])

STACKABLE_QUANTITIES = ACTIVITY_QUANTITIES | set([
    MIX_QUANTITY, GROUPED_MIX_QUANTITY, POPULATION_QUANTITY, FLOOR_AREA_QUANTITY, CURRENCY_QUANTITY
])

KNOWN_QUANTITIES = ACTIVITY_QUANTITIES | set([
    EMISSION_FACTOR_QUANTITY, CURRENCY_QUANTITY, NUMBER_QUANTITY, UNIT_PRICE_QUANTITY,
    PER_CAPITA_QUANTITY, FLOOR_AREA_QUANTITY, MIX_QUANTITY, CONSUMPTION_FACTOR_QUANTITY,
    VALUE_PROFILE_QUANTITY,
    'population', 'per_capita', 'fuel_consumption', 'ratio',
    'exposure', 'exposure-response', 'disease_burden', 'case_burden',
    'mass', 'consumption', 'mass_concentration', 'body_weight', 'incidence', 'fraction',
    'probability', 'ingestion', 'energy_per_area', 'area', 'occupancy_factor',
    'energy_factor',
])

DEFAULT_METRIC = 'default'


def ensure_known_quantity(quantity: str):
    if quantity not in KNOWN_QUANTITIES:
        raise Exception(f"Quantity {quantity} is unknown")


class DecisionLevel(Enum):
    MUNICIPALITY = 1
    NATION = 2
    EU = 3


def get_quantity_icon(quantity: str) -> str | None:
    if quantity == EMISSION_QUANTITY:
        return '💨'
    elif quantity == ENERGY_QUANTITY:
        return '⚡'
    elif quantity == MILEAGE_QUANTITY:
        return '🚗'
    elif quantity in (EMISSION_FACTOR_QUANTITY, 'energy_factor'):
        return '✖'
    elif quantity == POPULATION_QUANTITY:
        return '👪'
    return None

