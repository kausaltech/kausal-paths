from enum import Enum

FORECAST_COLUMN = 'Forecast'
YEAR_COLUMN = 'Year'
VALUE_COLUMN = 'Value'

VALUE_WITH_ACTION_GROUP = 'Action'
VALUE_WITHOUT_ACTION_GROUP = 'WithoutAction'
IMPACT_GROUP = 'Impact'


FORECAST_x = 'Forecast_x'
FORECAST_y = 'Forecast_y'
VALUE_x = 'Value_x'
VALUE_y = 'Value_y'

EMISSION_UNIT = 'kg'
BASELINE_VALUE_COLUMN = 'BaselineValue'

#
# Quantities
#
EMISSION_QUANTITY = 'emissions'
ENERGY_QUANTITY = 'energy'
MILEAGE_QUANTITY = 'mileage'
EMISSION_FACTOR_QUANTITY = 'emission_factor'
CURRENCY_QUANTITY = 'currency'
UNIT_PRICE_QUANTITY = 'unit_price'
FLOOR_AREA_QUANTITY = 'floor_area'
NUMBER_QUANTITY = 'number'
PER_CAPITA_QUANTITY = 'per_capita'
POPULATION_QUANTITY = 'quantity'
ACTIVITY_QUANTITIES = [EMISSION_QUANTITY, ENERGY_QUANTITY, MILEAGE_QUANTITY, 'mass']

KNOWN_QUANTITIES = ACTIVITY_QUANTITIES + [
    EMISSION_FACTOR_QUANTITY, CURRENCY_QUANTITY, NUMBER_QUANTITY, UNIT_PRICE_QUANTITY,
    PER_CAPITA_QUANTITY, FLOOR_AREA_QUANTITY,
    'population', 'per_capita', 'fuel_consumption', 'ratio',
    'exposure', 'exposure-response', 'disease_burden', 'case_burden',
    'mass', 'consumption', 'mass_concentration', 'body_weight', 'incidence', 'fraction',
    'probability', 'ingestion', 'energy_per_area', 'area',
]

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
    elif quantity == EMISSION_FACTOR_QUANTITY:
        return '✖'
    elif quantity == POPULATION_QUANTITY:
        return '👪'
    return None

