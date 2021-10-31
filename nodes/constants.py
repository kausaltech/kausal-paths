from enum import Enum

FORECAST_COLUMN = 'Forecast'
YEAR_COLUMN = 'Year'
VALUE_COLUMN = 'Value'
IMPACT_COLUMN = 'Impact'
EMISSION_UNIT = 'kg'
BASELINE_VALUE_COLUMN = 'BaselineValue'

#
# Quantities
#
EMISSION_QUANTITY = 'emissions'
ENERGY_QUANTITY = 'energy'
MILEAGE_QUANTITY = 'mileage'
EMISSION_FACTOR_QUANTITY = 'emission_factor'
ACTIVITY_QUANTITIES = [EMISSION_QUANTITY, ENERGY_QUANTITY, MILEAGE_QUANTITY, 'mass']

KNOWN_QUANTITIES = ACTIVITY_QUANTITIES + [
    'emission_factor', 'population', 'per_capita', 'fuel_consumption',
    'ratio', 'exposure', 'exposure-response', 'disease_burden',
]



class DecisionLevel(Enum):
    MUNICIPALITY = 1
    NATION = 2
    EU = 3
