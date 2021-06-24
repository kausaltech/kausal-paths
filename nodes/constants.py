from enum import Enum

FORECAST_COLUMN = 'Forecast'
YEAR_COLUMN = 'Year'
VALUE_COLUMN = 'Value'
IMPACT_COLUMN = 'Impact'
EMISSION_UNIT = 'kg'
BASELINE_VALUE_COLUMN = 'BaselineValue'
KNOWN_QUANTITIES = [
    'emissions', 'energy', 'emission_factor', 'mileage', 'population', 'per_capita'
]
ACTIVITY_QUANTITIES = ['emissions', 'energy', 'mileage']


class DecisionLevel(Enum):
    MUNICIPALITY = 1
    NATION = 2
    EU = 3
