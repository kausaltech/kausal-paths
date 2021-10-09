from enum import Enum

FORECAST_COLUMN = 'Forecast'
YEAR_COLUMN = 'Year'
VALUE_COLUMN = 'Value'
IMPACT_COLUMN = 'Impact'
FORECAST_x = 'Forecast_x'
FORECAST_y = 'Forecast_y'
VALUE_x = 'Value_x'
VALUE_y = 'Value_y'

EMISSION_UNIT = 'kg'
BASELINE_VALUE_COLUMN = 'BaselineValue'
KNOWN_QUANTITIES = [
    'emissions', 'energy', 'emission_factor', 'mileage', 'population', 'per_capita',
    'fuel_consumption', 'ratio', 'exposure', 'exposure-response', 'disease_burden', 'case_burden',
    'mass', 'consumption', 'mass_concentration', 'body_weight', 'incidence', 'fraction',
    'probability', 'ingestion'
]
ACTIVITY_QUANTITIES = ['emissions', 'energy', 'mileage', 'mass']


class DecisionLevel(Enum):
    MUNICIPALITY = 1
    NATION = 2
    EU = 3
