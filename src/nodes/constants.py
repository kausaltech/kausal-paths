from __future__ import annotations

from enum import Enum

import strawberry as sb

BASELINE_SCENARIO = 'baseline'

FORECAST_COLUMN = 'Forecast'
YEAR_COLUMN = 'Year'
VALUE_COLUMN = 'Value'
NODE_COLUMN = 'Node'
SCENARIO_COLUMN = 'ScenarioName'

# Per-row provenance that rides through to DVC as literal column values but is never a
# dimension, a metric or an index column. `upload_new_dataset` keeps these out of the
# parquet's index_columns, `load_dvc_dataset` reads them back into DataSource and
# DataPointComment links, and `DVCDataset` drops them on load so that a dataset read
# from DVC has the same columns as the same dataset read from the database.
#
# Matched case-insensitively, because the CSV spells them capitalised (`Source`,
# `Comment`) and the parquet lowercases them.
RESERVED_ROW_COLUMNS = frozenset({'source', 'comment', 'description'})

# A data source cited by a dataset can attach to either of two targets, and the target is
# carried through the whole CSV -> DVC -> DB path: the `Target` column of the sources
# registry read by `upload_new_dataset`, the `target` key of each `metadata['sources']`
# entry in DVC, and finally the `data_point` / `dataset` fork in `DatasetSourceReference`.
#
# `target` rather than `scope`, which is taken twice over: `DataSource.scope` is the
# instance that owns the source, and `Scope` in a data file is the GHG scope. It is also
# what the read side has always called it -- `DatasetSourceReferenceTarget` in GraphQL,
# `reference_target` in REST.
#
# `data_point` is the default and the historical behaviour: the source is named in a
# row's `Source` cell and is linked to the data points made from that row. `dataset`
# attaches the source to the dataset as a whole, for data that arrives from one
# publication in one update and has nothing per-row to say. A dataset may carry any
# number of dataset-scoped sources; they are a set, not a single attribution.
SOURCE_TARGET_DATA_POINT = 'data_point'
SOURCE_TARGET_DATASET = 'dataset'
SOURCE_TARGETS = frozenset({SOURCE_TARGET_DATA_POINT, SOURCE_TARGET_DATASET})

# The in-cell separators for the two reserved columns, defined once here because three
# commands now have to agree on them: `upload_new_dataset` writes them, `load_dvc_dataset`
# splits on them, and `export_dataset` joins on them again on the way back out. They were
# duplicated in the first two, each with a comment saying it must match the other.
#
# They are deliberately different from each other. Source names are short identifiers, so
# '; ' is safe. Comments are prose and prose contains semicolons -- splitting those on
# '; ' would fragment single sentences -- so a comment cell carrying several notes joins
# them with ' ;; '.
SOURCE_NAME_SEPARATOR = '; '
COMMENT_SEPARATOR = ' ;; '

# Joins the dataset names in the sources registry's `Datasets` column. Same string as
# SOURCE_NAME_SEPARATOR and kept separate anyway: they are different vocabularies, and a
# change to one has no business moving the other.
DATASET_NAME_SEPARATOR = '; '

# Impact constants
IMPACT_COLUMN = 'Impact'
# Probability iterations for Monte Carlo
UNCERTAINTY_COLUMN = 'iteration'

# Action as it is in the active scenario
SCENARIO_ACTION_GROUP = 'Scenario'
# With action disabled
WITHOUT_ACTION_GROUP = 'WithoutAction'
# Reference scenario (for scenario impact)
REFERENCE_SCENARIO_GROUP = 'Reference'
# Impact of action or scenario
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
EMPLOYEES_QUANTITY = 'employees'
MIX_QUANTITY = 'mix'
GROUPED_MIX_QUANTITY = 'grouped_mix'
UTILITY_QUANTITY = 'utility'
VALUE_WEIGHT_QUANTITY = 'value_weight'
FRACTION_QUANTITY = 'fraction'

ACTIVITY_QUANTITIES = {
    EMISSION_QUANTITY,
    ENERGY_QUANTITY,
    MILEAGE_QUANTITY,
    VEHICLE_MILEAGE_QUANTITY,
    PASSENGER_MILEAGE_QUANTITY,
    FREIGHT_MILEAGE_QUANTITY,
    UTILITY_QUANTITY,
    'fuel_consumption',
    'consumption',
    'mass',
    'volume',
    'area',
    'employment',
    'activity',
    'traffic_volume',
}

ACTIVITY_FACTOR_QUANTITIES = {
    ENERGY_FACTOR_QUANTITY,
    CONSUMPTION_FACTOR_QUANTITY,
    'energy_per_area',
    'occupancy_factor',
    'fuel_factor',
    'demand_factor',
    'time_factor',
    'factor',
}

UNIT_PRICE_QUANTITIES = {
    UNIT_PRICE_QUANTITY,
    'energy_unit_price',
    'floor_area_unit_price',
    'fuel_unit_price',
    'mileage_unit_price',
    'price',
}

STACKABLE_QUANTITIES = ACTIVITY_QUANTITIES | {
    MIX_QUANTITY,
    GROUPED_MIX_QUANTITY,
    POPULATION_QUANTITY,
    EMPLOYEES_QUANTITY,
    FLOOR_AREA_QUANTITY,
    CURRENCY_QUANTITY,
    NUMBER_QUANTITY,
    'area',
    'disease_burden',
    'health_effect',
    'length',
    'volume',
}

KNOWN_QUANTITIES = (
    ACTIVITY_QUANTITIES
    | ACTIVITY_FACTOR_QUANTITIES
    | UNIT_PRICE_QUANTITIES
    | STACKABLE_QUANTITIES
    | {
        EMISSION_FACTOR_QUANTITY,
        CURRENCY_QUANTITY,
        NUMBER_QUANTITY,
        PER_CAPITA_QUANTITY,
        FLOOR_AREA_QUANTITY,
        MIX_QUANTITY,
        GROUPED_MIX_QUANTITY,
        POPULATION_QUANTITY,
        VALUE_WEIGHT_QUANTITY,
        'ratio',
        'exposure',
        'exposure_response',
        'disease_burden',
        'case_burden',
        'mass_concentration',
        'concentration',
        'body_weight',
        'incidence',
        'fraction',
        'probability',
        'ingestion',
        'area',
        'effect',
        'health_effect',
        'rate',
        'speed',
        'argument',
        'duration',
        'distance',
        'elasticity',
        'quality_of_data',
        'temperature',
    }
)


DEFAULT_METRIC = 'default'


def ensure_known_quantity(quantity: str):
    if quantity not in KNOWN_QUANTITIES:
        raise Exception(f'Quantity {quantity} is unknown')


@sb.enum(name='DecisionLevel', description='Which governance level is applicable for an action')
class DecisionLevel(Enum):
    MUNICIPALITY = 1
    NATION = 2
    EU = 3

    def as_str(self) -> str:
        match self:
            case DecisionLevel.MUNICIPALITY:
                return 'municipality'
            case DecisionLevel.NATION:
                return 'nation'
            case DecisionLevel.EU:
                return 'eu'


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
    EMPLOYEES_QUANTITY: '👷',
    MIX_QUANTITY: '💯',
    GROUPED_MIX_QUANTITY: '💯',
    CURRENCY_QUANTITY: '💰',
    FRACTION_QUANTITY: '➗',
    VALUE_WEIGHT_QUANTITY: '⚓',  # balance scale: weighing values, not money
    UTILITY_QUANTITY: '♥',
}


def get_quantity_icon(quantity: str) -> str | None:
    return QUANTITY_ICONS.get(quantity)
