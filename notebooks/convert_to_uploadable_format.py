from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd
import polars as pl
from use_ids import load_yaml_mappings, replace_labels_with_ids  # type: ignore

sys.path.append('..')

from nodes.constants import (
    CURRENCY_QUANTITY,
    EMISSION_FACTOR_QUANTITY,
    EMISSION_QUANTITY,
    ENERGY_QUANTITY,
    FLOOR_AREA_QUANTITY,
    MILEAGE_QUANTITY,
    NUMBER_QUANTITY,
    VALUE_COLUMN,
    YEAR_COLUMN,
)

SLICE_NAME_MAPPING = {
    'datenzeitreihen': "Emissionsfaktoren für die Energieerzeugung",
    'datenzeitreihen-2': "Emissionsfaktoren für die Industrie",
    'datenzeitreihen-3': "Emissionsfaktoren für den Verkehr",
    'datenzeitreihen-4': "Aktivitäten, Verkehr und Gebäude",
}

UNIT_MAPPING = {
    'Mt CO2äqu': ('Mt/a', 'BISKO' , EMISSION_QUANTITY),
    'kt CO2äqu': ('kt/a', 'BISKO' , EMISSION_QUANTITY),
    't CO2äqu': ('t/a', 'BISKO' , EMISSION_QUANTITY),
    'kg CO2äqu': ('kg/a', 'BISKO' , EMISSION_QUANTITY),
    'GWh': ('GWh/a', 'EEV', ENERGY_QUANTITY),
    'MWh': ('MWh/a', 'EEV', ENERGY_QUANTITY),
    'kWh': ('kWh/a', 'EEV', ENERGY_QUANTITY),
    'Euro': ('EUR/a', 'Energiekosten', CURRENCY_QUANTITY),
    "t CO2-Äqu./MWh": ("t/MWh", 'BISKO', EMISSION_FACTOR_QUANTITY),
    "t/MWh": ("t/MWh", 'BISKO', EMISSION_FACTOR_QUANTITY),
    "g CO2-Äqu./Wh": ("g/Wh", 'BISKO', EMISSION_FACTOR_QUANTITY),
    "GJ": ("GJ/a", 'EEV', ENERGY_QUANTITY),
    "Anzahl": ("pcs", '', NUMBER_QUANTITY),
    "Wege/Person/d": ("trips/person/d", '', NUMBER_QUANTITY),
    "km/Weg": ("km/trip", '', NUMBER_QUANTITY),
    "%": ("%", '', NUMBER_QUANTITY),
    "Mio. Fz-km": ("Mvkm", '', MILEAGE_QUANTITY),
    "Mio. Zug-km": ("Mvkm", '', MILEAGE_QUANTITY),
    "m²": ("m²", '', FLOOR_AREA_QUANTITY),
}

CATEGORY_MAPPING = {
    'energy_carrier': {
        'klär-, deponie-, grubengas': 'biogas',
        'klärgas': 'biogas',
        'deponiengas': 'biogas',
        'grubengas': 'biogas',
        'flüssiggas': 'lpg',
        'erdgas': 'natural_gas',
        'biogas': 'biogas',
        'gas': 'natural_gas',
        'feste biomasse': 'biomass',
        'flüssige biomasse': 'biomass',
        'biomasse': 'biomass',
        'braunkohle': 'brown_coal',
        'steinkohle': 'hard_coal',
        'kohle': 'hard_coal',
        'sonstige erneuerbare energieträger': 'other_renewables',
        'erneuerbare energieträger': 'other_renewables',
        'sonstige erneuerbare': 'other_renewables',
        'erneuerbare': 'other_renewables',
        'sonstige konventionelle energieträger': 'other_conventional',
        'konventionelle energieträger': 'other_conventional',
        'sonstige konventionelle': 'other_conventional',
        'konventionelle': 'other_conventional',
        # Other specific terms
        'geothermie': 'environmental_heat',
        'heizöl': 'heating_oil',
        'photovoltaik': 'electricity',
        'wasserkraft': 'electricity',
        'windkraft': 'electricity',
        'abfall': 'other_conventional',
        'abwärme': 'district_heating',
        'strom': 'electricity',
        'electricity': 'electricity',
    },
    'fuel_type': {
        'erneuerbare energieträger': 'non_fossil',
        'erneuerbare': 'non_fossil',
        'konventionelle energieträger': 'fossil',
        'konventionelle': 'fossil',
        'biogas': 'non_fossil',
        'biomasse': 'non_fossil',
        'geothermie': 'non_fossil',
        'photovoltaik': 'non_fossil',
        'wasserkraft': 'non_fossil',
        'windkraft': 'non_fossil',
        'erdgas': 'fossil',
        'braunkohle': 'fossil',
        'steinkohle': 'fossil',
        'kohle': 'fossil',
        'heizöl': 'fossil',
        'flüssiggas': 'fossil',
        'abfall': 'mixed',
    },
    'sector': {
        'stromerzeugungsanlage': 'electricity',
        'stromerzeugung': 'electricity',
        'wärmeerzeugungsanlage': 'buildings',
        'wärmeerzeugung': 'buildings',
        'kraftwerk': 'electricity',
        # Transport
        'verkehr': 'transport',
        'transport': 'transport',
        # Buildings and heating
        'gebäude': 'buildings',
        'heizung': 'buildings',
        'wärme': 'buildings',
        # Industry
        'industrie': 'industry',
        'industrial': 'industry',
        # Households
        'haushalt': 'private_households',
        'privat': 'private_households',
        # Commercial
        'gewerbe': 'commerce_trade_services',
        'handel': 'commerce_trade_services',
        'dienstleistung': 'commerce_trade_services',
        # Municipal
        'kommunal': 'municipal_facilities',
        'municipal': 'municipal_facilities',
        # Waste
        'abfall': 'waste',
        'waste': 'waste',
        # Energy/Electricity
        'electricity': 'electricity',
        'strom': 'electricity',
    },
    'ghg': {
        'äquivalente': 'co2e',
        'co2': 'co2e',
    },
    'scope': {
        'vorkette': 'scope3',
        'kraftwerke': 'scope1',
        'stromerzeugung': 'scope2',
        'wärmeerzeugung': 'scope1',
    },
    'inventory_method': {
        'bisko': 'bisko',
    },
    'energy_process': {
        'stromerzeugungsanlage': 'electricity_generation',
        'stromerzeugung': 'electricity_generation',
        'wärmeerzeugungsanlage': 'heat_generation',
        'wärmeerzeugung': 'heat_generation',
        'blockheizkraftwerk': 'chp',
        'kraft-wärme-kopplung': 'chp',
        'kraftwerk': 'power_plants',
        'kwk': 'chp',
        'bhkw': 'chp',
    },
    'household_size': {
        'haushalte mit mehr als 5 personen': 'five_plus_person',
        '1-personen-haushalte': 'single_person',
        '2-personen-haushalte': 'two_person',
        '3-personen-haushalte': 'three_person',
        '4-personen-haushalte': 'four_person',
        '5-personen-haushalte': 'five_person',
        'mehr als 5 personen': 'five_plus_person',
        '1-personen': 'single_person',
        '2-personen': 'two_person',
        '3-personen': 'three_person',
        '4-personen': 'four_person',
        '5-personen': 'five_person',
    },
    'trip_type': {
        'schienenpersonenfernverkehr': 'long_distance_rail',
        'schienenpersonennahverkehr': 'local_rail',
        'schienengüterverkehr': 'rail_freight',
        'binnenschifffahrt': 'inland_shipping',
        'straßenverkehr': 'road_transport',
        'personenverkehr': 'passenger_transport',
        'güterverkehr': 'freight_transport',
        'flugverkehr': 'aviation',
        'luftverkehr': 'aviation',
        'seeverkehr': 'maritime_shipping',
    },
    'road_type': {
        'stadtstraße': 'city_streets',
        'landstraße': 'country_roads',
        'bundesstraße': 'federal_roads',
        'autobahn': 'highway',
        'außerorts': 'rural_roads',
        'innerorts': 'urban_roads',
    },
    'heating_type': {
        'einzel- oder mehrraumöfen': 'individual_room_heaters',
        'nachtspeicherheizung': 'night_storage_heating',
        'blockheizkraftwerk': 'chp',  # If heating context
        'blockheizung': 'district_heating_local',
        'zentralheizung': 'central_heating',
        'etagenheizung': 'floor_heating',
        'fernheizung': 'district_heating',
        'mehrraumöfen': 'multi_room_heaters',
        'solarheizung': 'solar_heating',
        'wärmepumpe': 'heat_pump',
        'ohne heizung': 'no_heating',
        'einzelöfen': 'individual_heaters',
    },
    'building_age': {
        'baujahr 1951-1969': 'age_1950_1969',
        'baujahr 1950 - 1969': 'age_1950_1969',
        'baujahr 1970-1989': 'age_1970_1989',
        'baujahr 1970 - 1989': 'age_1970_1989',
        'baujahr vor 1950': 'before_1950',
        'baujahr nach 1990': 'after_1990',
        '1951-1969': 'age_1950_1969',
        '1950 - 1969': 'age_1950_1969',
        '1970-1989': 'age_1970_1989',
        '1970 - 1989': 'age_1970_1989',
        '1990-2000': 'age_1990_2000',
        '2000-2010': 'age_2000_2010',
        'vor 1950': 'before_1950',
        'nach 1990': 'after_1990',
        'nach 2000': 'after_2000',
        'nach 2010': 'after_2010',
    },
    'building_type': {
        'einer und zwei wohnungen': 'single_two_family',
        '7-12 und 13 und mehr wohnungen': 'large_multifamily',
        '13 und mehr wohnungen': 'large_multifamily',
        '1-2 wohnungen': 'single_two_family',
        '3-6 wohnungen': 'small_multifamily',
        '7-12 wohnungen': 'medium_multifamily',
        'zweifamilienhaus': 'two_family',
        'einfamilienhaus': 'single_family',
        'mehrfamilienhaus': 'multifamily',
        'reihenhaus': 'row_house',
        'wohnblock': 'apartment_block',
        'hochhaus': 'high_rise',
    },
    'vehicle_type': {
        'leichte nutzfahrzeuge': 'light_commercial',
        'schwere nutzfahrzeuge': 'heavy_commercial',
        'motorisierte zweiräder': 'motorcycles',
        'personenkraftwagen': 'passenger_cars',
        'schienenfahrzeuge': 'rail_vehicles',
        'lastkraftwagen': 'trucks',
        'motorräder': 'motorcycles',
        'fahrräder': 'bicycles',
        'busse': 'buses',
        'züge': 'trains',
        'pkw': 'passenger_cars',
        'lkw': 'trucks',
        'bus': 'buses',
    }
}

def extract_metadata_from_file(df: pl.DataFrame) -> dict[str, str | None]:
    def _find_meta(df: pl.DataFrame, title) -> str | None:
        first_col_name = df.columns[0]
        second_col_name = df.columns[1] if len(df.columns) > 1 else None
        for i in range(len(df)):
            first_col_value = str(df[first_col_name][i]).strip()
            if first_col_value.startswith(title) and second_col_name is not None:
                meta = str(df[second_col_name][i]).strip()
                return meta
        return None

    unit = _find_meta(df, 'Einheit:')
    if unit is not None:
        unit, method, quantity = UNIT_MAPPING.get(unit, (unit, '', None))
    else:
        unit, method, quantity = (None, '', None)

    weather_correction = _find_meta(df, 'Witterungskorrektur:')
    inventory_method = _find_meta(df, 'Bilanzierungsmethode:')
    municipality = _find_meta(df, 'Gemeinde:')
    sector = _find_meta(df, 'Sektoren:')

    if inventory_method is not None and inventory_method != method:
        raise KeyError(f"The method {inventory_method} does not match what was expected ({method}).")

    return {
        'unit': unit,
        'quantity': quantity,
        'weather_correction': weather_correction,
        'inventory_method': inventory_method,
        'municipality': municipality,
        'sector': sector,
        }

def transform_slice_name(filename_stem) -> str: # TODO This is needed in 1-dimensional tables with several years
    if filename_stem.startswith('energietraeger-'):
        suffix = filename_stem[len('energietraeger-'):]
        return f"Energieverbrauch nach Energieträgern, {suffix}"
    return filename_stem

def is_energietraeger_file(filename) -> bool:
    return filename.startswith('energietraeger-')

def read_csv_fixed_width(input_file, separator) -> pl.DataFrame:
    """Make the line lengths match data."""

    with open(input_file, 'r', encoding='utf-8') as f:  # noqa: PTH123, UP015
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        line_ = line.strip()
        if line_.startswith(('Eingabefeld', 'Energieträger')):
            data_start = i
            file_type = line_[0:len('Energieträger')]
            break

    if data_start is None:
        raise KeyError(f"Cannot figure out data rows in {input_file.stem}")
    expected_cols = len(lines[data_start].split(separator))

    def _fix(line) -> list:
        if len(line) < expected_cols:
            return line + [''] * (expected_cols - len(line))
        return line[:expected_cols]

    dfpd = pd.read_csv(input_file, sep=separator, encoding='utf-8',
                    quotechar='"', on_bad_lines=_fix, engine='python', decimal=',',
                    skiprows=1)

    df = pl.from_pandas(dfpd)
    df = df.with_columns(pl.lit(file_type).alias('File type'))
    return df

def read_xlsx_fixed_width(input_file) -> pl.DataFrame:
    """Make the line lengths match data."""
    df = pl.from_pandas(pd.read_excel(input_file, header=0))

    data_start = None
    first_col = df.columns[0]
    for i in range(len(df)):
        rowname = df[first_col][i]
        if rowname and rowname.startswith(('Eingabefeld', 'Energieträger')):
            data_start = i
            file_type = rowname
            break

    if data_start is None:
        raise KeyError(f"Cannot figure out data rows in {input_file.stem}")

    heads = list(zip(df.columns, df.row(data_start), strict=True))
    oldhead = {old: str(new) for old, new in heads if new is not None}
    newhead = [str(new) for old, new in heads if new is not None]
    df = df.rename(oldhead).select(newhead).slice(data_start)
    df = df.with_columns(pl.lit(file_type).alias('File type'))
    return df

def get_single_ksp_file(input_file, separator) -> pl.DataFrame:  # noqa: C901
    if input_file.suffix.lower() in ['.xlsx', '.xls']:
        df = read_xlsx_fixed_width(input_file)
    else:
        df = read_csv_fixed_width(input_file, separator)

    meta = extract_metadata_from_file(df)

    data_end = None
    first_col = df.columns[0]

    for i in range(len(df)):
        rowname = str(df[first_col][i]).strip()
        if rowname.startswith(('Einheit:', 'Witterungskorrektur:', 'Bilanzierungsmethode:', 'Gemeinde:', 'Feldtyp')):
            data_end = i
            break
    if data_end is None:
        data_end = len(df)

    df = df.slice(1, data_end - 1)

    if len(df) == 0:
        return pl.DataFrame()

    if meta['unit'] is not None:
        df = df.with_columns(pl.lit(meta['unit']).alias('Unit'))
    if meta['quantity'] is not None:
        df = df.with_columns(pl.lit(meta['quantity']).alias('Quantity'))
    if meta['weather_correction'] is not None:
        df = df.with_columns(pl.lit(meta['weather_correction']).alias('Weather correction'))
    if meta['inventory_method'] is not None:
        df = df.with_columns(pl.lit(meta['inventory_method']).alias('Inventory method'))
    if meta['sector'] is not None:
        df = df.with_columns(pl.lit(meta['sector']).alias('Sektoren'))

    return df

def process_explanatory_column(df: pl.DataFrame) -> pl.DataFrame:
    """Extract units and BISKO."""
    pattern = r'^(.+?)(?:,\s*BISKO)?\s*\(([^)]+)\)\s*\[([^\]]+)\]$'
    col = 'Eingabefeld'

    df = df.with_columns([
        pl.col(col).str.extract(pattern, group_index=1).str.strip_chars().alias('Description'),
        pl.col(col).str.extract(pattern, group_index=2).str.strip_chars().alias('BISKO code'),
        pl.col(col).str.extract(pattern, group_index=3).str.strip_chars().alias('Unit'),
    ])

    def map_unit_info(unit) -> tuple:
        if unit is None:
            return (None, None, None)

        mapping = UNIT_MAPPING.get(unit, (unit, '', None))
        return mapping

    df = df.with_columns([
        pl.col("Unit").map_elements(lambda x: map_unit_info(x)[0], return_dtype=pl.Utf8).alias("updated_unit"),
        pl.col("Unit").map_elements(lambda x: map_unit_info(x)[1], return_dtype=pl.Utf8).alias("inventory_method"),
        pl.col("Unit").map_elements(lambda x: map_unit_info(x)[2], return_dtype=pl.Utf8).alias("Quantity")
    ])
    df = df.with_columns(pl.col('updated_unit').alias('Unit')).drop([col, 'updated_unit'])

    return df

def find_category_matches(description: str | None, category_mapping: dict):
    """Find dimension matches in description text."""
    # Initialize all dimensions to None
    matches: dict[str, str | None] = {}
    for dimension_id in category_mapping.keys():
        matches[dimension_id] = None
    if description is None:
        return matches

    description_lower = description.lower()

    for dimension_id, keyword_mapping in category_mapping.items():
        for keyword, category_id in keyword_mapping.items():
            if keyword.lower() in description_lower:
                matches[dimension_id] = category_id
                break  # Take first match for each dimension

    return matches

def process_hidden_categories(df: pl.DataFrame, category_mapping: dict) -> pl.DataFrame:
    """Add dimension columns based on keyword matching."""

    all_dimensions = set(category_mapping.keys())
    print(all_dimensions)
    # Apply matching function
    df = df.with_columns(
        pl.col("Description").map_elements(
            lambda x: find_category_matches(x, category_mapping),
            return_dtype=pl.Struct([pl.Field(dim, pl.Utf8) for dim in all_dimensions])
        ).alias("category_matches")
    )

    # Extract each dimension as a separate column
    for dimension in all_dimensions:
        df = df.with_columns(
            pl.col("category_matches").struct.field(dimension).alias(dimension)
        )

    return df.drop("category_matches")

def create_slice_column(df: pl.DataFrame, input_file, slice_column) -> pl.DataFrame:
    """Transform filename to slice name according to the rules."""
    slice_name = SLICE_NAME_MAPPING.get(input_file.stem)
    if slice_name:
        df = df.with_columns(pl.lit(slice_name).alias('Slice'))
    elif slice_column in df.columns:
        df = df.with_columns(pl.col(slice_column).alias('Slice'))
    else:
        raise ValueError(f"Slice column {slice_column} not found. Has {df.columns}")
    assert isinstance(slice_name, str)
    return df

def process_single_ksp_file(input_file, separator) -> pl.DataFrame:
    print('Processing file: ', input_file)
    df = get_single_ksp_file(input_file, separator)

    for col in df.columns:
        try:
            year = int(float(col))
            if 1900 <= year <= 2100:
                df = df.rename({col: str(year)})
        except (ValueError, TypeError):
            continue

    keys = [df.columns[0], 'Quantity', 'Unit', 'Weather correction', 'Inventory method', 'File type', 'Sektoren']
    year_columns = [col for col in df.columns if (col.isdigit() and 1900 <= int(col) <= 2100)]
    if year_columns:
        variable_name = YEAR_COLUMN
    else:
        variable_name = 'Sector'
        obs_year = str(input_file.stem[-4:])
        if not obs_year.isdigit():
            raise KeyError("With sektoren-energieträger files, the file name must contain"
                           " the year as the last four characters of the filename.")
        year_columns = [obs_year]
        df = df.with_columns(pl.lit(obs_year).cast(int).alias(YEAR_COLUMN))
        keys += [YEAR_COLUMN]
    primary_keys = [col for col in df.columns if col in keys]
    df = df.unpivot(
        index=primary_keys,
        variable_name=variable_name,
        value_name=VALUE_COLUMN,
    )

    df = process_explanatory_column(df)
    df = process_hidden_categories(df, CATEGORY_MAPPING)

    dim_mappings = load_yaml_mappings()
    df = replace_labels_with_ids(df, dim_mappings)
    print(df)
    print(df.columns)
    print('Success.')
    return df

def convert_multiple_energietraeger_files(input_patterns, separator, slice_column, output_file) -> pl.DataFrame:
    input_files = []
    for pattern in input_patterns:
        files = glob.glob(pattern)  # noqa: PTH207
        input_files.extend([Path(f) for f in files])

    if not input_files:
        print("Input files not found.")
        return pl.DataFrame()

    all_dataframes = []

    for input_file in input_files:
        df = process_single_ksp_file(input_file, separator)
        df = create_slice_column(df, input_file, slice_column)
        if len(df) > 0:
            all_dataframes.append(df)
        else:
            print(f"File {input_file} had no content.")

    if not all_dataframes:
        raise ValueError("No valid data found in any input files")

    df = pl.concat(all_dataframes, how='vertical')
    df = df.with_columns(pl.col(VALUE_COLUMN).cast(pl.Float64, strict=False))
    df = df.filter(~pl.col(VALUE_COLUMN).is_null())
    if 'sector' in df.columns:
        df = df.filter(pl.col('sector') != 'total')

    df = df.with_columns(pl.col('Quantity').alias('Metric Group'))

    assert isinstance(df, pl.DataFrame)
    missing = df.filter(pl.col('Quantity').is_null())['Unit'].unique().to_list()
    if missing:
        raise ValueError(f"These units do not have quantities defined: {missing}")

    df.write_csv(output_file, separator=',',
                      quote_style='necessary', null_value='')
    return df

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_energietraeger_to_uploadable_format.py <input_pattern1> [input_pattern2] ..."
              " <slice_column> <output_file>")
        sys.exit(1)

    input_patterns = sys.argv[1:-3]
    separator = sys.argv[-3]
    slice_column = sys.argv[-2]
    output_file = sys.argv[-1]

    df = convert_multiple_energietraeger_files(input_patterns, separator, slice_column, output_file)

    print(df)
    print(df.columns)
    print('Successfully saved everything to', output_file)

if __name__ == "__main__":
    main()

# python convert_energietraeger_to_uploadable_format.py "/Users/jouni/Downloads/energie/sektoren-energietraeger_*"
#   "," sector energietraeger.csv
# python upload_new_dataset.py "energietraeger.csv" ',' NONE potsdam/energie en
