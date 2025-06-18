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

    if slice_column not in df.columns:
        raise ValueError(f"Slice column {slice_column} not found. Has {df.columns}")
    df = df.with_columns([
        pl.col(slice_column).alias('Slice'),
        pl.col('Quantity').alias('Metric Group'),
    ])

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
