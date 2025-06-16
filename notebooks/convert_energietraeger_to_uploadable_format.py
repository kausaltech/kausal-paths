from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd
import polars as pl
from use_ids import load_yaml_mappings, replace_labels_with_ids  # type: ignore

sys.path.append('..')

from nodes.constants import CURRENCY_QUANTITY, EMISSION_QUANTITY, ENERGY_QUANTITY, VALUE_COLUMN, YEAR_COLUMN


def extract_metadata_from_file(lines) -> dict[str, str]:
    units = {
        'Mt CO2äqu': ('Mt/a','BISKO' , EMISSION_QUANTITY),
        'kt CO2äqu': ('kt/a','BISKO' , EMISSION_QUANTITY),
        't CO2äqu': ('t/a','BISKO' , EMISSION_QUANTITY),
        'kg CO2äqu': ('kg/a','BISKO' , EMISSION_QUANTITY),
        'GWh': ('GWh/a', 'EEV', ENERGY_QUANTITY),
        'MWh': ('MWh/a', 'EEV', ENERGY_QUANTITY),
        'kWh': ('kWh/a', 'EEV', ENERGY_QUANTITY),
        'Euro': ('EUR/a', 'Energiekosten', CURRENCY_QUANTITY),
    }

    def _find_meta(lines, title) -> str:
        for _, line in enumerate(lines):
            meta_line = line.strip()
            if meta_line.startswith(title):
                break
        parts = meta_line.split(',')
        if len(parts) > 1:
            meta = parts[1].strip().strip('"')
        else:
            meta = ''
        return meta

    unit = _find_meta(lines, 'Einheit:')
    unit, method, quantity = units.get(unit, (unit, '', ''))
    weather_correction = _find_meta(lines, 'Witterungskorrektur:')
    inventory_method = _find_meta(lines, 'Bilanzierungsmethode:')

    if inventory_method != method:
        raise KeyError(f"The method {inventory_method} does not match what was expected ({method}).")

    return {
        'unit': unit,
        'quantity': quantity,
        'weather_correction': weather_correction,
        'inventory_method': inventory_method,
        }

def transform_slice_name(filename_stem) -> str: # TODO This is needed in 1-dimensional tables with several years
    if filename_stem.startswith('energietraeger-'):
        suffix = filename_stem[len('energietraeger-'):]
        return f"Energieverbrauch nach Energieträgern, {suffix}"
    return filename_stem

def is_energietraeger_file(filename) -> bool:
    return filename.startswith('energietraeger-')

def get_single_ksp_file(input_file) -> pl.DataFrame:
    with open(input_file, 'r', encoding='utf-8') as f:  # noqa: PTH123, UP015
        lines = f.readlines()

    meta = extract_metadata_from_file(lines)

    data_start = 2
    data_end = None

    for i, line in enumerate(lines):
        line_ = line.strip()
        if line_.startswith(('Einheit:', 'Witterungskorrektur:', 'Bilanzierungsmethode:')):
            data_end = i
            break

    if data_end is None:
        data_end = len(lines)

    data_lines = lines[data_start:data_end]

    import io
    data_string = ''.join(data_lines)

    try:
        dfpd = pd.read_csv(io.StringIO(data_string), sep=',', encoding='utf-8', quotechar='"', decimal=",")
        df = pl.from_pandas(dfpd)
    except Exception:
        return pl.DataFrame()

    if len(df) == 0:
        return pl.DataFrame()

    df = df.with_columns(pl.lit(meta['unit']).alias('Unit'))
    df = df.with_columns(pl.lit(meta['quantity']).alias('Quantity'))
    df = df.with_columns(pl.lit(meta['weather_correction']).alias('Weather correction'))
    df = df.with_columns(pl.lit(meta['inventory_method']).alias('Inventory method'))

    return df

def process_single_ksp_file(input_file) -> pl.DataFrame:
    print('Processing file: ', input_file)
    print(input_file.stem)
    df = get_single_ksp_file(input_file)
    print(df)

    first_col_name = df.columns[0]

    year_columns = [col for col in df.columns if (col.isdigit() and 1900 <= int(col) <= 2100)]
    if year_columns:
        variable_name = YEAR_COLUMN
    else:
        variable_name = 'Sector'
        year = str(input_file.stem[-4:])
        if not year.isdigit():
            raise KeyError("With sektoren-energieträger files, the file name must contain"
                           " the year as the last four characters of the filename.")
        year_columns = [year]
    keys = [first_col_name, 'Quantity', 'Unit', 'Weather correction', 'Inventory method']
    primary_keys = [col for col in df.columns if col in keys]
    df = df.unpivot(
        index=primary_keys,
        variable_name=variable_name,
        value_name=VALUE_COLUMN,
    )
    df = df.with_columns(pl.lit(year).cast(int).alias(YEAR_COLUMN))

    print(df)

    # filename_stem = input_file.stem
    # is_energietraeger = is_energietraeger_file(filename_stem)
    # slice_value = transform_slice_name(filename_stem)

    dim_mappings = load_yaml_mappings()
    df = replace_labels_with_ids(df, dim_mappings)
    print('Success.')
    return df

def convert_multiple_energietraeger_files(input_patterns, slice_column, output_file):
    input_files = []
    for pattern in input_patterns:
        files = glob.glob(pattern)  # noqa: PTH207
        input_files.extend([Path(f) for f in files])

    if not input_files:
        print("Input files not found.")
        return

    all_dataframes = []

    for input_file in input_files:
        df = process_single_ksp_file(input_file)
        if len(df) > 0:
            all_dataframes.append(df)
        else:
            print(f"File {input_file} had no content.")

    if not all_dataframes:
        raise ValueError("No valid data found in any input files")

    combined_df = pl.concat(all_dataframes, how='vertical')
    combined_df = combined_df.filter(~pl.col(VALUE_COLUMN).is_null())
    combined_df = combined_df.filter(pl.col('sector') != 'total')
    combined_df = combined_df.with_columns([
        pl.col(slice_column).alias('Slice'),
        pl.col('Quantity').alias('Metric Group'),
    ])
    assert isinstance(combined_df, pl.DataFrame)

    combined_df.write_csv(output_file, separator=',',
                      quote_style='necessary', null_value='')
    print(combined_df)

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_energietraeger_to_uploadable_format.py <input_pattern1> [input_pattern2] ..."
              " <slice_column> <output_file>")
        sys.exit(1)

    input_patterns = sys.argv[1:-2]
    slice_column = sys.argv[-2]
    output_file = sys.argv[-1]

    try:
        convert_multiple_energietraeger_files(input_patterns, slice_column, output_file)
        print('Successfully saved everything to', output_file)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# python convert_energietraeger_to_uploadable_format.py "/Users/jouni/Downloads/energie/sektoren-energietraeger_*"
#   sector energietraeger.csv
# python upload_new_dataset.py "energietraeger.csv" ',' NONE potsdam/energie en
