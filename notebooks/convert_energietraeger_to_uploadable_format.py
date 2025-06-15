from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd
import polars as pl
from use_ids import load_yaml_mappings, replace_labels_with_ids  # type: ignore

sys.path.append('..')
from typing import Any

from nodes.constants import ENERGY_QUANTITY


def extract_unit_from_file(lines) -> str:
    try:
        unit_line = lines[-5].strip()
        parts = unit_line.split(',')
        if len(parts) > 1:
            unit = parts[1].strip().strip('"')
            return unit
        return "MWh"  # noqa: TRY300
    except (IndexError, AttributeError):
        return "MWh"

def format_unit_for_energy(unit) -> str:
    unit = unit.strip()
    if unit and not unit.endswith('/a') and not unit.endswith('/year'):
        return f"{unit}/a"
    return unit

def transform_slice_name(filename_stem) -> str:
    if filename_stem.startswith('energietraeger-'):
        suffix = filename_stem[len('energietraeger-'):]
        return f"Energieverbrauch nach Energieträgern, {suffix}"
    return filename_stem

def is_energietraeger_file(filename) -> bool:
    return filename.startswith('energietraeger-')

def get_single_ksp_file(input_file) -> tuple[pd.DataFrame, str]:
    with open(input_file, 'r', encoding='utf-8') as f:  # noqa: PTH123, UP015
        lines = f.readlines()

    raw_unit = extract_unit_from_file(lines)

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
        df = pd.read_csv(io.StringIO(data_string), sep=',', encoding='utf-8', quotechar='"')
    except Exception:
        return pd.DataFrame(), ''

    if df.empty:
        return pd.DataFrame(), ''

    return df, raw_unit

def process_single_ksp_file(input_file) -> pl.DataFrame:
    print('Processing file: ', input_file)
    df, raw_unit = get_single_ksp_file(input_file)

    first_col_name = df.columns[0]

    year_columns = []
    year_column_mapping = {}

    for col in df.columns[1:]:
        try:
            col_str = str(col).strip()
            if col_str.isdigit() and 1900 <= int(col_str) <= 2100:
                year_str = col_str
                year_columns.append(year_str)
                year_column_mapping[col] = year_str
        except (ValueError, TypeError):
            continue

    if not year_columns:
        return pl.DataFrame()

    filename_stem = input_file.stem
    is_energietraeger = is_energietraeger_file(filename_stem)
    slice_value = transform_slice_name(filename_stem)

    if is_energietraeger:
        formatted_unit = format_unit_for_energy(raw_unit)
    else:
        formatted_unit = raw_unit

    df = build_gpc_from_ksp(df, first_col_name, formatted_unit, slice_value, is_energietraeger, year_column_mapping)
    dim_mappings = load_yaml_mappings()
    dfpl = pl.from_pandas(df)
    dfpl = replace_labels_with_ids(dfpl, dim_mappings)
    print('Success.')
    return dfpl

def build_gpc_from_ksp(  # noqa: C901, PLR0912
        df: pd.DataFrame,
        first_col_name: str,
        formatted_unit: str,
        slice_value: str,
        is_energietraeger: bool,
        year_column_mapping: dict,
        ) -> pd.DataFrame:

    output_rows = []

    for _, row in df.iterrows():
        metric_name = str(row[first_col_name])
        if pd.isna(row[first_col_name]) or metric_name.strip() == '' or metric_name == 'nan':
            continue

        metric_name = metric_name.strip()
        output_row: dict[str, Any]

        if is_energietraeger:
            output_row = {
                'Metric Group': 'Energy',
                'Energieträger': metric_name,
                'Quantity': ENERGY_QUANTITY,
                'Unit': formatted_unit,
                'Slice': slice_value
            }
        else:
            output_row = {
                'Metric Group': metric_name,
                'Quantity': '1',
                'Unit': formatted_unit,
                'Slice': slice_value
            }

        has_data = False
        for actual_col, year_str in year_column_mapping.items():
            if actual_col in row.index:
                raw_value = row[actual_col]

                if pd.notna(raw_value) and str(raw_value).strip() != '' and str(raw_value) != 'nan':
                    try:
                        if isinstance(raw_value, str):
                            clean_value = raw_value.replace(',', '.')
                        else:
                            clean_value = str(raw_value)

                        output_row[year_str] = float(clean_value)
                        has_data = True
                    except (ValueError, TypeError):
                        output_row[year_str] = None
                else:
                    output_row[year_str] = None
            else:
                output_row[year_str] = None

        if has_data:
            output_rows.append(output_row)

    if not output_rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(output_rows)
    return result_df

def convert_multiple_energietraeger_files(input_patterns, output_file):
    input_files = []
    for pattern in input_patterns:
        files = glob.glob(pattern)  # noqa: PTH207
        input_files.extend([Path(f) for f in files])

    if not input_files:
        print("Input files not found.")
        return

    all_dataframes = []

    for input_file in input_files:
        try:
            df = process_single_ksp_file(input_file)
            if df is not None:
                all_dataframes.append(df)
        except Exception:  # noqa: S112
            continue

    if not all_dataframes:
        raise ValueError("No valid data found in any input files")

    combined_df = pl.concat(all_dataframes, how='vertical')
    assert isinstance(combined_df, pl.DataFrame)

    combined_df.write_csv(output_file, separator=',',
                      quote_style='necessary', null_value='')
    print(combined_df)

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_energietraeger_to_uploadable_format.py <input_pattern1> [input_pattern2] ... <output_file>")
        sys.exit(1)

    input_patterns = sys.argv[1:-1]
    output_file = sys.argv[-1]

    try:
        convert_multiple_energietraeger_files(input_patterns, output_file)
        print('Successfully saved everything to', output_file)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
