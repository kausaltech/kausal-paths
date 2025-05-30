import pandas as pd
import re
import sys
import csv
from pathlib import Path
import glob

sys.path.append('..')
from nodes.constants import ENERGY_QUANTITY

def extract_unit_from_file(lines):
    try:
        unit_line = lines[-5].strip()
        parts = unit_line.split(',')
        if len(parts) > 1:
            unit = parts[1].strip().strip('"')
            return unit
        else:
            return "MWh"
    except (IndexError, AttributeError):
        return "MWh"

def format_unit_for_energy(unit):
    unit = unit.strip()
    if unit and not unit.endswith('/a') and not unit.endswith('/year'):
        return f"{unit}/a"
    return unit

def transform_slice_name(filename_stem):
    if filename_stem.startswith('energietraeger-'):
        suffix = filename_stem[len('energietraeger-'):]
        return f"Energieverbrauch nach Energieträgern, {suffix}"
    else:
        return filename_stem

def is_energietraeger_file(filename):
    return filename.startswith('energietraeger-')

def process_single_energietraeger_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    raw_unit = extract_unit_from_file(lines)

    data_start = 2
    data_end = None

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('Einheit:') or line.startswith('Witterungskorrektur:') or line.startswith('Bilanzierungsmethode:'):
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
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

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
        return pd.DataFrame()

    filename_stem = input_file.stem
    is_energietraeger = is_energietraeger_file(filename_stem)
    slice_value = transform_slice_name(filename_stem)

    if is_energietraeger:
        formatted_unit = format_unit_for_energy(raw_unit)
    else:
        formatted_unit = raw_unit

    output_rows = []

    for _, row in df.iterrows():
        metric_name = str(row[first_col_name])
        if pd.isna(row[first_col_name]) or metric_name.strip() == '' or metric_name == 'nan':
            continue

        metric_name = metric_name.strip()

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
                'Quantity': 1,
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
        files = glob.glob(pattern)
        input_files.extend([Path(f) for f in files])

    if not input_files:
        return

    all_dataframes = []

    for input_file in input_files:
        try:
            df = process_single_energietraeger_file(input_file)
            if not df.empty:
                all_dataframes.append(df)
        except Exception:
            continue

    if not all_dataframes:
        raise ValueError("No valid data found in any input files")

    combined_df = pd.concat(all_dataframes, ignore_index=True)

    year_columns = []
    for col in combined_df.columns:
        if col not in ['Metric Group', 'Energieträger', 'Quantity', 'Unit', 'Slice']:
            try:
                year = int(col)
                if 1900 <= year <= 2100:
                    year_columns.append(col)
            except (ValueError, TypeError):
                continue

    year_columns = sorted(year_columns, key=int)

    base_columns = ['Metric Group', 'Quantity', 'Unit', 'Slice']
    if 'Energieträger' in combined_df.columns:
        base_columns.insert(1, 'Energieträger')

    final_columns = base_columns + year_columns
    combined_df = combined_df.reindex(columns=final_columns)

    temp_file = str(output_file) + '.tmp'
    combined_df.to_csv(temp_file, sep=';', index=False, encoding='utf-8',
                      quoting=csv.QUOTE_ALL, na_rep='')

    with open(temp_file, 'r', encoding='utf-8') as infile:
        content = infile.read()

    content = content.replace('""', '')

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

    Path(temp_file).unlink()

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_energietraeger_to_uploadable_format.py <input_pattern1> [input_pattern2] ... <output_file>")
        sys.exit(1)

    input_patterns = sys.argv[1:-1]
    output_file = sys.argv[-1]

    try:
        convert_multiple_energietraeger_files(input_patterns, output_file)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()