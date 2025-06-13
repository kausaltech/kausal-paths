# Usage:
# python convert_koln_export_to_uploadable_format.py "/Users/jouni/Downloads/KSP
#  Export/datenzeitreihen*" datenzeitreihen_combined.csv
# python convert_energietraeger_to_uploadable_format.py  "/Users/jouni/Downloads/KSP
#  Export/energietraeger*" combined_energietraeger.csv
# python combine_csvs.py datenzeitreihen_combined.csv combined_energietraeger.csv combined_bisko.csv
from __future__ import annotations

import csv
import glob
import re
import sys
from pathlib import Path

import pandas as pd

# Import the quantity constants
sys.path.append('..')
from nodes.constants import EMISSION_FACTOR_QUANTITY, ENERGY_QUANTITY, FLOOR_AREA_QUANTITY, MILEAGE_QUANTITY, NUMBER_QUANTITY


def extract_metric_and_unit(metric_text):
    """Extract metric name and unit from text like 'Metric name [unit]'."""
    unit_match = re.search(r'\[([^\]]+)\]$', metric_text)
    if unit_match:
        unit = unit_match.group(1)
        metric = metric_text[:unit_match.start()].strip()
    else:
        metric = metric_text.strip()
        unit = ""
    return metric, unit

def extract_bisko_code(metric_text):
    """Extract any value in parentheses at the end and add to BISKO column."""
    bisko_code = ""
    cleaned_text = metric_text

    match = re.search(r'\(([^)]+)\)\s*$', metric_text)
    if match:
        code_with_bisko = match.group(1).strip()

        if code_with_bisko.upper().startswith('BISKO'):
            bisko_code = re.sub(r'^BISKO\s*', '', code_with_bisko, flags=re.IGNORECASE).strip()
        else:
            bisko_code = code_with_bisko

        cleaned_text = metric_text[:match.start()].strip()
        cleaned_text = re.sub(r',\s*BISKO\s*$', '', cleaned_text, flags=re.IGNORECASE).strip()

    return cleaned_text, bisko_code

def convert_unit_and_get_quantity(unit):
    """Convert unit according to the rules and determine the appropriate quantity."""
    unit = unit.strip()

    if unit == "t CO2-Äqu./MWh":
        return "t/MWh", EMISSION_FACTOR_QUANTITY
    elif unit == "g CO2-Äqu./Wh":
        return "g/Wh", EMISSION_FACTOR_QUANTITY
    elif unit == "MWh":
        return "MWh/a", ENERGY_QUANTITY
    elif unit == "GJ":
        return "GJ/a", ENERGY_QUANTITY
    elif unit == "Anzahl":
        return "pcs", NUMBER_QUANTITY
    elif unit == "Wege/Person/d":
        return "trips/person/d", NUMBER_QUANTITY
    elif unit == "km/Weg":
        return "km/trip", NUMBER_QUANTITY
    elif unit == "%":
        return "%", NUMBER_QUANTITY
    elif unit in ["Mio. Fz-km", "Mio. Zug-km"]:
        return "Mvkm", MILEAGE_QUANTITY
    elif unit == "m²":
        return "m²", FLOOR_AREA_QUANTITY
    else:
        return unit, 1

def transform_slice_name(filename_stem):
    """Transform filename to slice name according to the rules."""
    if filename_stem.startswith('datenzeitreihen'):
        if filename_stem == 'datenzeitreihen':
            return "Emissionsfaktoren für die Energieerzeugung"
        elif filename_stem == 'datenzeitreihen-2':
            return "Emissionsfaktoren für die Industrie"
        elif filename_stem == 'datenzeitreihen-3':
            return "Emissionsfaktoren für den Verkehr"
        elif filename_stem == 'datenzeitreihen-4':
            return "Aktivitäten, Verkehr und Gebäude"
        else:
            return filename_stem
    else:
        return 'datenzeitreihen'

def extract_region_and_clean_metric(metric_text):
    """Extract region (Lokal/National) and clean the metric text."""
    region = "National"
    cleaned_text = metric_text

    if metric_text.lower().startswith('lokaler'):
        region = "Lokal"
        cleaned_text = re.sub(r'^lokaler\s*', '', metric_text, flags=re.IGNORECASE).strip()

    return cleaned_text, region

def extract_haushalte_info(metric_text):
    """Extract household size information from metric text."""
    if "1-Personen-Haushalte" in metric_text:
        return "1-Personen"
    elif "2-Personen-Haushalte" in metric_text:
        return "2-Personen"
    elif "3-Personen-Haushalte" in metric_text:
        return "3-Personen"
    elif "4-Personen-Haushalte" in metric_text:
        return "4-Personen"
    elif "5-Personen-Haushalte" in metric_text:
        return "5-Personen"
    elif "mehr als 5 Personen" in metric_text or "Haushalte mit mehr als 5 Personen" in metric_text:
        return "Mehr als 5 Personen"
    return ""

def extract_strassentyp_info(metric_text):
    """Extract street type information from metric text."""
    metric_lower = metric_text.lower()

    if "auf autobahn" in metric_lower:
        return "Autobahn"
    elif "außerorts" in metric_lower:
        return "Außerorts"
    elif "innerorts" in metric_lower:
        return "Innerorts"

    return ""

def extract_wohnflaeche_info(metric_text):
    """Extract WohnFläche information from metric text."""
    if metric_text.startswith("Wohnfläche in "):
        # Remove "Wohnfläche in " and return the rest
        return metric_text[len("Wohnfläche in "):].strip()

    return ""

def process_single_file(input_file):
    """Process a single Excel/CSV file and return the converted DataFrame."""
    if input_file.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file, header=0)
    else:
        df = pd.read_csv(input_file, sep=';', encoding='utf-8', quotechar='"',
                        skipinitialspace=True, quoting=csv.QUOTE_ALL)

    header_row = None
    for i, row in df.iterrows():
        if pd.notna(row.iloc[0]) and 'Eingabefeld' in str(row.iloc[0]):
            header_row = i
            break

    if header_row is not None:
        new_columns = []
        for j, val in enumerate(df.iloc[header_row]):
            if pd.notna(val):
                new_columns.append(str(val))
            else:
                new_columns.append(f"Unnamed_{j}")

        df.columns = new_columns
        df = df.iloc[header_row + 1:].reset_index(drop=True)

    df = df.dropna(how='all')

    metric_col_name = df.columns[0]

    year_columns = []
    year_column_mapping = {}

    for col in df.columns[1:]:
        try:
            year = int(float(str(col)))
            if 1900 <= year <= 2100:
                year_str = str(year)
                year_columns.append(year_str)
                year_column_mapping[col] = year_str
        except (ValueError, TypeError):
            continue

    if not year_columns:
        return pd.DataFrame()

    filename_stem = input_file.stem
    slice_value = transform_slice_name(filename_stem)

    output_rows = []

    for _, row in df.iterrows():
        metric_text = str(row[metric_col_name])
        if pd.isna(row[metric_col_name]) or metric_text.strip() == '' or metric_text == 'nan':
            continue

        # Extract household and street type info from original text
        haushalte = extract_haushalte_info(metric_text)
        strassentyp = extract_strassentyp_info(metric_text)

        # Extract metric name and unit (keeping commas intact)
        metric_name, unit = extract_metric_and_unit(metric_text)

        # Extract region and clean the metric name
        cleaned_metric_name, region = extract_region_and_clean_metric(metric_name)

        # Check if it's an emission factor and remove "Emissionsfaktor" from the text
        is_emission_factor = cleaned_metric_name.startswith("Emissionsfaktor")
        if is_emission_factor:
            cleaned_metric_name = re.sub(r'^Emissionsfaktor\s*', '', cleaned_metric_name).strip()

        # Remove "für" from the beginning if present
        if cleaned_metric_name.startswith("für"):
            cleaned_metric_name = re.sub(r'^für\s*', '', cleaned_metric_name).strip()

        # Extract BISKO code and clean the metric name further
        final_metric_name, bisko_code = extract_bisko_code(cleaned_metric_name)

        # Now extract WohnFläche info from the final cleaned metric name
        wohnflaeche = extract_wohnflaeche_info(final_metric_name)

        # Convert unit and get quantity
        converted_unit, quantity = convert_unit_and_get_quantity(unit)

        if is_emission_factor:
            quantity = EMISSION_FACTOR_QUANTITY
            metric_group = "Emission Factor"
            energietraeger = final_metric_name
        else:
            metric_group = final_metric_name
            energietraeger = None

        output_row = {
            'Metric Group': metric_group,
            'Energieträger': energietraeger,
            'Haushalte': haushalte if haushalte else None,
            'Straßentyp': strassentyp if strassentyp else None,
            'WohnFläche': wohnflaeche if wohnflaeche else None,
            'Quantity': quantity,
            'Unit': converted_unit,
            'Slice': slice_value,
            'Region': region,
            'BISKO': bisko_code if bisko_code else None
        }

        has_data = False
        for actual_col, year_str in year_column_mapping.items():
            if actual_col in row.index:
                value = row[actual_col]
                if pd.notna(value) and str(value).strip() != '' and str(value) != 'nan':
                    try:
                        output_row[year_str] = float(value)
                        has_data = True
                    except (ValueError, TypeError):
                        output_row[year_str] = None
                else:
                    output_row[year_str] = None
            else:
                output_row[year_str] = None

        if has_data:
            output_rows.append(output_row)

    file_df = pd.DataFrame(output_rows)

    if not file_df.empty:
        year_cols_in_output = [col for col in file_df.columns if col.isdigit()]
        file_df = file_df.dropna(subset=year_cols_in_output, how='all')

    return file_df

def convert_multiple_files(input_patterns, output_file):
    """Convert multiple BISKO Excel/CSV files to a single upload script format CSV."""

    all_files = []
    for pattern in input_patterns:
        if '*' in pattern or '?' in pattern:
            matched_files = glob.glob(pattern)
            all_files.extend([Path(f) for f in matched_files])
        else:
            file_path = Path(pattern)
            if file_path.exists():
                all_files.append(file_path)

    if not all_files:
        raise ValueError("No input files found")

    all_dataframes = []
    for input_file in all_files:
        try:
            df = process_single_file(input_file)
            if not df.empty:
                all_dataframes.append(df)
        except Exception as e:
            continue

    if not all_dataframes:
        raise ValueError("No valid data found in any input files")

    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)

    year_cols = sorted([col for col in combined_df.columns if col.isdigit()])
    metadata_cols = ['Metric Group', 'Energieträger', 'Haushalte', 'Straßentyp', 'WohnFläche', 'Quantity',
                     'Unit', 'Slice', 'Region', 'BISKO']
    final_columns = metadata_cols + year_cols
    combined_df = combined_df[final_columns]

    combined_df = combined_df.drop_duplicates()

    temp_file = str(output_file) + '.tmp'
    combined_df.to_csv(temp_file, index=False, sep=';', encoding='utf-8',
                      quotechar='"', quoting=csv.QUOTE_NONNUMERIC, na_rep='')

    with open(temp_file, 'r', encoding='utf-8') as infile:  # noqa: PTH123, UP015
        content = infile.read()

    content = content.replace('""', '')

    with open(output_file, 'w', encoding='utf-8') as outfile:  # noqa: PTH123
        outfile.write(content)

    Path(temp_file).unlink()

    return combined_df

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_koln_export_to_uploadable_format.py <input_file1> [input_file2] [input_file3] ... <output_file>")
        sys.exit(1)

    input_patterns = sys.argv[1:-1]
    output_file = Path(sys.argv[-1])

    try:
        convert_multiple_files(input_patterns, output_file)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
