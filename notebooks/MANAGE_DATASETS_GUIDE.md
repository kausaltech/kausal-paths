# Dataset Management Guide

This guide explains how to use `manage_datasets.py` to convert, transform, and upload datasets using YAML configuration files.

## Overview

`manage_datasets.py` provides a flexible, YAML-driven pipeline for:
- Loading data from CSV, Excel, and Parquet files
- Transforming data using a sequence of operations
- Extracting metadata (units, metrics, descriptions)
- Converting dimension names to category IDs
- Pushing datasets to DVC repositories

## Quick Start

```bash
# Recommended: Run as module to ensure Django/environment setup
python -m notebooks.manage_datasets your_config.yaml

# Alternative: Direct script execution (may not work for operations requiring Django context)
python notebooks/manage_datasets.py your_config.yaml
```

**Note**: Use `python -m notebooks.manage_datasets` when you need access to environment-specific objects like dimensions (e.g., for `convert_names_to_cats` operation). This ensures proper Django setup and context initialization.

## Configuration Structure

### Basic Structure

```yaml
# Top-level settings (optional, can be inherited by all datasets)
output_file_path: "output/dataset.csv"
instance: "mainz-bisko"  # Instance ID for context and dimensions
metrics:
- name: Energy Consumption
  id: energy_consumption
  quantity: energy
  unit: kWh
  column: energy  # Optional: custom column name

# Single dataset or multiple datasets
datasets:
- input_file_path: "data/file.xlsx"
    # ... dataset configuration
```

### Property Inheritance

When multiple datasets are defined, each dataset inherits properties from the previous one. This allows you to define common settings once:

```yaml
instance: "mainz-bisko"  # Inherited by all datasets
metrics:
- name: Energy Consumption
  id: energy_consumption
  quantity: energy
  unit: kWh

datasets:
- input_file_path: "data/file1.xlsx"
  # Inherits instance and metrics from top level

- input_file_path: "data/file2.xlsx"
  # Also inherits instance and metrics
  # Can override: instance: "other-instance"
```

## File Formats

### CSV Files

```yaml
datasets:
- input_file_path: "data/file.csv"
  file_type: "csv"  # Auto-detected if not specified
```

### Excel Files

```yaml
datasets:
- input_file_path: "data/file.xlsx"
  file_type: "excel"  # Auto-detected
  
  # Option 1: Process single sheet
  sheet_name: "2020"
  
  # Option 2: Process multiple sheets with year mapping
  sheet_year_mapping:
    "2020": 2020
    "2021": 2021
    "2022": 2022
  
  # Option 3: Select specific Excel range
  excel_range: "A3:D20"  # Start at A3, end at D20
```

### Parquet Files

```yaml
datasets:
- input_file_path: "data/file.parquet"
  file_type: "parquet"  # Auto-detected
  # No schema needed - data loaded as-is
```

## Schema-Based Processing

For structured data (CSV/Excel), you can define a schema to extract data:

```yaml
datasets:
- input_file_path: "data/energy_data.xlsx"
  row_identifier_name: "EnergyCarrier"  # Name for identifier column
  identifier_column: 0  # 0-based column index
  skip_rows: 4  # Skip header/metadata rows
  has_header: true  # First data row is header
  
  dimension_names:
  - "EnergyCarrier"
  - "Sector"
  
  column_specs:
    # Column 0: Identifier column (EnergyCarrier)
    
    # Column 1: Sector data
    1:
      quantity: "Energy consumption"
      unit: "kWh"
      dimensions:
        Sector: "Industry"
    
    # Column 2: Another sector
    2:
      quantity: "Energy consumption"
      unit: "kWh"
      dimensions:
        Sector: "Transport"
```

## Operations

Operations are executed in the order they appear in the `operations` list. Each operation transforms the DataFrame (we use Polars DataFrames) and passes it to the next.

### Data Manipulation Operations

#### `filter`
Filter rows using Polars expressions.

```yaml
operations:
- type: filter
  params:
    expr: "pl.col('value') > 0"
- type: filter
  params:
    expr: "pl.col('year') >= 2020"
```

#### `with_columns`
Add new columns using Polars expressions.

```yaml
operations:
- type: with_columns
  params:
    expr: "pl.col('value') * 1000"  # Single expression

# Or multiple expressions:
- type: with_columns
  params:
    exprs:
    - "pl.col('value') * 1000"
    - "pl.col('year').cast(pl.Int64)"
```

#### `drop`
Remove columns.

```yaml
operations:
- type: drop
  params:
    columns: ["temp_col", "unused_col"]
```

#### `rename`
Rename columns.

```yaml
operations:
- type: rename
  params:
    mapping:
      old_name: "new_name"
      "Old Name": "new_name"
```

#### `select`
Select specific columns.

```yaml
operations:
- type: select
  params:
    columns: ["year", "sector", "value"]
```

### Metadata Operations

#### `define_metrics`
Create `metric_col` column from YAML-defined metrics. Maps Metric/Quantity values to metric IDs.

```yaml
metrics:
- name: Energy Consumption
  id: energy_consumption
  quantity: energy
  unit: kWh
  column: energy  # Optional: custom column name

datasets:
- input_file_path: "data.csv"
  operations:
  - type: define_metrics
    # Requires metrics to be defined in config
```

#### `extract_units`
Extract units from DataFrame using `metric_col` and `Unit` columns.

```yaml
operations:
- type: extract_units
  # Requires metric_col and Unit columns
  # Stores units internally for later use
```

#### `extract_units_from_row`
Extract units from the first row if it contains only strings.

```yaml
operations:
- type: extract_units_from_row
  # Checks if first row is metadata (no numbers)
  # Extracts units and removes that row
```

#### `extract_metadata`
Remove metadata columns (Quantity, Unit, Metric, metric_col) from DataFrame.

```yaml
operations:
- type: extract_metadata
  # Removes: Quantity, Unit, Metric, metric_col
  # Prepares DataFrame for final output
```

#### `print_metadata`
Print metadata summary for verification before pushing to DVC.

```yaml
operations:
- type: print_metadata
  params:
    sample_rows: 10  # Optional: number of sample rows to show
    # units: {...}    # Optional: override units to print
```

### Data Transformation Operations

#### `clean_dataframe`
Remove metadata columns and empty columns.

```yaml
operations:
- type: clean_dataframe
```

#### `convert_to_standard_format`
Convert DataFrame to standard format with Year column if needed.

```yaml
operations:
- type: convert_to_standard_format
```

#### `pivot_by_compound_id`
Pivot metrics to columns (one column per metric).

```yaml
operations:
- type: pivot_by_compound_id
  # Transforms long format to wide format
  # Metrics become separate columns
```

#### `prepare_for_dvc`
Standardize column names for DVC.

```yaml
operations:
- type: prepare_for_dvc
  params:
    units:
      energy: "kWh"
      emissions: "tCO2e"
```

#### `to_snake_case_columns`
Convert column names to snake_case.

```yaml
operations:
- type: to_snake_case_columns
  params:
    exclude: ["Year"]  # Optional: columns to exclude
```

### Dimension Operations

#### `extract_dimensions_from_text`
Extract dimension columns from unstructured text using keyword matching.

```yaml
operations:
- type: extract_dimensions_from_text
  params:
    column: "Description"  # Optional, defaults to 'Description'
    mapping:
      sector:
        'industrie': 'industry'
        'verkehr': 'transport'
        'gebäude': 'buildings'
      energy_carrier:
        'erdgas': 'natural_gas'
        'strom': 'electricity'
        'biomasse': 'biomass'
    verbose: true  # Optional: print extraction summary
```

#### `convert_names_to_cats`
Convert names in dimension columns to category IDs using instance context.

```yaml
instance: "mainz-bisko"  # Required for this operation

datasets:
- input_file_path: "data.csv"
  operations:
  - type: convert_names_to_cats
    params:
      units:
        energy: "kWh"
        emissions: "tCO2e"
```

### DVC Operations

#### `push_to_dvc`
Push dataset to DVC repository.

```yaml
instance: "mainz-bisko"
metrics:
- name: Energy Consumption
  id: energy_consumption
  quantity: energy
  unit: kWh

datasets:
- input_file_path: "data.csv"
  operations:
  - type: define_metrics
  - type: extract_units
  - type: pivot_by_compound_id
  - type: extract_metadata
  - type: push_to_dvc
    params:
      output_path: "datasets/my_dataset"
      dataset_name: "Energy Data"
      language: "en"
      # units: {...}  # Optional: override extracted units
      # description: "Custom description"  # Optional
```

## Complete Examples

### Example 1: Excel File with Multiple Sheets

```yaml
instance: "mainz-bisko"
output_file_path: "output/energy_data.csv"

metrics:
- name: Energy Consumption
  id: energy_consumption
  quantity: energy
  unit: kWh
  column: energy

datasets:
- input_file_path: "data/energy_by_year.xlsx"
  sheet_year_mapping:
    "2020": 2020
    "2021": 2021
    "2022": 2022
  
  row_identifier_name: "EnergyCarrier"
  identifier_column: 0
  skip_rows: 4
  has_header: true
  
    dimension_names:
    - "EnergyCarrier"
    - "Sector"
    
    column_specs:
      1:
        quantity: "Energy consumption"
        unit: "kWh"
        dimensions:
          Sector: "Industry"
      2:
        quantity: "Energy consumption"
        unit: "kWh"
        dimensions:
          Sector: "Transport"
    
    operations:
    - type: define_metrics
    - type: extract_units
    - type: pivot_by_compound_id
    - type: extract_metadata
    - type: print_metadata
    - type: push_to_dvc
      params:
        output_path: "datasets/energy"
        dataset_name: "Energy Consumption by Sector"
```

### Example 2: Parquet File with Operations

```yaml
instance: "mainz-bisko"

metrics:
- name: Energy Consumption
  id: energy_consumption
  quantity: energy
  unit: kWh
- name: CO2 Emissions
  id: co2_emissions
  quantity: emissions
  unit: tCO2e

datasets:
- input_file_path: "data/klimaschutz_data.parquet"
  operations:
  - type: filter
    params:
      expr: "pl.col('year') >= 2020"
  - type: define_metrics
  - type: extract_units
  - type: pivot_by_compound_id
  - type: extract_metadata
  - type: convert_names_to_cats
    params:
      units:
        energy_consumption: "kWh"
        co2_emissions: "tCO2e"
  - type: print_metadata
  - type: push_to_dvc
    params:
      output_path: "datasets/climate_data"
      dataset_name: "Climate Data"
```

### Example 3: CSV with Text-Based Dimension Extraction

```yaml
datasets:
- input_file_path: "data/unstructured_data.csv"
  operations:
  - type: extract_dimensions_from_text
    params:
      column: "Description"
      mapping:
        sector:
          'industrie': 'industry'
          'verkehr': 'transport'
          'gebäude': 'buildings'
        energy_carrier:
          'erdgas': 'natural_gas'
          'strom': 'electricity'
      verbose: true
  - type: filter
    params:
      expr: "pl.col('sector').is_not_null()"
  - type: to_snake_case_columns
```

### Example 4: Multiple Datasets with Inheritance

```yaml
instance: "mainz-bisko"
output_file_path: "output/combined.csv"

metrics:
- name: Energy Consumption
  id: energy_consumption
  quantity: energy
  unit: kWh

datasets:
# First dataset: Excel with schema
- input_file_path: "data/energy_2020.xlsx"
  sheet_name: "2020"
  row_identifier_name: "EnergyCarrier"
  identifier_column: 0
  skip_rows: 4
  has_header: true
  dimension_names: ["EnergyCarrier", "Sector"]
  column_specs:
    1:
      quantity: "Energy consumption"
      unit: "kWh"
      dimensions:
        Sector: "Industry"
  operations:
  - type: define_metrics
  - type: extract_units

# Second dataset: Inherits instance, metrics, and operations
- input_file_path: "data/energy_2021.xlsx"
  sheet_name: "2021"
  # Inherits all schema and operations from first dataset
  # Can override specific properties if needed
```

## Operation Execution Order

Operations are executed sequentially. Typical order:

1. **Data Loading**: File is loaded (automatic)
2. **Filtering**: `filter` - Remove unwanted rows
3. **Metric Definition**: `define_metrics` - Create metric_col
4. **Unit Extraction**: `extract_units` or `extract_units_from_row`
5. **Data Transformation**: `pivot_by_compound_id` - Convert to wide format
6. **Metadata Cleanup**: `extract_metadata` - Remove metadata columns
7. **Dimension Conversion**: `convert_names_to_cats` - Convert to category IDs
8. **Verification**: `print_metadata` - Check before pushing
9. **DVC Push**: `push_to_dvc` - Upload to repository

## Best Practices

1. **Define metrics at top level**: If multiple datasets use the same metrics, define them once at the top level.

2. **Use property inheritance**: Define common settings (instance, metrics) at the top level and let datasets inherit them.

3. **Verify before pushing**: Always use `print_metadata` before `push_to_dvc` to verify units and structure.

4. **Extract units early**: Run `extract_units` before `pivot_by_compound_id` or `extract_metadata` (which remove the Unit column).

5. **Use Excel ranges**: For large Excel files, use `excel_range` to select only the data area.

6. **Test incrementally**: Start with simple operations and add complexity gradually.

## Troubleshooting

### Units are empty after push_to_dvc
- Ensure `extract_units` runs before `pivot_by_compound_id` or `extract_metadata`
- Or define units in YAML metrics (they'll be used automatically)
- Or provide units explicitly in `push_to_dvc` params

### Metric column not found
- Run `define_metrics` before operations that need `metric_col`
- Ensure metrics are defined in YAML config

### Context not found for convert_names_to_cats
- Set `instance` at top level or in dataset config
- Ensure the instance ID exists in your configs directory

### Excel range parsing fails
- Use format: "A3:D20" (start cell:end cell)
- Column letters must be uppercase
- Row numbers start at 1 (Excel-style, not 0-based)

## See Also

- `notebooks/upload_new_dataset.py` - Functions used by operations
- `configs/*.yaml` - Example instance configurations
