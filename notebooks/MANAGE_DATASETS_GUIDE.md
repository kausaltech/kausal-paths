# Dataset Management Guide

This guide explains how to use `manage_datasets.py` to convert, transform, and upload datasets using YAML configuration files.

## Overview

`manage_datasets.py` provides a flexible, YAML-driven pipeline for:
- Loading data from **files** (CSV, Excel, Parquet), **DVC**, or the **database**
- Transforming data using a sequence of operations
- Extracting metadata (units, metrics, descriptions)
- Converting dimension names to category IDs
- Pushing datasets to DVC repositories

## Schema vs DataFrame metadata (meta)

Two different concepts affect how your data is interpreted:

| Concept | When it applies | What it describes |
|--------|----------------------------------|-------------------|
| **Schema** | Only when `source: file` and you define `row_identifier_name` (and related keys) in YAML | How to **ingest** the file: which columns are identifiers, dimensions, or data; skip rows; header row; column specs (quantity, unit per column). Used to *collect* rows from Excel/CSV into a normalized table. |
| **Meta** | When you load from **DVC** or the **database** (`source: dvc` or `source: db`) | **Semantics** of the already-normalized table: **units** per metric column and **primary keys** (Year + dimensions). Carried by `PathsDataFrame`; captured before operations and reattached after so it is preserved through the pipeline. |

- **File without schema**: You load a CSV/Excel/Parquet as-is (no schema). No meta. Units come from YAML `metrics`, from `extract_units` / `extract_units_from_row`, or from `push_to_dvc` params.
- **File with schema**: You describe the layout of the file; the pipeline collects and transforms it. Units can come from the schema’s `column_specs`, from `extract_units`, or from YAML metrics / `push_to_dvc`.
- **DVC or DB**: You get a `PathsDataFrame` with **meta** (units + primary keys). That meta is kept across operations. To **change a unit**, use the `set_unit` operation (see below).

### How to set or change units

- **File pipeline (no schema)**  
  - Define **metrics** in YAML with `unit` and (if needed) `column`.  
  - Or run **`extract_units`** / **`extract_units_from_row`** and optionally override in **`push_to_dvc`** params (`units: { metric_id: unit }`).

- **File pipeline (with schema)**  
  - Units can come from **`column_specs`** in the schema, from **`extract_units`**, or from YAML **metrics** / **`push_to_dvc`** params.

- **DVC or DB pipeline**  
  - Units are in the **meta** of the loaded `PathsDataFrame` and are preserved through operations.  
  - To **change** a unit, add a **`set_unit`** operation (see [set_unit](#set_unit)) with `mapping: { column_name: "new_unit" }` or `column` + `unit`.

## Pipeline order and standard objects (design)

A clear, recommended order of operations and use of standard types keeps the pipeline predictable and reusable.

### Recommended process order

1. **Load data** from file, DVC, or DB (raw bytes / DVC dataset / DB row set).
2. **If relevant, use schema** to ingest the right data into a `pl.DataFrame` (only for file source with `row_identifier_name` etc.; otherwise the loaded table is already the DataFrame).
3. **Collect other metadata** that is not already in the DataFrame: dataset **name**, **description**, and any metric labels/quantity info that will become **NodeMetric** (see below). This is the place for “dataset-level” and “metric-level” metadata that does not live in PathsDataFrame meta.
4. **Convert `pl.DataFrame` into `PathsDataFrame`** with proper **meta**: units per metric column and primary keys (Year + dimensions). So all *structural* and *unit* information lives in the PathsDataFrame.
5. **Create `NodeMetric`** for each quantity: id, column_id, unit, quantity, and multilingual **label** (e.g. `I18nString` / `dict[str, str]`). These are the standard metric descriptors used elsewhere (e.g. in nodes); they carry quantity and labels, and units can be taken from PathsDataFrame meta or from the metric when pushing.
6. **Run data-management operations** (filter, rename, set_unit, etc.). PathsDataFrame supports normal Polars-style operations; meta is preserved where implemented (and reattached after the operation list when loading from DVC/DB).
7. **Save to CSV** if needed (for inspection or downstream tools).
8. **Push to DVC** with **data + metadata**: the PathsDataFrame, dataset name, description, and list of **NodeMetric** (instead of ad-hoc MetricData). The DVC layer can derive `units` and `metadata.metrics` from NodeMetric + PathsDataFrame meta.

This order keeps: **schema** for ingestion (steps 1–2), **meta** for table semantics (step 4, then preserved in 6), and **NodeMetric + dataset name/description** for the rest of the metadata used at push time (steps 3, 5, 8).

### Standard objects to use

- **PathsDataFrame** (from `common.polars`): the main data type after step 4. It carries **meta** (units, primary keys). Use it through the pipeline so that units and index structure are explicit and preserved.
- **NodeMetric** (from `nodes.node`): metric descriptor with `id`, `column_id`, `unit`, `quantity`, and optional multilingual `label`. Use this instead of ad-hoc types (e.g. `MetricData` in `upload_new_dataset.py`) so that dataset metrics align with the rest of the system (nodes, quantities).
- **Dataset-level metadata**: a single place for **dataset name** and **description** (and optionally language/tags). These are not columns or PathsDataFrame meta; they belong to the “dataset” entity and should be collected in step 3 and passed to step 8 (push_to_dvc) together with the PathsDataFrame and the list of NodeMetric.

### Implemented behaviour

- **PathsDataFrame**: Use the **`to_paths_dataframe`** operation (with `units` and `primary_keys` params) after loading from file to get a PathsDataFrame. Data loaded from DVC/DB is already a PathsDataFrame and meta is preserved across operations.
- **NodeMetric**: `push_to_dvc` accepts **`list[NodeMetric]`**; metrics are built from YAML `metrics` (MetricSpec) or from **`extract_metrics`** (converted to NodeMetric). DVC metadata stores id, column_id, quantity, unit, and label.
- **Dataset name and description**: Set **`dataset_name`** and **`description`** on the dataset config (or in `push_to_dvc` params). They are passed through the executor into `push_to_dvc`.
- **Units**: When the DataFrame is a PathsDataFrame, `push_to_dvc` uses **units from its meta** unless overridden by params.

### Implementation notes (what happens under the hood)

- **Django**: When you run `python -m notebooks.manage_datasets config.yaml`, the script calls **`django.setup()`** at the start of `main()`. That makes the app registry ready for operations that need it (e.g. **NodeMetric**, loading from **DVC/DB**, **get_context** for `convert_names_to_cats`). Always run the script as the main program (e.g. `python -m notebooks.manage_datasets`) so that setup runs; then all operations work.
- **PathsDataFrame and dvc-pandas**: The **dvc-pandas** library expects a Polars DataFrame whose **`.to_pandas()`** returns a pandas DataFrame with **columns** for the index (it then calls **`set_index(index_columns)`** itself). **PathsDataFrame.to_pandas()** instead moves primary keys into the pandas index. So when you pass a PathsDataFrame into **`push_to_dvc`**, the code **converts it to a plain Polars DataFrame** (using the underlying `_df`) before creating the dvc-pandas `Dataset`. That way dvc-pandas gets normal columns and can do `set_index` correctly. You don’t need to change your config; this conversion is automatic. Your pipeline can use PathsDataFrame all the way; only the final write to DVC uses the plain frame.
- **Lazy imports**: **NodeMetric** and Django-dependent code are imported only when an operation needs them (e.g. when **`push_to_dvc`** or DVC/DB loading runs). That avoids **AppRegistryNotReady** if the module is imported without running as main.

## Quick Start

```bash
# Recommended: run as module so the project root is on the path
python -m notebooks.manage_datasets your_config.yaml

# Or run the script directly (Django is set up automatically when the script runs)
python notebooks/manage_datasets.py your_config.yaml
```

**Note**: The script calls **Django setup** at startup when run as main, so operations that need the app (e.g. **NodeMetric**, DVC/DB loading, **convert_names_to_cats** with instance context) work without extra steps.

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

# Optional dataset-level metadata (used e.g. by push_to_dvc)
dataset_name: "My Dataset"
description: "Optional description for DVC metadata"

# Single dataset or multiple datasets
datasets:
- input_file_path: "data/file.xlsx"
  dataset_name: "Overridden name"   # optional per-dataset
  description: "Optional description"
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

### Loading from DVC or the database

Instead of a file, you can load a dataset from DVC or from the database. The result is a **PathsDataFrame** with **meta** (units and primary keys), which is preserved across operations.

```yaml
# From DVC (requires instance for context)
datasets:
- source: dvc
  dataset_id: "longmont/greenhouse_gas_emissions_by_subsector"
  instance: "longmont"
  output_file_path: "from_dvc.csv"
  operations: []   # optional: filter, set_unit, etc.

# From database
datasets:
- source: db
  dataset_id: "longmont/greenhouse_gas_emissions_by_subsector"
  instance: "longmont"
  output_file_path: "from_db.csv"
  operations:
  - type: set_unit
    params:
      mapping:
        value: "tCO2e"   # change unit for column "value"
```

- **`instance`** is required for both DVC and DB (used to resolve context and instance config).  
- **Schema-based processing** is not used for `source: dvc` or `source: db`; only the operation list runs on the loaded table.

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

#### `to_paths_dataframe`
Convert a `pl.DataFrame` into a **PathsDataFrame** with explicit units and primary keys (pipeline step 4). Use this after loading from file and applying initial operations so that later steps (e.g. `set_unit`, `push_to_dvc`) can use PathsDataFrame meta.

```yaml
operations:
- type: to_paths_dataframe
  params:
    units:
      value: "kWh"
      emissions: "tCO2e"
    primary_keys: ["Year", "sector", "category"]
```

- **`units`**: dict of column name → unit string (e.g. `"kWh"`, `"tCO2e"`).
- **`primary_keys`**: list of column names that form the index (typically `Year` plus dimension columns).

#### `set_unit` {#set_unit}
Set or change units on metric columns. **Only works when the DataFrame has unit metadata** (i.e. it is a PathsDataFrame from `source: dvc` or `source: db`, or after `to_paths_dataframe`). Use `metrics` in YAML or `push_to_dvc` params for file-based pipelines.

```yaml
# After loading from DVC or DB:
operations:
- type: set_unit
  params:
    mapping:
      value: "tCO2e"
      energy: "GWh"

# Or a single column:
- type: set_unit
  params:
    column: "value"
    unit: "EUR/vehicle"
```

- **`mapping`**: dict of `column_name -> unit string` (e.g. `"kWh"`, `"tCO2e"`).  
- **`column`** and **`unit`**: alternative to `mapping` for a single column.  
- Existing units are overwritten.

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
Push dataset to DVC repository. Metrics are sent as **NodeMetric** (from YAML `metrics` or from **`extract_metrics`**). Units are taken from the **PathsDataFrame meta** when the data is a PathsDataFrame (e.g. after **`to_paths_dataframe`** or load from DVC/DB), or from params / YAML metrics. If the data is a PathsDataFrame, it is converted to a plain Polars DataFrame internally before calling dvc-pandas, so index columns remain as columns and the DVC write succeeds (see [Implementation notes](#implementation-notes-what-happens-under-the-hood)).

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

### How do I set or change a unit?
- **File-based data**: Use YAML **metrics** (`unit` per metric), or **`extract_units`** / **`extract_units_from_row`**, or pass **`units`** in **`push_to_dvc`** params. See [Schema vs DataFrame metadata (meta)](#schema-vs-dataframe-metadata-meta) and [How to set or change units](#how-to-set-or-change-units).
- **Data from DVC or DB**: The table already has units in its **meta**. Use the **`set_unit`** operation to change them (see [set_unit](#set_unit)).

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

### "None of ['col1', 'col2'] are in the columns" when pushing to DVC
- This happened when passing a **PathsDataFrame** to dvc-pandas: PathsDataFrame’s **`.to_pandas()`** moves primary keys into the index, so dvc-pandas’ later **`set_index(index_columns)`** failed. The code now **converts PathsDataFrame to a plain Polars DataFrame** before creating the dvc-pandas Dataset, so index columns stay as columns. If you see this error on an older version, upgrade so that this conversion is in place (see [Implementation notes](#implementation-notes-what-happens-under-the-hood)).

### AppRegistryNotReady when running or importing
- Run the script as main (e.g. **`python -m notebooks.manage_datasets config.yaml`**) so that **`django.setup()`** runs at startup. Do not rely on importing the module and calling functions without going through **`main()`** unless Django is already set up elsewhere. NodeMetric and other Django-dependent code are imported lazily so that importing the module alone does not trigger the app registry.

## See Also

- `notebooks/upload_new_dataset.py` - Functions used by operations
- `configs/*.yaml` - Example instance configurations
