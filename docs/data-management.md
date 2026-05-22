# Data Management

This document describes the standard operating procedure for integrating
customer data into Kausal Paths models. Follow these steps whenever a city
or region provides new data files.

---

## Step 1 — Receive data from the customer

Accepted formats: Excel (`.xlsx`), CSV, JSON, Parquet.

Save the file(s) under `data/<city-id>/` (e.g. `data/bayreuth/`). Do not
transform or rename anything yet — keep the original file intact as the
source of record.

---

## Step 2 — Study the data with AI

Open a conversation and ask the AI to read the file and summarise:

- What sheets or sections exist and what each covers.
- Which years are present and whether values are historical measurements,
  targets, or projections.
- Which sectors, categories, or dimensions the data uses.
- Units and any unit conversions that will be needed.
- Whether the data is absolute values or relative (%, ratios, indices).

For Excel files, pay attention to:

- Which cells are **measured values** vs. **model-computed values** (formulas).
- Whether multiple scenarios or pathways exist in the same sheet.
- Embedded metadata rows (headers, units rows, totals) that must be skipped.

Record the relevant cell ranges for each dataset you intend to extract.

---

## Step 3 — Map data to the model

Read the customer's YAML config (e.g. `configs/bayreuth-bisko.yaml`) and
any included module (e.g. `configs/modules/bisko/model.yaml`) to identify:

- Which **dataset IDs** are referenced by which nodes
  (`input_datasets: - id: ...`).
- What **dimensions** each node expects (`input_dimensions`, `output_dimensions`).
- What **unit** each node works in (`unit:`).
- Whether the dataset is a direct replacement for a module default
  (`dataset_replacements`).

Cross-reference the customer data against this list. For each dataset the
model needs, determine:

- Which cells in the customer file provide that data.
- What dimension values the rows represent and how they map to the model's
  dimension category IDs.
- Whether units match or need conversion.

If the customer data covers a concept that has no matching node yet, note it
for Step 4.

---

## Step 4 — Design model structure (new models only)

If the customer has no model yet, draft a node graph based on the data:

- Group data by the natural sectors/categories the file uses.
- Identify which quantities are inputs (energy, activity) vs. outputs
  (emissions) and which are parameters (emission factors).
- Look for an existing module (`modules/bisko/`, `modules/nzc/`, etc.) that
  covers the methodology and can be included with dataset replacements.
- Sketch the required nodes, their types, units, and connections before
  touching any YAML.

---

## Step 5 — Inspect existing datasets

Before writing new data, check what is already in DVC for this customer.
Run `load_nodes.py` to list datasets in use:

```bash
./load_nodes.py -i <instance-id>
```

To inspect the structure of a specific existing dataset, use a Python
snippet:

```bash
python -c "
import dvc_pandas
repo = dvc_pandas.Repository(repo_url='https://github.com/kausaltech/dvctest.git', dvc_remote='kausal-s3')
ds = repo.load_dataset('<city>/<dataset-name>')
print(ds.df.head())
print(ds.meta)
"
```

Check:

- Column names (dimension IDs, metric column name).
- Index columns (dimension columns + `Year`).
- Units stored in the metadata.
- Whether a `Forecast` column is present.

This tells you the exact format a replacement dataset must match.

---

## Step 6 — Extract data to a standard CSV

Write a Python script `data/<city-id>/create_<topic>_csv.py` that reads the
source file and produces a CSV in the standard wide format.

### Standard CSV format

| Column | Description |
|--------|-------------|
| `Metric` | Always `Value` for single-metric datasets |
| `Unit` | Pint-compatible unit string, e.g. `t_co2e/a`, `MWh/a` |
| `Quantity` | Leave empty (`""`) unless needed |
| `Dataset` | Dataset name (becomes the DVC file name in snake_case) |
| *dimension columns* | One column per dimension (e.g. `sector`, `energy_carrier`). Values must be the model's category IDs. |
| *year columns* | One column per year (e.g. `2019`, `2020`, …). Empty cell = no data for that year. |

Multiple datasets can share one CSV file by using different values in the
`Dataset` column.

### Key rules

- **Units**: keep values in the unit you declare. Do not silently convert
  (e.g. do not divide t → kt and then label the column `t_co2e/a`).
- **Dimension values**: must match the model's category IDs exactly, not
  the human-readable labels from the customer file.
  Use `python -m notebooks.upload_new_dataset ... -n <instance-id>` (Step 7)
  to let the upload script validate and convert names to IDs automatically.
- **No Forecast column needed** in the CSV — it is derived at upload time
  via `forecast_from` (see Step 7 and Step 8).
- Prefer explicit, narrow cell-range extraction over reading whole sheets,
  so the script breaks loudly if the source file layout changes.

### Example extraction script structure

```python
import openpyxl, polars as pl
from pathlib import Path

XLSX = Path(__file__).parent / "Customer File.xlsx"
OUT  = Path(__file__).parent / "topic.csv"

wb = openpyxl.load_workbook(XLSX, data_only=True)
ws = wb["Sheet name"]
rows = list(ws.iter_rows(values_only=True))

# Extract ranges, build records, write CSV
# ...
df.write_csv(OUT, null_value="")
```

---

## Step 7 — Upload to DVC

Run `upload_new_dataset` from the repo root:

```bash
python -m notebooks.upload_new_dataset \
  --input-csv data/<city-id>/<topic>.csv \
  --output-dvc <city-id> \
  --language de \
  --instance <instance-id>
```

- `--output-dvc <city-id>` sets the DVC directory; dataset files will be
  created as `<city-id>/<dataset_name_in_snake_case>.parquet`.
- `--language` sets the metadata language for the dataset name.
- `--instance <instance-id>` loads the model context so the script can
  validate dimension values and convert category names to IDs.

The script will print the units it extracted and the number of rows per
dataset. Verify these before continuing.

After upload, **update the `commit:` field** in the instance YAML to the
new DVC repository HEAD commit so the model picks up the new data:

```yaml
dataset_repo:
  url: https://github.com/kausaltech/dvctest.git
  commit: <new-commit-hash>   # update this
  dvc_remote: kausal-s3
```

---

## Step 8 — Wire datasets to nodes and verify

Add or update the relevant node definitions in the instance YAML. For each
new dataset, check:

**`column: Value`** — required when the metric was uploaded as `Metric=Value`.
Without it the node cannot find the unit in the DataFrame metadata.

```yaml
input_datasets:
- id: <city>/<dataset-name>
  column: Value
```

**`forecast_from`** — required when the parquet does not contain a `Forecast`
column (which is the normal case for data extracted from customer files).
Set it to the first year that should be treated as a forecast. For datasets
that are purely historical, set it to one year after the last data year.

```yaml
input_datasets:
- id: <city>/<dataset-name>
  column: Value
  forecast_from: 2025   # years < 2025 → Forecast=False
```

**Unit compatibility** — the node's `unit:` and the dataset's unit must be
compatible (pint will convert automatically, e.g. `t` → `kt`). If they are
dimensionally incompatible the node raises an `ensure_unit` error.

### Verification

Run the node directly and check that output is printed without errors:

```bash
./load_nodes.py -i <instance-id> --node <node-id>
```

For actions, also run the outcome node to confirm the full pipeline works:

```bash
./load_nodes.py -i <instance-id> --node net_emissions
```

Common errors and their causes:

| Error | Cause |
|-------|-------|
| `KeyError: 'Value'` in `ensure_unit` | Missing `column: Value` in the input_dataset definition |
| `Forecast column missing` | Missing `forecast_from` in the input_dataset definition |
| `Series with type X is not compatible with Y` | Dimensionally incompatible units between node and dataset |
| `Input dataset has duplicate index rows` | Two rows with the same (Year, dimension) combination in the parquet |
| `No input datasets, but node requires one` | Dataset ID typo or wrong `dataset_replacements` mapping |
