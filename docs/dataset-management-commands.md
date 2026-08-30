# Dataset Management Commands

This document describes the main tools for managing datasets in Kausal Paths and explains what data is transformed or lost at each step.

## Overview

There are two layers of dataset storage:

- **DVC** (Data Version Control): Parquet files stored in git + S3. This is the persistent, version-controlled store for input datasets that nodes read from during computation.
- **Django DB**: Datasets stored in PostgreSQL via the `kausal_common.datasets` models. These are the structured datasets (with DataPoints) used by the admin UI and the diffsync-based import/export system.

The two layers are separate: uploading to DVC does not automatically populate the DB, and the DB commands do not touch DVC.

---

## 1. `upload_new_dataset.py` — CSV → DVC

**Location:** `tools/upload_new_dataset.py`  
**Invocation:** `python -m tools.upload_new_dataset -i input.csv -o dvc/path -l de [-n instance-id] [--keep-empty-cells]`

Reads a wide-format CSV and pushes one or more datasets to the DVC parquet store.

### Processing pipeline

1. Load CSV
2. Detect metric column (`Metric` or `Quantity`)
3. Canonicalize metric column values: `"Value"` → `"default"`, others → `snake_case`
4. Extract units and metric metadata (labels, quantity, unit)
5. Remove metadata columns (`Unit`, `Metric`, `Description`)
6. Convert wide format (year columns `2019`, `2020`, …) → long format (one row per year)
7. Pivot on metric column (one column per metric value)
8. Standardize dimension column names to `snake_case`
9. Optionally convert dimension category names to IDs (requires `--instance`)
10. Push to DVC as parquet, with metadata (name, identifier, metrics, units) stored in the `.yaml` sidecar

### What is transformed

| Source (CSV) | Result (DVC parquet) |
|---|---|
| `Metric: "Value"` | Column named `default` |
| `Metric: "Some Label"` | Column named `some_label` (snake_case) |
| `Unit` column | Moved to `.yaml` sidecar `units` dict; key is the canonical column name |
| `Metric` human name | Moved to sidecar `metrics[].label` |
| `Quantity` column | Moved to sidecar `metrics[].quantity` (snake_cased) |
| `Dataset` column | Becomes part of the DVC path (`dvc/path/dataset_name`) |
| Wide year columns | Melted into `Year` (int) column |
| Dimension category names | Category IDs, if `--instance` is used |

### What is lost

- **Empty cells in year columns** — rows with `None`, `"."` or `"-"` values are skipped by default; those year/dimension combinations will be absent from the parquet file. A year column that is *entirely* empty is dropped before the unpivot, so an all-blank file fails with *"No year columns found"* rather than uploading as nothing.
  **`--keep-empty-cells` reverses this for `None`** (never for `"."`/`"-"`, which mean "no data" in a source file). Use it for a template a city is meant to fill in, where the blank cells are the deliverable and a presence check has to tell an entered zero from an untouched one — see [`dataset-csv-format.md`](dataset-csv-format.md) §5, which also covers the `empty_to_zero` binding tag the consuming node then needs. Wide format only; a long-format file with a `Year` column was never affected.
- **`Description` column** — moved to DVC metadata but does not appear in the admin UI (known limitation, marked `FIXME` in code).
- **`Dataset` column** — consumed to split into separate DVC datasets; not stored in the parquet.
- **Historical vs forecast distinction** — parquet files have no `Forecast` column. The split is applied at read time via `forecast_from` in the node's `input_datasets` YAML config.
- **`--datasets` filter** (`-d`) — not implemented; all `Dataset` values in the CSV are always processed.

### Required YAML wiring after upload

After uploading, the node that reads the dataset needs explicit configuration:

```yaml
input_datasets:
  - id: dvc/path/dataset_name
    forecast_from: 2025          # year from which rows are marked as forecast
    column: Value                # required when Metric was "Value" (maps "default" parquet column)
```

Without `column: Value`, the node will raise `KeyError: 'Value'` because the parquet column is named `default`.

---

## 2. `fetch_dataset.py` — DVC parquet → CSV

**Location:** `notebooks/fetch_dataset.py`  
**Invocation:** `python notebooks/fetch_dataset.py <ekey> <output.csv>`

Fetches a single parquet file directly from the S3 bucket by its ekey (either a full S3 object key or an MD5/ETag hash) and writes it as a flat CSV. Useful for inspecting what is actually stored in DVC without checking out the full repository.

### What it does

1. Lists the S3 bucket XML index at `https://s3.kausal.tech/datasets/`
2. Searches for the ekey as a direct S3 key match, then as an ETag match
3. Downloads the parquet file and writes it with `polars.DataFrame.write_csv()`

### What is transformed

| Source (S3 parquet) | Result (CSV) |
|---|---|
| Index columns (dimensions + Year) | Expanded back into regular CSV columns |
| Value columns (e.g. `default`) | Written as-is with their parquet column names |

### What is lost

- **DVC metadata** (`.yaml` sidecar) — units, metric labels, and identifiers are not included in the CSV output.
- **Parquet index** — the parquet file stores dimensions + Year as the DataFrame index; after `write_csv()` these appear as regular columns, which is correct.
- **Forecast flag** — not stored in parquet, so not present in output (same as in the DVC store).

---

## 3. `copy_datasets` — DB → DB (in-instance copy)

**Location:** `nodes/management/commands/copy_datasets.py`  
**Invocation:** `python manage.py copy_datasets <source> <destination> [-N] [-y]`

Copies structured datasets from one `InstanceConfig` scope to another using diffsync. Nothing touches DVC; this operates entirely on the Django DB.

### What is copied

The diffsync adapter loads and syncs these models:

- `Dimension` and `DimensionCategory`
- `DatasetSchema` and `DatasetMetric`
- `Dataset` and `DataPoint`

### What is transformed

- The diffsync sync operation computes a diff between source and destination and applies create/update/delete operations. Existing destination records that match source records by UUID are updated in place.
- If `--yes` (`-y`) is passed, `allow_related_deletion = True` enables deletion of destination records that have no counterpart in the source.

### What is lost / not copied

- **DimensionScope and DatasetSchemaScope** — scope bindings (which instances own a dimension/schema) are not part of the sync. The destination instance must have its own scope bindings, or they are created by the `ScopeAwareDjangoDiffModel` logic when the dimension is new.
- **`--datasets` filter** (`-d`) — option is accepted on the CLI but not yet implemented; all datasets in the source scope are always synced.
- **Dry-run rollback** — with `--dry-run`, the entire transaction is rolled back after the diff is shown. No partial state is persisted.

---

## 4. `sync_datasets` — JSON ↔ DB, DB → CSV

**Location:** `nodes/management/commands/sync_datasets.py`  
**Invocation:** `python manage.py sync_datasets <action> <instance> [file] [-N] [-y]`

Three actions:

### `import` — JSON file → DB

Reads a JSON file produced by `export` and syncs it into the DB for the given instance. The JSON adapter and Django adapter use the same diffsync models, so all fields round-trip without loss.

**What is lost:**
- Same as `copy_datasets`: DimensionScopes are not included.
- `--datasets` filter not implemented.

### `export` — DB → JSON file

Serializes the instance's datasets to a JSON file via `TypedAdapter.save_json()`. The JSON contains all diffsync model types as arrays:

```json
{
  "dimension": [...],
  "dimension_category": [...],
  "dataset_schema": [...],
  "dataset_metric": [...],
  "dataset": [...],
  "data_point": [...]
}
```

All UUIDs are preserved. The file can be re-imported to the same or a different instance.

**What is lost:**
- DimensionScope and DatasetSchemaScope (scope bindings are not serialized).
- `--datasets` filter not implemented.

### `csv` — DB → wide-format CSV

Exports all datasets in the instance scope to a standard wide-format CSV, the same format that `upload_new_dataset.py` reads. Each row represents one metric × dimension combination; years become columns.

**Output column order:** `Metric`, `Unit`, `Quantity` (empty), `Dataset`, then dimension columns in schema-defined order, then year columns in ascending order.

```
Metric,Unit,Quantity,Dataset,sector,2019,2020,...
Value,t_co2e/a,,Historical emissions,heating,330770,...
```

**What is transformed:**

| Source (DB) | Result (CSV) |
|---|---|
| `DatasetSchema.name` | `Dataset` column |
| `DatasetMetric.label` | `Metric` column |
| `DatasetMetric.unit` | `Unit` column |
| `DimensionCategory.identifier` (or label if no identifier) | Dimension column value |
| `DataPoint.date.year` | Year column header |
| `DataPoint.value` | Cell value |

**What is lost:**
- **`Quantity` column** — `DatasetMetric` has no quantity field equivalent; the column is written but left empty.
- **Forecast flag** — not stored in the DB `DataPoint` model; the round-trip through `upload_new_dataset` requires `forecast_from` to be set again in the node's YAML config.
- **`--datasets` filter** not implemented.

---

## 5. `sync_instance_to_db` — YAML instance spec → DB

**Location:** `nodes/management/commands/sync_instance_to_db.py`  
**Invocation:** `python manage.py sync_instance_to_db <instance-id> [--all] [--dry-run] [--start-after <id>] [--runtime-export]`

Parses an instance's YAML config into a structured snapshot and writes its
computation graph into the Django DB (`InstanceConfig` + `NodeConfig`). The
`--runtime-export` option selects the legacy full-runtime introspection path.
This is separate from dataset management: it syncs node kinds, parameters,
ports and related graph configuration, not data values.

This is the step that makes an instance available as a DB-sourced instance, so that it can be served without reading YAML at runtime.

See also `docs/trailhead/tools.md` for the full documentation.

---

## 6. `collect_city_data.py` — Computed node outputs → CSV

**Location:** `tools/collect_city_data.py`  
**Invocation:** `python -m tools.collect_city_data config.yaml`

Not a management command but a standalone CLI. Run it as a module, not as a script path, so the repo root stays on `sys.path`; and run it from the repo root, because a collector config's `output_path` is relative to the working directory. It runs the node computation engine across multiple instances and collects the computed outputs of specified nodes — not the raw input datasets.

### What it does

1. Reads a YAML config listing instances, nodes, target units, and processors.
2. For each instance, loads the context and calls `node.get_output_pl()` on each specified node.
3. Applies a configurable pipeline of processors (e.g. `find_target_values`, `convert_to_target_units`, `sum_over_dims`, `sum_over_instances`, `calculate_difference`).
4. Writes summary CSVs and a log file.

### What is transformed

| Source | Result |
|---|---|
| Node output DataFrame | Filtered to specific years (observation year + target year) |
| Original units | Converted to target units specified in config |
| Multi-dimensional output | Optionally summed over non-year dimensions |
| Values | Rounded to 8 decimal places in summaries |

### What is lost

- **All intermediate years** — unless `find_target_values` is not applied, only the newest observation year and the target year are kept.
- **All dimensions** — `sum_over_dims` collapses all non-year dimensions.
- **Forecast column** — not propagated through the summary pipeline.
- **Node computation errors** — nodes that raise `ValueError` or `NodeComputationError` are skipped and logged; their data is absent from the output without warning in the CSV.
- **`Quantity` metadata** — computed DataFrames carry units via Pint but the output CSV has no unit header row.

---

## Summary: Data loss matrix

| Tool | Loses forecast flag | Loses units | Loses empty-year rows | Loses scope bindings | Filter by dataset |
|---|---|---|---|---|---|
| `upload_new_dataset` | Yes (use `forecast_from`) | No (sidecar) | Yes (silently) | N/A | Not implemented |
| `fetch_dataset` | Yes (not in parquet) | Yes (metadata only) | No | N/A | N/A |
| `copy_datasets` | No | No | No | Yes | Not implemented |
| `sync_datasets import/export` | No | No | No | Yes | Not implemented |
| `sync_datasets csv` | Yes (use `forecast_from`) | No | No | N/A | Not implemented |
| `sync_instance_to_db` | N/A (node specs only) | N/A | N/A | N/A | N/A |
| `collect_city_data` | Yes | Converted | Yes (non-target years) | N/A | N/A |
