# NZC City Data Migration: Architecture and Status

## Overview

The NZC (Net Zero Cities) framework runs ~150 city instances from a single framework
YAML. Each city's results must reflect its own measured data rather than the generic
reference model. This document describes the new data-flow architecture introduced in
`lucia.yaml` (now `nzc.yaml`) and the migration work required to make it work correctly
for all city instances.

---

## Data tiers

Every node in the framework can draw values from three tiers, in priority order:

1. **City-specific (DB)** — values entered by city staff as `MeasureDataPoint` records in
   the database, linked to `MeasureTemplate` rows via UUID.  These are the most
   authoritative values for a metric.

2. **Comparable-city placeholder (DVC)** — default values stored in the DVC repository,
   representing averages or typical values from comparable reference cities.  Used when
   a city has not yet entered its own data.

3. **Framework reference (computation)** — the NZC reference model's own computed values.
   This is what the standalone `nzc` framework instance produces when no city-specific
   or placeholder data is present.

---

## Old data-flow (`nzc copy.yaml` + `nzc/*.yaml`)

The old system used a single wide DVC dataset `nzc/defaults` as the entry point for
city data.  It had a `UUID` column (uppercase) and a `Sector` column identifying the
metric.

**Node types that provided city data:**

- `gpc.DatasetNode` with `input_datasets: [nzc/defaults]` automatically triggered
  `FrameworkMeasureDVCDataset` in the instance loader, regardless of tags.
- `formula.FormulaNode` with `tags: [framework_measure_data]` also triggered
  `FrameworkMeasureDVCDataset`.

**`FrameworkMeasureDVCDataset._override_with_measure_datapoints`:**

1. Queries DB for `MeasureDataPoint` records for this city's `FrameworkConfig`, matched
   by UUID.
2. Overlays city-specific (tier 1) or default (tier 2) values onto every metric row in
   `nzc/defaults`.
3. Adds `ObservedDataPoint` and `FromMeasureDataPoint` boolean flag columns.

Because every input node throughout the graph (`existing_building_stock`,
`passenger_kilometres`, `energy_use_intensity`, etc.) was a `gpc.DatasetNode` loading
from `nzc/defaults`, city-specific data penetrated the entire computation chain.

**Scenario switching for observable metrics:**

Five "observable" nodes (`building_heat_energy_use_observed`, `passenger_km_observed`,
`freight_transport_need_observed`, `total_electricity_consumption`,
`emissions_from_other_sectors_observed`) used a `formula.FormulaNode` with:

```
select_port(condition, observed_only_extend_all(observed), modeled)
```

- `measure_data_override = false` (default/decarbonisation scenario): returns `modeled`,
  which was city-specific because the sub-nodes loaded city data from `nzc/defaults`.
- `measure_data_override = true` (progress_tracking scenario): extends the observed DVC
  or DB value across all years.

---

## New data-flow (`nzc.yaml` / `lucia.yaml`)

The new architecture replaced `nzc/defaults` with multiple narrow DVC datasets, one per
domain (e.g., `nzc/population`, `nzc/buildings_stock_renovation`,
`nzc/passenger_transport_need_fleet`, etc.).  City data is injected at two distinct
levels.

### Level 1 — ObservableNode (5 aggregate sector metrics)

**Datasets:** `nzc/buildings_heating_fuel_tech`, `nzc/passenger_transport_need_fleet`,
`nzc/freight_transport`, `nzc/other_sectors`, `nzc/electricity`.

**Tag in YAML:** `observation_dataset`

**Dataset class:** `ObservationDataset` (in `frameworks/datasets.py`)

**Mechanism:**

1. DVC file is loaded; rows must have a `uuid` column with `drop_col: false`.
2. `_overlay_observations` queries DB for `MeasureDataPoint` records by UUID (hyphen
   format; DVC stores underscores — conversion is applied automatically).
3. Rows with DB data get `observed = True`; rows with only a DVC placeholder get
   `placeholder = True`; rows with neither get `observed = False`.
4. For lookup-table datasets where the DVC file uses `Year = 0` (reference year) and
   the DB observation is stored at the city's actual reference year, a UUID-only
   fallback join is used when year sets do not overlap.
5. The dataset output (with uuid, observed, placeholder columns retained) is consumed
   by `generic.ObservableNode.apply_observations`:
   - `use_observations = false` (default scenario): uses the modelled trajectory for all
     years; overrides only the reference year from a DB observation if one is present.
   - `use_observations = true` (progress_tracking): extends the observed (or placeholder)
     value across all years.

### Level 2 — CityDataset (all other input nodes)

**Datasets:** `nzc/buildings_stock_renovation`, `nzc/population`, and all other narrow
datasets used by plain `GenericNode` or action nodes.

**Tag in YAML:** `city_data`

**Dataset class:** `CityDataset` (in `frameworks/datasets.py`)

**Mechanism:**

1. Uses `GenericDataset`'s loading path (`_filter_and_process_df`, `_transform_data`,
   `_index_data`) — essential because GenericNode consumers expect the metric-column
   setup that `GenericDataset.load_internal` provides.
2. Between `_filter_and_process_df` and `_transform_data`, calls `_overlay_observations`
   (inherited from `ObservationDataset`) to inject city-specific DB values.
3. Falls back to original DVC data if the overlay empties the frame (e.g. when all uuid
   values are null in the DVC file, meaning no DB lookup is possible for that row).
4. Strips `uuid`, `observed`, and `placeholder` columns before returning, so any
   consuming node type receives clean data with the city-specific value already in the
   `Value` column.

### Year encoding in DVC files

Narrow DVC datasets use **relative years**:

- `Year = 0` → city's `reference_year` (e.g. 2019 for most cap-2030 cities)
- `Year = 1..89` → `reference_year + Year`
- `Year = 90..199` → `target_year + (Year - 100)`

`DVCDataset._filter_and_process_df` converts these to absolute years before any overlay
or computation runs.  Because the DB stores absolute years (e.g. 2019) and the DVC
data's relative year 0 maps to the same absolute year, the year-based join in
`_overlay_observations` generally succeeds without special handling.

**Exception — lookup-table datasets:** Some datasets (e.g. `nzc/population`,
`nzc/other_sectors`) store a single time-invariant row with `Year = 0`.  After
conversion this becomes the reference year, which may differ from the year stored in the
DB observation (particularly when the framework `reference_year` differs from the
city instance's `reference_year`).  `_overlay_observations` now detects this case
(no overlap between DVC year set and DB year set) and falls back to a UUID-only join
using the most recent DB observation.

### UUID format

| Location | Format | Example |
|---|---|---|
| DVC files | underscores | `3779efa4_9eb0_4f4b_b5d5_eb510461bed8` |
| Database (`MeasureTemplate.uuid`) | hyphens | `3779efa4-9eb0-4f4b-b5d5-eb510461bed8` |

The conversion `uuid.replace('_', '-')` is applied in both `ObservationDataset` and
`CityDataset` before querying the DB.

### `use_datasets_from_db`

`lucia.yaml` sets `features.use_datasets_from_db: true`.  This causes `GenericNode`
datasets to prefer `DBDataset` (loaded from the database) over DVC files when a matching
`DatasetConfig` record exists.  City instances served directly from the `lucia` DB
instance benefit from this for full-category data.  Framework instances (like `nzc`)
that lack DB dataset records fall back to DVC.

---

## Migration work: issue, problems, what has been done, what remains

### The core issue

Replacing `nzc.yaml` with the `lucia.yaml`-derived content broke city-specific results
for all ~150 NZC city instances.  The new YAML uses different node types and dataset IDs
that do not support the old `FrameworkMeasureDVCDataset` mechanism for injecting
city data.

### Key problems identified

#### Problem 1 — UUID dtype mismatch (FIXED)

**Symptom:** 104 cap-2030 city instances failed entirely with:

```
polars.exceptions.SchemaError: datatypes of join keys don't match -
`uuid`: cat on left does not match `uuid`: str on right
```

**Cause:** DVC Parquet files store `uuid` as `cat` (Categorical); the `obs_raw`
DataFrame built from DB query results has `uuid` as `str`.
`FrameworkMeasureDVCDataset._overlay_observations` tried to join them.

**Fix:** After building `obs_raw`, cast its `uuid` column to match `df.schema['uuid']`:
```python
uuid_dtype = df.schema['uuid']
if obs_raw.schema['uuid'] != uuid_dtype:
    obs_raw = obs_raw.with_columns(pl.col('uuid').cast(uuid_dtype))
```

**File:** `frameworks/datasets.py`, `_overlay_observations`

---

#### Problem 2 — City-specific data not reaching input nodes (PARTIALLY FIXED)

**Symptom:** After fixing Problem 1, most cities computed successfully but all got the
same framework reference value (~1532 kt/a for "newest", ~371 kt/a for 2030 target)
instead of city-specific results.

**Root cause:**

- Old: `gpc.DatasetNode` with `nzc/defaults` → `FrameworkMeasureDVCDataset` → city data
  at every input node throughout the graph.
- New: `generic.GenericNode` with narrow datasets → plain `DVCDataset` or
  `GenericDataset` → only framework reference values.  The `framework_measure_data` tag
  that would trigger `FrameworkMeasureDVCDataset` is absent.  The new narrow datasets
  also use lowercase `uuid` (incompatible with `FrameworkMeasureDVCDataset` which
  requires uppercase `UUID`).

**UUID matching confirmed:** All 64 UUIDs in the 5 new narrow datasets match
`MeasureTemplate` records in the DB (after `_` → `-` conversion).

**Fix applied:**

1. Created `CityDataset(GenericDataset, ObservationDataset)` class in
   `frameworks/datasets.py`.  Inherits `GenericDataset`'s loading path
   (`_transform_data`, `_index_data`) and `ObservationDataset`'s `_overlay_observations`.
   Strips `uuid`/`observed`/`placeholder` before returning.

2. Added `use_city_ds = 'city_data' in ds_def.tags` check in
   `nodes/instance_loader.py` to instantiate `CityDataset` for tagged datasets.

3. Added `tags: [city_data]` to 118 dataset entries in `nzc.yaml` (scripted).

4. Added `drop_col: false` to 112 uuid filter entries in those dataset entries
   (scripted), ensuring uuid is kept through `_filter_and_process_df` so the overlay
   can use it.

5. Added `features.use_datasets_from_db: true` to `nzc.yaml` (was already present in
   `lucia.yaml`; had no immediate effect since DB lacks the new-format datasets for
   framework instances, but is required for forward compatibility).

---

#### Problem 3 — Lookup-table datasets produce year mismatches (FIXED)

**Symptom:** `nzc/population` and `nzc/other_sectors` datasets store a single row with
`Year = 0`, which `_filter_and_process_df` converts to the city's `reference_year`
(e.g. 2018 for berlin, 2019 for aachen).  When the DB observation is stored at a
different year (e.g. 2019 for berlin's reference), the year-based join in
`_overlay_observations` finds no overlap → city-specific value is not applied.

**Fix:** In `_overlay_observations`, check if DVC and DB year sets have any overlap.
If not (lookup-table case), fall back to a UUID-only join using the latest DB
observation:

```python
years_overlap = dvc_years & obs_years
if years_overlap:
    # normal year+uuid join
else:
    # lookup-table fallback: uuid-only join, latest DB observation
    latest_obs = obs_raw.sort(YEAR_COLUMN, descending=True).group_by('uuid').first()...
```

**File:** `frameworks/datasets.py`, `_overlay_observations`, step 6.

---

#### Problem 4 — Null uuid rows empty the DataFrame (FIXED)

**Symptom:** Some DVC dataset rows (e.g. `demolition_rate_existing_buildings` in
`nzc/buildings_stock_renovation`) have `uuid = null`, because the metric was defined
without a UUID linkage.  `_overlay_observations` step 2 filters to
`uuid.is_not_null()`, emptying the frame → `_linear_interpolate` then fails with
`assert isinstance(min_year, int)`.

**Fix:** In `CityDataset.load_internal`, if the overlay empties a non-empty frame,
fall back to the pre-overlay data:

```python
df_before = df
df = self._overlay_observations(df)
if df.is_empty() and not df_before.is_empty():
    df = df_before
```

**File:** `frameworks/datasets.py`, `CityDataset.load_internal`.

---

#### Problem 5 — `CityDataset` used wrong loading path (FIXED)

**Symptom:** `CityDataset` originally inherited only from `ObservationDataset(DVCDataset)`.
`DVCDataset.load_internal` calls `self.post_process(df)` — but `GenericNode` consumers
expect data shaped by `GenericDataset.load_internal`'s `_transform_data` / `_index_data`
steps, which `DVCDataset.load_internal` does not call.  This caused "No metric columns
in DF" errors.

**Fix:** Changed `CityDataset` to inherit from `GenericDataset, ObservationDataset`
(multiple inheritance, MRO: CityDataset → GenericDataset → ObservationDataset →
DVCDataset).  Overrides `load_internal` to replicate `GenericDataset`'s path but
injects `_overlay_observations` between `_filter_and_process_df` and `_transform_data`.

---

#### Problem 6 — `drop_col` scripts missed some entries (PARTIALLY FIXED)

**Symptom:** The script that added `drop_col: false` to uuid filters only searched for
`city_data` in the 15-line context **before** the `- column: uuid` line.  In some YAML
entries the `tags:` key comes **after** the `filters:` key, so `city_data` appeared
**after** the uuid filter line and was not detected.  These entries kept
`drop_col: true`, meaning uuid was silently dropped before `CityDataset` could use it.

**Status:** Fixed.  Six entries were missing `drop_col: false`: the historical dataset
entries in `a31_renovation_improvements`, `a32_new_building_improvements`,
`a33_do_efficient_appliances`, `a341_increase_district_heating`,
`a342_...` (share_of_heating_fuel), and `a343_change_heating_fossil_share`.
All now have `drop_col: false`.

---

### What remains to be done

1. ~~**Fix `drop_col: false` for remaining uuid filters where `tags:` follows `filters:`.**~~
   **DONE.** Six entries were found and fixed (see Problem 6 above).

2. ~~**Investigate `old_building_renovation_rate_observed` dimension error.**~~
   **FIXED** in `nodes/actions/linear.py`.

   **Root cause:** `CityDataset._index_data` calls `extend_last_historical_value_pl`,
   which fills data up to `model_end_year` and marks those extended rows `FORECAST=True`.
   `DatasetReduceAction._get_metric_data` then overrides all historical rows to
   `FORECAST=False`, making `df[YEAR_COLUMN].max()` return `model_end_year` instead of
   the last actual observation year.  The resulting `gdf.filter(Year > model_end_year)`
   was empty, so the delta computation produced a 0-row wide dataframe, and after
   `to_narrow()` this became an empty `(Year, Forecast)` frame with no dimension columns.

   **Fix:** For the three dataset-based historical code paths in `_get_metric_data`,
   filter to `~FORECAST_COLUMN` rows before overriding FORECAST to False — matching the
   existing behaviour of the three node-based historical paths (lines 198, 221-222, 231).

3. **Full regression test with `collect_city_data`.**
   Run `python -m notebooks.collect_city_data ../scripts/paths/collectors/emission_potential.yaml`
   and compare totals against the reference file
   `model-outputs/reduction_potential/sum_over_instances_net_emissions_kt-a_2026-06-07_base.csv`.
   Investigate remaining numerical differences.

4. **Document intentional behavioral changes.**
   The new `ObservableNode.apply_observations` with `use_observations = false` overlays
   the reference-year DB observation onto the modelled baseline (even in the default
   scenario).  The old `measure_data_override = false` did NOT apply any DB observations.
   Cities with reference-year DB observations will show slightly different "newest" and
   target values — this is intentional and represents a model improvement.

5. **Consider whether all 5 `ObservableNode` sectors need the same calibration.**
   Currently `building_heat_energy_use_observed`, `passenger_kilometres_observed`,
   `freight_transport_need_observed`, `emissions_from_other_sectors_observed`, and
   `total_electricity_consumption` calibrate at reference year.  Verify that this
   behaviour is correct for each in the context of the default and progress_tracking
   scenarios.

---

## Files changed

| File | Change |
|---|---|
| `frameworks/datasets.py` | UUID dtype cast fix; lookup-table year fallback; `CityDataset` class; null-uuid fallback; `_reattach_passthrough` for null-uuid rows; coalesce now includes `_obs_default` (comparable-city tier) so cities without entered values use their city estimate rather than the raw DVC reference |
| `nodes/instance_loader.py` | `city_data` tag → `CityDataset` |
| `nodes/actions/linear.py` | `DatasetReduceAction._get_metric_data`: filter historical datasets to `~FORECAST` rows before returning, so `max_hist_year` = last actual observation, not model_end_year |
| `configs/nzc.yaml` | `use_datasets_from_db: true`; `city_data` tags on 118 datasets; uuid filters cleaned up (removed redundant `- column: uuid` filters; remaining ones use `drop_col: false`) |
