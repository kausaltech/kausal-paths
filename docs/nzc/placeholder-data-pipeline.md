# NZC Placeholder Data Pipeline: Issues, Fixes, and Open Questions

## Background

The NZC framework pre-populates default values for new city instances using
two datasets stored in DVC:

- **`nzc/placeholders`** — single cluster-averaged values (one row per measure),
  built from the 42-city input dataset by `notebooks/NZC-CCV-Dataset-Processor.py`
- **`nzc/placeholders_yearly`** — yearly time-series values (2018–2024) with
  confidence bounds, built from `data/nzc/placeholders_yearly_canonical.csv`
  by `data/nzc/lucia/nzc_process_yearly_placeholders.py`

### How defaults reach city instances

The pipeline has two stages:

**Stage 1 — framework-level import** (`tools/import_nzc_yearly_placeholders.py`):
Reads `nzc/placeholders_yearly` from DVC (or a CSV file if given explicitly) and
writes `MeasureTemplateDefaultDataPoint` records — one row per
(template, year, cluster group). Also sets `MeasureTemplate.default_value_scaling`
to `'population'` for population-dependent measures, or `None` for others.
These records live at the framework level, independent of any city instance.

**Stage 2 — instance-level propagation**:
When `populate_measure_defaults_from_default_data_points()` is called (either
at instance creation or via `python manage.py populate_nzc_defaults`), it
selects the best-matching `MeasureTemplateDefaultDataPoint` for each measure
(choosing the city's cluster group by `FrameworkConfig.categories`), applies
population scaling where `default_value_scaling == 'population'`, and writes
the result into `MeasureDataPoint.default_value` for `year == baseline_year` only.

---

## Issues Found

### 1. `waste_recycling_future_baseline_shares` IndexError on new instances

**Symptom:** Creating a new NZC instance (e.g. jouniville-5) crashed with
`IndexError: index 0 is out of bounds for sequence of length 0` inside
`gpc.DatasetNode.implement_unit_col()`.

**Root cause:** `DatasetNode.get_correct_baseline()` filters rows to those
where `Year == reference_year OR Year > maximum_historical_year OR Forecast`.
The `maximum_historical_year` was calculated as
`MeasureDataPoint.objects.aggregate(max_year=Max('year'))` across all data
points for the framework config.  For a brand-new instance, all data points
are defaults-only (future projections up to 2050), so `max_hist_year` became
2050.  The filter `Year > 2050` then excluded the only relevant row (Year=2030),
returning an empty DataFrame.

**Fix** (`nodes/instance_loader.py`): filter the aggregate query to
`value__isnull=False` so only user-entered observations contribute to the
historical year range.  New instances with no user data fall back to
`fwc.baseline_year`.

```python
mdp_data = MeasureDataPoint.objects.filter(
    measure__framework_config=fwc, value__isnull=False,
).aggregate(min_year=Min('year'), max_year=Max('year'))
max_hist_year = mdp_data['max_year'] or fwc.baseline_year
min_hist_year = mdp_data['min_year'] or fwc.baseline_year
```

---

### 2. `PerCapita=TRUE` errors in the yearly source CSV

**Symptom:** "Share of fleet that is less than 2 years old" (measure 047,
UUID `f0479ebd-e0b9-43ee-ad71-67f958529c54`) and two emission factor measures
(F171, F172) were being multiplied by population when populating defaults,
producing nonsensical values.

**Root cause:** `data/nzc/placeholders_30042026_11_34_PM.csv` has
`PerCapita=TRUE` for measures 047, F171, and F172.  These are intensive
quantities (a dimensionless fraction and g/kWh emission factors) — not
population-dependent flows.  The authoritative source is the static
`placeholders.csv` produced by `NZC-CCV-Dataset-Processor.py`, which derives
`PerCapita` from its explicit `pop_measures` list and does **not** include 047,
F171, or F172.

**Fix:** The canonical CSV builder (`tools/build_canonical_yearly_placeholders.py`)
uses `placeholders.csv` as the authority for `PerCapita`, overriding the source
CSV.  It reports any corrections automatically.  On the current source CSV it
corrects measures **047, F171, F172** from TRUE to FALSE.

---

### 3. Malformed UUIDs in the source CSV

**Symptom:** Several measure rows had UUIDs that were either 35 characters
(missing the last character) or 37 characters (Excel auto-increment artefacts).

**What was fixed:**
The canonical CSV builder (`tools/build_canonical_yearly_placeholders.py`)
resolves all UUIDs from `placeholders.csv` by joining on Measure ID, so UUID
errors in the source CSV never reach the DVC dataset or the DB import.

The import script (`tools/import_nzc_yearly_placeholders.py`) retains a
one-off fixup for UUID `00924063-1dd8-43b1-a118-f3e29edbecf` for the case
where an old CSV is passed as explicit input; this fixup is irrelevant when
loading from DVC.

---

### 4. Skipped measures (104, 105, 117, 118) in the yearly dataset

Measures 104, 105, 117, 118 (cost of renovation; new building costs) appear
in `placeholders_30042026_11_34_PM.csv` but have no entry in `placeholders.csv`
and therefore no canonical UUID.  They are silently skipped by the canonical
CSV builder.

---

### 5. Multi-year bug in `_select_default_data_points`

**Symptom:** After the Stage 1 import writes `MeasureTemplateDefaultDataPoint`
records for years 2018–2024, Stage 2 propagation was writing `default_value`
into `MeasureDataPoint` rows for all seven years instead of only the city's
`baseline_year`.

**Fix** (`frameworks/models.py`):
`_select_default_data_points` now filters to `year=self.baseline_year` before
selecting the best-matching cluster record.  `populate_measure_defaults_from_default_data_points`
also runs a stale-record cleanup that clears `default_value`/bounds on non-baseline-year
`MeasureDataPoint` rows (those with `value__isnull=True`) for affected templates,
removing any remnants left by previous runs.

---

### 6. Test failure after fixing issue 1

**Symptom:** `test_create_nzc_framework_config_mutation_creates_instance_and_defaults`
failed with `assert set() == {<FrameworkDimensionCategory: Temperature - Low>}`.

**Root cause:** A previous fix had removed `fwc.categories.add()` entirely to
avoid a "Framework dimension not found" crash on instances that do not define
`renewable_mix`/`temperature` as `FrameworkDimension` objects.

**Fix** (`frameworks/schema.py`): restore category assignment but make it
optional — loop over expected dimensions, skip with `continue` if the
dimension or category is not found.

---

## What Was Done

### Canonical yearly placeholders CSV

`tools/build_canonical_yearly_placeholders.py` produces
`data/nzc/placeholders_yearly_canonical.csv` by joining five sources:

| Source file | Purpose |
|---|---|
| `placeholders.csv` | **Authority** for UUID and PerCapita (overrides source CSV) |
| `42_cities_input_9_clusters.csv` | Description text (from column-header prefix `MID/description`) |
| `nzc_updated_datasets.csv` | Unit and Metric column name (primary) |
| `Draft Matias_...defaults.csv` | Unit and Metric (fallback for 3 measures absent above) |
| `placeholders_30042026_11_34_PM.csv` | Actual yearly values; UUID and PerCapita columns discarded |

Output column order:
`UUID | MeasureID | Unit | PerCapita | Year | 0_ccv … 3_max | Metric | Description`

Running the script reports any PerCapita corrections automatically.  On the
current source CSV it corrects measures **047, F171, F172** from TRUE to FALSE.

A separate override mechanism (`PERCAPITA_OVERRIDES` dict in the script) handles
the **population measure (015, UUID `3779efa4...`)**:

- In the static `nzc/placeholders` dataset the value is stored as `1.0` with
  `PerCapita=True`, so `_calculate_placeholders` recovers the city population
  by computing `1.0 × population`.
- In the yearly dataset the values are actual comparable-city population counts
  (e.g. 433 340 for 2018), so `PerCapita` must be `False` — multiplying those
  counts by population again would produce nonsense.

### DVC upload script simplified

`data/nzc/lucia/nzc_process_yearly_placeholders.py` was rewritten to simply
read `data/nzc/placeholders_yearly_canonical.csv` and push it to DVC as
`nzc/placeholders_yearly`.  All complex parsing (UUID resolution, EU decimal
format, PerCapita correction) was already handled by
`build_canonical_yearly_placeholders.py`; this script just sets correct polars
types and pushes.

### Stage 1 import script moved and improved

`data/nzc/convert_nzc_yearly_placeholders.py` was moved to
`tools/convert_nzc_yearly_placeholders.py`.

Key improvements:
- **Default input is now the DVC dataset** `nzc/placeholders_yearly` instead of
  a local CSV file.  Pass an explicit path argument to use a CSV instead.
- The `convert()` function was refactored to accept a pre-loaded row list,
  decoupling input loading from parsing.
- A `load_rows_from_dvc()` function loads the DVC dataset via the framework's
  first FrameworkConfig instance and converts the typed polars DataFrame to the
  string-dict format expected by `convert()`.

### Old DVC-based pipeline retired

`populate_nzc_yearly_defaults` management command and the underlying
`FrameworkConfig.populate_measure_defaults_from_nzc_yearly()` method have been
deleted.  They implemented a parallel approach that read DVC directly, applied
population scaling inline, and wrote per-instance `MeasureDataPoint.default_value`
records — overlapping with Stage 2 of the current pipeline.

### New management command

`python manage.py populate_nzc_defaults` replaces `populate_nzc_yearly_defaults`.
It calls `fc.populate_measure_defaults_from_default_data_points()` with no DVC
dependency.  Arguments: `--framework-config PK`, `--instance IDENTIFIER`, or
no arguments to target all NZC FrameworkConfigs.

### New management command parameter (earlier)

`populate_nzc_yearly_defaults` had gained an `--instance <IDENTIFIER>` option
before it was retired.  The new `populate_nzc_defaults` command carries the same
interface.

---

## Re-running the pipeline after any data fix

When values or PerCapita flags in the source data need to be corrected:

1. **Rebuild canonical CSV:**
   ```
   python tools/build_canonical_yearly_placeholders.py
   ```
2. **Push to DVC:**
   ```
   python data/nzc/lucia/nzc_process_yearly_placeholders.py
   ```
3. **Write framework-level default data points:**
   ```
   python tools/import_nzc_yearly_placeholders.py
   ```
4. **Propagate to all city instances:**
   ```
   python manage.py populate_nzc_defaults
   ```

---

## Remaining Open Questions

### A. Should measures 104, 105, 117, 118 be in the yearly dataset?

These cost-of-renovation and new-building-cost measures exist in the source
CSV but have no canonical UUID.  Either they were intentionally excluded from
`placeholders.csv` (costs handled by a separate pipeline), or they were
overlooked.  **Verify with the data team** before adding them.

### B. Are the following measures missing from the yearly dataset intentionally?

The static `placeholders.csv` has 109 measures that do not appear in the
yearly source CSV.  Most are clearly correct omissions (action lever parameters
286–452, derived GHG totals 254–279, static city-level inputs).  The following
~15 measures are less obvious and may have been overlooked:

| Measures | Description |
|---|---|
| 186, 193, 194 | Total collected waste (total / organic / other) |
| 218, 223–225, 229–231 | Waste treatment shares (incineration / landfill / compost) |
| 236–239 | Emission factors from waste incineration (CO₂, NOₓ, PM2.5, PM10) |

All of these could plausibly have yearly time-series data.
