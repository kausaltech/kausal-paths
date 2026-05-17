# NZC Placeholder Data Pipeline: Issues, Fixes, and Open Questions

## Background

The NZC framework pre-populates default values for new city instances using
two datasets stored in DVC:

- **`nzc/placeholders`** — single cluster-averaged values (one row per measure),
  built from the 42-city input dataset by `notebooks/NZC-CCV-Dataset-Processor.py`
- **`nzc/placeholders_yearly`** — yearly time-series values (2018–2024) with
  confidence bounds, built by `data/nzc/lucia/nzc_process_yearly_placeholders.py`
  from the source file `data/nzc/placeholders_30042026_11_34_PM.csv`

There is also a parallel import path written independently by a colleague:
`data/nzc/convert_nzc_yearly_placeholders.py`, which reads from a CSV and
writes directly to `MeasureTemplateDefaultDataPoint` DB records and
`MeasureTemplate.default_value_scaling`.

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

**Root cause — two interacting paths:**

**Path A (DVC / `populate_nzc_yearly_defaults`):**
`data/nzc/placeholders_30042026_11_34_PM.csv` has `PerCapita=TRUE` for
measures 047, F171, and F172.  `nzc_process_yearly_placeholders.py` copies
this flag verbatim into the DVC parquet.  `frameworks/nzc.py`
`_calculate_yearly_placeholders()` then multiplies values by population
wherever `PerCapita=True`.

**Path B (direct DB import / `convert_nzc_yearly_placeholders.py`):**
The colleague's script reads the same source CSV and sets
`MeasureTemplate.default_value_scaling = 'population'` for every measure
where `PerCapita=TRUE`.  This is a persistent DB change that affects all
future calls to `populate_measure_defaults_from_default_data_points()`,
including at instance-creation time, regardless of the DVC dataset.

**Why 047 is wrong:** The authoritative source is the static `placeholders.csv`
produced by `NZC-CCV-Dataset-Processor.py`.  That script derives `PerCapita`
from its explicit `pop_measures` list, which does **not** include 047
(a dimensionless fraction: share of fleet), F171, or F172 (g/kWh emission
factors).  These are intensive quantities — not population-dependent flows.
The source CSV introduced the error; both downstream pipelines inherited it.

**Fix:** See the canonical CSV section below.

---

### 3. Malformed UUIDs in the source CSV

**Symptom:** Several measure rows had UUIDs that were either 35 characters
(missing the last character) or 37 characters (Excel auto-increment artefacts,
e.g. `0ae5fb06-...-ccfa4e8f44a10`).

**What was fixed:**
- `nzc_process_yearly_placeholders.py` resolves UUIDs from `placeholders.csv`
  by joining on Measure ID, so UUID errors in the source CSV do not reach DVC.
- `convert_nzc_yearly_placeholders.py` hardcodes a fix for one specific
  truncated UUID (`00924063-1dd8-43b1-a118-f3e29edbecf` → `...ecf7`) but
  does not handle the multi-character sequence UUIDs.

**Measures affected by unresolved UUID errors in the colleague's script:**
`0ae5fb06-...`, `3779efa4-...-bed10` through `bed14`, `e2f0c38d-...-a10/11`.
These rows will silently fail to match any `MeasureTemplate` in the DB when
the colleague's script is run against the original source CSV.

---

### 4. Skipped measures (104, 105, 117, 118) in the yearly dataset

Measures 104, 105, 117, 118 (cost of renovation; new building costs) appear
in `placeholders_30042026_11_34_PM.csv` but have no entry in `placeholders.csv`
and therefore no canonical UUID.  They are silently skipped by both the lucia
script and the new canonical CSV builder.

---

### 5. Test failure after fixing issue 1

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

### New management command parameter

`populate_nzc_yearly_defaults` gained an `--instance <IDENTIFIER>` option so
users can target a specific city by its `InstanceConfig.identifier` rather
than having to look up the `FrameworkConfig` primary key.

### Canonical yearly placeholders CSV

`data/nzc/lucia/build_canonical_yearly_placeholders.py` produces
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

---

## Remaining Questions

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

### C. Colleague's script UUID fixes are incomplete

`convert_nzc_yearly_placeholders.py` only fixes the `00924063...` UUID.
If the script is ever re-run against the original source CSV
(`placeholders_30042026_11_34_PM.csv`), rows with multi-digit sequence UUIDs
will silently produce no DB update.  The script should either be updated to
use the canonical CSV as input, or have its UUID fix list extended to cover
all known malformed UUIDs.

### D. The two import pipelines need a shared source of truth

The lucia DVC pipeline and the colleague's direct-DB import pipeline currently
both derive `PerCapita` from the source CSV independently.  Now that a
canonical CSV exists, both scripts should be updated to use it as input so
that PerCapita, UUID, and other metadata are never duplicated or drift apart.

### E. Re-running both pipelines after any data fix

When PerCapita or values in the canonical CSV are corrected, the fix must
propagate through **both** pipelines:

1. Re-run `build_canonical_yearly_placeholders.py` to regenerate the canonical CSV (done)
2. Re-run `nzc_process_yearly_placeholders.py` (or equivalent) and push to DVC (pushed on 2026-05-17)
3. Re-run `python manage.py populate_nzc_yearly_defaults` on all affected instances
4. Re-run `convert_nzc_yearly_placeholders.py` on the test/production server
   to reset any `MeasureTemplate.default_value_scaling` values that were set
   incorrectly
