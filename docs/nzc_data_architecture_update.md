# NZC Data Architecture Update

## Context

We have a generic model `nzc.yaml` that is a basis for 100+ city-level models. The city users cannot edit the model structure but they can enter city-specific values via a special UI for some data points (specifically: those that have UUID, which links the user entry with a model dataset datapoint). If the user does not know a value, they can leave a cell blank and it will be replaced by a comparable city value (CCV; it depends on which of the four groups the city belongs to but not the year). If no CCV is available, a default value from the DVC dataset is used.

We are implementing two interlinked updates.

**Update A: Structural improvements to the model**
1. Split the single dataset into meaningful parts
2. Make the datasets compatible with DB dataset structure and independent of the outdated `gpc.DatasetNode`
3. Give the super-admin access to manage DB datasets via the generic admin UI
4. Give city admin users access to a city-specific dataset where they can edit also those data points that do not have UUID
5. Update the model structure in the new version `lucia.yaml` while keeping the mathematical outputs intact.

**Update B: New yearly comparable city values**

A new dataset `nzc/placeholders_yearly` is introduced alongside the existing `nzc/placeholders`. The new dataset has these features:
1. It still has separate values for each of the four city groups
2. It has yearly values for some data points (but does not cover all data points of `nzc/placeholders`)
3. It has values for plausible lower and upper bounds, used in the city user UI to give a warning if the user enters a value outside the range

There are also logical boundaries that give an error if violated (e.g. negative values not allowed), but those are not in `nzc/placeholders_yearly` — they stay as scalar fields on `MeasureTemplate`.

The model should use available data in this priority order:
1. Observation given by the city user
2. Data given by the city admin in the city-specific DB dataset
3. Comparable city value (CCV)
4. Data given by the super-admin in the generic DB dataset
5. DVC dataset

---

## Data Sources and Priority Order

| Priority | Source | Who controls it | Where stored |
|---|---|---|---|
| 1 | City user observation | City user (UI) | `MeasureDataPoint.value` |
| 2 | City admin dataset | City admin (UI) | `MeasureDataPoint.city_value` |
| 3 | Comparable city value (CCV) | Framework (DVC) | *see schema change below* |
| 4 | Super-admin generic dataset | Framework super-admin (admin UI) | `MeasureTemplateDefaultDataPoint` |
| 5 | DVC base dataset | Framework (DVC file) | DVC parquet |

Priority 3 is what changes. Currently CCVs are computed at city-creation time and stored as `MeasureDataPoint.default_value` (per-city copies). They should be framework-level data stored once, not duplicated per city.

---

## Schema Changes

### Problem: migration 0020 state

Migration 0020 added `min_value` and `max_value` to `MeasureDataPoint`. It was deployed to staging but failed, so its state there is uncertain. It has NOT reached production. **First step: determine the exact state of the staging DB** (run `\d frameworks_measuredatapoint` in psql), then create a corrective migration.

### Problem: CCVs and confidence bounds are in the wrong model

Currently `MeasureDataPoint` stores:
- `default_value` — CCV (duplicated per city, same for all cities in the same group)
- `min_value`, `max_value` — plausible confidence bounds (migration 0020, wrong place)

These are framework-level data, not city-level. A city with no user entries should still see CCVs and confidence bounds — they should be available on `MeasureTemplate` before any city datapoints have been created. The existing scalar `MeasureTemplate.min_value` / `MeasureTemplate.max_value` are for absolute logical bounds (UI errors); the per-year plausible confidence bounds are a different concept and need a different home.

### Problem: city group membership is not persisted

`NZCPlaceholderInput` (population, renewmix, temperature) is passed at city-creation time to compute CCVs, then discarded. If CCVs need to be refreshed (e.g. when the framework dataset updates), the group is gone. It must be stored on `FrameworkConfig`.

---

## Target Schema

### New: `MeasureTemplateCCVDataPoint`

```
template    FK → MeasureTemplate
group       IntegerField (0–3, matching renewmix × temperature matrix)
year        IntegerField
value       FloatField              — the CCV for this group and year
min_value   FloatField, nullable    — lower plausible bound (UI warning)
max_value   FloatField, nullable    — upper plausible bound (UI warning)
```

The four groups are:
```
0 = renewmix:low  × temperature:low
1 = renewmix:low  × temperature:high
2 = renewmix:high × temperature:low
3 = renewmix:high × temperature:high
```

Logical absolute bounds (error if violated, e.g. negative values) stay as the existing scalar fields `MeasureTemplate.min_value` / `MeasureTemplate.max_value`.

### Changes to `FrameworkConfig`

Add three fields:
- `renewmix` — `CharField(choices=['low', 'high'])`, nullable
- `temperature` — `CharField(choices=['low', 'high'])`, nullable
- `population` — `IntegerField`, nullable

These are set when the city is created (already available in `CreateNZCFrameworkConfigMutation`). They allow CCV refresh at any time without re-running the creation flow.

### Changes to `MeasureDataPoint`

Remove: `default_value`, `min_value`, `max_value` (all three move to `MeasureTemplateCCVDataPoint`).  
Keep: `measure`, `year`, `value` (city user), `city_value` (city admin, migration 0021).

### `MeasureTemplateDefaultDataPoint` — unchanged

This already correctly serves priority 4 (super-admin generic dataset, per year). No changes needed.

---

## What the UI Receives (API Contract)

The UI needs, for each measure template in the city's framework:

**Per-template (static):**
- Absolute min/max (error bounds) — already on `MeasureTemplate`
- List of CCV data points for the city's group: `[{year, value, min_value, max_value}]` — from `MeasureTemplateCCVDataPoint` filtered by `FrameworkConfig.group`

**Per-year, per-city (dynamic, may be empty):**
- `value` — what the city user entered (from `MeasureDataPoint`)
- `city_value` — what the city admin entered (from `MeasureDataPoint`)

The UI resolves what to display using the same priority order: if `value` exists, show it; else if `city_value` exists, show it; else show the CCV for the matching year.

For the confidence range warning: compare the user's entered value against `min_value` / `max_value` from `MeasureTemplateCCVDataPoint` for the matching year. Show a warning (not an error) if outside range.

---

## Implementation Steps

### Step 1 — Staging cleanup (full-stack)

Assess migration 0020 state on staging. Create a corrective migration or roll it back cleanly so the branch and staging are in sync.

### Step 2 — New schema migration (full-stack)

- Add `MeasureTemplateCCVDataPoint` model
- Add `renewmix`, `temperature`, `population` to `FrameworkConfig`
- Remove `default_value`, `min_value`, `max_value` from `MeasureDataPoint`
- Update `CreateNZCFrameworkConfigMutation` to save group fields on `FrameworkConfig` and populate `MeasureTemplateCCVDataPoint` instead of `MeasureDataPoint.default_value`

### Step 3 — Populate CCV data (impact modeler + full-stack)

Import `nzc/placeholders` and `nzc/placeholders_yearly` into `MeasureTemplateCCVDataPoint` for all four groups at once. This replaces the current per-city `create_measure_defaults()` + `create_measure_defaults_yearly()` approach. The data is the same for all cities in a group, so it is stored once per framework.

### Step 4 — Update `ObservationDataset` overlay (full-stack)

In `_overlay_observations`, replace the query for `MeasureDataPoint.default_value` (priority 3) with a query for `MeasureTemplateCCVDataPoint` filtered by `FrameworkConfig.group`. Add `group` to `FrameworkConfigData` in `context.py`.

### Step 5 — GraphQL schema update (full-stack + UI designer)

Expose `MeasureTemplateCCVDataPoint` on `MeasureTemplate` as `ccv_data_points`. The city's group (from `FrameworkConfig`) determines which subset to return. Agree on exact field names with the UI designer before implementing.

### Step 6 — Migration of 100+ existing DB cities

Existing cities have `MeasureDataPoint.default_value` populated but no stored group on `FrameworkConfig`. Two options — **team must choose one**:

**(A) Infer group from existing data**: compare each city's `default_value` against known group CCVs to reconstruct `renewmix` / `temperature`. Write a one-time management command. Cleaner, but requires a data reconstruction script.

**(B) Accept null group temporarily**: existing cities show no CCV in the new UI until a city admin re-confirms their group in a new UI step. CCVs for priority 3 fall back to nothing (DVC remains priority 5). Simpler, but disrupts existing cities until they re-confirm.

After group membership is resolved: run `sync_instance_to_db --all` to push updated node specs to all DB instances.

---

## Separate: nzc.yaml → lucia Structural Migration (Update A)

This is independent of the schema changes above and can proceed in parallel:

1. Merge lucia.yaml node definitions into nzc.yaml (`GenericNode` / `ObservableNode` replacing `FormulaNode` / `DatasetNode`), keeping `id: nzc` and all instance metadata unchanged
2. Update `nzc/*.yaml` module files similarly, so YAML-file city configs pick up the changes
3. Verify numerical equivalence: run `test_instance --state-dir` against a saved baseline
4. Run `sync_instance_to_db --all` to push updated node specs to all 100+ DB instances
5. Create the lucia framework in the DB (separate identifier from nzc), populate its `MeasureTemplate` and `MeasureTemplateCCVDataPoint` from the new datasets

---

## Open Decisions Needing Team Agreement

1. **Existing city group reconstruction** (Step 6): infer from data (A) or require re-confirmation (B)?
2. **CCV model name**: `MeasureTemplateCCVDataPoint`, `MeasureTemplatePlaceholderDataPoint`, or something else?
3. **GraphQL field names for confidence bounds** on the template: coordinate with UI designer before Step 5.
4. **Sequencing**: can the lucia structural migration (Update A) go out before the schema changes, or do they need to be one release?
