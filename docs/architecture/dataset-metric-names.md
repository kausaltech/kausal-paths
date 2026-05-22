# DatasetMetric.name vs .label — and the NZC template-clone bug

## The two fields and their roles

`DatasetMetric` has two string fields that look similar but have very different purposes:

| Field | Purpose | Expected value |
|-------|---------|----------------|
| `name` | **Column identifier** — maps directly to the DataFrame column name used in calculations | Snake-case identifier matching the DVC column name, e.g. `share_of_trucks_fully_electric` |
| `label` | **Display label** — human-readable, translatable | Anything readable, e.g. `"Share of trucks fully electric"` |

`nodes/datasets.py:DBDataset.deserialize_df` uses `metric.name` as the pivot column name when it reconstructs a `PathsDataFrame` from database `DataPoint` rows. If `name` is a human-readable label instead of the DVC column identifier, the resulting DataFrame will have wrong column names and any node that references the dataset by its DVC column ID will raise `DatasetError: Column '...' not found`.

## How datasets are created — two paths

### Path A: `load_dvc_dataset` management command

```
python manage.py load_dvc_dataset <instance> <dataset_id>
```

`nodes/management/commands/load_dvc_dataset.py` line 329:

```python
metric = DatasetMetric(schema=schema, name=col, label=col, unit=str(unit))
```

`col` is the physical DVC column name (already snake_case). `name` is set correctly.
If DVC metadata contains translated labels, they are applied to `label` afterwards.

### Path B: Trailhead framework template cloning

When a new city instance is created via Trailhead's "Create Instance" UI flow, it is
cloned from the NZC framework template instance via `nodes/instance_serialization.py`:

**Export** (`MetricSnapshot.from_model`, line ~112):

```python
identifier=obj.name or str(obj.uuid),
```

**Import** (`_import_dataset`, line ~557):

```python
metric = DatasetMetricModel.objects.create(
    schema=schema,
    name=m_snap.identifier,   # <- comes from snapshot.identifier
    ...
)
```

`name` is faithfully round-tripped from the template's `metric.name`. If the template's
`metric.name` was set to a human-readable label (as happened during the initial NZC
framework setup), every instance cloned from that template inherits the bad value.

## The NZC bug

The NZC framework template datasets were originally created through the Trailhead UI
before `load_dvc_dataset` was used for NZC. At that point, `DatasetMetric.name` was
set to the human-readable label (e.g. `"Share of trucks fully electric (not including
hybrids)"`) instead of the DVC column identifier
(`share_of_trucks_fully_electric_not_including_hybrids`).

Every city instance later cloned from the template via "Create Instance" inherited
these bad values. The symptom appears only when `use_datasets_from_db = true` is set
on the instance config (or for any DB-sourced instance), because that is when
`DBDataset.deserialize_df` is called and tries to pivot by metric name.

### Detecting the issue

```
DatasetError: [<dataset_id>: Column '<human_readable_label>' not found ...]
```

or

```
DatasetError: No metric columns in DF (units dict has no matching column)
```

### Fixing the issue

Run the management command to repair metric names for one or more instances:

```bash
# Fix a single instance
python manage.py fix_dataset_metric_names lucia

# Fix multiple instances
python manage.py fix_dataset_metric_names lucia potsdam-dev

# Preview changes without writing them
python manage.py fix_dataset_metric_names lucia --dry-run

# Fix all instances (slow; loads every instance's DVC repo)
python manage.py fix_dataset_metric_names --all
```

The command (`nodes/management/commands/fix_dataset_metric_names.py`):

1. For each DB dataset in the instance, loads the corresponding DVC dataset.
2. Compares each `DatasetMetric.name` against the set of actual DVC column names.
3. If a name is wrong but a normalized form matches (lowercasing + replacing
   non-alphanumeric runs with `_` + collapsing section-number dots like `1.3` → `13`),
   it updates `metric.name` to the correct DVC column identifier.
4. Reports any metrics it could not match automatically — those need manual correction.

**After fixing an instance, the template itself still has bad values.** Any new
instance cloned from the template will need fixing again unless the template's metrics
are repaired first (e.g. by re-running `load_dvc_dataset --force` on the template
instance).

## The NZC relative-year offset

Related issue also affecting NZC DB datasets: `load_dvc_dataset` stores NZC relative
years (DVC `Year=0` through `Year=100`) as `date.year = year_value + 1` because
Python's `date` type does not allow `year=0`. `DBDataset.deserialize_df` subtracts 1
back when it reads years `≤ 101`, restoring the original DVC convention:

| DVC year | Stored date year | Meaning |
|----------|-----------------|---------|
| 0 | 1 | reference year (baseline) |
| 1–99 | 2–100 | intermediate relative years |
| 100 | 101 | target year |
| 2020+ | 2020+ | absolute calendar years (unchanged) |

If this correction is ever removed or the threshold is changed, `RelativeYearScaledNode`
and other nodes that look up year-0 data will silently return empty DataFrames.
