# Dataset-level data sources

*Produced by Claude Opus 5 on 2026-08-29.*
*Responsible: Jouni Tuomisto.*

A data source can be attached to a dataset as a whole, not only to its individual
data points. This is the right shape whenever the provenance is uniform — one
publication, or a handful of them, in one update, with nothing per row to
distinguish. It replaces the earlier practice of citing the same source from
every row, which stored one attribution tens of thousands of times and still did
not say the thing it meant: that the *dataset* came from there.

A dataset may carry any number of dataset-level sources. Co-equal sources for one
body of data are ordinary — a table assembled from a statistical release and
validated against a second one cites both, and neither is per-row.

Per-row citations are unchanged and remain the right tool when a value's origin
really does vary by row. The two can coexist in one dataset; what a single source
may not do is claim both levels at once (see *Refusals* below).

## 1. Declaring one

Dataset-level sources are declared in the sources registry CSV passed to
`--sources-csv`, not in the data file — there is nothing in the data to hang them
on, and typically no `Source` column at all. The registry gains three columns
beyond the ones described in [`dataset-csv-format.md`](dataset-csv-format.md) §2:

| column | meaning |
|---|---|
| `Target` | `data_point` (the default, and what a registry without the column means) or `dataset` |
| `Edition` | `DataSource.edition` — the vintage of the release, which is usually what the "single update" is |
| `Datasets` | which datasets a dataset-level source applies to; `'; '`-joined, matching the `Dataset` column's values. Empty means all of them |

```csv
Name,Authority,URL,Edition,Target,Datasets,Licensing
Energiebilanz Rheinland-Pfalz,Statistisches Landesamt,https://example.org/eb,2024,dataset,,Open data
Verkehrsmodell,Stadt Mainz,,2023,dataset,Fahrleistung,Internal
Handbuch Emissionsfaktoren,UBA,https://example.org/hb,,,,Open data
```

Read that as: the energy balance is a source for every dataset this upload
produces; the traffic model is one for `Fahrleistung` alone; the emission-factor
handbook is cited the old way, from the `Source` cells of individual rows.

**`Target`, not `Scope`.** The word is taken twice over already:
`DataSource.scope` is the instance that owns the source, and a data file's
`Scope` column is the GHG scope. `Target` is also what the read side has always
called it — `DatasetSourceReferenceTarget` in GraphQL, `reference_target` in
REST.

`Target` and `Datasets` had to be added to the reader's list of understood
columns, because anything it does not understand is folded into `Description` as
labeled text. A registry written for an older checkout is unaffected: no `Target`
column means every source is `data_point`, which is what those registries meant.

### Which datasets a source lands on

One upload run can produce many datasets — the input CSV's `Dataset` column
splits them — and one registry serves the whole run. A dataset-level source with
an empty `Datasets` cell attaches to every dataset in the run. That is the common
case, since a registry accompanies one upload of one workbook, and it is why the
column is optional.

Two guards, because an inert entry is invisible:

- A `Datasets` cell naming a dataset the run does not produce is **refused**,
  listing the names it did find. Otherwise the source is attached to nothing and
  the dataset it was meant for imports with no provenance at all. This check runs
  only where the full set of names is known — a run that splits the file on its
  `Dataset` column. A run narrowed to one dataset with `-d` cannot tell a typo
  from the datasets it was not asked to build.
- `Datasets` on a `data_point`-targeted source is **refused**. A per-row source is
  placed by the cells that cite it; a restriction there would read as though it
  did something.

`-d plain_csv` and `-d plain_csv_wide` upload the file as a single dataset and
pass no name at all. There is nothing to choose between, so every dataset-level
source applies and a `Datasets` restriction is moot rather than exclusionary.

## 2. What reaches DVC

Each entry of the dataset's `metadata['sources']` list now carries `edition` and
`target`:

```yaml
sources:
- name: Energiebilanz Rheinland-Pfalz
  authority: Statistisches Landesamt
  url: https://example.org/eb
  description: null
  edition: '2024'
  target: dataset
```

Still a list of dicts rather than a mapping keyed by name: a long human-readable
name used as a YAML *key* can be line-wrapped by the writer into YAML that will
not read back.

A missing `target` key means `data_point` — which is what every `.dvc` file
written before this feature says, and what those files meant. Nothing needs
republishing.

The one structural change on this side is that `build_sources_metadata` no longer
returns early when the data has no `Source` column. A dataset whose provenance is
entirely dataset-level has no citations to collect, and used to reach DVC with no
`sources` metadata at all.

## 3. What the import does

`load_dvc_dataset` reads the target back and creates
`DatasetSourceReference(dataset=…)` rows instead of one reference per data point.

**The set is replaced, not extended.** Per-point references CASCADE away with the
data points that `refresh_dataset_in_place` deletes, so they are naturally
rebuilt from scratch on every `--force`. A reference hanging off the `Dataset`
row survives that, so the import deletes the existing set first. Without it, a
re-import would add a duplicate every time and would keep a source the registry
had dropped, forever.

**Source rows are reused and refreshed.** A `DataSource` is identified by its
name within the instance's scope, so an existing one keeps its row and with it
every reference — from this dataset and from every other dataset citing it. Its
authority, URL, description and edition are updated from the incoming metadata
when they differ, and the change is printed. For an imported dataset the registry
in DVC is the source of truth ([`data-management.md`](data-management.md)), and
an edition that can never update is worse than no edition. A source that must
survive as a distinct entity — a superseded edition still cited elsewhere — needs
a distinct `Name`.

**`--plan` reports it.** Alongside the metric lines, the plan prints the
dataset-level sources it would keep, add and drop:

```
mainz/energiebilanz
  row          pk=412 uuid=…
  commit       9f2c1ab -> 3ee40d7
  data points  3783 -> 3783
  metrics kept value
  source keep  Energiebilanz Rheinland-Pfalz (dataset-level)
  source add   Verkehrsmodell (dataset-level)
  source drop  Altes Handbuch (dataset-level)
```

The drops are the reason this is worth printing: a wholesale replacement is
otherwise silent, and provenance disappearing is not the kind of change anyone
notices from the data.

## 4. Reading it back

Both levels are exposed everywhere they were already modeled, so nothing on the
read side had to be built:

- **GraphQL** — a dataset's `sourceReferences` takes a target of `DATASET`,
  `DATA_POINT` or all; `dataSources` unions the two.
- **REST** — `/datasets/<uuid>/sources/` takes `reference_target`
  (`dataset`, the default, `data_point`, or `all`);
  `/datapoints/<uuid>/sources/` is the per-point list.
- **Instance export/import** — `_export_dataset_provenance` emits a snapshot with
  `point: null` for a dataset-level reference and restores it on import, so a
  copied or published instance keeps it.

**Worth verifying before converting a live dataset:** a consumer that reads only
a data point's own `sourceReferences` will render every point of such a dataset
as unsourced. It has to fall back to the dataset's references. The failure is
silent and looks like data loss.

## 5. Converting an existing dataset

Per dataset, and deliberately not in bulk — whether the provenance really is
uniform is an editorial judgement about that table:

1. Add `Target,dataset` for the source in the registry CSV (and `Edition`, while
   you are there).
2. **Remove the `Source` column from the data**, or at least the cells naming
   that source. Keeping both is refused, not merged.
3. Re-run `python -m notebooks.upload_new_dataset … --sources-csv …` and push.
4. `python manage.py load_dvc_dataset <instance> <dataset> --plan`, check the
   source lines, then `--force`.

The old per-point references disappear with the data points the refresh deletes;
nothing has to clean them up.

## 6. Refusals, and why each one is not a merge

| situation | what happens |
|---|---|
| `Target` is neither `data_point` nor `dataset` | Registry load fails. Falling back to the default would read as dataset-level in the CSV while the import quietly made nothing |
| A `dataset`-targeted source is also cited by a `Source` cell | Upload fails. Both attributions is a duplicated claim, not a stronger one. (Hand-edited DVC metadata that reaches the import anyway gets the dataset-level link and a warning) |
| `Datasets` names a dataset the run does not produce | Upload fails, listing the names found |
| `Datasets` on a `data_point`-targeted source | Registry load fails |
| A `Source` cell names something absent from the registry | Warns at upload and at import; the name survives, the provenance does not |

## 7. The database constraint, and the bug it found

`DatasetSourceReference` now requires exactly one of `data_point` and `dataset`
(`source_reference_targets_exactly_one`). Both columns are nullable because
either may be the one in use, which left two invalid states reachable — and one
of them was being written routinely.

**References created through the REST API named both.** The data-point endpoint
is nested under a dataset (`/datasets/<uuid>/data_points/<uuid>/sources/`), and
the serializer took the dataset from the URL as well as the data point. So every
source added to a single value also read as a source for the *whole dataset*: it
came back from `reference_target=dataset` and from GraphQL's `DATASET` target,
which is exactly the claim this feature is now making deliberately. The
serializer sets only the target the endpoint is about, and the migration repairs
the existing rows by clearing the dataset side — it was the URL's shape, not a
claim.

**"Neither" has nothing to repair it to**, so the constraint simply fails on a
database that holds such a row. Check before deploying, including in Kausal
Watch, which shares this model through `kausal_common`:

```python
DatasetSourceReference.objects.filter(data_point__isnull=True, dataset__isnull=True).count()
```

## Where the code is

| file | what |
|---|---|
| `nodes/constants.py` | `SOURCE_TARGET_DATA_POINT` / `SOURCE_TARGET_DATASET` / `SOURCE_TARGETS`, the one definition both sides read |
| `notebooks/upload_new_dataset.py` | `SourceRegistryEntry`, `load_sources_registry`, `check_registry_dataset_names`, `build_sources_metadata` |
| `nodes/management/commands/load_dvc_dataset.py` | `source_target`, `dataset_level_source_names`, `ResolvedSource`, `Command.sync_dataset_source_references`, the plan's source lines |
| `kausal_common/datasets/models.py` | `DatasetSourceReference`, the constraint, `DataSource.get_label` |
| `kausal_common/datasets/api.py` | `BaseSourceReferenceSerializer.to_internal_value`, which used to write both targets |
| `kausal_common/datasets/migrations/0037_…` | the constraint and the repair of rows naming both |
| `datasets/tests/test_dataset_rest_api.py` | that the data-point endpoint no longer writes a dataset reference |
| `nodes/tests/test_upload_dataset_level_sources.py` | registry parsing and metadata building |
| `nodes/tests/test_load_dvc_dataset_provenance.py` | import, re-import, mixed targets, plan, constraint |
