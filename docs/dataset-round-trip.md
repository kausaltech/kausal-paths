# Lossless dataset round trip: DB or DVC → CSV → DVC → DB

*Produced by Claude Opus 5.0 on 2026-08-30.*
*Version 2 produced by Claude Opus 5.0 on 2026-08-30, after building it.*
*Responsible: Jouni Tuomisto.*

One thing that used to need several partial tools and some hand work: take a dataset
out of the database *or* out of DVC as CSV, edit it, push it back with
`upload_new_dataset`, import it with `load_dvc_dataset`, and get the same dataset —
values, dimensions, units, comments and provenance included.

**Status.** §7's steps 1-5 are built and tested. §2 is the survey that motivated them,
kept because it names what the older tools still lose. §3.1 was thought to be a defect and
is not — it is the empty-template mechanism, and the documentation has been corrected to
match. §3.2 is open: the representation is intended, what the importer and the loader then
do with it loses values, and the measurement is recorded there.

## 1. What already round-trips

`upload_new_dataset` carries `Source` and `Comment` into the stored parquet as
literal per-row columns, and the sources registry into `metadata['sources']`.
`load_dvc_dataset` reads them back: one `DatasetSourceReference` per `'; '`-joined
source name, one `DataPointComment` per `' ;; '`-joined note, at either target
level ([`dataset-level-data-sources.md`](dataset-level-data-sources.md)).

Long format (explicit `Year` and `Value` columns) is accepted throughout, so
provenance that varies by year survives as well
([`dataset-csv-format.md`](dataset-csv-format.md) §"Long format").

So **CSV → DVC → DB loses nothing that the CSV format can express**. Everything
below is about what the CSV format cannot express, and about the missing exporter.

## 2. Where it leaked

The survey that motivated the work. The `sync_datasets` rows still stand — that command
is untouched (§6) — and the rest are closed by §4 and §5.

| hop | what is lost | where |
|---|---|---|
| DB → CSV | comments, per-point source references, dataset-level sources, and the registry behind them | `nodes/management/commands/sync_datasets.py:204` — the fieldnames are `Metric, Unit, Quantity, Dataset` + dimensions + years, and nothing else |
| DB → CSV | `Quantity`, written but always empty | same, line 149 |
| DB → CSV | dimension columns keyed by **translated display name** (`d.name_i18n or d.name`), not by a stable identifier | same, line 192 |
| DB → CSV | dataset-level attributes: `forecast_from`, `time_resolution`, `is_editable` | no slot in the format — **closed by §4** |
| DVC → CSV | everything — no exporter wrote the upload format from DVC | `tools/fetch_dataset.py` dumps the raw parquet, which does carry `Source`/`Comment`, but not `metadata['sources']` — **closed by §5's `--from dvc`** |
| CSV → DVC | dataset-level attributes | `push_to_dvc` wrote only `description`, `metrics`, `sources`, `index_columns`, `updated_at` — **closed by `metadata['dataset']`** |

The registry line is the one that matters most. Even if `sync_datasets csv` wrote
a `Source` column tomorrow, the names would arrive at the next import with no
authority, URL, description or edition behind them — the registry is a separate
file and nothing generates it.

## 3. What the holes are, and are not

The first two subsections correct an earlier draft of this plan, and are kept rather than
quietly dropped. The rest are what building it turned up.

### Data-point UUIDs are not an identity hole

`load_dvc_dataset` mints a fresh `DataPoint.uuid` on every import
(`load_dvc_dataset.py:785` — `DataPoint.objects.create(dataset, date, metric,
value)`), and that is correct rather than lossy. The codebase has already chosen
natural-key identity for data points: `DataPointKey` in
`nodes/instance_serialization.py:189` is documented as *"Natural key locating a
DataPoint within its dataset (id-free, restore-stable)"* and is
`(year, metric identifier, sorted category ids)`. The export/import layer
deliberately does not use point UUIDs, and a CSV row **is** that natural key.

So the round trip needs no point identifier, and `load_dvc_dataset` should not
grow one.

The `UUID` *column* is a separate, older thing and must not be conscripted for
this. It exists for the NZC framework, where it matched city-level framework
measures against a pre-existing model in the days of one monolithic DVC dataset.
That dataset is gone; `configs/nzc.yaml:735` now declares `uuid` as an ordinary
**dimension** whose categories are the UUIDs, which is why the normal upload path
needs no special case — `RESERVED_ROW_COLUMNS` (`nodes/constants.py:23`) is
`{'source', 'comment', 'description'}` and does not contain `uuid`, so a `UUID`
column falls into the default-catch family and is read as a dimension.

Two loose ends follow from that, neither blocking:

- `dataset-csv-format.md` §1 listed `UUID` in the table of *"Reserved per-row metadata
  columns"*. That was wrong: it is not reserved, and in the normal path it is a
  dimension. Corrected on 2026-08-30 (§7 step 5).
- The one real special case left is `non_index = {'Value', 'Unit', 'UUID'}` in the
  `plain_csv_wide` branch (`tools/upload_new_dataset.py:902`). It is legacy. It
  should be checked against the remaining `plain_csv_wide` users and probably
  removed, but separately from this work.

### The forecast flag is a format gap, not a missing fact

`DataPoint` has no forecast field, but the boundary is not lost: it lives in
`Dataset.spec['forecast_from']` (read at `nodes/datasets.py:1117`) and
`DatasetSnapshot.forecast_from` already maps it. It is a dataset-level attribute
with nowhere to go in the CSV format, which is §4 rather than a separate problem. §4 now
gives it one.

### 3.1 Provenance on empty cells is the template, not a leak

An earlier draft of this document called this a defect. It is not, and the reason is the
product: **an empty dataset is shipped to a city with the instruction, or the source to
obtain the figure from, written in the `Comment` of the cell they are being asked to
fill.** A `Comment` is scoped to its row, a row spans every metric, and a comment that
only reached cells which already held values could not say "we need this, here is where
to get it".

So `create_data_points` attaching a row's `Source` and `Comment` to every metric of that
row, empty cells included, is correct. `dataset-csv-format.md` said "every **non-null**
metric's data point"; that wording was the error and has been corrected.
`test_provenance_reaches_valueless_cells` now holds the behaviour in place rather than
flagging it.

### 3.2 A split row duplicates a cell, and the duplicate is resolved against the value

Held by `test_metric_specific_provenance_does_not_duplicate_a_cell` and
`test_a_split_row_does_not_lose_the_value_a_node_reads`, both `xfail(strict=True)`.
**Undecided, and recorded here with the measurement rather than as a verdict.**

The representation is not in question. When two metrics of one (Year, dimensions) carry
different provenance they cannot share a row in a table that is wide by metric, so the
pivot emits two rows, each holding the other metric's cell as null. That is the only way
to express metric-specific provenance, and it is intended.

What follows from it was expected to be harmless — an empty row, dropped on load. It is
not what happens. The importer makes a data point for every metric of *both* rows, so a
real value and a null land under one natural key; and `DBDataset.deserialize_df`
(`nodes/datasets.py:1255`) resolves that collision like this:

```python
dupes = df.group_by(uniq_cols).agg(pl.count()...).filter(pl.col('_count') > 1)
if len(dupes) > 0:
    capture_error('Dataset %s (pk %d) has %s duplicate rows' % ...)
    df = df.group_by(uniq_cols).first()
```

`uniq_cols` is `[Year, *dimensions, metric]`. **The value is not in it**, so which of the
two rows survives is arbitrary — and the null wins about half the time. Measured on the
test fixture, round-tripped through wide format:

| cell | in the database | what a node reads back |
|---|---|---|
| `value` 2020 residential/gas | 10.5 | **null** |
| `value` 2021 residential/gas | 11.25 | **null** |
| `quality` 2020 residential/gas | 3.0 | 3.0 |

Two of the three values gone, before any node sees the dataset, and every load reported
to Sentry as duplicate rows.

No production dataset is in this state today — `mainz/final_energy` has 3896 points and
3896 distinct natural keys — because it takes two metrics of one cell disagreeing about
provenance to trigger, and no upload has done that yet. `export_dataset` makes it easier
to reach, since a round trip now reproduces whatever provenance the database holds.

A fix that keeps everything §3.1 and this section want: **per (Year, dimensions) group and
per metric, create one data point — from the row that has a non-null value for that metric
if any row does, otherwise from the first row.** Templates are untouched (nothing else
fills the cell, so the empty row with its instruction is still the one that lands).
Metric-specific provenance still works, because each metric's point still takes the
provenance of the row it came from. The duplicate, the Sentry report and the coin-flip
disappear.

### 3.3 Two asymmetries that are not defects

- **A wide export must be uploaded with `--keep-empty-cells`.** Blank cells are dropped
  otherwise, and a valueless data point comes back absent rather than empty — the very
  distinction §3.1 is about. `export_dataset` counts them and puts the flag in the command
  it prints.
- **The DVC copy carries a fuller registry than the database.** `metadata['sources']` keeps
  every source the upload declared; the database keeps only those some data point still
  cites. Exporting `mainz/final_energy` gives 10 sources `--from dvc` and 1 `--from db`.
  Neither is wrong: they answer different questions.

## 4. The format gap: dataset-level attributes

Some facts belong to the dataset rather than to any row, and the wide CSV has no place
for them: `forecast_from`, `time_resolution`, `is_editable`, and the dataset's name.

They must not become per-row columns. A per-row column for a dataset-level fact
is exactly the mistake that dataset-level sources were introduced to undo, and it
invites the same contradiction — two rows disagreeing about a fact the dataset can
only have one of.

**Built as a third sidecar file, `<name>_dataset.csv`**, one row per dataset:

```csv
Dataset,Identifier,Name,ForecastFrom,TimeResolution,IsEditable
endenergie,mainz/endenergie,Endenergieverbrauch,2024,yearly,true
```

The other two hops carry it:

1. `upload_new_dataset --dataset-csv` folds it into a new `metadata['dataset']` key
   (`build_dvc_metadata`).
2. `load_dvc_dataset` reads that key back onto the row and its schema
   (`Command.apply_dataset_attributes`).

A missing key means "unchanged", exactly as a missing `target` key means `data_point` —
so nothing already in DVC needs republishing, and
`test_dvc_file_without_the_dataset_key_changes_nothing` holds that line.

**`external_ref` and `is_external_placeholder` are deliberately not carried.** They are
not lost: the import writes them itself, stamping the commit it actually read. Carrying a
value round the loop would replace a true statement with a stale one.

## 5. The new command

```bash
python manage.py export_dataset <instance> <identifier> --out DIR \
    [--from db|dvc|auto] [--repo-from auto|yaml|db] [--format auto|wide|long]
```

Writes up to three files into `DIR`: `<name>.csv` (upload format), `<name>_sources.csv`
(the registry), and `<name>_dataset.csv` (§4). The output is exactly what
`upload_new_dataset -i <name>.csv --sources-csv <name>_sources.csv` reads, so the
loop closes by construction rather than by agreement between two authors.

**`--from auto`** reads the DB row when one holds data points and DVC otherwise, and
**refuses when both hold data**, naming the counts and the pin and pointing at
`dataset_inventory`. `--from db` and `--from dvc` state the intent and are never refused
on those grounds. The DVC side goes through `Context.load_dvc_dataset()`
(`nodes/context.py:375`) — identifier → the instance's pinned commit → dataframe — the
same call `dataset_inventory.py:209` already makes. No hash needed.

**A wide export of a dataset holding valueless cells prints `--keep-empty-cells` in the
upload command it suggests.** Without the flag the uploader drops blank cells, and the
next import brings the cell back as absent rather than as empty (§3.3).

**`--format auto`** emits wide when `Source` and `Comment` are uniform across the
years of each row, long when they vary. That is the choice `dataset-csv-format.md`
documents; making it from the data removes the one decision a person currently has
to get right by hand.

**Dimension columns are keyed by category identifier**, not by translated display
name. The current exporter's use of `name_i18n` is a live round-trip hazard on a
non-English instance, and the same class of bug as the dataset labels described in
`docs/trailhead/tools.md` §`rename_dataset`.

### What it means for `fetch_dataset`

`--from dvc` covers the use it is put to — reading the exact content of what is stored —
without needing the hash first, and it returns the upload format rather than a raw parquet
dump.

One case does not survive the generalisation: a bare hash with no idea which dataset it
belongs to, from a bucket listing or an error message. The earlier draft of this plan put
that on `export_dataset` as an `--ekey` flag. **It was not done that way.** A raw hash dump
and an upload-format export are different output shapes to mix in one command, and
`fetch_dataset` needs no Django — a real virtue in a tool whose job is to say what is
*actually* in the bucket, independently of any model or pin.

So it stays its own tool, moved to `tools/fetch_dataset.py` and now covered by ruff and
mypy. Fixing what that turned up also fixed a real bug: the MD5-path fallback was tried
*inside* the ETag loop, on the first entry whose ETag did not match, so the ETag search was
effectively skipped on any bucket whose first object was not the one wanted. The three
strategies — exact key, ETag, then the conventional `files/md5/<2>/<rest>` path — are now
tried in order, once each.

## 6. Deliberately not touched

- **`sync_datasets`** (`--action csv` and the rest). It overlaps §5 and does the
  job less completely, but it predates this work by a year and its other actions
  have users whose needs have not been surveyed. Leave it. The overlap is recorded
  here so that whoever surveys it later starts from the analysis rather than
  redoing it.
- **`delete_dataset --dump-to`**. A long-format rescue dump of data about to be
  destroyed, with lowercase headers and no registry. It is deliberately not the
  upload format: its job is to lose nothing about *rows*, at a moment when
  re-importability is not the point. Keep it as it is.

## 7. What was built

All five done on 2026-08-30.

1. **`export_dataset --from db`**, emitting the data CSV and the sources registry —
   `nodes/management/commands/export_dataset.py`.
2. **`--from dvc|auto`**, via `Context.load_dvc_dataset()`. `tools/fetch_dataset.py` is
   *not* retired into an `--ekey` flag; see §5 for why.
3. **The `<name>_dataset.csv` sidecar**, plus `metadata['dataset']` in
   `upload_new_dataset` (`--dataset-csv`, `build_dvc_metadata`) and the reader in
   `load_dvc_dataset` (`apply_dataset_attributes`).
4. **The round-trip test** — `nodes/tests/test_dataset_round_trip.py`. Seven passing, four
   `xfail(strict=True)` for §3.1 and §3.2 across both formats. It does not push to DVC:
   `build_dvc_frame` (split out of `process_dataset` for this) produces exactly the frame
   that would be stored, and the test hands it to `create_data_points` with the units and
   index columns the push would have recorded. Every transformation that can lose
   something is on this side of the store.
5. **The `UUID` row in `dataset-csv-format.md` §1** corrected, with the history from §3.

One incidental tidy: `SOURCE_NAME_SEPARATOR`, `COMMENT_SEPARATOR` and
`DATASET_NAME_SEPARATOR` were defined twice, in `upload_new_dataset` and
`load_dvc_dataset`, each with a comment saying it must match the other. They now live in
`nodes/constants.py` beside `RESERVED_ROW_COLUMNS`, which three commands read.

### Still open

- **§3.2**, the split-row collision. A change in `create_data_points`, sketched at the end
  of that section. It changes what `load_dvc_dataset --force` writes for every city, so it
  wants its own review.
- **The legacy `UUID` special case** in the `plain_csv_wide` branch
  (`non_index = {'Value', 'Unit', 'UUID'}`), to be checked against remaining
  `plain_csv_wide` users and probably removed.

## 8. Settled

- **No multi-dataset mode.** The format can hold several datasets in one file,
  keyed by the `Dataset` column, but the command exports one named dataset. If a
  whole-instance export is wanted later it comes out of the `sync_datasets` survey
  (§6), which is where the requirement would originate.
- **`--from auto` refuses when the two copies disagree.** It names the drift and
  points at `dataset_inventory`, rather than picking a side. A silent choice
  between two populated copies is the failure this whole document is about: the
  export would look complete and be half of one dataset and half of another.
  `--from db` or `--from dvc` states which one you mean and is never refused on
  these grounds.
