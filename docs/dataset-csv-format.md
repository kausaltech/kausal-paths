# The Wide CSV Upload Format (with per-row metadata)

The format `tools/upload_new_dataset.py` reads: one CSV holding **several
datasets**, each row a value series for one metric × dimension combination, with
per-row provenance (`Source`, `Comment`, `UUID`) that survives the round trip into
`DataSource` and `DataPointComment` records.

Until now this was documented only implicitly — the authoritative behaviour lives
in `upload_new_dataset.py` (`RESERVED_ROW_COLUMNS`, `load_sources_registry`,
`build_sources_metadata`), and `data/cork/README.md` describes a *different* wide
format used by the NZP pipeline. This document is the spec for the NZC-style one.

Related: [`manage-datasets-guide.md`](manage-datasets-guide.md) covers the
YAML-configured *processing pipeline* (a different entry point);
[`dataset-management-commands.md`](dataset-management-commands.md) covers
`load_dvc_dataset`.

---

## 1. Column families

Every column falls into exactly one of five families. Which family a column
belongs to is decided by name, so the names matter.

### Routing columns

| column | meaning |
|---|---|
| `Dataset` | Human-readable dataset name. Rows are grouped by this; each group becomes one DVC dataset. Snake-cased for the dataset id (`Other Sectors` → `other_sectors`, published as `<namespace>/other_sectors`). |
| `Metric` | The metric within the dataset. Becomes a metric column in the resulting dataframe. `Quantity` is accepted as a fallback when `Metric` is absent. |

### Unit and quantity columns

| column | meaning |
|---|---|
| `Quantity` | The Paths quantity (`emissions`, `energy`, `mass`, `fraction`, …). Must be a quantity the model knows. |
| `Unit` | Pint-parseable unit string (`kt/a`, `GWh/a`, `kg/a`, `dimensionless`). One unit per (`Dataset`, `Metric`) group — mixed units within a metric are an error. |

### Dimension columns

Any column that is **not** in another family becomes a dimension (index) column.
Values must be category labels (or ids) of a dimension registered in the
instance's context, or loading fails. Blank means "not applicable to this row".

This is the default-catch family, which is why the reserved names below exist:
without them, `Source` would be read as a dimension called "Source".

### Reserved per-row metadata columns

`RESERVED_ROW_COLUMNS = {'source', 'comment', 'description'}` (matched
case-insensitively). These ride through to DVC as literal per-row values but are
**never** treated as dimensions. In new files use `Source` and `Comment` only;
`description` remains in the set for backward compatibility with
`plain_csv_wide` files that predate this rule.

| column | meaning |
|---|---|
| `Source` | One or more names from the sources registry (§2), joined by `SOURCE_NAME_SEPARATOR` (`'; '`). `load_dvc_dataset` splits on the same separator and creates one `DatasetSourceReference` per name. |
| `Comment` | Free text attached to the data point. This is where a value's rationale, a superseded value, a source-cell reference, or a known-error note belongs. Several notes about one data point are joined by `COMMENT_SEPARATOR` (`' ;; '`) and become separate `DataPointComment` records. |
| `Description` | **Deprecated — never use alongside `Comment`.** `upload_new_dataset` drops it in `clean_dataframe()` before writing to DVC, folding it into the dataset-level description instead, so per-row `Description` text does not survive. Both `upload_new_dataset` (in `validate_required_columns`) and `load_dvc_dataset` raise if a file carries both columns, rather than discarding one silently. |
| `UUID` | Stable identifier for the metric × dimension combination, letting a row be tracked across file revisions. Not a dimension; not required. |

#### Several notes on one data point

`COMMENT_SEPARATOR` is `' ;; '`, defined in both `upload_new_dataset.py` and
`load_dvc_dataset.py`. It is deliberately **not** `'; '` like the source
separator: source names are short identifiers, but comments are prose and prose
contains semicolons — 162 of the 12,899 comment cells already under `data/`
contain `'; '` mid-sentence, and splitting on it would fragment them.

This is a separate concern from CSV quoting. Quoting is what lets a *field*
contain commas, semicolons and newlines at all, and it applies automatically;
but a quoted field is still one string, and CSV has no notion of structure
*within* a field. Delivering N notes in one cell needs an in-cell convention
whatever the quoting does.

#### Granularity: per dimension-combination *and* per metric

`Source` and `Comment` are part of the pivot index
(`pivot_by_compound_id`: `dim_cols` is every column except `Quantity`, `Value`
and `metric_col`). Two consequences:

- **Metric-specific comments are preserved.** If two metric rows share a
  dimension combination but carry *different* comments, they do not collapse
  into one stored row — they stay as two rows, each with the other metric's
  column null. On load, each metric's data point gets its own comment.
- **Identical comments collapse**, as intended: metric rows sharing a dimension
  combination *and* a comment become one stored row, and `load_dvc_dataset`
  attaches that comment to every non-null metric's data point in the row. Nine
  rows × three metrics then yields 27 commented data points.

So giving every metric row of a dimension combination the same `Comment` (as
`data/cork/make_kpmg_building_heat.py` does) keeps the stored table dense; giving
them different comments costs sparsity but loses nothing.

For **year-specific** comments and sources, use the long format (§"Long format"
below) — in wide format one `Comment` cell necessarily spans every year column
of its row.

### Year columns

Any column whose name is a year (`2018`, `2023`, `2030`…) is a value column, and
gets unpivoted into `Year` / `Value`. Small integers (`0`, `1`, `100`) are
relative-year offsets in the NZC framework — see
[`architecture/dataset-metric-names.md`](architecture/dataset-metric-names.md)
§"The NZC relative-year offset".

Leave a year cell blank where there is no observation; blanks are dropped, not
zero-filled. **A blank and a zero mean different things** — a zero is an
assertion that the value is zero. When the blanks are the point, because the file
is a template a city has yet to fill in, see §5.

### Long format

A file may instead carry explicit `Year` and `Value` columns rather than one
column per year. `convert_to_standard_format` returns such a file unchanged
(it only unpivots when no `Year` column exists), and `Year` then behaves as an
ordinary index column in the pivot.

```csv
Dataset,Metric,Quantity,Year,Value,Unit,Sector,Source,Comment
T,Energy,energy,2020,10,GWh/a,A,S1,metered
T,Energy,energy,2021,11,GWh/a,A,S2,estimated from the 2020 reading
```

Column *order* is free — the pipeline keys on names throughout — so put `Year`
and `Value` next to `Quantity` as above rather than at the end of the row. The
fact then reads as one phrase (quantity, year, value, unit) instead of being
pushed past every dimension column and the comment prose.

**Use it when `Source` or `Comment` must vary by year.** In wide format one
comment cell necessarily applies to every year column of its row; in long format
the year is part of the row, so provenance can differ per (dimension
combination, metric, year). The example above stores as two rows, each keeping
its own source and comment.

The cost is the same sparsity described above: every distinct
(dimensions, year, source, comment) tuple is its own stored row, so a dataset
whose metrics have year-varying comments gets one row per metric rather than one
row per dimension combination. Values and metadata all survive; the parquet is
just wider and more null-filled. Prefer wide format when provenance is uniform
across years, which is the common case.

## 2. The sources registry

Provenance is **not** written inline. `Source` cells hold names that key into a
separate registry CSV — one row per source, so a source cited by fifty rows is
described once.

Registry columns:

| column | maps to |
|---|---|
| `Name` | the key `Source` cells refer to (required) |
| `Authority` | `DataSource.authority` |
| `URL` | `DataSource.url` |
| `Description` | `DataSource.description` |
| `Edition` | `DataSource.edition` |
| `Target` | `data_point` (default) or `dataset` — see below |
| `Datasets` | for a dataset-level source, which datasets it applies to |
| *anything else* | appended to `Description` as `'<Column>: <value>'` rather than dropped |

Cork's registry is `data/cork/cork_sources.csv`, which carries the extra columns
`Attachment`, `Format`, `Licensing`, `Geographic Coverage`, `Update Frequency`,
`Date Last Updated`, `Updated by` — all folded into the description.

The registry is passed to the uploader and the used subset is written into the
DVC dataset's `metadata['sources']` as a **list** of `{name, authority, url,
description, edition, target}` dicts. It is a list, not a mapping keyed by name,
deliberately: a long human-readable name used as a YAML *key* can get
line-wrapped by the writer and produce YAML that will not read back. Names are
safe as values.

### Sources that belong to the whole dataset

A source with `Target,dataset` attaches to the dataset itself rather than to its
rows. Use it when the provenance is uniform — one publication, one update,
nothing per row to distinguish — instead of repeating the same name in every
`Source` cell. Such a source needs no citations and no `Source` column at all,
and a dataset may carry several of them. A single source may not do both: a
dataset-level source that is also cited from a `Source` cell is refused rather
than merged. (`Target`, not `Scope`: `DataSource.scope` is the owning instance,
and `Scope` in a data file is the GHG scope.)

[`dataset-level-data-sources.md`](dataset-level-data-sources.md) has the whole
mechanism — the `Datasets` restriction, what the import does on a re-import, and
how to convert a dataset that currently cites one source from every row.

## 3. Worked example

```csv
Dataset,Metric,Quantity,Unit,Scope,Ghg,UUID,Source,Comment,2018,2023
Other Sectors,Emissions from other sectors,Emissions,kt/a,Scope 1,co2e,,CorkCity_BEI_v2_Agriculture_20260323.xlsx,,,69.96530264
Other Sectors,Emissions from other sectors,Emissions,kt/a,Scope 1,co2e,,,Set to zero to reproduce the inventory; modelled value was 64.45.,,0.0
```

Both rows are the same metric of the same dataset, distinguished by their
dimension values; the second carries a `Comment` recording why its value is zero
and what it replaced.

## 4. Upload

```bash
python -m tools.upload_new_dataset \
  --input-csv data/<city>/<file>.csv \
  --output-dvc <namespace> \
  --instance <instance-id> \
  --language en \
  --source data/<city>/<city>_sources.csv \
  [-d <dataset_name>]      # omit to upload every Dataset group in the file
```

Three things that are easy to get wrong:

- **Run it as a module** (`python -m tools.upload_new_dataset`), not as a
  script path, so the project root is on `sys.path`.
- **`--instance` takes an *instance* identifier, not a framework one** — e.g.
  `cork-nzc`, not `nzc`. It is what the dimension and category names in the file
  are validated against, so the wrong value validates against the wrong
  dimensions.
- **`--source` (`--sources-csv`) is not optional in practice.** Without it the
  `Source` names are still written to `metadata['sources']`, but with
  `authority`, `url` and `description` all null — the registry is never read. The
  names survive; the provenance does not.

Full flag list:

| short | long | notes |
|---|---|---|
| `-i` | `--input-csv` | required |
| `-o` | `--output-dvc` | DVC namespace, may be nested (`cork/kpmg`) |
| `-c` | `--output-csv` | write locally instead of uploading |
| `-s` | `--csv-separator` | |
| `-e` | `--encoding` | |
| `-l` | `--language` | default `en` |
| `-d` | `--dataset` | filter to one `Dataset` group |
| `-n` | `--instance` | validates dimensions/categories |
| — | `--sources-csv` | the source registry; `--source` is an accepted prefix |

Afterwards, `--pull-datasets` advances the YAML commit pointer to the new DVC
commit.

For flat NZC-style files where every dimension combination lives in one file, use
`-d plain_csv_wide`, which sets `index_columns` to every column except `Value`,
`Unit` and `UUID`. See `data/cork/README.md` §"Why plain_csv_wide and not
plain_csv" for why including `Value` breaks the round trip.

## 5. Shipping an empty template

Sometimes the blank cells *are* the deliverable: a dataset a city is meant to fill in, where
every value is empty on purpose. This is the opposite of the usual case, and it needs four
deliberate steps, because every layer of the pipeline drops empty cells by default and each one
is right to do so for ordinary data.

**Why it matters.** A presence check has to be able to tell "the municipality entered zero" from
"nobody has looked at this yet". Those are different facts, and standards that accept a reported
zero — BISKO among them — make the distinction load-bearing: a balance built on unexamined cells
must not report conformity. If the template ships zeros, no check can ever tell the two apart,
because a pre-filled zero and an entered zero are the same number.

### The four steps

**1. The CSV leaves the value cells empty.** Not `0`, not `.` — empty. Keep the `Comment` column
populated: it is what tells the city what belongs in the cell, and it survives to the editor.

```csv
Metric,Unit,Quantity,Dataset,vehicle_type,road_type,2023,Comment
Value,Mvkm/a,mileage,vehicle_kilometers,passenger_cars,highway,,"Fahrleistung Pkw auf Autobahnen, in Mio. Fahrzeugkilometern pro Jahr."
```

**2. Upload with `--keep-empty-cells`.** Without it the upload fails with *"No year columns found
and no Year column exists"*: `clean_dataframe` drops a year column that is entirely null, and
`convert_to_standard_format` then filters out every null value. Both are correct defaults —
dropping blanks is what keeps a sparse wide file from becoming a dense one — so the template case
is opt-in.

```bash
python -m tools.upload_new_dataset -i template.csv -o <repo> --language de \
    --instance <instance> --keep-empty-cells
```

Note this affects **wide format only**. A long-format file (one that already has a `Year` column)
returns early from `convert_to_standard_format` and never hits either step.

**3. Import keeps the empty cell as a null data point.** `load_dvc_dataset` creates a `DataPoint`
with a null `value` rather than skipping the row, so the cell exists in the editor with its
dimension categories, source link and comment intact. `DataPoint.value` is nullable and GraphQL
types it `float | None`; nothing extra is needed.

**4. Tag the *consuming* binding `empty_to_zero`.** This is the step that is easy to get wrong.

A node that reads the template still has to compute while the template is empty — often it has no
choice, because `select_port` evaluates both of its branches even when the switch selects the
other one. But an empty frame has no rows and therefore no dimensions, which fails far away as
*"Dimensions (set()) do not match in output"*.

```yaml
input_datasets:
- id: bisko/vehicle_kilometers
  column: Value
  tags: [mileage, empty_to_zero]
```

**The tag belongs to the binding, not to the dataset**, and that is exactly what makes this work.
The same dataset is read twice, and the two readers want opposite things:

| Reader | Tag | Sees | Why |
| --- | --- | --- | --- |
| the node that computes with the data | `empty_to_zero` | zeros | so it computes instead of failing |
| a `DataAvailabilityNode` checking the data | *(none)* | empty cells | so it reports 0 and the balance is not declared conformant |

Tagging the *dataset* would give both readers zeros and silently defeat the presence check — the
tool would report "the city supplied this" about a template nobody has touched.

### Checking it

After the upload, the availability node should read `0.0` in every required cell while the
consuming node computes without error. If the availability node reads `1.0`, the cells are zeros
rather than blanks somewhere along the chain: check the CSV first, then whether the tag ended up
on the availability binding by mistake.

## 6. Rules worth stating

- **Any unrecognised column becomes a dimension.** A typo in a metadata column
  name turns it into a dimension and the load fails on an unknown dimension —
  which is the good outcome. A typo in a *dimension* column's value fails on an
  unknown category, also good. Silent misreads come from column names that
  collide with reserved names by accident.
  **Dimensions are unchecked without --instance** However, if the upload_new_dataset command does not have `--instance` defined, dimensions and categories are uploaded as is, without checking anything. This approach is depreciated and useful only for a dataset that is never meant to be loaded to DB.
- **One unit per metric.** Convert before writing the CSV, not after.
- **A metric name must not collide with a dimension column name.** Both end up
  as columns of the pivoted frame, keyed by `to_snake_case`, so a `Metric` of
  `Livestock` alongside a `Livestock` dimension raises
  `DuplicateError: column 'livestock' is duplicate` — and only *after* earlier
  datasets in the same file have already been written. Rename the metric.
- **Blank ≠ zero.** See §1 — and §5 for how to ship a file that is deliberately
  all blanks, which takes an explicit flag at upload and an explicit tag at the binding.
- **Comments are the home for adopted-error documentation.** When a model must
  reproduce a source's mistake, the `Comment` cell is where the correct value and
  the reasoning go; see
  [`nzp-kpmg`-style error registers](matching-a-model-to-an-inventory.md#10-when-the-city-wants-the-inventorys-errors-reproduced).
  It is also where the `source_ref` (`workbook!sheet!cell`) belongs — there is no
  separate column for it, and `Description` will not carry it (§1). Join it to
  the other notes with `COMMENT_SEPARATOR`.
- **`Comment`, never `Comment` + `Description`.** Two comment columns in one
  file is an error, not a merge. `upload_new_dataset` raises up front, because
  that is where the per-row `Description` text would otherwise be dropped;
  `load_dvc_dataset` raises too, for datasets that reach it by another path.
- **Prefer a dataset over inline YAML `historical_values`.** Values with
  provenance belong in a dataset where the `Source`/`Comment` machinery applies
  and the admin UI can show them; `historical_values` in a config file carries
  none of that.
- **DB records shadow DVC.** If a `DBDatasetModel` record exists for a dataset
  id, it is loaded instead of the DVC version, so a DVC upload will not be
  visible until the DB record is refreshed or deleted. See
  `data/cork/README.md` §"Procedure for updating cork-nzc DB datasets".
- **Uploading is not importing.** `upload_new_dataset.py` puts the data in DVC;
  `Source` and `Comment` become `DataSource` links and `DataPointComment` records
  only when `load_dvc_dataset` runs. Until then the provenance is in the dataset
  but invisible in the admin UI — which reads as "the comments were lost".

  ```bash
  python manage.py load_dvc_dataset <instance> <dataset-id>
  ```
