# The Wide CSV Upload Format (with per-row metadata)

The format `notebooks/upload_new_dataset.py` reads: one CSV holding **several
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
**never** treated as dimensions.

| column | meaning |
|---|---|
| `Source` | One or more names from the sources registry (§2), joined by `'; '`. `load_dvc_dataset` splits on the same separator and creates one `DatasetSourceReference` per name. |
| `Comment` | Free text attached to the data point. This is where a value's rationale, a superseded value, or a known-error note belongs. |
| `Description` | Longer per-row description. Folded into metric metadata. |
| `UUID` | Stable identifier for the metric × dimension combination, letting a row be tracked across file revisions. Not a dimension; not required. |

### Year columns

Any column whose name is a year (`2018`, `2023`, `2030`…) is a value column, and
gets unpivoted into `Year` / `Value`. Small integers (`0`, `1`, `100`) are
relative-year offsets in the NZC framework — see
[`architecture/dataset-metric-names.md`](architecture/dataset-metric-names.md)
§"The NZC relative-year offset".

Leave a year cell blank where there is no observation; blanks are dropped, not
zero-filled. **A blank and a zero mean different things** — a zero is an
assertion that the value is zero.

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
| *anything else* | appended to `Description` as `'<Column>: <value>'` rather than dropped |

Cork's registry is `data/cork/cork_sources.csv`, which carries the extra columns
`Attachment`, `Format`, `Licensing`, `Geographic Coverage`, `Update Frequency`,
`Date Last Updated`, `Updated by` — all folded into the description.

The registry is passed to the uploader and the used subset is written into the
DVC dataset's `metadata['sources']` as a **list** of `{name, authority, url,
description}` dicts. It is a list, not a mapping keyed by name, deliberately: a
long human-readable name used as a YAML *key* can get line-wrapped by the writer
and produce YAML that will not read back. Names are safe as values.

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
python notebooks/upload_new_dataset.py \
  --input-csv data/<city>/<file>.csv \
  --output-dvc <namespace> \
  --instance <instance-id> \
  --language en \
  [-d <dataset_name>]      # omit to upload every Dataset group in the file
```

`-d` filters to one `Dataset` group. Afterwards, `--pull-datasets` advances the
YAML commit pointer to the new DVC commit.

For flat NZC-style files where every dimension combination lives in one file, use
`-d plain_csv_wide`, which sets `index_columns` to every column except `Value`,
`Unit` and `UUID`. See `data/cork/README.md` §"Why plain_csv_wide and not
plain_csv" for why including `Value` breaks the round trip.

## 5. Rules worth stating

- **Any unrecognised column becomes a dimension.** A typo in a metadata column
  name turns it into a dimension and the load fails on an unknown dimension —
  which is the good outcome. A typo in a *dimension* column's value fails on an
  unknown category, also good. Silent misreads come from column names that
  collide with reserved names by accident.
  **Dimensions are unchecked without --instance** However, if the upload_new_dataset command does not have `--instance` defined, dimensions and categories are uploaded as is, without checking anything. This approach is depreciated and useful only for a dataset that is never meant to be loaded to DB.
- **One unit per metric.** Convert before writing the CSV, not after.
- **Blank ≠ zero.** See §1.
- **Comments are the home for adopted-error documentation.** When a model must
  reproduce a source's mistake, the `Comment` cell is where the correct value and
  the reasoning go; see
  [`nzp-kpmg`-style error registers](matching-a-model-to-an-inventory.md#10-when-the-city-wants-the-inventorys-errors-reproduced).
- **Prefer a dataset over inline YAML `historical_values`.** Values with
  provenance belong in a dataset where the `Source`/`Comment` machinery applies
  and the admin UI can show them; `historical_values` in a config file carries
  none of that.
- **DB records shadow DVC.** If a `DBDatasetModel` record exists for a dataset
  id, it is loaded instead of the DVC version, so a DVC upload will not be
  visible until the DB record is refreshed or deleted. See
  `data/cork/README.md` §"Procedure for updating cork-nzc DB datasets".
