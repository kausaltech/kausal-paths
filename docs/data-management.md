# Data Management

This document describes the standard operating procedure for integrating
customer data into Kausal Paths models. Follow these steps whenever a city
or region provides new data files.

---

## Step 1 — Receive data from the customer

Accepted formats: Excel (`.xlsx`), CSV, JSON, Parquet.

Save the file(s) under `data/<city-id>/` (e.g. `data/bayreuth/`). Do not
transform or rename anything yet — keep the original file intact as the
source of record.

---

## Step 2 — Study the data with AI

Open a conversation and ask the AI to read the file and summarise:

- What sheets or sections exist and what each covers.
- Which years are present and whether values are historical measurements,
  targets, or projections.
- Which sectors, categories, or dimensions the data uses.
- Units and any unit conversions that will be needed.
- Whether the data is absolute values or relative (%, ratios, indices).

For Excel files, pay attention to:

- Which cells are **measured values** vs. **model-computed values** (formulas).
- Whether multiple scenarios or pathways exist in the same sheet.
- Embedded metadata rows (headers, units rows, totals) that must be skipped.

Record the relevant cell ranges for each dataset you intend to extract.

---

## Step 3 — Map data to the model

Read the customer's YAML config (e.g. `configs/bayreuth-bisko.yaml`) and
any included module (e.g. `configs/modules/bisko/model.yaml`) to identify:

- Which **dataset IDs** are referenced by which nodes
  (`input_datasets: - id: ...`).
- What **dimensions** each node expects (`input_dimensions`, `output_dimensions`).
- What **unit** each node works in (`unit:`).
- Whether the dataset is a direct replacement for a module default
  (`dataset_replacements`).

Cross-reference the customer data against this list. For each dataset the
model needs, determine:

- Which cells in the customer file provide that data.
- What dimension values the rows represent and how they map to the model's
  dimension category IDs.
- Whether units match or need conversion.

If the customer data covers a concept that has no matching node yet, note it
for Step 4.

---

## Step 4 — Design model structure (new models only)

If the customer has no model yet, draft a node graph based on the data:

- Group data by the natural sectors/categories the file uses.
- Identify which quantities are inputs (energy, activity) vs. outputs
  (emissions) and which are parameters (emission factors).
- Look for an existing module (`modules/bisko/`, `modules/nzc/`, etc.) that
  covers the methodology and can be included with dataset replacements.
- Sketch the required nodes, their types, units, and connections before
  touching any YAML.

---

## Step 5 — Inspect existing datasets

Before writing new data, check what is already in DVC for this customer.
Run `load_nodes.py` to list datasets in use:

```bash
./load_nodes.py -i <instance-id>
```

For YAML-backed instances, `--update-nodes` compiles the YAML directly into an
`InstanceExport` and synchronizes its node metadata into `NodeConfig`; it no
longer reads the values back from the already-overlaid runtime nodes. Existing
flags retain their meanings:

```bash
# Preview a structured, read-only JSON diff
./load_nodes.py -i <instance-id> --update-nodes --overwrite --dry-run

# Apply the same YAML-authored NodeConfig changes
./load_nodes.py -i <instance-id> --update-nodes --overwrite
```

Omit `--overwrite` to fill only missing fields. `--skip-descriptions` excludes
description updates, and `--delete-stale-nodes` includes nodes absent from the
YAML as deletions. This compatibility workflow leaves `config_source`
unchanged and does not synchronize dataset contents; it only maintains legacy
relations to DB datasets that already exist.

To inspect the structure of a specific existing dataset, use a Python
snippet:

```bash
python -c "
import dvc_pandas
repo = dvc_pandas.Repository(repo_url='https://github.com/kausaltech/dvctest.git', dvc_remote='kausal-s3')
ds = repo.load_dataset('<city>/<dataset-name>')
print(ds.df.head())
print(ds.meta)
"
```

Check:

- Column names (dimension IDs, metric column name).
- Index columns (dimension columns + `Year`).
- Units stored in the metadata.
- Whether a `Forecast` column is present.

This tells you the exact format a replacement dataset must match.

---

## Step 6 — Extract data to a standard CSV

Write a Python script `data/<city-id>/create_<topic>_csv.py` that reads the
source file and produces a CSV in the standard wide format.

### Standard CSV format

| Column | Description |
|--------|-------------|
| `Metric` | Always `Value` for single-metric datasets |
| `Unit` | Pint-compatible unit string, e.g. `t_co2e/a`, `MWh/a` |
| `Quantity` | Leave empty (`""`) unless needed |
| `Dataset` | Dataset name (becomes the DVC file name in snake_case) |
| *dimension columns* | One column per dimension (e.g. `sector`, `energy_carrier`). Values must be the model's category IDs. |
| *year columns* | One column per year (e.g. `2019`, `2020`, …). Empty cell = no data for that year. |

Multiple datasets can share one CSV file by using different values in the
`Dataset` column.

### Key rules

- **Units**: keep values in the unit you declare. Do not silently convert
  (e.g. do not divide t → kt and then label the column `t_co2e/a`).
- **Dimension values**: must match the model's category IDs exactly, not
  the human-readable labels from the customer file.
  Use `python -m tools.upload_new_dataset ... -n <instance-id>` (Step 7)
  to let the upload script validate and convert names to IDs automatically.
- **No Forecast column needed** in the CSV — it is derived at upload time
  via `forecast_from` (see Step 7 and Step 8).
- Prefer explicit, narrow cell-range extraction over reading whole sheets,
  so the script breaks loudly if the source file layout changes.

### Example extraction script structure

```python
import openpyxl, polars as pl
from pathlib import Path

XLSX = Path(__file__).parent / 'Customer File.xlsx'
OUT = Path(__file__).parent / 'topic.csv'

wb = openpyxl.load_workbook(XLSX, data_only=True)
ws = wb['Sheet name']
rows = list(ws.iter_rows(values_only=True))

# Extract ranges, build records, write CSV
# ...
df.write_csv(OUT, null_value='')
```

---

## Step 7 — Upload to DVC

Run `upload_new_dataset` from the repo root:

```bash
python -m tools.upload_new_dataset \
  --input-csv data/<city-id>/<topic>.csv \
  --output-dvc <city-id> \
  --language de \
  --instance <instance-id>
```

- `--output-dvc <city-id>` sets the DVC directory; dataset files will be
  created as `<city-id>/<dataset_name_in_snake_case>.parquet`.
- `--language` sets the metadata language for the dataset name.
- `--instance <instance-id>` loads the model context so the script can
  validate dimension values and convert category names to IDs.

The script will print the units it extracted and the number of rows per
dataset. Verify these before continuing.

After upload, **update the `commit:` field** in the instance YAML to the
new DVC repository HEAD commit so the model picks up the new data:

```yaml
dataset_repo:
  url: https://github.com/kausaltech/dvctest.git
  commit: <new-commit-hash>   # update this
  dvc_remote: kausal-s3
```

### Refreshing the DB row (usually required)

Uploading puts data in DVC; it does not make the model see it. Most instances
carry at least some **DB datasets that shadow their DVC counterpart** — where a
`DBDatasetModel` row exists for a dataset id, that row is loaded *instead of* the
DVC version. This is deliberate: it is what lets a dataset be edited in the admin
UI and what `use_datasets_from_db: true` selects. It also means a fresh upload
stays invisible until the DB row is refreshed from DVC:

```bash
# Diagnose first: what would change? Writes nothing.
python manage.py load_dvc_dataset <instance> <city>/<dataset-id> --plan

# Apply. Refreshes in place -- same pk and UUID -- so DatasetPort,
# NodeDataset and revision-pin references stay valid.
python manage.py load_dvc_dataset <instance> <city>/<dataset-id> --force
```

**Refresh one dataset at a time.** Naming the dataset explicitly is the normal
workflow, not a workaround: the plan output is only readable per dataset, and
`--all` deliberately finds nothing once the ids have DB rows (they load as
`DBDataset`, so `get_all_dvc_dataset_ids()` is empty) — the command says so
rather than silently doing nothing.

Provenance also arrives at this step, not at upload: `Source` and `Comment`
cells become `DataSource` links and `DataPointComment` records only when
`load_dvc_dataset` runs. Before that they are in the parquet but invisible in
the admin UI, which reads as "the comments were lost".

Mechanics in full — which commit gets imported, what stops a run, `--recreate` —
are in [`trailhead/tools.md`](trailhead/tools.md#load_dvc_dataset).

---

## Step 8 — Wire datasets to nodes and verify

Add or update the relevant node definitions in the instance YAML. For each
new dataset, check:

**`column: Value`** — required when the metric was uploaded as `Metric=Value`.
Without it the node cannot find the unit in the DataFrame metadata.

```yaml
input_datasets:
- id: <city>/<dataset-name>
  column: Value
```

**`forecast_from`** — required when the parquet does not contain a `Forecast`
column (which is the normal case for data extracted from customer files).
Set it to the first year that should be treated as a forecast. For datasets
that are purely historical, set it to one year after the last data year.

```yaml
input_datasets:
- id: <city>/<dataset-name>
  column: Value
  forecast_from: 2025   # years < 2025 → Forecast=False
```

**Unit compatibility** — the node's `unit:` and the dataset's unit must be
compatible (pint will convert automatically, e.g. `t` → `kt`). If they are
dimensionally incompatible the node raises an `ensure_unit` error.

### Verification

Run the node directly and check that output is printed without errors:

```bash
./load_nodes.py -i <instance-id> --node <node-id>
```

For actions, also run the outcome node to confirm the full pipeline works:

```bash
./load_nodes.py -i <instance-id> --node net_emissions
```

Common errors and their causes:

| Error | Cause |
|-------|-------|
| `KeyError: 'Value'` in `ensure_unit` | Missing `column: Value` in the input_dataset definition |
| `Forecast column missing` | Missing `forecast_from` in the input_dataset definition |
| `Series with type X is not compatible with Y` | Dimensionally incompatible units between node and dataset |
| `Input dataset has duplicate index rows` | Two rows with the same (Year, dimension) combination in the parquet |
| `No input datasets, but node requires one` | Dataset ID typo or wrong `dataset_replacements` mapping |

---

## Step 9 — When the model and its data disagree

A DB dataset row is a *copy* of the DVC data taken at a point in time. The
model moves on — categories get renamed, metrics get renamed, columns get
added — and the copy does not follow. Bumping the `commit:` pin does **not**
move it. Everything in this section follows from that one fact.

### Telling a stale spec from a stale dataset

These fail differently and are fixed differently. Diagnose which one you have
before touching anything.

| Symptom | What is stale | Fix |
|---------|---------------|-----|
| Unknown node class, e.g. `ChpAction` | The **spec** — a node type the deployed code no longer defines | `sync_instance_to_db <instance>` |
| `Unknown categories in dimension column 'X': <cat>` | The **dataset** — the row stores a category id the dimension has since renamed or merged | Refresh the row (below) |
| `Nothing left after filter_dimension` | The **dataset** — an edge filters to a category the row no longer contains | Refresh the row (below) |
| `No metric <column> in dataset <id>` (from sync) | The **dataset** — DVC metric ids changed, the row still has the old ones | Refresh the row (below) |
| `Dimension <x> not found`, loading from the database | The **spec** — emptied by a `database` → `yaml` flip (below) | `sync_instance_to_db <instance>` |

Three traps worth knowing before you reach for them:

- **`sync_instance_to_db` does not fix stale data.** It rewrites the spec and
  the port bindings; it never touches dataset rows.
- **Switching to `config_source: yaml` does not bypass a stale dataset.**
  `use_datasets_from_db: true` applies regardless of config source, so the
  runtime still reads the DB row. Switching to YAML is the escape hatch for a
  broken *spec* only:

  ```bash
  python -m tools.debug_instance -i <instance> --source yaml --save   # then --source db --save to revert
  ```

- **Flipping to `yaml` empties the stored spec, and flipping back does not refill it.**
  This is the trap the row above names. While `config_source` is `yaml`,
  `ensure_spec()` rebuilds `InstanceConfig.spec` from the config file with
  `make_minimal_instance_spec()` and saves it, whenever the YAML's mtime hash
  changes — which a deploy guarantees. *Minimal* means identity, params,
  scenarios and pages, but **no dimension catalogue**: the YAML runtime reads
  dimensions from the config file and never consults the spec, so it does not
  need one. The previous, sync-generated spec is overwritten in the process.

  Nothing breaks while the instance stays yaml-sourced. The moment anything
  loads it through the *database* path, the snapshot gets a spec with an empty
  dimension list and the load dies on the first node that declares one:

  ```
  NodeError: Node net_emissions: Dimension sector not found
  ```

  `build_instance_snapshot` now refuses this up front and names the fix instead
  of letting it surface as a missing dimension. Two ways to meet it:

  - **`sync_instance_to_db <instance>` after flipping back to `database`.** Not
    optional — without it the live site gets the error, not just your terminal.
  - **`debug_instance --source db` against a yaml-sourced instance.** The tool
    forces the database path when you ask for it explicitly, and warns. Without
    `--source` it now follows the instance's own `config_source`, which is what
    you want almost always.

### Confirming a row is stale

Compare the DB row against the DVC data at the pin. Metric names and data
point counts are the quickest tell:

```bash
python manage.py shell_plus --quiet-load -c "
from kausal_common.datasets.models import Dataset, DatasetMetric
for d in Dataset.objects.filter(identifier='<city>/<dataset-id>'):
    ms = list(DatasetMetric.objects.filter(schema=d.schema).values_list('name', flat=True))
    print(f'{str(d.scope):40} {ms} {d.data_points.count()}')
"
```

Rows are scoped per instance, so every city that uses the dataset gets its own
copy and they go stale independently. Then check what the model actually sees:

```bash
python -m tools.debug_instance -i <instance> --source db -c "
node = ctx.get_node('<node-id>')
for ds in node.input_dataset_instances:
    df = ds.get_copy()
    print(ds.id, type(ds).__name__, len(df), {d: sorted(df[d].unique().to_list()) for d in df.dim_ids})
"
```

`SerializedDBDataset` in the output confirms the data came from the DB, not DVC.

### Refreshing the row

Start with `--plan`; it writes nothing and names the blockers:

```bash
python manage.py load_dvc_dataset <instance> <city>/<dataset-id> --plan
```

A renamed metric produces this:

```
metrics drop Value (bound by 1 dataset port(s))
blocker   metric 'Value' would be dropped but 1 dataset port(s) still bind it
```

That is a deadlock by construction: the import refuses because a port binds the
old metric, and the sync keeps binding the old metric because it resolves
against the DB schema, which still has it. Break it by deleting the binding
first — `DatasetPort` rows are derived state that `sync_instance_to_db`
deletes and rebuilds wholesale, so removing them costs nothing:

```bash
python manage.py shell_plus --quiet-load -c "
from nodes.models import InstanceConfig, DatasetPort
from nodes.input_bindings import sync_input_bindings
ic = InstanceConfig.objects.get(identifier='<instance>')
print(DatasetPort.objects.filter(instance=ic, dataset__identifier='<city>/<dataset-id>').delete())
print('mirror resync:', sync_input_bindings(ic))
"
```

**Do not skip `sync_input_bindings`.** `NodeInputPortBinding` is a derived
mirror of `NodeEdge` + `DatasetPort`, normally refreshed by a hook at the
transaction boundary. A raw queryset `.delete()` bypasses that hook, leaving an
orphaned mirror row that still protects the metric:

```
ProtectedError: ... referenced through protected foreign keys: 'NodeInputPortBinding.metric'
```

Then finish. The instance is missing bindings between the delete and the sync,
so run it straight through:

```bash
python manage.py load_dvc_dataset <instance> <city>/<dataset-id> --force
python manage.py sync_instance_to_db <instance>
```

Never use `--recreate` for this. It mints a new UUID and orphans the dataset
references held by published instance revisions.

### When a DVC fetch hangs

A run that stops at `Fetching`, right after a few
`aiobotocore.credentials … Found credentials` lines, is stuck in DVC's S3
transfer. `Found credentials` only means a file was read — it says nothing
about whether the transfer works.

Two properties make this worse than it looks: the stuck call is inside a C
extension, so **Ctrl-C does nothing**, and `dvc_pandas` holds a `filelock`
acquired with no timeout, so **every later process that touches the DVC repo
blocks silently and forever** — other management commands and the web workers
alike. A stuck command can therefore degrade the running site, not just your
terminal.

Recover from a second shell:

```bash
D=$(python -c "from dvc_pandas.utils import cache_dir_for_url; print(cache_dir_for_url('https://github.com/kausaltech/dvctest.git'))")

pgrep -af "manage.py|load_nodes"     # find the holder
kill -9 <pid>                        # SIGTERM will not land either
```

A leftover `.dvc-pandas.lock` file is **not** itself a lock — the kernel drops
the `fcntl` lock when the process dies, so the file is inert. Only remove lock
files when a `kill -9` left DVC's own state behind and the next run complains:

```bash
pgrep -af "manage.py|load_nodes"     # must be empty first
rm -f $D/.dvc-pandas.lock $D/.dvc/tmp/lock $D/.dvc/tmp/rwlock $D/.dvc/tmp/rwlock.lock
```

Then sidestep the fetch entirely by populating the cache by hand. The cache is
content-addressed at `files/md5/<first 2>/<remaining 38>`, and dvc_pandas only
checks that the path exists — so a correct file at the correct path is enough.
The `datasets` bucket is anonymously readable over HTTPS, which avoids the S3
path that wedges:

```bash
REV=<pin from the instance YAML>
cd $D && git fetch origin
for f in $(git ls-tree -r --name-only $REV | grep '^<city>/.*\.parquet\.dvc$'); do
  H=$(git show $REV:$f | awk '/md5:/{print $NF; exit}')
  P=$D/.dvc/cache/files/md5/${H:0:2}/${H:2}
  [ -f "$P" ] && continue
  mkdir -p "$(dirname $P)"
  curl -sSfL --max-time 120 -o "$P" "https://s3.kausal.tech/datasets/files/md5/${H:0:2}/${H:2}" \
    && chmod 444 "$P" && echo "fetched $f" || echo "FAILED $f"
done
```

With every object present the fetch branch is skipped and the command runs
normally. `dvc fetch -r public --jobs 4 <city>` does the same via the CLI —
`fetch`, not `pull`, so it fills the cache without writing the working tree.

To check whether the S3 path itself is healthy, without involving DVC:

```bash
python - <<'EOF'
import time, s3fs
t = time.time()
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'https://s3.kausal.tech'})
print('OK', fs.info('datasets/files/md5/<2>/<38>')['size'], round(time.time()-t, 2), 's')
EOF
```

Missing credentials raise `NoCredentialsError` immediately — a hang here means
something else.

### Avoiding all of this

- **Treat a pin bump as two steps, not one.** Update `commit:`, then refresh
  the DB row of every dataset whose data or metadata changed. A renamed
  category or metric that only lands in DVC is a crash waiting for the next
  deploy.
- **Rename categories with aliases, not replacements.** Adding the retired id
  to the surviving category's `aliases:` lets old rows keep resolving. Only
  drop the alias once every environment's rows have been refreshed.
- **Test from a cold cache before deploying.** Evict one object locally and
  run the sync; it exercises the fetch path that only ever fails on a cold
  cache, which is exactly the state a freshly deployed pod is in:

  ```bash
  rm -f $D/.dvc/cache/files/md5/<2>/<38>
  python manage.py sync_instance_to_db <instance> --dry-run
  ```

- **Pre-flight the target environment.** Before deploying, check the dataset
  rows (metric names and point counts), the S3 path, and whether
  `$D` exists. Three read-only commands turn a deploy from discovery into
  execution.
- **Give the DVC cache a persistent volume.** If it lives on the container's
  writable layer it is wiped on every restart, so each new pod re-clones and
  re-fetches, and any deploy opens a window where that first cold fetch can
  wedge. Note also that `Repository.__init__` clones *before* it creates the
  lock, so a cold pod can have a management command and a web worker cloning
  into the same directory concurrently, unguarded.
- **Never run a sync or import in production without a second shell open.**
