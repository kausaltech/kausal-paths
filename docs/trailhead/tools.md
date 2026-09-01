# Trailhead Migration: Tools & Tips

## debug_instance.py

The main tool for investigating DB-backed vs YAML-backed model
instances. Lives at `tools/debug_instance.py`.

**Invoke it as a module, not as a script** — `python -m tools.debug_instance`,
never `python tools/debug_instance.py`. The script form puts `tools/` on
`sys.path` instead of the repo root, so the `kausal_common` / `nodes` imports
only resolve where the repo happens to be installed editable (which is what
`mise prepare` gives you locally). On a deployment without that editable
install it fails at import. The `-m` form keeps the working directory on the
path and works in both places.

### Diff a node's spec between YAML and DB

The most useful operation — parses the current YAML into a snapshot and
diffs the node's `NodeSpec` against the stored `NodeConfig.spec`:

```bash
python -m tools.debug_instance -i espoo --diff-node building_type_index
```

A diff means the DB mirror is stale (re-sync) or the parse changed.

### Which source it loads from

With no `--source`, the instance is loaded the way the site loads it — from its
own `config_source`. Pass `--source yaml` or `--source db` to override.

Overriding to `db` on a yaml-sourced instance warns, because its stored spec is
usually the minimal one `ensure_spec()` derives from the YAML, which carries no
dimensions; see [`data-management.md`](../data-management.md) §Step 9. Give it a
full spec with `sync_instance_to_db` first if you actually want that path.

### Switch an instance between YAML and DB sources

```bash
# Switch to YAML (useful when DB spec is stale or broken)
python -m tools.debug_instance -i budget --source yaml --save

# Switch back to DB
python -m tools.debug_instance -i budget --source db --save
```

### Evaluate Python with instance/ctx/node in scope

```bash
# List input datasets for all nodes
python -m tools.debug_instance -i espoo --source db -c "
    for n in ctx.nodes.values():
        if not n.input_dataset_instances:
            continue
        print(f'{n.id}: {[ds.id for ds in n.input_dataset_instances]}')
"

# Inspect a specific node's output metrics
python -m tools.debug_instance -i budget --source yaml -c "
    node = ctx.get_node('building_renovations')
    for k, m in node.output_metrics.items():
        print(f'key={k!r}, column_id={m.column_id!r}, unit={m.unit}')
"
```

### Compute a node from a specific source

```bash
python -m tools.debug_instance -i espoo --source db --node net_emissions
```


## load_dvc_dataset

Imports a DVC dataset into the DB as a `Dataset` row. Two phases:
diagnose, then apply.

```bash
# Phase 1 -- what exists, and what would change? Writes nothing.
python manage.py load_dvc_dataset espoo espoo/buildings --plan

# Phase 2 -- apply. Existing rows are refreshed in place (same pk and UUID),
# so DatasetPort / NodeDataset / revision-pin references stay valid.
python manage.py load_dvc_dataset espoo espoo/buildings --force
```

The plan reports the row's pk/UUID, the commit its data came from versus the
one about to be imported, the data-point count on both sides, and which
metrics are kept, added or dropped.

### Which commit gets imported

Every run prints the repository and commit it is reading, and where that pin
came from. `--repo-from` chooses:

- `auto` (default) — the instance's declared `config_source`. For a
  DB-sourced instance that is the DB spec's pin, which lags the YAML until
  `sync_instance_to_db` runs.
- `yaml` — the pin in `configs/<instance>.yaml`.
- `db` — the pin in the stored `InstanceConfig.spec`.

When the two pins disagree the command warns and names the other one. This
matters because importing from the wrong commit is otherwise invisible: it
surfaces much later as `No metric <column> in dataset <id>` from
`sync_instance_to_db`, which reads the YAML regardless of `config_source`.

### Things that will stop a run

- **A dropped metric that model bindings still hold.** Refused up front, with
  the count, rather than leaving a node bound to input that no longer arrives.
  Both `DatasetPort.metric` and `NodeInputPortBinding.metric` are counted —
  each is a `PROTECT`ed reference, and counting only one turns a clean refusal
  into a `ProtectedError` halfway through the sync. Fix it with
  `rename_dataset_metrics` (below), which is usually what the upstream change
  actually was.
- **`--all` with `use_datasets_from_db`.** Any identifier that already has a
  DB row loads as a `DBDataset`, so `ctx.get_all_dvc_dataset_ids()` is empty
  and `--all` has nothing to do. Name the datasets explicitly, or ask
  `dataset_status` (below) which ones need naming.
- **An index column no dimension resolves.** `sync_dimensions` looks every index
  column up in `ctx.dimensions`, so a dataset carrying a raw key column dies with
  a bare `KeyError: '<column>'` partway through. That is not a fault to fix in the
  dataset: some datasets keep a raw key on purpose — `mainz/municipal_building_energy`
  and `mainz/municipal_water_use` carry `we_from`/`we_to` so a reader can re-derive
  the property-group assignment instead of inheriting it, and the BISKO transport
  datasets carry `district`/`ags`. **These are DVC-only by design**: they stay
  external placeholders, the model reads them straight from DVC, and they must be
  left out of a `load_dvc_dataset` list rather than forced in. A city user cannot
  edit them in the admin either, which is part of the same trade.

  Each dataset syncs inside its own transaction, so a failure here rolls that one
  back and leaves the datasets already processed committed — re-run without the
  offender rather than starting over.

`--recreate` restores the old delete-and-rebuild behaviour. It mints a new
UUID, which orphans the dataset references held by published instance
revisions, and it fails outright when anything references the row under
`PROTECT` — use it only when you want a genuinely fresh row.


## dataset_status

Answers "what do I still have to load here, and in what order?" — the question
`load_dvc_dataset --all` cannot answer for a DB-sourced instance, because every
imported identifier loads as a `DBDataset` and disappears from the DVC list.
This command enumerates from the DB rows instead, so nothing is invisible.

```bash
python manage.py dataset_status mainz-bisko
python manage.py dataset_status bisko mainz-bisko augsburg-bisko --stale-only
```

Each dataset gets a verdict: `current`, `import`, `rename first` (a metric was
renamed upstream and bindings still hold the old name), `new` (declared but
never imported), `unreadable` (claims a DVC source that will not read) or
`db only` (authored in the admin, so nothing to import). It then prints the
commands to run in the order they have to happen — rename, import, sync.

It deliberately lists rows that carry no `external_ref`. Those were imported
before provenance stamping existed, and they are the ones most likely to be
silently stale: 15 of `mainz-bisko`'s 32 rows are in that state, including the
one that was blocking its update.


## dataset_inventory

Answers the question *before* `dataset_status`: not "what do I have to run?" but
**what exists, and where?** Per dataset it prints the data-point count and last
write date on *both* the database and DVC sides, the drift between them, and the
commit recorded in `external_ref`.

```bash
python manage.py dataset_inventory mainz-bisko
python manage.py dataset_inventory mainz-bisko --repo-from yaml --order drift
python manage.py dataset_inventory mainz-bisko --csv /tmp/datasets.csv
```

**Use it in place of `load_nodes.py`'s dataset listing.** That listing enumerates
the datasets the *runtime* resolved, so on an instance with
`use_datasets_from_db` every imported identifier loads as a `DBDataset` and never
appears — the listing looks short and reassuring while most of the model is
missing from it. This command enumerates from the DB rows and the declared set,
so nothing is invisible.

The two point counts are comparable: a `DataPoint` is one (metric, year,
dimension) cell, and the DVC count is non-null metric cells. `--order drift` puts
the mismatches first.

Three verdicts in the `where` column, and the distinction that matters:

- `both` — populated on both sides.
- `db only` — the DB has data the pinned DVC copy does not.
- `dvc only` — the reverse. **A `0` in `db pts` against a `—` separates an empty
  row from no row at all**: `load_dvc_dataset` creates the row and schema before
  it can fail to fill them, and a dataset indexed by columns the instance has no
  dimension for (`we_from`, `ags`) leaves exactly that — a row, a schema,
  bindings, and nothing in them.

The dates are different kinds of fact and the headings say so. `db written` is
the newest `last_modified_at` over the row's data points — when this database was
last written, not when the data was collected. `dvc committed` is the last commit
**reachable from the pin** that touched `<identifier>.parquet.dvc`, so a dataset
pushed after the pin is reported as the pin sees it, which is the state the model
computes on.


## delete_dataset

Deletes dataset rows and their data points from one instance. There was no
command for this and it was done from `shell_plus`, which is the wrong place: a
bare `Dataset.objects.filter(identifier=...).delete()` takes every city's row of
that name, leaves the schema behind as an orphan, and either raises
`ProtectedError` halfway through or silently removes a dataset the model still
binds.

```bash
python manage.py delete_dataset mainz-bisko bisko/energy_costs                   # plan
python manage.py delete_dataset mainz-bisko a b c --apply --dump-to /tmp/backup
```

Nothing is written without `--apply`, and **`--apply` refuses until you choose
`--dump-to DIR` or `--no-dump`**. A delete destroys data points, and for a row
with no DVC provenance there is no other copy anywhere — the plan says which rows
those are, because it changes what a mistake costs. The dump is a CSV per
dataset: dimensions as columns, comments and data-point UUIDs preserved.

A refusal stops the **whole set**, as in `rename_dataset`. What it refuses:

- **ambiguous** — the identifier resolves to more than one row visible here.
  Identifiers repeat across cities; never guess.
- **foreign scope** — the row's scope is not the named instance. Scope is a
  generic FK, so a framework-scoped row is merely *visible* here and deleting it
  would remove it from every holder.
- **published revision** — an `InstanceRevisionDatasetPin` names it. No override:
  the pin records what a published revision computed from.
- **model still binds it** — a `NodeInputPortBinding`, `DatasetPort` or
  `NodeDataset` points at the row. `--clear-bindings` overrides.
- **still declared in `configs/`** — because a deleted-but-declared dataset comes
  back on the next sync, empty, and an empty row the model binds is the failure
  mode in *A dataset can be current in DVC and absent from the model*.
  `--ignore-configs` overrides, which is right when the hit is another city's
  config or a module this instance does not include.

**The binding check is the one that cannot be replaced by inspecting the graph**,
and `dataset_replacements` is why. The loaded dataset object carries the
*module's* declared id (`kommune/fahrleistung_strassenverkehr`) while the binding
points at the row the replacement resolved to
(`mainz/fahrleistung_strassenverkehr`), so scanning node dataset ids reports such
a row as unused while three bindings hold it. The FK tables are the only signal
that survives a replacement.

The config scan separates **declarations from comment mentions**: a YAML comment
naming an identifier is not a declaration, and a comment explaining why an entry
was removed would otherwise refuse the delete on the strength of its own
documentation. Both are reported; only declarations block. Split rather than
stripped, because stripping everything after a `#` would under-count an
identifier inside a quoted string, and under-counting is the unsafe direction.

The schema goes with the dataset when it serves only that dataset, and is kept
when shared — `Dataset.schema` is `PROTECT`, so otherwise it survives as an
orphan carrying its metrics and dimensions. Data points, their comments and their
source references all `CASCADE`. `invalidate_cache()` runs at the end.


## rename_dataset_metrics

When DVC data comes back with a metric column under a new name, the import
wants to drop the old metric and add the new one — and cannot, because model
bindings hold the old one under `PROTECT`. Renaming the metric row in place is
better than clearing the bindings and re-syncing: the bindings keep pointing at
the same row, the metric UUID survives for anything that pinned it, and the
import then sees the metric as *kept*, so the conflict never arises.

```bash
# plan only; the mapping is inferred from the DVC data when unambiguous
python manage.py rename_dataset_metrics mainz-bisko bisko/weather_correction

python manage.py rename_dataset_metrics mainz-bisko --all
python manage.py rename_dataset_metrics mainz-bisko bisko/energy_shares --rename Value=default --apply
```

Nothing is written without `--apply`. It refuses rather than guess when two
columns were renamed at once (say which with `--rename`), when the target name
already exists, or when the schema is shared with other datasets and the rename
would silently affect them too.


## rename_dataset

Renames a dataset *identifier* in place, across every scope that holds it. The sibling
`rename_dataset_metrics` renames a metric column *within* a dataset; this renames the
dataset itself.

```bash
# Plan only -- writes nothing
python manage.py rename_dataset bisko/final_energy kommune/endenergieverbrauch

python manage.py rename_dataset bisko/final_energy kommune/endenergieverbrauch --apply

# A whole namespace at once, as one transaction
python manage.py rename_dataset --from-file data/bisko/renames.yaml --apply
```

The mapping file is a flat `old: new` YAML document:

```yaml
bisko/final_energy: kommune/endenergieverbrauch
bisko/energy_shares: de/energieanteile_verkehr
```

### Why in place

The model graph references datasets **by row, not by name**: `NodeInputPortBinding.dataset`
and `DatasetPort.dataset` are foreign keys, no binding's `dataset_spec` embeds an
identifier, and node and instance specs carry none either. So an in-place rename keeps
every binding, every pinned UUID and every published revision intact, and the graph cannot
observe it at all. Deleting and re-importing under the new name would mint a new UUID and
orphan all of that.

`Dataset.external_ref['dataset_id']` moves with the identifier when it names the old one,
so the DVC provenance stamp does not go on claiming a path that no longer exists. The
commit in that stamp is provenance about the *data* and is left alone.

Two things deliberately do not move:

- **`DatasetSchema.name`**, the human-readable label, unless an entry names it. Labels are
  worth setting during the rename, because today they are inconsistent (`Endenergie`,
  `Energy shares`, `weather_correction`) and monolingual. A mapping entry is then a table
  rather than a bare identifier:

  ```yaml
  bisko/endenergie_emissionsfaktoren:
    to: de/emissionsfaktoren_endenergie
    name_de: Emissionsfaktoren Endenergie
    name_en: End energy emission factors
  ```

  The label goes on every row of the rename, because the schema is one-to-one with the
  dataset and each city's row is the same logical dataset. Storage follows `modeltrans`:
  the value for `settings.LANGUAGE_CODE` goes in the `name` column, the rest into `i18n`.
  An entry naming labels **must** include the default language, or the column keeps a stale
  value while the translations move on — which is how the dimension categories came to hold
  German in a column that is read as English. `--set-name` does the single-language case
  from the command line.

  **This command is the only way to set a dataset label on a non-English instance.** The
  Wagtail admin form binds the plain `name` field, so it edits the value for
  `settings.LANGUAGE_CODE` — English — whatever the editor's UI language is set to. The
  failure is quiet, and looks like success half the time: editing a dataset that has no
  translated name appears to work, because the other language falls back to the column and
  displays what was typed, while the text is stored under the wrong language; editing one
  that *does* have a translated name appears to do nothing, because the column moves and the
  translation stays put. Both were observed on `bisko` within the same hour on 2026-08-24.

  A label-only run is supported: give an entry whose `to` is its own identifier.
  `build_rename_plan` blocks a same-identifier run only when no labels are supplied, and
  nothing is renamed, so the "Still referenced in configs" warning does not apply and no
  `sync_instance_to_db` is needed afterwards.
- **`InstanceRevisionDatasetPin.identifier`**, a denormalized record of what the dataset
  was called when that revision was published. The pin's identity is its foreign key and
  `dataset_uuid`, and nothing resolves a pin by identifier, so rewriting it would only
  falsify the manifest. Pins are reported (`pin:N`) and left alone.

### Things that will stop a run

- **The target name already taken in the same scope.** The constraint is
  `unique_identifier_per_dataset_scope`, so a clash is only a clash *within* one scope --
  two cities may legitimately both hold the new name, and that is not refused.
- **An identifier no dataset carries**, which is almost always a typo or an already-applied
  rename. `--allow-missing` downgrades it to a no-op.
- **An invalid target.** Identifiers are `namespace/name` in `[a-z0-9_-]`, so German names
  must be transliterated: `de/fernwaerme`, never `de/fernwärme`.

Any refusal stops the **whole** set — a half-renamed namespace is harder to reason about
than one that has not moved.

### The rename must precede any sync of a database-sourced instance

If `sync_instance_to_db` runs on a database-sourced instance whose deployed config already
names the *new* identifiers, it creates a row for each of them. The scope then holds two
rows per dataset — the original with the data, and a new empty one — and because the spec
named the new one, the **bindings move to the empty row**. The instance keeps computing, so
nothing announces the problem; it just computes from nothing.

This happened to `bisko` in production on 2026-08-20: 28 duplicate rows, 11 of them
shadowing real data including `bisko/final_energy` with 3783 data points.

The recovery is `--replace-empty-target`, which deletes a target row that holds no data —
clearing the `PROTECT`ed `NodeInputPortBinding`, `DatasetPort` and `NodeDataset` rows first —
and renames the real row into its place, keeping its pk, UUID and data:

```bash
python manage.py rename_dataset --from-file renames.yaml --allow-missing --replace-empty-target
python manage.py rename_dataset --from-file renames.yaml --allow-missing --replace-empty-target --apply
python manage.py sync_instance_to_db <instance>   # rebuilds the bindings that were cleared
```

The final sync is not optional: the deletion drops bindings, and `reconcile_input_bindings`
(`nodes/spec_sync.py`) rebuilds the set from the spec, now resolving to the renamed rows.

It refuses rather than guesses in two cases: a target that holds data (merging two populated
datasets is not this command's business), and a target pinned by a published revision (the
pin records what that revision used, so deleting it would falsify history).

To verify afterwards, check that the **original pks** carry the data under the new names — a
row whose pk is in the range the sync minted is the placeholder, not the real dataset.

### Order: the DVC copy comes first

The database is one of three sides, and the order is forced rather than a preference:

1. **Copy** the DVC paths, leaving the old ones in place, and bump the pins
   (`data/bisko/copy_dataset_paths.py --push` does the copy, reading the same mapping file).
2. `rename_dataset --from-file … --apply`.
3. Update `configs/`, deploy.
4. Delete the old DVC paths — the only irreversible step, and only once 1–3 are verified.

Step 1 must precede step 2 because a `Dataset` row that is an external placeholder carries
its DVC path in `external_ref['dataset_id']`, and this command restamps it: after step 2
those rows read the *new* path, so it has to exist. And step 4 must follow step 3 because a
yaml-sourced instance resolves datasets by identifier — `augsburg-bisko` resolves all 32 of
its datasets from DVC and holds no DB rows at all, so it would stop computing the moment the
old paths vanished from under a config that still named them.

Copying rather than moving is what removes the window: while both paths resolve, the three
sides do not have to land together, and everything up to step 4 can be reverted by reverting
the config.

The command finishes by listing the config files that still name the old identifier. After
step 3, re-run `sync_instance_to_db` for database-sourced instances and `dataset_status` to
confirm nothing went stale.

## sync_instance_to_db

Exports runtime node specs from YAML-loaded instances into the DB.

```bash
# Sync a single instance
python manage.py sync_instance_to_db espoo

# Sync all non-framework instances
python manage.py sync_instance_to_db --all

# Dry run (shows summary without writing)
python manage.py sync_instance_to_db espoo --dry-run
```

After changing spec models (adding/removing fields, changing
serialization), all DB-sourced instances need re-syncing. The typical
workflow:

1. Make the schema change
2. `python manage.py sync_instance_to_db --all`
3. Verify with `test_instance`


## test_instance

Validates that instances can initialize and compute correctly from
their current config source (YAML or DB).

```bash
# Dry run (no state comparison, just init + compute)
python manage.py test_instance --state-dir model-outputs/ --dry-run

# Start from a specific instance, if previous run was interrupted. Tolerate some failures.
... test_instance ... --start-from longmont --maxfail 5

# Spec-only mode (only tests initialization, not computation)
... test_instance ... --spec-only
```


## copy_instance

Copies a whole instance — its model (spec/nodes/edges/datasets), its
Wagtail page tree (including draft revisions and `InstanceSiteContent`),
and a new `Site` — under a new identifier. Node references in copied
pages (the `OutcomePage.outcome_node` FK and `NodeChooserBlock` PKs in
StreamField bodies) are repointed from the source's `NodeConfig` rows to
the copy's, on both the live rows and every copied revision.

```bash
# Auto mode (default): follows the source's config_source
python manage.py copy_instance zuerich zuerich-copy \
    --site-url https://zuerich-copy.paths.kausal.dev/

# Dry run: does everything in a transaction, then rolls back
#   (and removes any YAML file it wrote)
python manage.py copy_instance zuerich zuerich-copy --dry-run
```

### Choosing a representation: `--mode {auto,db,yaml}`

- **`db`** — `export_instance` → `import_instance` into a fresh
  `config_source='database'` InstanceConfig. A self-contained snapshot of
  the source's *current DB state*, including admin/UI edits and
  DB-resident datasets. Use for instances that are already
  database-backed.
- **`yaml`** — copies `configs/<src>.yaml` → `configs/<dst>.yaml`
  (rewriting only the instance `id` / `name*`), creates a
  `config_source='yaml'` InstanceConfig, then materialises its `NodeConfig`
  rows — and the editor graph (`NodeEdge` / `DatasetPort`) — from the
  source's DB snapshot, so admin-authored fields the YAML can't express are
  carried over (without flipping `config_source` to `database`). When the
  source has no DB spec to snapshot, it falls back to `sync_nodes()`. These
  DB rows come from the source's mirror (its last `sync_instance_to_db`),
  which can lag the YAML — the command warns about this, and the copied YAML
  (not the mirror) governs the runtime. Preserves full YAML fidelity
  for instances whose features aren't yet fully expressible in the DB
  spec. Per-instance `include` fragments (the `configs/<src>/*.yaml`
  node-group files) are **copied** into `configs/<dst>/` and the
  `include[].file` paths in the new top-level config are repointed there,
  so the copy owns its model source; `include`s that don't live under
  `configs/<src>/` (shared library fragments) are left shared. DVC datasets
  ride along by reference (the dataset paths in the YAML — including the
  `<src>/…` ids inside the copied fragments — are deliberately left pointing
  at the shared repo); only DB-resident (admin-edited) datasets are copied
  into the DB.
- **`auto`** (default) — picks `yaml` if the source is YAML-backed,
  otherwise `db`.

### Things to know

- `--name "Foo"` sets both the DB row name and every YAML `name*` field to
  that one value, so the model/runtime name and the DB/page title match
  exactly. Without `--name`, each name independently gets ` (copy)`
  appended (per-language YAML names are preserved), so the primary title
  matches only insofar as the source's DB name already agreed with its YAML.
- yaml mode writes files under `configs/` — the top-level `configs/<dst>.yaml`
  plus the copied `configs/<dst>/` include fragments — a side effect *outside*
  the DB transaction. They are removed again on `--dry-run` or on failure.
  Commit them afterwards if you want to keep the copy.
- Framework-backed instances can't be yaml-copied (their YAML is the
  shared framework file) — use `--mode db`.
- After remapping, the command scans copied pages/revisions for leftover
  source-node references and **fails** if any remain (e.g. a node that
  wasn't materialised in the copy). Pass `--allow-dangling-refs` to
  downgrade that to a warning.
- `--no-pages` skips **all** Wagtail content (the page tree, the `Site`, and
  `InstanceSiteContent`); `--sync-source` (db mode) refreshes the source's DB
  mirror from YAML first (this mutates the source mirror and is not reverted).
- A copy is fully reversible: `InstanceConfig.delete()` removes its nodes,
  datasets, pages and Site (delete the `configs/<dst>.yaml` file and the
  `configs/<dst>/` fragment directory too for a yaml-mode copy).

### Copying a yaml-backed instance into production

A yaml-backed copy needs its `configs/<dst>.yaml` committed to the repo and
deployed — production filesystems are immutable/ephemeral, so the command
can't write it there. Split the operation across the two environments with
`--write-config-only` (write the file, no DB changes) and
`--use-existing-yaml` (apply the DB side from an already-committed file,
without rewriting it):

```bash
# 1. Locally: write the config only (no DB changes), then review + commit it.
#    This also copies the include fragments into configs/zuerich-copy/.
python manage.py copy_instance zuerich zuerich-copy --mode yaml \
    --write-config-only --site-url https://zuerich-copy.paths.kausal.dev/
git add configs/zuerich-copy.yaml configs/zuerich-copy/
git commit -m "Add zuerich-copy instance"

# 2. Deploy so configs/zuerich-copy.yaml (and configs/zuerich-copy/) are present
#    in production.

# 3. In production: apply the DB side from the committed config.
python manage.py copy_instance zuerich zuerich-copy --mode yaml \
    --use-existing-yaml --site-url https://zuerich-copy.paths.kausal.dev/
```

Notes:
- In the `--use-existing-yaml` (production) stage, `--site-url` and `--name`
  default to the values in the committed `configs/<dst>.yaml` when omitted, so
  the DB row and routing can't silently disagree with the deployed file. Pass
  them explicitly only to override. (For the `--write-config-only` stage they
  still come from the CLI, since that's what gets written.)
- `--write-config-only` keeps the file (it's the deliverable) and can't be
  combined with `--dry-run`. `--use-existing-yaml` never writes the file, so
  a `--dry-run` in production rolls back the DB and leaves the committed
  config untouched.
- Both flags are yaml-mode only and mutually exclusive; the plain (neither
  flag) yaml run still does file-write + DB together for local use.
- The db-mode equivalent needs no file step at all — `--mode db` run once in
  production is self-contained, if a database-backed copy is acceptable.


## Common workflows

### Bringing an instance's data up to date

Start from the report rather than from memory — for a DB-sourced instance there
is no other way to see the whole list:

```bash
python manage.py dataset_status <instance> --stale-only
```

Then run what it prints, which is this sequence:

1. `rename_dataset_metrics <instance> <datasets> --apply` — only for anything it
   marked `rename first`, and it has to come first: the import refuses while a
   binding still holds the old metric name.
2. `load_dvc_dataset <instance> <datasets> --force`
3. `sync_instance_to_db <instance>`
4. `dataset_status <instance> --stale-only` again, which should now be quiet.

Check the pin before starting. `dataset_status` prints it, and for a DB-sourced
instance it is the DB spec's pin, which lags the YAML until step 3 — so a first
pass can legitimately import from an older commit than the YAML names. Use
`--repo-from yaml` on both `dataset_status` and `load_dvc_dataset` when you mean
the YAML's.

### Retiring a dataset an instance no longer uses

The order matters, because a dataset the config still declares comes back on the
next sync as an empty row — and an empty row the model binds is worse than the
dataset that was there.

1. `dataset_inventory <instance> --order drift` — find the candidates. A row that
   is `db only` with a populated DB side, or one nothing else explains, is worth a
   look.
2. **Check it is really unused.** `delete_dataset <instance> <ids>` with no
   `--apply` does this for you and reports every reason it would refuse. Trust the
   FK tables over reading the graph: `dataset_replacements` means a row can be
   bound under a different identifier than the one the model declares.
3. **Remove the declaration first** — the node, the `input_datasets` entry, or the
   `dataset_replacements` line — then commit, deploy and
   `sync_instance_to_db <instance>`. The sync moves the bindings off the row,
   which is what turns a refusal into a clean delete.
4. `delete_dataset <instance> <ids> --apply --dump-to DIR`.
5. `dataset_inventory <instance>` again to confirm the count dropped and nothing
   else moved.

If a removed declaration was load-bearing you want to know before step 4, not
after. The cheap test: capture the outputs of the affected nodes, remove the
declaration, recompute, and diff. On `mainz-bisko` that showed a
`dataset_replacements` entry between two value-free datasets to be completely
inert — every node output byte-identical, `total_weighted_data_quality` included —
which is what made removing it safe rather than plausible.

### Verifying a spec model change

1. Make the change (e.g. add a field to `OutputPortDef`)
2. Re-sync: `python manage.py sync_instance_to_db --all`
3. Test init: `python manage.py test_instance --state-dir model-outputs/ --dry-run --spec-only`
4. Test compute: `python manage.py test_instance --state-dir model-outputs/ --dry-run`
5. Spot-check a node diff: `python -m tools.debug_instance -i espoo --diff-node some_node`

### Debugging a DB-sourced instance that fails to load

1. Check the error: `python manage.py test_instance --start-from the_instance --dry-run`
2. Diff a suspicious node: `python -m tools.debug_instance -i the_instance --diff-node the_node`
3. Switch to YAML to verify it works: `python -m tools.debug_instance -i the_instance --source yaml --save`
4. Fix the serialization in `instance_from_db.py`
5. Re-sync and switch back: `python manage.py sync_instance_to_db the_instance`

### The ClusterableModel save() trap

`NodeConfig.save()` goes through Wagtail's `ClusterableModel` which
can silently revert changes to modeltrans `i18n` fields. When updating
`NodeConfig` fields programmatically, use `queryset.update()` instead
of `instance.save()`. See
[graphql-mutations.md](../architecture/graphql-mutations.md) for details.
