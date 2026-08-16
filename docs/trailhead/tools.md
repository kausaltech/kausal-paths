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
