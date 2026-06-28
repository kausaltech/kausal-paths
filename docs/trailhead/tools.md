# Trailhead Migration: Tools & Tips

## debug_instance.py

The main tool for investigating DB-backed vs YAML-backed model
instances. Lives at `tools/debug_instance.py`.

### Diff a node's config dict between YAML and DB

The most useful operation — shows exactly what the DB serialization
produces vs what the YAML loader would see:

```bash
python tools/debug_instance.py -i espoo --diff-node building_type_index
```

Use this to verify that `instance_from_db.py` serialization produces
config dicts that the InstanceLoader can consume correctly.

### Switch an instance between YAML and DB sources

```bash
# Switch to YAML (useful when DB spec is stale or broken)
python tools/debug_instance.py -i budget --source yaml --save

# Switch back to DB
python tools/debug_instance.py -i budget --source db --save
```

### Evaluate Python with instance/ctx/node in scope

```bash
# List input datasets for all nodes
python tools/debug_instance.py -i espoo --source db -c "
    for n in ctx.nodes.values():
        if not n.input_dataset_instances:
            continue
        print(f'{n.id}: {[ds.id for ds in n.input_dataset_instances]}')
"

# Inspect a specific node's output metrics
python tools/debug_instance.py -i budget --source yaml -c "
    node = ctx.get_node('building_renovations')
    for k, m in node.output_metrics.items():
        print(f'key={k!r}, column_id={m.column_id!r}, unit={m.unit}')
"
```

### Compute a node from a specific source

```bash
python tools/debug_instance.py -i espoo --source db --node net_emissions
```


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

### Verifying a spec model change

1. Make the change (e.g. add a field to `OutputPortDef`)
2. Re-sync: `python manage.py sync_instance_to_db --all`
3. Test init: `python manage.py test_instance --state-dir model-outputs/ --dry-run --spec-only`
4. Test compute: `python manage.py test_instance --state-dir model-outputs/ --dry-run`
5. Spot-check a node diff: `python tools/debug_instance.py -i espoo --diff-node some_node`

### Debugging a DB-sourced instance that fails to load

1. Check the error: `python manage.py test_instance --start-from the_instance --dry-run`
2. Diff a suspicious node: `python tools/debug_instance.py -i the_instance --diff-node the_node`
3. Switch to YAML to verify it works: `python tools/debug_instance.py -i the_instance --source yaml --save`
4. Fix the serialization in `instance_from_db.py`
5. Re-sync and switch back: `python manage.py sync_instance_to_db the_instance`

### The ClusterableModel save() trap

`NodeConfig.save()` goes through Wagtail's `ClusterableModel` which
can silently revert changes to modeltrans `i18n` fields. When updating
`NodeConfig` fields programmatically, use `queryset.update()` instead
of `instance.save()`. See
[graphql-mutations.md](../architecture/graphql-mutations.md) for details.
