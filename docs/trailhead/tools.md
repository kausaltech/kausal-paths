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
