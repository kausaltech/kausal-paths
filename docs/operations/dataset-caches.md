# Persistent dataset caches

Paths keeps rebuildable DVC metadata and prepared input frames in the `datasets`
Python package. Its Django app label is `paths_datasets`; `datasets` is already
used by `kausal_common.datasets`, which owns authored datasets.

`DVCSourceManifest` stores one dvc-pandas `DatasetManifest` per repository URL,
full Git commit, selected remote, dataset identifier and manifest format version.
Reads and missing-entry resolution are batched per model context. A valid row
lets dvc-pandas fetch the content-addressed Parquet object without opening Git.
Unpinned or abbreviated revisions retain the repository loading path. Runtime
credentials come from the environment, not persisted metadata.

`PreparedDataset` stores a deterministic prefix of a DVC binding as compressed
Arrow IPC, together with units, ordered primary keys and explanations. Its key
includes the source content hash and interpretation metadata, ordered operation
recipes and their declared runtime dependencies, the selected metric and the
`empty_to_zero` binding flag. It does not include the deployment ID, instance ID,
or repository commit: unchanged inputs can be reused across deployments and
instances. Prepared rows are batch-read once per context; bindings with identical
recipes share a row. A later parameter change can request a different recipe.

The prefix ends before the first temporal-fill operation or an operation that
has not opted into persistent caching, whichever comes first. Framework measure
values, other source overlays and sampling always execute afterward. Legacy tag
operations can inspect arbitrary context and therefore stop the prefix. The
separate `GenericDataset` preparation pipeline is not yet cached. Authored DB
payloads continue to use their existing materialization path.

When changing conversion or serialization semantics (including unit definitions),
bump `PREPARATION_VERSION` in `datasets/prepared.py`. When changing an opted-in
operation, update its `cache_version` and audit `cache_hash_data(context)` for all
runtime dependencies. New operations default to not being persistently cacheable.
Do not move the boundary across an instance overlay without re-auditing it.

## Deployment and warm-up

Migrate before starting pods. The deployment-only `docker/post-migrate` hook runs
`prepare_dvc_manifests --in-customer-use`, resolving missing or invalid metadata
without downloading Parquet data. It has a bounded timeout and logs failures;
runtime resolution remains available. Ordinary `manage.py migrate` does not run
this hook.

Pod startup still runs `compute_instances --in-customer-use --warm-dvc-cache`.
Manifest-backed prefetch downloads missing Parquet objects without decoding them,
even when another pod has populated the external computation cache. Git is only
needed when metadata cannot be resolved from the database. Prepared frames are
populated by computations that miss the normal in-memory/external caches.

## Backups and cleanup

Both tables have the `__rebuildable` suffix. Shared logical backup commands use:

```sh
pg_dump --exclude-table-data='*.*__rebuildable' ...
```

This retains their schema while omitting rows. A restore starts with empty caches;
the post-migration preparation and runtime misses rebuild them. Do not use this
suffix for authored data or anything that cannot be regenerated.

Inspect unused entries with:

```sh
python manage.py prune_dataset_caches
python manage.py prune_dataset_caches --delete --older-than-days 7
```

The command scans effective draft and published configurations, including
YAML-backed instances and inactive instances. It does not clone Git or create
manifests during the scan, and aborts deletion if any configuration cannot load.
Entries younger than the grace period are retained. Old revisions and non-current
parameter variants are not permanent roots; they can regenerate inputs on demand.
Cleanup is explicit, not part of every migration or pod startup.

## Measuring

Compare fresh processes with external computation caching disabled:

```sh
python load_nodes.py -i INSTANCE --node NODE --disable-ext-cache --show-perf
python load_nodes.py -i INSTANCE --node NODE --disable-ext-cache --disable-prepared-cache --show-perf
```

The `prepare` dataset operation measures prefix lookup or computation. Use a first
run to populate the prepared cache. `--skip-cache` also bypasses prepared frames,
but changes ordinary model caching too, so it is not an isolated comparison.
When comparing outputs, align rows by primary keys: downstream parallel reductions
can change row order and floating-point rounding between otherwise identical runs.
