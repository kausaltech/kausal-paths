# Dataset revision closure

## Goal

Make DB-backed dataset computation use the same serialized-data path for the
current draft and published instance revisions. Publishing an instance must
produce an immutable closure over its model structure and every DB dataset
used by that structure, without adding dataset-shaped N+1 queries to model
loading.

After this work, the supported sequence is:

1. Edit a DB-sourced instance and its datasets.
2. Publish the current draft.
3. Continue editing the draft, including its datasets.
4. Resolve public calculations from the published model and dataset payloads.
5. Resolve authenticated draft calculations from the current materialized
   payloads.

The public result must remain unchanged by step 3 until the next publication.

## Architectural decision

Normalized `Dataset` / `DataPoint` rows remain canonical for editing. A
serialized calculation payload is canonical for runtime reads.

Both runtime modes instantiate the same snapshot-backed dataset implementation:

| Mode | Payload source | Runtime cache identity |
| --- | --- | --- |
| Draft | One-to-one current `DatasetMaterialization` | dataset UUID + generation + content hash |
| Published | Immutable Wagtail dataset `Revision` | dataset revision ID |

The payload decoder and the subsequent binding transformations are shared.
Only payload selection differs.

## Data model

### Materialization generation

Keep the Paths-specific monotonically increasing generation on the
materialization rather than adding it to the shared `Dataset` model. Every
logical refresh increments that generation. Freshness against the normalized
source rows is checked by storing the exact `Dataset.last_modified_at` written
by the same atomic operation as `source_modified_at` on the materialization.
The content hash supplies an independent payload identity.

### DatasetMaterialization

Add a Paths-side model with:

- `dataset`: one-to-one FK to `Dataset`, cascading on dataset deletion;
- `content`: serialized `DatasetSnapshot` JSON;
- `generation`: monotonically increasing materialization generation;
- `content_hash`: SHA-256 of a canonical serialization;
- `source_modified_at`: exact parent-dataset timestamp represented by
  `content`;
- `updated_at`.

The core invariant is:

```text
Dataset.last_modified_at == DatasetMaterialization.source_modified_at
```

A mismatch is stale state. Once the rollout/backfill period ends, runtime
loading and publication must reject it rather than silently reading ORM rows.

### InstanceRevisionDatasetPin

Add a relational manifest for the dataset revisions retained by an instance
revision:

- `instance_config`: FK with `CASCADE`;
- `instance_revision`: Wagtail `Revision` FK with `CASCADE`;
- `dataset`: FK with `PROTECT`;
- `dataset_revision`: Wagtail `Revision` FK with `PROTECT`;
- the publication-time identifier for diagnostics and the transitional loader.

Use a uniqueness constraint on `(instance_revision, dataset)` and indexes that
support lookup by instance config, instance revision, dataset, and dataset
revision.

The instance snapshot also contains a portable manifest entry with dataset
UUID, identifier, and revision ID. The relational table enforces lifecycle
integrity; the JSON manifest remains the snapshot contract.

## Logical dataset write boundary

Do not materialize from model signals. Signals would observe intermediate M2M
state, serialize once per row during bulk changes, and miss operations that use
bulk queryset methods.

Provide one explicit Paths-side write boundary used by GraphQL, REST, imports,
and sync code:

```python
with dataset_change(dataset, user):
    # Apply the complete logical operation.
```

The outermost boundary:

1. locks the dataset row;
2. applies the logical mutation;
3. writes the dataset modification timestamp;
4. increments the materialization generation;
5. builds one serialized payload from the final database state;
6. upserts `DatasetMaterialization` with the same source timestamp and a canonical
   content hash;
7. invalidates the owning instance cache;
8. commits atomically.

Nested operations must materialize only once. Bulk imports materialize once per
affected dataset after the bulk operation, not once per datapoint.

## Draft runtime path

Introduce a lazy bulk current-payload store. Instance construction knows all
DB dataset UUIDs needed by its bindings but does not fetch payload JSON yet.

On the first dataframe request, fetch all required materializations in one
query. Validate source timestamp equality, parse each distinct payload once, and
cache decoded dataframes in the runtime context. Multiple bindings and ports
for the same dataset share the result.

Metadata-only GraphQL operations must not fetch materialization content.

## Publication

Publication is one transaction:

1. lock the `InstanceConfig` and distinct bound DB datasets;
2. verify every current materialization is present and source-current;
3. create or reuse an immutable Wagtail dataset revision from each current
   materialized payload;
4. build the `InstanceSnapshot` using exactly those revision IDs;
5. save the instance revision;
6. create `InstanceRevisionDatasetPin` rows;
7. publish the instance revision and invalidate public caches.

The first implementation may create a new dataset revision at each instance
publication. Content-hash reuse is safe as a later optimization.

Publication must never rebuild dataset payloads from live datapoint rows. That
keeps the cut fast and ensures the pinned content is exactly what draft runtime
was already using.

## Published runtime path

Introduce a lazy bulk revision-payload store. It uses the snapshot/relational
manifest and fetches all required Wagtail revision payloads with one query on
first dataframe access.

Published computation must perform no queries against live `Dataset`, schema,
metric, dimension-category, or datapoint rows. Revision IDs are immutable cache
keys.

Do not use `Revision.as_object()`: reconstructed row-shaped objects retain ORM
relationship behavior and invite both live-state leakage and N+1 queries.

## Compatibility

New instance revisions require complete pins for every DB-resolved dataset.
Missing or mismatched pins are errors.

Older revisions without pins may use one explicit compatibility fallback to
live datasets, accompanied by a warning/metric. Republishing upgrades them.
The fallback stays at the revision boundary and must not spread into runtime
dataset or GraphQL resolvers.

## Dataframe decoding and performance

Use the existing pandas JSON-table decoder first, so the architecture and data
format do not change simultaneously. Instrument payload fetch, Pydantic
validation, pandas decoding, and conversion separately.

Implement a direct Polars decoder only if representative measurements show at
least about 100 ms saved per loaded model, or a similarly material reduction in
server CPU.

Required query properties:

- metadata-only model queries: zero payload-content queries;
- first draft dataframe access: one materialization query, independent of the
  number of datasets;
- first published dataframe access: one revision query, independent of the
  number of datasets;
- subsequent accesses: zero payload queries;
- published computation: zero live dataset/datapoint/metric queries;
- each distinct payload decoded at most once per runtime context.

Measure both read and write costs on representative dataset sizes. In
particular, record materialization latency and JSONB/WAL size after a one-point
edit and a bulk grid edit. Prefer batching logical edits before introducing a
second runtime data path.

## Rollout

1. Add generation, materialization, and pin schema.
2. Add the explicit dataset-change boundary and migrate GraphQL and REST writes.
3. Backfill current materializations in batches; keep the operation resumable.
4. Switch draft computation to the lazy current-payload store.
5. Add publication-time immutable revisions and relational pins.
6. Switch published computation to the lazy revision-payload store.
7. Migrate active import, copy, repair, and Wagtail-admin writes to the
   boundary. The unused DiffSync-based `sync_datasets` and `copy_datasets`
   commands are intentionally out of scope.
8. Remove transitional live-row fallbacks after production backfill and
   verification.

## Verification gates

Correctness:

1. Set dataset value A and publish.
2. Change the draft value to B.
3. Published calculation still returns A.
4. Draft calculation returns B.
5. Republish; published calculation returns B.

Integrity and lifecycle:

- publication and concurrent dataset editing cannot produce a mixed cut;
- a stale current materialization is rejected;
- a missing or mismatched revision pin is rejected for new snapshots;
- pinned dataset revisions and datasets cannot be deleted;
- deleting an instance cascades its pin rows and permits normal cleanup;
- repeated publication and pruning leave no invalid pins.

Performance:

- query-count tests cover one and many datasets;
- repeated bindings decode only once;
- metadata-only queries remain payload-lazy;
- representative timing compares live-row loading with serialized draft and
  published loading;
- full pytest, mypy, Ruff, and `git diff --check` pass.
