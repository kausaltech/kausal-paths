# Datasets owned by a node

Status: agreed with Juha 2026-09-24. Step 1 is done (2026-09-25), Paths side
uncommitted: scope types, `Dataset.scope_instance`/`scope_node`, the query split,
the scope-delegating policy with dataset `ObjectRole`s, `NodeSnapshot.datasets`
(snapshot v12), graph-level exclusive binding, deletion, and `Dataset.scope`
NOT NULL with the backfill. Step 2 is next.

## Why

Models keep wide tables for many nodes because an instance dataset was the only
way to make a node's numbers editable. For example, the Mainz KNSV actions each
filter their own row out of `mainz/knsv_massnahmen_*` by a `measure` dimension
that mirrors the action list. An action should instead have its own effect
table, edited in the data studio, which lives and travels with the action.

## Decisions

1. **The dataset is fully owned by its node.** `Dataset.scope` is the
   `NodeConfig`. Lifecycle and permissions follow from the scope. kausal_common
   already supports object scopes: Watch scopes datasets to `Action`,
   `Category` and `Indicator`, and takes the plan from the parent object.
2. **The snapshot records ownership.** `NodeSnapshot.datasets` holds the node's
   own catalog entries (`DatasetMeta`), moved out of `InstanceSnapshot.datasets`.
   The node is the unit that is copied, deleted, reverted and exported, so its
   data travels inside it. `build_instance_graph` collects them into the flat
   graph catalog once; consumers of the graph see no change.
3. **The data stays in the dataset tables.** Data points, evidence, sources,
   comments and validation work as for any dataset. Publishing pins each
   dataset revision in `InstanceSnapshot.dataset_revisions`, keyed by uuid,
   unchanged. No data points in `NodeSnapshot`: that would repeat every table
   in every revision blob and bypass the data-studio machinery.
4. **Exports carry the data with its node.** `InstanceExport` groups the
   node's `DatasetSnapshot` bodies under their node, or at least marks them
   with `owner_node`. Import creates the node and its data as a unit.
5. **A node-owned dataset has no identifier.** Bindings refer to it by foreign
   key. The runtime falls back to the uuid (`dataset.identifier or
   str(dataset.uuid)`, already the rule), so two nodes can each own their data
   without their identifiers colliding.
6. **Edits go through the dataset's own revisions**, as for instance datasets,
   and the node's history shows them by comparing the node's pins.
7. **The live tables are the latest draft.** Deleting a node hard-deletes its
   datasets and schemas. `NodeConfigQuerySet.delete_related()` deletes, in
   bulk, the datasets and schemas scoped to the nodes in the queryset, and
   their unpinned dataset revisions. `NodeConfig.delete()` and
   `InstanceConfig.delete()` both call it. `InstanceConfig.delete()` calls it
   before `self.nodes.all().delete()`, which, being a queryset delete, never
   runs `NodeConfig.delete()`. No publish-time cleanup is needed.
8. **Reverting a node deletion is built now, for this case only.** The UI finds
   deleted nodes among recent `InstanceChangeOperation`s (`action='node.delete'`,
   not superseded). Reverting the operation recreates the node, its bindings
   (both into and out of it) and its datasets from the log entries, by uuid,
   and marks the operation superseded. Dataset revisions cannot be the source:
   a dataset `Revision` is written only at publish, so draft edits and
   never-published datasets would be lost.
   - Each owned dataset's log entry holds a `DatasetSnapshot`. Today that
     snapshot has no uuids (dataset, schema and metrics are matched by
     identifier, and import gives them fresh uuids). Owned datasets have no
     identifier, and bindings refer to metrics, so the snapshot gains
     `uuid`, `schema_uuid` and metric `uuid`s. Import and export can use them
     too (step 3).
   - Revert is all or nothing and refuses on conflict: the identifier is taken
     by another node, a source node or dimension category is gone, or a port it
     would rebind is now bound elsewhere.
   - This is the first real `EditableInstanceChild.apply_snapshot`.
9. **Pins keep the revision, not the row.** The pin design
   (`docs/plans/dataset-revision-closure.md`) uses `PROTECT` on both
   `dataset` and `dataset_revision` for "lifecycle integrity": pinned datasets
   and pinned revisions cannot be deleted. Only the revision needs keeping. The
   published runtime reads only `pin.revision_id` and `dataset_uuid`
   (`InstanceLoader.from_snapshot` passes `dataset_pk=0`). The dataset FK is
   also used for the `(instance_revision, dataset)` uniqueness and for copying
   inherited template pins, and `dataset_uuid` serves both. So:
   - drop `pin.dataset` (or make it `SET_NULL`), key the uniqueness on
     `dataset_uuid`, and keep `dataset_revision` under `PROTECT`;
   - stop the dataset deleting its pinned revisions. Wagtail's
     `RevisionMixin._revisions` is a `GenericRelation` that exists only to
     delete an object's revisions with it. Removing it on the Paths `Dataset`
     (`_revisions = None` in the Paths branch; a `GenericRelation` has no
     column) turns that into explicit work: `delete_related()` and
     `InstanceConfig.delete()` delete the unpinned revisions themselves. A
     pinned revision outlives its row until its pin goes.

10. **Node-owned datasets are internal to their node.** `for_instance_config()`
    returns only instance-scoped rows; `for_node()` sits next to it, and
    `governed_by_instance()` returns both for the few callers that need
    everything the instance governs (runtime loading, materialization,
    `dataset_status`, `delete_dataset`, `instance_export_sync`, permissions).
    Instance-level UI lists leave them out; a list field can gain an argument
    if a use case appears. Binding is exclusive by construction: `bindDataset`
    resolves among `for_instance_config(ic) | for_node(nc)`.
11. **The schema is node-scoped too.** `DatasetSchemaScope` points at the node,
    so a schema edit cannot reach another node's data. Its dimensions stay
    instance-scoped, which they must be anyway: the port's dimensions are the
    instance's. That works because every dimension lookup already goes through
    the dataset's scope and will go through `dataset_instance()`. The schema
    permission policy (`construct_perm_q`, `construct_state_perm_q`,
    `get_instance_configs_for_obj`) matches only instance and framework scopes.
    It walks the node scope to the instance in step 1; until then a
    node-scoped schema is visible to superusers only. Deletion is
    `delete_related()` (decision 7). Catalogue listings
    (`frameworks/catalogue.py`) leave them out, as they should.
12. **A dataset's permissions are its scope's.** An instance dataset delegates
    to `InstanceConfigPermissionPolicy`, a node-owned one to
    `NodeConfigPermissionPolicy` (which inherits from the instance and adds the
    node's edit lock). Explicit grants come on top through dataset
    `ObjectRole`s: `DatasetPersonPermission` and `DatasetGroupPermission`
    replace the schema ones (there are no schema grants), so a climate expert
    can delegate a dataset's data to a domain expert who cannot touch its
    schema. The schema governs structure only (metrics, dimensions, category
    domain); a node-owned schema's structure delegates to the node too.
    Blocks: locked instance, locked node, and `DatasetSchema.is_editable=False`,
    which stays for now with a FIXME: it was a quick lock for a BISKO
    certification round and goes when shared BISKO data is injected from the
    framework template instead of copied.

## Work

### Step 1: scope, lookup, permissions

- `DatasetScopeType` (Paths branch in `kausal_common/datasets/models.py`)
  becomes `InstanceConfig | NodeConfig`, and schemas may be scoped to a
  node. Committed to kausal_common `main`; Juha pushes after review.
- `Dataset.scope` becomes NOT NULL. Scopeless placeholders predate a910cd46
  (2026-06-01); `nodes/0077` backfills them from their schema's single instance
  scope and runs before the kausal_common migration. Watch has none.
- **Resolver (done).** `Dataset.scope_instance` and `Dataset.scope_node` in the
  Paths branch, with `Dataset.instance_scope_q()` for querysets.
- **Query split (done)** (decision 10). Each `for_instance_config` caller keeps
  instance-only or moves to `governed_by_instance`; `frameworks/conversion.py`
  keeps instance-only.
- **Permissions (done)** (decision 12): the scope-delegating dataset policy, dataset
  `ObjectRole` models replacing the schema ones, structure-only schema policy.
- **The snapshot (done).** `NodeSnapshot.datasets` holds the node's own
  `DatasetMeta` entries, bound or not; `InstanceSnapshot.all_datasets()` and
  `build_instance_graph` flatten them into the graph catalog, and the graph
  rejects another node binding them. Snapshot schema version 12; the field
  default upgrades older snapshots. The draft loader keys DB datasets by
  `identifier or uuid`, as the published one already did.
- **GraphQL (done).** `Dataset.ownerNodeId`, `NodeEditor.datasets`.
- **Deletion (done early).** `NodeConfigQuerySet.delete_related()`,
  `NodeConfig.delete()`, and `InstanceConfig.delete()` calling it first.
- **Tests.** Permissions for owned datasets and schemas (view, edit, comment,
  cite, dataset grants), loading, and the snapshot round trip.

### Step 2: creating and moving data

- **An editor mutation that creates a port's own data.** It takes a schema
  derived from the port (unit, quantity, dimensions) and binds the port to
  the new dataset.
- **`extract_node_dataset`.** Move the held command onto the real scope, with
  no identifier.
- **Node deletion and its revert**, as in decisions 7–9: `delete_related()`,
  the pin change, uuids in `DatasetSnapshot`, the log entries, and the revert
  mutation. Both delete paths (the editor mutation and `instance_export_sync`)
  record the same entries.

### Step 3: export, import, copy

- `instance_serialization.py` writes and reads the instance scope in about ten
  places (export ranking and dataset query around :1655-1671, and creation at
  :1752, 1813, 1848, 1865, 1927, 2101, 2112, 2197-2207, 2248). Owned datasets
  carry their node's uuid, and import resolves it to the node row by uuid.
- **Import is lossless: `export(import(x)) == x`.** A model developer imports
  an instance locally, edits it, and exports it to a remote deployment, so
  import keeps every uuid and every value. The remote push is
  `instance_export_sync`, which today matches nodes by uuid or identifier and
  datasets by identifier only; it matches everything by uuid. The round-trip
  test is the contract.
  - Every `*Snapshot` carries the uuids of what it describes: dataset,
    schema, metrics, dimensions, categories, data points, sources, comments.
    Today import mints fresh ones for all of them, and categories travel as
    `dim_id/cat_id` identifiers.
  - `DatasetSnapshot.data` is today `JSONDataset.serialize_df` of the
    runtime frame: wide pandas `to_json(orient='table')`, one row per year and
    category combination, one column per metric. It has no place for a data
    point's uuid, and pandas rounds to 10 significant digits
    (`double_precision=10`), while `DataPoint.value` is `Decimal(32, 16)`.
    Replace it with a Pydantic model: long form, one entry per data point
    (uuid, date, metric and category uuid references, `Decimal` value), with
    comments, evidence and source references nested under it instead of
    matched by natural key. The model converts to and from `PathsDataFrame`;
    `JSONDataset` delegates to it. If long form is too large for revision
    storage, the model can store columns without changing its interface.
  - `DatasetSnapshot` is also the materialization content, the hash input and
    the published revision payload (`serialize_dataset`). The format change
    needs a `DatasetSnapshot.schema_version` upgrader for stored revisions,
    and every content hash changes once.
- **Copying rekeys first.** A copy (`copy_instance`, a future single-node
  copy) calls `ModelSnapshot.rekeyed(mapping)` and then imports as usual.
  - Uuid fields are one of three kinds, marked on the field type:
    - *identity*: minted anew, with the old→new pair added to the mapping;
    - *reference*: rewritten if its target is in the mapping and kept if not
      (template datasets, framework schemas);
    - *provenance* (`copy_of`): never rewritten; a copy sets it to the source.
    A generic walker that replaced every uuid found in the mapping would
    rewrite `copy_of` wrongly, so the kinds are needed.
  - References cross snapshot boundaries (binding → node, dataset or metric;
    `NodeSnapshot.datasets`; `dataset_revisions`), so the mapping is shared
    across one call: collect identities first, then rewrite references.
  - Uuids inside untyped `dict[str, Any]` fields (`spec`, `data`,
    `external_ref`) are invisible to the walker. The `Any` clean-up makes them
    typed.
  - Invariant test: in a rekeyed dump of a real instance, no source identity
    uuid appears except in provenance fields. It scans the JSON text for
    uuid-shaped strings, so it also catches uuids that are not typed as such.
- **Read-only BISKO copies.** The 121 datasets on `is_editable=False` schemas in
  paths-de are 26 identifiers (70 placeholders). The `de/*` copies in `bisko`,
  `mainz-bisko` and `augsburg-bisko` have drifted apart (different values and
  row counts), so replacing them with the template's shared data needs a
  reconciliation, not only a delete.
- **Framework templates.** A template node's data reaches inheriting
  instances through the publish pins. Whether a municipality may override it
  follows that input's `binding_owner`, which is still to be decided.

## Starting point

Uncommitted in the working tree (branch `feat/data-studio-backend`):

- `src/datasets/defs.py` `DatasetSpec.owner_node` and
  `src/datasets/graphql/types.py` `DatasetType.ownerNodeId`: the marker design.
  Discard it; nothing stored uses it, so the snapshot upgrader only adds
  `NodeSnapshot.datasets = []`.
- `src/nodes/management/commands/extract_node_dataset.py` and its test. It
  runs a binding's transformations up to the first temporal fill, stores the
  result with the forecast year the binding used, rebinds the node, and checks
  that the node and all outcome nodes are unchanged before keeping anything.
  A dry run on `mainz-bisko` (`paths-de`) for
  `knsv_dekarbonisierung_strommix` gives a 64-row `energy_carrier ×
  measure_scenario` table starting its forecast in 2025, with every result
  unchanged.
- `docs/architecture/node-owned-data.md`: the architecture page, written for
  the marker design. Rewrite its node-data section when step 1 lands.

Facts checked while planning:

- `sync_instance_to_db` updates `NodeConfig` rows in place by uuid and marks
  leftover nodes `is_stale`. The editor's `delete` mutation
  (`nodes/graphql/editor.py`) and `instance_export_sync` hard-delete.
- `NodeConfig` has no `delete()` override. `InstanceConfig.delete()` deletes
  its nodes and its own dataset graph.
- `revert_to_published` and undo are not implemented.
- A dataset `Revision` is created only at publish (`InstanceConfig.publish`).
  The draft data exists only in the live rows and `DatasetMaterialization`.
