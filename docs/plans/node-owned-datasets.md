# Datasets owned by a node

Status: agreed with Juha 2026-09-24. Step 1 is done (95b8af88, 2026-09-25):
scope types, `Dataset.scope_instance`/`scope_node`, the query split, the
scope-delegating policy with dataset `ObjectRole`s, `NodeSnapshot.datasets`
(snapshot v12), graph-level exclusive binding, deletion, and `Dataset.scope`
NOT NULL with the backfill. Decisions 13–15 and the split of the former step 3
into steps 3–6 were agreed on 2026-09-27. On 2026-10-09 steps 4–6 were moved
ahead of step 2 and the rest of step 3: lossless export and import are needed
for other work too, and step 4 depends on neither (see step 4). The step 3
stopgap is done (ca60343a), as is the re-import identity that step 4 assumes
(881c9a1a). Step 4 is next.

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
     too (steps 4 and 5).
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
13. **Persisted references are uuids; column names are a runtime detail.** A
    metric has three roles that `DatasetMetric.name` used to conflate:
    identity (`uuid`), the dataframe column (a handle), and display (`label`,
    optionally with `spec.quantity`). Nothing persisted names a DB dataset's
    metric column: a binding selects its metric by `metric_uuid`, and the
    dataset loader both names the columns and resolves the selection, so the
    two cannot drift.
    - The column name is generated on `DatasetMetricMeta`, the graph-bound
      catalogue entry the computation uses: the authored `name` if present,
      else a slug of the label, else of the quantity; deduplicated with a
      suffix in a deterministic order (`order`, then uuid), so cache keys are
      stable across graph builds. Metric columns carry a fixed prefix (for
      example `m_`) so a user-controlled name cannot collide with a dimension
      column or a reserved column (`Year`, `Forecast`, `Value`, `node`). The
      prefix exists only in the raw dataset frame; `select_metric` renames the
      column to the port's.
    - A label edit renames the column on the next load. Nothing persisted
      notices, and the frames stay readable while debugging. Neither the
      label nor the quantity kind is unique within a dataset (two `currency`
      metrics are ordinary), which is why neither can be the handle; the
      editor asks for distinct sibling labels as a usability rule.
    - An external dataset's column name is the source's identifier, not
      ours: it goes into `DatasetMetricSpec` (for example `external_column`)
      and serves DVC import, re-sync and materialization.
      `fix_dataset_metric_names` and `rename_dataset_metrics` become
      maintenance of that field. This replaces step 3 of the metric-spec plan
      ("move `name` into `spec`"): `name` moves as the external column, and its
      handle role disappears.
    - Why: `ostersund-c4c` `net_costs` failed because `DBDataset` named a
      column `Coalesce(name, label, uuid)` (`Mileage`) while the binding
      selected by `metric.name` (`None`), so a two-metric port was never
      narrowed. 42 metrics had no name, with 11,280 bindings to them. Any rule
      computed in two places drifts; the fix is one place, not a better rule.
14. **A dataset's structure is described once.** `DatasetSnapshot` contains a
    `DatasetMeta` for the structure (composition, so the meta stays a frozen
    value) and adds the body: data, sources, references, comments, evidence.
    `InstanceExport.datasets` carries only bodies keyed by dataset uuid; the
    structure is already in the `InstanceSnapshot`. Today one export describes
    each dataset twice, and the two disagree: `DatasetMetricSnapshot.identifier`
    is `metric_column_id()`, `DatasetMetricMeta.identifier` is raw
    `metric.name`.
    - `DatasetMeta` becomes pure structure. `revision_id` is binding state (which
      revision the graph pinned) and moves to `DatasetRevisionPinSnapshot`;
      inside a dataset's own revision it would be circular. `is_editable` (the
      deprecated BISKO lock) stays out of it.
    - The fields only the snapshot has today are structure and move into the
      meta: `name`, `forecast_from`, `time_resolution`. `dimension_columns`
      is a persisted column-name mapping, so by decision 13 it is external-source
      metadata, like `external_column`.
    - Dimensions already have this shape: `InstanceSnapshot.dimensions` stores
      `DimensionMeta` directly. The other Snapshot/Meta pairs (`NodeSnapshot`
      and its graph counterpart) have not been compared yet.
15. **Authored identifiers are preferred, never required, never
    load-bearing.** The rule of decision 13 applies one level up to dimension
    columns and category identifiers (the values in those columns): use the
    authored identifier if present and unique in its scope, else generate one.
    Their scope is the instance, not the dataset, because nodes and category
    filters use them too, so their naming authority is the instance-level
    `DimensionMeta`. `filter_column` refers to dimension columns by identifier
    today; like the metric references, those become uuid references.

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
  - The uuids `DatasetSnapshot` gains here (`uuid`, `schema_uuid`, metric
    `uuid`s) take `DatasetMeta`'s field names and types, so that step 4
    moves them into the contained meta instead of renaming them.

Steps 3–6 were one step. They are split where each can land and be verified
on its own: 3 fixes the class of bug that motivated decision 13 and needs no
format break; 4 is the one `DatasetSnapshot` format break; 5 and 6 build on
the uuid-complete format.

### Step 3: metric columns named on the catalogue

Decision 13. Independent of the export format; it changes `DatasetMetricMeta`
(an `InstanceSnapshot` version bump) but not `DatasetSnapshot`.

- **Stopgap, if needed before the rest.** Build every metric selector with
  `metric_column_id()`: `dataset_meta_from_model` (`datasets/catalogue.py`),
  the `external_metric_id=F('metric__name')` in
  `NodeConfigQuerySet.annotate_ports` (as a `Coalesce`), and the GraphQL
  `external_metric_id=...metric.name` sites (`nodes/graphql/bindings.py`,
  `nodes/graphql/types/instance.py`). Fixes `ostersund-c4c` without the
  design change.
- **Generated column names** on `DatasetMetricMeta`, as in decision 13.
  `DBDataset.deserialize_df` takes its column names from the meta instead of
  its SQL `Coalesce`; `metric_column_id()` and the `Coalesce` go.
- **Selection by uuid.** `_load_dataset_value` (`nodes/runtime_input.py`)
  selects by `metric_uuid` through the meta. A binding to a multi-metric
  dataset whose selection resolves to nothing fails loudly instead of
  delivering the wide frame. `external_metric_id` then means only the id in an
  external source.
- **Rewrite the string references.** Of 5,790 dataset bindings (local DB,
  2026-09-27), 6 pipeline ops name a metric column: all `filter_column`,
  presumably dropping unwanted metrics. They become uuid references or a
  keep-these-metrics selection. `SelectMetricOp` already takes no parameters.
- **Remove the identifier fallback** in `build_instance_graph`
  (`nodes/instance_graph.py`: a binding without `metric_uuid` matched by
  `m.identifier`). An `InstanceSnapshot` upgrader resolves old bindings once
  against the catalogue the snapshot carries, as
  `instance-graph-dimension-constraints.md` requires ("must not silently fall
  back to identifier lookup").
- **External column in the spec.** `DatasetMetricSpec.external_column`,
  written by `load_dvc_dataset.create_metric` and
  `placeholders._create_metric` from the DVC metadata `column_id`. The same
  metadata (`metrics: [{column_id, id, label, quantity}]`) also has
  `quantity`, which import does not write into the spec today; it should. The
  label fallback becomes the capitalized column name instead of the raw one.
- **Backfill** the 42 nameless metrics, then check sibling-label clashes and
  report them rather than failing.
- **Check Watch** before moving or dropping `DatasetMetric.name`: the model is
  in kausal_common.

### Step 4: one dataset description, uuid-keyed data

Decision 14 and the long-form data, in one `DatasetSnapshot.schema_version`
bump, so stored revisions are upgraded and content hashes change once.

- **`DatasetSnapshot` contains `DatasetMeta`.** `revision_id` moves to the pin;
  `name`, `forecast_from` and `time_resolution` move into the meta.
  `InstanceExport.datasets` carries bodies keyed by dataset uuid.
- **Every `*Snapshot` carries the uuids of what it describes:** dataset,
  schema, metrics, dimensions, categories, data points, sources, comments.
  Today import mints fresh ones for all of them, and categories travel as
  `dim_id/cat_id` identifiers. `DataPointKey` (a natural key, so that the
  export-then-import copy could mint fresh uuids) goes.
- **Long-form data.** `DatasetSnapshot.data` is today
  `JSONDataset.serialize_df` of the runtime frame: wide pandas
  `to_json(orient='table')`, one row per year and category combination, one
  column per metric. It has no place for a data point's uuid, and pandas
  rounds to 10 significant digits (`double_precision=10`), while
  `DataPoint.value` is `Decimal(32, 16)`. Replace it with a Pydantic model:
  long form, one entry per data point (uuid, date, metric and category uuid
  references, `Decimal` value), with comments, evidence and source references
  nested under it instead of matched by natural key. The model converts to and
  from `PathsDataFrame`; `JSONDataset` delegates to it. If long form is too
  large for revision storage, the model can store columns without changing its
  interface.
- **One upgrader.** `DatasetSnapshot` is also the materialization content, the
  hash input and the published revision payload (`serialize_dataset`). The
  `schema_version` upgrader converts stored revisions; every content hash
  changes once, here.
  - Revisions frozen before this have no data-point uuids, and the true ones
    cannot be recovered. The upgrader borrows the uuid of the live data point
    with the same coordinates where one still exists, and derives one
    (`uuid3(dataset_uuid, coordinates)`) where it does not. Old revisions are
    lossless only where that is possible.
- **Order.** This step does not depend on step 2: the only part of step 2 it
  uses is "uuids in `DatasetSnapshot`", which it delivers itself by containing
  `DatasetMeta`. Nor does it depend on step 3: long-form data refers to metrics
  by uuid, so the snapshot no longer depends on column names at all.
- **Prerequisite (done, 881c9a1a).** Uuid-keyed data is only worth having if
  a data point keeps its uuid while its cell exists. `load_dvc_dataset --force`
  used to delete every data point and create it again; it now upserts by
  coordinates through `DBDataset.upsert_df` (`datasets/runtime/db.py`), which
  is also where a frame from any other source (`extract_node_dataset`, framework
  conversion) should be written. The DVC boundary is the right place for a
  natural key, because the source has no uuids.

### Step 5: lossless export and import

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

### Step 6: copy by rekeying

Today `copy_instance` is export plus an import that mints fresh uuids, which is
what `DataPointKey` served. With step 5's identity-preserving import, a copy
rekeys the export first and then imports it as usual.

- **`ModelSnapshot.rekeyed(mapping)`**, used by `copy_instance` and a future
  single-node copy.
  - Uuid fields are one of three kinds, marked on the field type:
    - *identity*: minted anew, with the old→new pair added to the mapping;
    - *reference*: rewritten if its target is in the mapping and kept if not
      (template datasets, framework schemas);
    - *provenance* (`copy_of`): never rewritten; a copy sets it to the source.
    A generic walker that replaced every uuid found in the mapping would
    rewrite `copy_of` wrongly, so the kinds are needed.
  - References cross snapshot boundaries (binding → node, dataset or metric;
    `NodeSnapshot.datasets`; `dataset_revisions`), so the mapping is shared
    across one call: first collect identities and mint their new uuids, then
    rewrite references.
  - Uuids inside untyped `dict[str, Any]` fields (`spec`, `data`,
    `external_ref`) are invisible to the walker. The `Any` clean-up makes them
    typed.
- **Invariant test:** in a rekeyed dump of a real instance, no source identity
  uuid appears except in provenance fields. It scans the JSON text for
  uuid-shaped strings, so it also catches uuids that are not typed as such.

### Afterwards: the editor reads datasets from its edition

Not part of this plan, but it waits on step 4, so it is recorded here. The editor's
`Dataset` type reads every field from the live row, so a query against a published
revision (`InstanceEditorFields._source`) already returns the draft's dataset
metadata and data; `editor.dataset(s)`, the dataset of a port binding and
`Node.datasets` all ignore the source. `DataEntryQuery.load()` already resolves the
right edition per dataset (the live row when unpinned in the draft, the pinned
revision otherwise), but only for the data-entry views, which is why
`DataEntryDataset` duplicates `Dataset`.

After step 4 a `DatasetSnapshot` is "`DatasetMeta` plus body", so the refactor is:

- a request-scoped resolver in `InstanceRequestResources` (next to
  `dataset_models_for_graph`), keyed by `ResolvedInstanceSource`, returning each
  dataset's edition: the meta, the snapshot (lazily) and the live row only for an
  unpinned draft dataset. `DataEntryQuery` uses it instead of owning it.
- `DatasetType` rooted on the edition. Content fields (name, dimensions, metrics,
  category domain, `shape` from `graph.shapes`, data) come from the meta and the
  snapshot even in the draft, so draft and published cannot read differently. Fields
  that only exist live (permissions, editability, timestamps) come from the row and
  are false or null without one. `portBindings` comes from `graph.bindings`.
- every `Dataset` reference resolved through the edition of the editor's source;
  `DataEntryTable.dataset` becomes a `Dataset` and `DataEntryDataset` is deprecated.

Open: what `editor.datasets` lists in a frozen edition (only the graph's datasets,
while the draft lists every row in the scope), and whether a mutation's result is
built from the row it wrote or from a refreshed request graph.

### Later

- **Generated dimension and category identifiers** (decision 15), on the
  instance-level `DimensionMeta`. Authored identifiers are more deliberate and
  stable than metric labels, so this is not urgent.
- **Compare the remaining Snapshot/Meta pairs** (decision 14) and merge them the
  same way where they duplicate.
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
