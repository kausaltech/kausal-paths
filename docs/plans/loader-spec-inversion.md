# Loader inversion: specs as the load-time contract

*Status: the architectural inversion is CLOSED. Snapshot serving, dataset
revision closure, native snapshot-to-runtime construction and YAML-on-the-
common-path have all landed; the config-dict shim is deleted. What remains
is the step-1 tail (transform executor, `EdgeDimension`), the
`get_input(port)` class migration, the YAML edge-declaration codemod and
three-way sync. Updated 2026-09-21 after the step-11 wrap-up (`66f6f539`).*

## Goal

Make `InstanceSnapshot` the single structural contract between configuration
sources and the runtime:

```text
YAML ──parse──┐
              ├──> InstanceSnapshot ──load──> runtime Instance
DB draft ─────┤
published rev ┘
```

YAML-shaped dictionaries should exist only at the YAML compatibility boundary.
The runtime loader should consume typed specs and snapshots directly. Draft and
published reads must differ only in which snapshot and dataset payloads they
select, not in how they build the runtime or GraphQL model state.

The snapshot also obeys a closure property: everything reachable from a
published snapshot is immutable or version-pinned.

## Current state

### Landed foundation

| Area | State | Notes |
| --- | --- | --- |
| Parse-only YAML sync | Complete | YAML parses directly into `InstanceSnapshot`; `sync_instance_to_db` writes it through `nodes/spec_sync.py`. The runtime-introspection exporter and the parse oracle were retired on 2026-09-15. |
| Node identity/computation split | Complete | `NodeConfig` owns identity/display metadata; `NodeSpec` owns computation. Snapshot schema v4 and migration/bootstrap landed in `00e31e3a`. |
| Stable snapshot references | Complete | Node references use UUIDs; shared editor layout is snapshot-backed. |
| Snapshot-backed public GraphQL | Complete | Draft and published runtimes bind their selected `InstanceSnapshot`/`NodeSnapshot`; non-editor GraphQL reads model content from those snapshots. Landed in `54fa1dd3`. |
| Revisioned instance content | Complete | Snapshot v6 includes instance lead content and node body, removing the remaining structured-revision metadata leaks for newly published revisions. |
| Dataset revision closure | Complete | Snapshot v7 pins immutable dataset revisions. Draft reads use current materializations; published reads use pinned revision payloads. Landed in `e70b70ad`. |
| Dataset writer coverage | Complete for active writers | GraphQL, REST, Wagtail, snapshot import, DVC import, repair commands, metadata fan-out, and forecast-default promotion refresh materializations at explicit atomic boundaries. Unused DiffSync commands remain intentionally out of scope. |
| Native snapshot loader | Complete | `InstanceLoader.from_snapshot()` builds `Instance`, `Context`, nodes, edges and bindings from the typed snapshot. Delivered as step 10 of the [instance-graph plan](instance-graph-dimension-constraints.md) over four gated stages, 2026-08-13 → 2026-08-16 (`0b7e4df4`, `b1661436`, `5255eae9`, `5796d2b3`, `82387590`). |
| YAML on the common path | Complete | `from_yaml()` is parse → `InstanceSnapshot` → native build, with `snapshot_transform` as the framework overlay hook. `snapshot_to_config_dict`, `serialize_instance_to_dict`, `_init_instance`, `make_node`, `Edge.from_config`, `EdgeDimension.from_config`, emission-sector generation and every config-dict branch are deleted. |
| Compatibility-state removal | Mostly complete | Step 11 of the instance-graph plan: legacy `NodeEdge`/`DatasetPort` tables dropped, `supported_dimensions` gone, the identifier-era GraphQL editor surface removed, the `NodeExplanationSystem` dict shim retired, `spec_export` + `parse_oracle` retired, `dataset_spec`/`dataset_index` retired from the binding path. Open: legacy role inference, snapshot upgraders. |

### The inversion is closed

The construction path is now:

```text
InstanceSnapshot
  -> InstanceLoader._init_instance_from_snapshot  (Instance + Context)
  -> build_instance_graph(snapshot)               (InstanceGraph, alongside)
  -> _setup_nodes_from_snapshot / _setup_edges_from_snapshot
  -> _setup_runtime_inputs                        (from InstanceGraph bindings)
```

No YAML-shaped dictionary exists between snapshot selection and the runtime.
`InstanceGraph` is built during construction and is the source of truth for
runtime input bindings and for the explanation system's typed inputs; the
loader cross-checks it against the snapshot binding list and fails on
disagreement. `Instance`, `Context`, nodes and edges are still built from the
snapshot objects directly rather than from the graph — closing that gap is the
remaining half of instance-graph step 10 (`InstanceGraph.create_context()`),
tracked there, not here.

Three internal helpers still take dict fragments — `_make_node_datasets`,
`_make_node_params` and `_make_node_visualizations`. They are fed per-piece
from typed specs, not from a reconstructed instance config, so they are a
typing debt rather than a structural one.

### Current read paths

#### DB draft

1. `build_instance_snapshot(InstanceConfig)` reads the current editor tables.
2. `InstanceLoader.from_snapshot(snapshot)` builds the runtime natively.
3. The runtime binds the source snapshot to the instance and nodes.
4. DB datasets resolve through `DatasetMaterialization` and the lazy
   `CurrentDatasetPayloadStore`.
5. A missing materialization is a hard error. The transitional live-row
   fallback is gone; `ensure_dataset_materializations()` repairs stale rows
   at the load boundary instead.

#### Published

1. The live Wagtail instance revision supplies `InstanceSnapshot`.
2. Snapshot v7's dataset manifest is checked against relational
   `InstanceRevisionDatasetPin` rows.
3. `InstanceLoader.from_snapshot(snapshot, published=True)` builds the same
   runtime through the same native path.
4. DB datasets resolve lazily from immutable pinned Wagtail revisions through
   `RevisionDatasetPayloadStore`.
5. Pre-v7 revisions retain a boundary-only compatibility path; republishing
   upgrades them.

#### YAML

`InstanceLoader.from_yaml()` parses into an `InstanceSnapshot` and delegates to
the same native build. Framework-configured instances ride it too, through
`FrameworkConfig.create_model_instance()` and the `snapshot_transform` overlay;
the loader has no `fw_config` parameter any more. All three source modes share
one construction path.

## Remaining work

Steps 2 and 3 are done. The chain that is left:

```text
1. Runtime ports/transformations tail (executor coverage, EdgeDimension)
     and the get_input(port) class migration
          -> 4. YAML edge-declaration normalization (needs structural port
                identity, so declaration side cannot affect binding identity)
          -> 5. Three-way YAML/DB sync (needs the same identity; simplified
                by step 4, otherwise independent)

6. Snapshot-native editor reads — optional, unsequenced.
```

### 1. Finish runtime ports and binding transformations

The spec side already stores typed `PortTransformOp` pipelines.

Landed 2026-08-06 (`6d798054`, `c21702bc`, `b8d9f4e3`):

- `Edge.to_transforms()` derives the typed pipeline from the edge, and
  `_get_output_for_target()` executes it through the shared
  `apply_port_transformations()` executor, with `PipelineEnv.node` carrying
  the edge-parity semantics (NodeError attribution, no NaN pruning before
  flatten-sum, all-null dimension tolerance, Categorical assigned columns).
- `flatten` ops are shape declarations, not executable operations — they are
  excluded from execution and only feed the output-dimension assertion,
  which still reads `edge.to_dimensions`. A bare `to_dimensions: [{id: x}]`
  parses into the same `EdgeDimension` as an executable `from_dimensions`
  flatten; only the declaration side distinguishes them (~5,300 such
  entries in configs).
- `_guard_not_empty()` fails only on a non-empty → empty transition;
  emptiness flows through (edges depend on this after metric selection).
- `slice_category_at_edge` deleted.
- Verified with focused parity tests plus a full `test_instance --compare`
  sweep (all remaining failures reproduce identically on main: stale
  lucia/muenchen-bisko baselines, muenchen-demo missing YAML entrypoint,
  dut-transport-nzc known-broken).

Also landed since:

- `Edge.from_config()` and `EdgeDimension.from_config()` are gone; the
  snapshot path builds `Edge` objects from typed bindings
  (`_setup_edges_from_snapshot` → `_make_edge_from_group` →
  `_edge_dimensions_from_transforms`).
- The `flatten` placeholder was retired with instance-graph step 2, and
  binding-level `DatasetPortSpec.output_dimensions` with it on 2026-08-31
  (`1a5df7a5`), as specified in
  [dimension constraints](../architecture/dimension-constraints.md).
- Edge iteration order is preserved through stored per-port `position`
  (snapshot v9) rather than declaration accident; snapshot production is the
  single ordering authority (`ordered_binding_snapshots()`).

Remaining:

- Move `_get_output_for_node()`'s node-column row filter and metric
  selection into the executor (multi-metric retention needs an op or an
  extended `select_metric`; the executor currently narrows to one column).
  `Node._get_output_for_target()` still runs `_get_output_for_node()` before
  handing the frame to `apply_port_transformations()`.
- Make the typed ops primary all the way down and delete `EdgeDimension`.
  It survives as a runtime intermediate that
  `_edge_dimensions_from_transforms()` reconstitutes from the ops on every
  load; the output-dimension assertion still reads `edge.to_dimensions`
  rather than declared port dimensions.
- Retire the legacy multiplicative role inference in `InstanceGraph`
  (`inferred_port_role` / `unclassified_port_role`). This is not a delete:
  the parser only persists roles for multi groups, so single-binding legacy
  ports need parser-side role assignment first. Fleet-wide as of 2026-09-15:
  5,162 inferred and 4,489 unclassified diagnostics across 70 buildable
  instances.

#### The `get_input(port)` node accessor

The canonical design and migration sequence now live in
[node-input-port-runtime-migration.md](node-input-port-runtime-migration.md),
which is **in progress**: the runtime foundation, the pilots and the first
bulk conversion wave landed 2026-08-27. `RuntimeInputBinding`,
`Node.get_input()` / `iter_inputs()` / `require_input()` and the declaration
constructors all exist, and the loader attaches graph-derived bindings to
every node. The class migration is early — roughly 24 `get_input()` call
sites against 57 `get_input_dataset*` and 95 `get_output_pl()` — and step 7
there (removing the legacy `Edge` grouping and `input_dataset_instances`) has
not started. Until it does, the loader builds the legacy edge/dataset views
independently of the runtime registry instead of projecting them from it.

In brief, node classes address semantic class declarations rather than UUIDs;
dataset-backed and edge-backed bindings deliver the same one-metric port
value; requiredness comes from port cardinality rather than a per-call
boolean; and binding transformations execute exactly once at the input
boundary.

Multiplicity does not by itself prescribe addition. A multi-port exposes
ordered homogeneous binding values, while an optional declared aggregation
allows `get_input(port)` to return one combined value. Repeatable roles create
distinct heterogeneous ports. This distinction is needed for additive sums,
multiplicative factors, ordered imputation, and future authored pipeline
operations.

The migration remains incremental. A temporary `RuntimeInputBinding` built
from the graph Def retains binding identity and source-specific resolution;
legacy node/dataset accessors become compatibility projections of that same
registry while node classes migrate by semantic pattern. After
`InstanceGraph` becomes the Context factory, revisit moving runtime
instantiation behavior closer to `PortBindingDef` and its source-specific
subclasses.

#### Port identity required for later three-way sync

Use structural keys rather than list positions or node identifiers:

- edge port: source node UUID + metric identifier;
- dataset port: dataset identifier + column;
- explicit authored port UUID wins when present;
- otherwise preserve the stored UUID during sync;
- only then derive a deterministic UUID.

Dataset binding identity must not depend on `input_datasets` order, and node
renames must not change edge identity.

Landed 2026-08-13 (`7c917d6f`) and hardened by the step-9 authority flip:
`match_preserved_uuids()` in `instance_serialization.py` does multi-pass
ordered matching (exact structural key first, then a loose fallback so edges
survive port changes and dataset ports survive index reordering), and every
sync writer carries matched UUIDs across a rewrite. Rebinding to a different
dataset mints a new identity by design. `reconcile_input_bindings()` then
replaced delete-and-recreate entirely, so pk and uuid both survive a re-sync.
This unblocks steps 4 and 5.

### 2. Build the runtime directly from `InstanceSnapshot`

**Done** (instance-graph step 10, stages 1–3, 2026-08-13 → 2026-08-16).
`from_snapshot()` initializes instance identity, languages, years, features,
dimensions, scenarios and action groups from `InstanceMetadata` +
`InstanceModelSpec`; instantiates nodes from `NodeSnapshot` + `NodeSpec`;
resolves node references by UUID; constructs edges and dataset bindings from
their typed snapshot forms; and binds the selected snapshot during
construction rather than as a corrective overlay. Dataset payload selection
stays an injected concern (`CurrentDatasetPayloadStore` versus
`RevisionDatasetPayloadStore`). Each stage was gated on old-vs-new parity
across all buildable DB instances — structure plus computed outputs at
rel_tol 1e-9.

`NodeSpec.extra` remains the one deliberate attic: `historical_values`,
`forecast_values`, `input_dataset_processors`, `tags`, and an `other`
catch-all. Each still needs either a typed runtime consumer or an explicit
legacy adapter; the desired end state is an empty class.

Not yet done, and tracked in the instance-graph plan rather than here:
`InstanceGraph.create_context()` and building nodes/edges *from the graph*
rather than alongside it.

### 3. Move YAML serving onto the common path and remove the shim

**Done** (stage 4, `5796d2b3` + `82387590`, 2026-08-16, −690 lines).
`from_yaml()` parses into `InstanceSnapshot` and delegates; the loader now
*requires* a snapshot. `snapshot_to_config_dict()`, `serialize_instance_to_dict()`,
`_init_instance()`, `make_node()`, `from_dict_config()`, emission-sector
generation, `Edge.from_config()` and `EdgeDimension.from_config()` are all
deleted. The framework carve-out dissolved with it: the loader's `fw_config`
parameter is gone, replaced by a `snapshot_transform` overlay, a
`Node.uses_framework_measure_data` ClassVar, and lazily derived
measure-datapoint years.

No narrowly named legacy YAML conversion entry point was needed. The last
dict consumer, `NodeExplanationSystem`, was converted to typed inputs on
2026-09-15 (`ddc2b42d`) and `snapshot_nodes_to_config_dicts` went with it.

**The architectural inversion is complete as of 2026-08-16.**

### 4. Normalize YAML edge declarations to the target node

Not started. YAML still lets an edge be authored on either endpoint: the
source's `output_nodes` or the target's `input_nodes`. Re-counted
2026-09-21, the repo configs hold roughly 3,700 source-side entries (about
450 bare node references, the rest dicts) against roughly 4,300 target-side
ones — the same order as the 2026-08-06 count, so the codemod has not been
outgrown. The binding — tags, `from_dimensions`/`to_dimensions`, metric
selection, and after step 1 the typed transformation pipeline — is a
property of the target's input port regardless of where it was declared,
so the authored form should live there too.

- Codemod the repo configs: rewrite each `output_nodes` entry as an
  `input_nodes` entry on the referenced node. Edge attributes move
  verbatim — they are defined relative to edge *direction*, not
  declaration side.
- Sequence after step 1's structural port identity, so declaration side
  cannot affect binding identity and the codemod is a structural no-op.
  `parse_oracle` is retired, so verify with `sync_instance_to_db --dry-run`
  snapshot equality instead.
- Ordering hazard: edges are created in declaration order and additive
  summation exposes float association order. The codemod must preserve the
  resulting per-target order (insert moved entries at the position the
  source-side declaration produced), and the change is gated by full
  `test_instance --compare` regardless. The `position` backfill of
  instance-graph step 9 has since landed, so per-port position is now
  stored and authoritative — the codemod must leave those positions
  unchanged rather than re-derive them.
- The parser keeps accepting `output_nodes` at the YAML compatibility
  boundary; the codemod plus a lint/CI nudge stop new source-side authoring.
  Open question: whether to keep attribute-less `output_nodes` (a bare node
  reference) as a permanent shorthand — actions declaring "I feed
  `net_emissions`" is an ergonomic idiom, and without attributes there is
  no binding content to misplace. Decide when the codemod is written.

### 5. Add three-way YAML/DB coexistence

Not started as three-way merge. The artifact, structured preview,
transactional apply, API, and staged three-way design live in
[`instance-export-sync-preview.md`](instance-export-sync-preview.md), which has
moved independently: the `load_nodes --update-nodes` compatibility slice,
`InstanceExport` documents over GraphQL (`ab3bda15`), and the packaged
devtool client with SSO/export/import (`9793d9d3`). None of that is the
merge itself. The summary below records this plan's dependency on stable
snapshot identity, which step 1 has now supplied.

Use kubectl-apply semantics:

- base: last successfully synced snapshot;
- incoming: newly parsed YAML snapshot;
- current: DB draft snapshot;
- apply fields changed only by YAML;
- preserve DB-only edits;
- report fields changed on both sides and fail unless an explicit conflict
  policy is supplied.

Required pieces:

- one-shot UUID stamping for existing YAML nodes, then UUIDs for newly authored
  nodes going forward;
- persisted `last_synced_snapshot` (or an equivalent immutable base revision);
- field-level diff/apply over the snapshot tree;
- explicit `--prefer db|yaml` conflict handling;
- YAML affordances for authored ports while retaining the current shorthand;
- eventually, dataset-content merge using revision pins as the common base.

Until this exists, `sync_instance_to_db` remains a replace-style computation
sync — it has no `--prefer`, no `last_synced_snapshot`, and no field-level
diff. A clobber guard based on change operations newer than the last sync is
still desirable. Note that binding *identity* now survives a re-sync even
though content is replaced, so the base-snapshot machinery no longer has to
re-establish identity first.

### 6. Optional follow-up: snapshot-native editor reads

Not started; still optional.

Non-editor GraphQL already reads model content from the runtime's selected
snapshot. Editor queries still use Strawberry-Django ORM types because they
need permission policies, governance fields, optimizer integration, and rows
that mutations can update.

Rewriting editor reads onto snapshots could simplify revision browsing and
eliminate ORM query classes, but it is no longer required to close draft versus
published leakage. Treat it as a separate measured refactor:

- snapshot serves model content;
- ORM serves governance (`is_stale`, permissions, audit history,
  publication/lock state, `created_by`);
- mutations continue writing rows and rebuild the returned snapshot view at a
  defined consistency seam.

## Work deliberately tracked elsewhere

Publish/revert/undo/named-draft mutations and replaying
`InstanceModelLogEntry.before` images belong to
[the draft/publish/revisions plan](../trailhead/plans/draft-publish-revisions.md).
Dataset revision closure now provides the immutable data substrate those
operations need, but loader inversion does not itself implement them.

The disabled post-edit lint hook is also a separate tooling track and no
longer belongs in this plan.

## Transitional behavior to retire

- ~~`snapshot_to_config_dict()` between snapshot selection and runtime
  build~~ (deleted 2026-08-16).
- ~~Direct YAML-dict runtime construction~~ (deleted 2026-08-16).
- ~~Draft live-row dataset fallback when a materialization is missing or
  stale~~ (gone; both payload stores now raise on a missing payload, and
  `ensure_dataset_materializations()` repairs stale rows at the boundary).
- Compatibility hydration for legacy structured revisions before snapshot v6,
  and the `from_serialized_data` upgraders for v3/v4/v9/v10/v11; republishing
  is the migration path, and removal waits on the supported revision window.
- `NodeSpec.extra` as a typed attic for runtime fields not yet modeled at their
  final boundary (five fields remain).
- `EdgeDimension` as a runtime intermediate reconstituted from the typed ops
  on every load.
- The dict-fragment loader helpers `_make_node_datasets`, `_make_node_params`
  and `_make_node_visualizations`.
- Legacy multiplicative role inference in `InstanceGraph`.

## Verification gates

For every phase:

- focused unit tests for the affected typed boundary;
- full `pytest --reuse-db`;
- full `mypy .`;
- `ruff check .` and `git diff --check`;
- for parse/sync changes: `sync_instance_to_db --dry-run` plus a double-sync
  idempotency check on a YAML-backed instance (pks *and* uuids stable).
  `tools/parse_oracle.py` and `nodes/spec_export.py` were retired on
  2026-09-15 — the parser is the only spec producer now, so there is no
  second implementation left to diff against;
- `export_schema` plus schema diff for GraphQL changes;
- full `test_instance --state-dir model-outputs/ --compare` for computation or
  ordering changes.

Loader-specific assertions:

- the same snapshot produces equivalent draft and published runtime graphs;
- published runtime construction and computation query no live dataset,
  datapoint, metric, or dimension-category rows;
- draft/published payload loading stays one bulk query on first access and zero
  thereafter;
- metadata-only GraphQL queries do not fetch payload JSON;
- repeated bindings decode each distinct dataset payload once;
- published output remains unchanged after further draft graph and dataset
  edits, until republished.

## Practical notes

- Do not add `from __future__ import annotations` to Strawberry modules.
- `NodeConfig.objects` defers `spec`; use `.with_spec()` where the spec is
  required.
- `NodeConfig.spec` writes use `queryset.update()` because
  `ClusterableModel.save()` can restore stale schema-field values.
- Format before committing: the pre-commit Ruff formatter intentionally stops
  the first commit attempt if it changes a file.
- Snapshot schema is currently v11 (`SNAPSHOT_SCHEMA_VERSION` in
  `nodes/instance_serialization.py`). Bump it only for a snapshot contract
  change, not for an internal loader refactor that accepts the existing shape.
- A reused test database can hold committed rows from a previous run; use
  `--create-db` before blaming code for a mass of unexplained errors.
- Graduate durable decisions to `docs/architecture/`; keep this file as the
  working sequence and state ledger.
