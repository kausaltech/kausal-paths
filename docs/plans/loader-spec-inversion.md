# Loader inversion: specs as the load-time contract

*Status: snapshot serving and dataset revision closure COMPLETE; direct
snapshot-to-runtime construction NOT STARTED. Updated 2026-08-06 after
`54fa1dd3` and `e70b70ad` landed.*

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
| Parse-only YAML sync | Complete | YAML parses directly into `InstanceSnapshot`; `sync_instance_to_db` writes it through `nodes/spec_sync.py`. The runtime-introspection exporter remains behind `--runtime-export`. |
| Node identity/computation split | Complete | `NodeConfig` owns identity/display metadata; `NodeSpec` owns computation. Snapshot schema v4 and migration/bootstrap landed in `00e31e3a`. |
| Stable snapshot references | Complete | Node references use UUIDs; shared editor layout is snapshot-backed. |
| Snapshot-backed public GraphQL | Complete | Draft and published runtimes bind their selected `InstanceSnapshot`/`NodeSnapshot`; non-editor GraphQL reads model content from those snapshots. Landed in `54fa1dd3`. |
| Revisioned instance content | Complete | Snapshot v6 includes instance lead content and node body, removing the remaining structured-revision metadata leaks for newly published revisions. |
| Dataset revision closure | Complete | Snapshot v7 pins immutable dataset revisions. Draft reads use current materializations; published reads use pinned revision payloads. Landed in `e70b70ad`. |
| Dataset writer coverage | Complete for active writers | GraphQL, REST, Wagtail, snapshot import, DVC import, repair commands, metadata fan-out, and forecast-default promotion refresh materializations at explicit atomic boundaries. Unused DiffSync commands remain intentionally out of scope. |

### The remaining inversion seam

`InstanceLoader.from_snapshot()` is not yet a native snapshot loader. It calls
`snapshot_to_config_dict()` and then enters the old YAML-dict constructor:

```text
InstanceSnapshot
  -> nodes/instance_from_db.py:snapshot_to_config_dict
  -> InstanceLoader(config=dict)
  -> legacy _init_instance path
```

This shim is now private and contained, but it is still load-bearing. Node and
edge construction, dimensions, parameters, scenarios, and several legacy node
fields are still interpreted from the reconstructed dict.

### Current read paths

#### DB draft

1. `build_instance_snapshot(InstanceConfig)` reads the current editor tables.
2. `InstanceLoader.from_snapshot(snapshot)` builds the runtime through the
   compatibility shim.
3. The runtime binds the source snapshot to the instance and nodes.
4. DB datasets resolve through `DatasetMaterialization` and the lazy
   `CurrentDatasetPayloadStore`.
5. Missing or timestamp-stale materializations still use an explicit
   transitional live-row fallback.

#### Published

1. The live Wagtail instance revision supplies `InstanceSnapshot`.
2. Snapshot v7's dataset manifest is checked against relational
   `InstanceRevisionDatasetPin` rows.
3. `InstanceLoader.from_snapshot(snapshot, published=True)` builds the same
   runtime through the compatibility shim.
4. DB datasets resolve lazily from immutable pinned Wagtail revisions through
   `RevisionDatasetPayloadStore`.
5. Pre-v7 revisions retain a boundary-only compatibility path; republishing
   upgrades them.

#### YAML

Direct YAML serving still uses `InstanceLoader.from_yaml()` and the legacy
dict path. YAML-to-DB sync does not: it parses straight into a snapshot.

## Remaining work

The dependency chain is:

```text
1. Runtime ports/transformations (incl. the get_input(port) accessor)
          -> 2. Native snapshot loader
          -> 3. Remove the dict reconstruction shim

4. YAML edge-declaration normalization depends on structural port identity
from step 1 (so declaration side cannot affect binding identity).

5. Three-way YAML/DB sync depends on stable structural identity from step 1
and is simplified by step 4, but can otherwise proceed independently of
steps 2-3.
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

Remaining:

- Move `_get_output_for_node()`'s node-column row filter and metric
  selection into the executor (multi-metric retention needs an op or an
  extended `select_metric`; the executor currently narrows to one column).
- Construct `Edge` from the typed binding (ops primary), then retire
  `EdgeDimension` and `Edge.from_config()`. The output-dimension assertion
  moves from `edge.to_dimensions` to declared port dimensions — landing
  early as step 2 of the
  [instance-graph plan](instance-graph-dimension-constraints.md), which
  also retires the `flatten` placeholder.
- Retire binding-level `DatasetPortSpec.output_dimensions` as specified in
  [dimension constraints](../architecture/dimension-constraints.md).
- Preserve edge iteration order: additive summation currently makes floating
  point association order observable in regression outputs.

#### The `get_input(port)` node accessor

Node computation code currently pulls its inputs through two ad-hoc
families: `self.get_input_dataset*()` (~80 call sites) and
`other_node.get_output_pl(target=self)` (~100 call sites), with selection
by tag or quantity and per-call `required=` booleans. Neither knows about
ports; the edge/dataset distinction and the transformation execution leak
into every node class.

Introduce one accessor as the node-facing side of the edge boundary:

- `self.get_input(port)` addresses one declared input port and returns the
  bound input with the binding's `apply_port_transformations()` pipeline
  already applied. Multi-ports (`InputPortDef.multi`) return the bound
  inputs as a list; arity and required-ness come from the port definition,
  not per-call-site arguments.
- Dataset-backed and edge-backed bindings are indistinguishable to the
  consuming node. The port is the abstraction; where the data comes from
  is the binding's business.
- Node *classes* declare their input ports (the way they already declare
  `output_metrics` and `allowed_parameters`); YAML/DB bindings attach to
  declared ports. `InputPortDef.identifier` is optional today because
  synced ports often have no meaningful name — migrating a node class to
  `get_input()` is the moment its ports gain identifiers. This typed
  class-level declaration is the same one the
  [instance-graph plan](instance-graph-dimension-constraints.md)'s
  `shape_rules()` resolves roles against: one role namespace for
  computation and constraint rules.
- This is where fault-tolerance's skip-don't-sum logic generalizes: move
  it out of `add_nodes_pl` into `get_input()` and delete the
  special-casing (see
  [fault-tolerance.md](../architecture/fault-tolerance.md), Deferred).

Migration is incremental: the accessor lands with the transformation
executor; the legacy accessors become compatibility wrappers over ports;
call sites migrate class by class (generic/simple first, region-specific
modules last). Tag- and quantity-based selection maps onto port
identifiers.

#### Port identity required for later three-way sync

Use structural keys rather than list positions or node identifiers:

- edge port: source node UUID + metric identifier;
- dataset port: dataset identifier + column;
- explicit authored port UUID wins when present;
- otherwise preserve the stored UUID during sync;
- only then derive a deterministic UUID.

Dataset binding identity must not depend on `input_datasets` order, and node
renames must not change edge identity.

### 2. Build the runtime directly from `InstanceSnapshot`

Introduce a native construction path whose input is the typed snapshot, not a
reconstructed config dict. This step is delivered as step 10 of the
[instance-graph plan](instance-graph-dimension-constraints.md): the loader
consumes the `InstanceGraph` built from the snapshot rather than growing a
second snapshot-reading path — the two plans describe one work item here.

- Initialize instance identity, languages, years, features, dimensions,
  scenarios, and action groups from `InstanceMetadata` and
  `InstanceModelSpec`.
- Instantiate nodes from `NodeSnapshot` + computation-only `NodeSpec`.
- Resolve node references by UUID at the boundary; identifiers remain runtime
  labels, not durable graph identity.
- Construct runtime edges and dataset bindings from their typed snapshot
  forms.
- Carry `NodeSpec.extra` fields deliberately. Each field must either gain a
  typed runtime consumer or remain in one explicit legacy adapter; do not
  spread dict fallbacks into the new path.
- Keep framework-specific initialization working without reintroducing live
  draft reads into a published runtime.
- Bind the selected snapshots during construction rather than as a corrective
  overlay afterward.

The existing `from_snapshot()` entry point should become the common path for
both draft and published snapshots. Dataset payload selection remains an
injected concern (`CurrentDatasetPayloadStore` versus
`RevisionDatasetPayloadStore`).

### 3. Move YAML serving onto the common path and remove the shim

Once the native loader has parity:

- make `from_yaml()` parse YAML into `InstanceSnapshot` and delegate to
  `from_snapshot()`;
- remove `snapshot_to_config_dict()` from normal loading;
- retain any unavoidable legacy YAML conversion behind a narrowly named
  compatibility entry point;
- remove obsolete config-dict branches from `_init_instance()` and node/edge
  construction.

This is the point at which the architectural inversion is complete.

### 4. Normalize YAML edge declarations to the target node

YAML currently lets an edge be authored on either endpoint: the source's
`output_nodes` or the target's `input_nodes`. As of 2026-08-06 the repo
configs hold ~3,900 source-side declarations (849 bare strings, 704 plain
dicts, 2,323 carrying tags/dimensions/metrics) against ~5,200 target-side
ones. The binding — tags, `from_dimensions`/`to_dimensions`, metric
selection, and after step 1 the typed transformation pipeline — is a
property of the target's input port regardless of where it was declared,
so the authored form should live there too.

- Codemod the repo configs: rewrite each `output_nodes` entry as an
  `input_nodes` entry on the referenced node. Edge attributes move
  verbatim — they are defined relative to edge *direction*, not
  declaration side.
- Sequence after step 1's structural port identity, so declaration side
  cannot affect binding identity and the codemod is a structural no-op,
  verifiable with `parse_oracle` snapshot equality.
- Ordering hazard: `_build_edges()` creates edges in declaration order,
  and additive summation exposes float association order. The codemod must
  preserve the resulting per-target edge order (insert moved entries at
  the position the source-side declaration produced), and the change is
  gated by full `test_instance --compare` regardless. Coordinate with the
  `position` backfill in the
  [instance-graph plan](instance-graph-dimension-constraints.md) step 9 —
  both reassign observed binding order.
- The parser keeps accepting `output_nodes` at the YAML compatibility
  boundary; the codemod plus a lint/CI nudge stop new source-side authoring.
  Open question: whether to keep attribute-less `output_nodes` (a bare node
  reference) as a permanent shorthand — actions declaring "I feed
  `net_emissions`" is an ergonomic idiom, and without attributes there is
  no binding content to misplace. Decide when the codemod is written.

### 5. Add three-way YAML/DB coexistence

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
sync. A clobber guard based on change operations newer than the last sync is
still desirable.

### 6. Optional follow-up: snapshot-native editor reads

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

- `snapshot_to_config_dict()` between snapshot selection and runtime build.
- Direct YAML-dict runtime construction.
- Draft live-row dataset fallback when a materialization is missing or stale;
  remove after production backfill and monitoring establish the invariant.
- Compatibility hydration for legacy structured revisions before snapshot v6,
  and legacy config-dict revisions; republishing is the migration path.
- `NodeSpec.extra` as a typed attic for runtime fields not yet modeled at their
  final boundary.

## Verification gates

For every phase:

- focused unit tests for the affected typed boundary;
- full `pytest --reuse-db`;
- full `mypy .`;
- `ruff check .` and `git diff --check`;
- `tools/parse_oracle.py` for parse/sync changes (`--refresh` when row UUIDs
  changed);
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
- Snapshot schema is currently v7. Bump it only for a snapshot contract change,
  not for an internal loader refactor that accepts the existing shape.
- Graduate durable decisions to `docs/architecture/`; keep this file as the
  working sequence and state ledger.
