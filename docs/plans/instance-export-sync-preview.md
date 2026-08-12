# InstanceExport-native model sync and preview

*Status: local `load_nodes` compatibility slice implemented; generic graph,
dataset-content, and API apply remain proposed. Written 2026-08-12.*

## Implemented compatibility slice

The first local slice now routes YAML-backed
`load_nodes.py --update-nodes` through `InstanceExport`:

- `compile_instance_export_from_yaml()` parses the raw YAML graph while using
  existing node and port catalogs only for stable identity. It does not
  reconcile `NodeConfig` display metadata into the export.
- `plan_load_nodes_instance_export_sync()` produces a structured, read-only
  NodeConfig and legacy NodeDataset diff using the existing `--overwrite`,
  `--skip-descriptions`, and `--delete-stale-nodes` meanings.
- `apply_load_nodes_instance_export_sync()` applies that plan atomically and
  leaves `config_source` unchanged.
- `load_nodes.py --update-nodes --dry-run` prints the structured JSON plan and
  exits before runtime initialization.
- YAML serving projects the reconciled `NodeConfig` display metadata onto the
  runtime node as well as its source snapshot, so metrics and direct GraphQL
  fields use the same effective values.

This slice intentionally leaves `InstanceExport.datasets` empty. Dataset-port
references remain in `InstanceExport.instance`, and legacy `NodeDataset`
relations to already-existing target DB datasets are synchronized. It does
not yet create/update dataset bodies, replace the generic
`sync_instance_to_db` writer, persist apply-plan hashes/change operations, or
provide standalone compile/apply management commands.

## Goal

Make `InstanceExport` the source-neutral contract for moving an authored
model, including its datasets, into the database representation associated
with an `InstanceConfig`:

```text
YAML ──compile──┐
                │
DB/export ──────┼──> export artifact ──plan──> structured diff ──apply──> DB draft
                │                            (read-only)          (atomic)
other tooling ──┘
```

This replaces the runtime-derived `load_nodes.py --update-nodes` workflow
with an explicit compilation, preview, and apply pipeline. The same planner
and applier must serve management commands, a future CLI, and the API.

The immediate use case is a model developer who changes YAML, previews the
semantic changes against an instance, and then deliberately applies them.
The design must also support exports produced by other tools without
giving YAML special semantics inside the persistence layer.

## Non-goals

- Do not make the runtime `Node` a synchronization transport.
- Do not upload raw YAML to the server. Includes and repository-relative
  files are resolved where the source tree exists; the portable artifact is
  the compiled export.
- Do not require every producer to materialize every external dataset body.
  An export may carry a dataset by value, carry an external reference, or
  leave a target-owned dataset outside its apply scope. These states must not
  be conflated.
- Do not implement automatic three-way merge in the first milestone. The
  planner must expose conflicts and retain enough provenance to add it later.
- Do not make applying an export publish it. Apply updates the draft;
  publication remains a separate validated operation.

## Current state

The main pieces already exist, but their responsibilities are mixed.

### YAML parsing

`nodes.instance_parser.parse_instance_snapshot()` parses merged YAML into the
`InstanceSnapshot` graph portion without initializing the computational
runtime.

Node UUID selection currently has this precedence:

1. an explicitly authored YAML UUID;
2. a UUID supplied by the target's identifier-to-UUID catalog;
3. `uuid3(instance_uuid, node_identifier)`.

Output ports, dataset ports, input-port groups, and edge-side identities also
have deterministic UUIDv3 fallbacks. Existing port-reference catalogs take
precedence where needed to preserve identities created before the current
derivation rules.

The fallback is deterministic, but it is not rename-stable: changing an
identifier changes a derived UUID. Existing target catalogs bridge legacy
renames; explicitly authored UUIDs are the durable contract. Compilation
therefore needs a stable instance UUID and may optionally use a target
identity catalog. It does not need runtime initialization.

### YAML-to-DB sync

`nodes.spec_sync.sync_parsed_instance_to_db()` currently:

1. loads and merges YAML;
2. obtains the target instance UUID, node UUIDs, and port-reference catalog;
3. parses an `InstanceSnapshot`;
4. reconciles existing `NodeConfig` metadata into that snapshot;
5. writes instance spec, dimensions, nodes, edges, placeholders, and dataset
   ports;
6. changes the instance to `config_source='database'`.

Step 4 means the intermediate snapshot no longer faithfully represents the
YAML. The write functions also embed preservation and stale-row policy rather
than accepting them explicitly, and dataset handling is limited to target-side
placeholders and binding resolution rather than a portable dataset payload.

The management command's current `--dry-run` executes the writes inside a
transaction and rolls the transaction back. It reports aggregate counts, not
a structured semantic diff.

### Runtime serving

YAML serving initializes runtime nodes from YAML and subsequently attaches
metadata reconciled from `NodeConfig`. Snapshot-backed GraphQL fields can
therefore see a different name from runtime consumers such as metric
generation. Serving needs one effective snapshot rather than separate runtime
and resolver views of metadata. That correction is related to this plan but
must not turn runtime nodes into a store for pre-reconciliation YAML values.

### Import/export

`build_instance_snapshot()` serializes the structural model. `export_instance()`
wraps that snapshot in an `InstanceExport` whose `datasets` contain dataset
schema, metrics, dimensions, datapoints, data sources, source references, and
comments. External placeholders carry their external reference and no body.

`import_instance()` populates a new instance, while the functions in
`nodes/spec_sync.py` can update existing graph rows. Dataset import is
creation-oriented: `import_instance_datasets()` preserves an existing
populated dataset rather than synchronizing changes into it. There is no
single generic, policy-driven apply service for an existing draft and its
datasets.

## Load-bearing decisions

### 1. Compilation produces a raw InstanceExport

Introduce a public compilation service along these lines:

```python
def compile_yaml_export(
    yaml_path: Path,
    *,
    instance_uuid: UUID,
    identity_catalog: SnapshotIdentityCatalog | None = None,
    dataset_mode: DatasetExportMode = DatasetExportMode.REFERENCES,
) -> InstanceSyncArtifact: ...
```

It loads includes, parses the merged configuration, and returns YAML-authored
values unchanged in `InstanceExport.instance`. It does not query `NodeConfig`,
reconcile target metadata, or construct runtime nodes.

The compiler also constructs `InstanceExport.datasets` according to the
producer's actual dataset authority:

- `OMIT`: the producer makes no claim about dataset objects or bodies. This is
  appropriate for target-authored DB datasets referenced by YAML.
- `REFERENCES`: external/DVC datasets are represented by schema metadata and
  immutable repository/dataset references, without copying their bodies.
- `CONTENTS`: materialize selected datasets into `DatasetSnapshot.data`, with
  their source references and comments where the producer owns them.

The mode is not a global assertion that all datasets have the same treatment.
The artifact records per-dataset authority. A YAML model using both DVC data
and target-authored DB data can carry the former by reference and leave the
latter out of scope. An export from a DB instance can carry its owned datasets
by value.

`SnapshotIdentityCatalog` contains only stable identity information needed at
the compilation boundary. It must not carry display metadata. Initially it
contains existing node UUIDs and port-reference identities. A future API may
return this catalog independently so local tools can compile against a remote
target.

### 2. Values and write authority are separate

A bare `InstanceExport` describes values but not which absent values the
producer intends to clear. This matters because YAML cannot author every
snapshot field or every dataset. For example, parse-side snapshots do not
carry an admin-authored `NodeConfig.body`, and a YAML model configured to use
DB datasets does not own those datasets merely because its ports reference
them.

Transport the export in an envelope:

```python
class InstanceSyncArtifact(BaseModel):
    artifact_version: int
    export: InstanceExport
    apply_scope: InstanceExportApplyScope
    provenance: InstanceExportProvenance
```

`InstanceExportApplyScope` declares the entity sets and fields for which this
artifact is authoritative. It is data, not an inferred check on
`provenance.kind`.

The YAML compiler's scope includes YAML-expressible instance spec, graph
structure, bindings, dimensions, node display fields, and only the datasets
it actually exports. It excludes admin-only content, target-owned datasets,
and governance fields. A DB export can declare a broader scope. An external
producer must state its scope explicitly.

For every scoped optional field, the artifact must distinguish:

- field not in scope: preserve the target;
- field in scope with a value: set the value;
- field in scope with an explicit null/empty value: clear the target.

Never overwrite organization, permissions, hostnames, ownership relations,
creation metadata, or other deployment governance through this model-sync
artifact.

`InstanceExport.pages` remains outside the initial apply scope. It currently
exists for verification and cloning support, and applying Wagtail page trees
has different ownership and publication semantics.

### 3. Planning is a pure, read-only operation

Introduce:

```python
def plan_instance_export_apply(
    current: TargetExportState,
    incoming: InstanceSyncArtifact,
    *,
    policy: InstanceExportApplyPolicy,
    source_transition: SourceTransitionPolicy,
    target_catalog: TargetCatalog,
    draft_head_token: UUID | None,
) -> InstanceExportApplyPlan: ...
```

Planning validates and resolves the incoming artifact against a captured
target state, but performs no writes. Dataset binding resolution must gain a
read-only planning form so missing datasets or metrics become diagnostics
rather than errors discovered halfway through application.

The plan is the authoritative description of the proposed transaction. It
contains:

- hashes of the incoming artifact and current target export state;
- the observed draft head token;
- the selected policy and apply scope;
- the selected source transition;
- entity-aware changes;
- preserved incoming values;
- conflicts, errors, and warnings;
- aggregate counts;
- the normalized desired export or equivalent resolved operations used by
  the applier.

The planner must be deterministic: identical current state, artifact, catalog,
and policy produce an identical plan and plan hash.

### 4. Diff by entity identity, then by field

Do not expose a raw positional diff of serialized JSON arrays. Match entities
first and report field changes within them.

Initial identities:

| Entity | Matching identity |
| --- | --- |
| Instance | instance UUID |
| Node | node UUID; identifier fallback only for legacy adoption |
| Node port | port UUID |
| Edge | resolved endpoint node/port tuple until edges gain their own UUID |
| Dataset binding | dataset-port UUID |
| Dimension | scoped dimension identifier/UUID contract |
| Dimension category | parent dimension + category identifier/UUID contract |
| Dataset reference | dataset UUID where available, otherwise identifier |
| Data point | `DataPointKey`: year + metric + sorted dimension categories |
| Data source | data-source UUID |

List order is reported as an order-field or ordering change, not as delete and
create churn. Computation specs receive semantic field paths after their node
has been matched.

Each field change reports at least:

```json
{
  "entityType": "node",
  "entityUuid": "...",
  "identifier": "private_household_emissions",
  "operation": "update",
  "fields": [
    {
      "path": "name.de",
      "before": "Private Haushalte Emissionen",
      "incoming": "Private Haushalte",
      "after": "Private Haushalte",
      "resolution": "incoming"
    }
  ]
}
```

Keeping `incoming` distinct from `after` makes preservation visible:

```json
{
  "before": "Database-authored name",
  "incoming": "YAML name",
  "after": "Database-authored name",
  "resolution": "preserve-target"
}
```

Dataset changes are divided into schema, payload, and provenance:

- schema: name, time resolution, dimensions/column names, metrics and units;
- payload: datapoints matched by `DataPointKey`;
- provenance: data sources, source references, and comments.

The preview summary reports payload row counts, hashes, and aggregate
create/update/delete counts. It may return a bounded sample or paginated
detail, but must not inline an unbounded dataset body into a GraphQL response.
The server-side plan retains the normalized incoming payload needed for apply.
Secrets must never appear in artifact or diff output.

Large graph values such as pipelines or dimension catalogs may likewise be
summarized in human output, while the machine-readable detail retains precise
structured changes.

### 5. Apply executes a validated plan atomically

Introduce one application service used by every entry point:

```python
def apply_instance_export_plan(
    ic: InstanceConfig,
    plan: InstanceExportApplyPlan,
    *,
    actor: User | None,
    source: InstanceChangeSource,
) -> InstanceExportApplyResult: ...
```

Inside one transaction it:

1. locks the `InstanceConfig`;
2. verifies the current draft head and target export hash still match the
   plan;
3. rejects plans containing unresolved errors or conflicts;
4. writes the exact resolved desired state represented by the plan;
5. records one `InstanceChangeOperation` such as `instance_export.apply`, with
   row-level log entries for its changes;
6. invalidates derived graph/runtime caches;
7. returns the new draft head and an applied summary.

The target export hash is required in addition to `draft_head_token`: existing
CLI, admin, or legacy writes may not yet create change operations. The head
token becomes sufficient only after all writers participate in change
tracking.

Application must not independently reinterpret source precedence. If target
state has changed, it fails with a stale-preview result and requires a new
plan.

For an API flow the server may recompute a plan from the submitted artifact
and precondition hashes rather than trusting client-supplied resolved
operations. The returned `planHash` is a review handle, not authorization to
execute arbitrary operations.

### 6. Apply policy is explicit

The first implementation supports two policies:

- `PRESERVE_TARGET`: incoming structural/computation fields apply, while
  existing target-authored display metadata is preserved according to the
  field policy. This retains current `sync_instance_to_db` behavior.
- `REPLACE_SCOPED`: every field in the artifact's apply scope takes the
  incoming value, including explicit clearing. This replaces
  `load_nodes.py --update-nodes --overwrite` for YAML-authored fields.

The policy cannot expand `apply_scope`; it only decides what happens where the
artifact has authority.

Later add `THREE_WAY`, using the last successfully applied source export as
the base:

```text
base     = last applied artifact from this synchronization lineage
incoming = newly compiled artifact
current  = current DB draft
```

Changes made only in incoming apply, changes made only in current survive,
and overlapping changes become structured conflicts unless the caller chooses
an explicit resolution. Store source lineage and artifact hash when an apply
succeeds; do not assume the most recent published revision is the sync base.

### 7. Source transition is independent of field precedence

Applying an artifact must not silently change which source constructs the
runtime. Add an explicit `SourceTransitionPolicy`:

- `KEEP_CURRENT`: update the DB representation but retain the current
  `config_source`. For a YAML-sourced instance, this replaces the
  `load_nodes.py --update-nodes` metadata/mirror workflow; runtime computation
  continues to come from YAML and persisted `NodeConfig` metadata remains its
  overlay.
- `ADOPT_DATABASE`: apply the complete scoped model state and set
  `config_source='database'`. This is the current `sync_instance_to_db`
  transition.

Field precedence and source transition are orthogonal. For example, a model
developer can use `REPLACE_SCOPED + KEEP_CURRENT` to push YAML-authored names
into `NodeConfig` without changing runtime computation ownership. A migration
can use `PRESERVE_TARGET + ADOPT_DATABASE` to preserve curated DB metadata
while moving computation into the database draft.

The plan and preview must state the resulting `config_source` prominently.
`ADOPT_DATABASE` is rejected unless the artifact scope is sufficient to build
a complete database runtime. `KEEP_CURRENT` must still keep the structural DB
mirror internally consistent even where those rows are dormant under YAML
serving.

### 8. Dataset application respects ownership and revision boundaries

`DatasetSnapshot` is currently optimized for clone/import, not repeated sync.
Before using it as a synchronization payload:

- add a durable dataset UUID to `DatasetSnapshot` (with identifier fallback
  only for legacy artifacts), aligned with `InstanceSnapshot.datasets`;
- record per-dataset transfer mode and authority in
  `InstanceExportApplyScope`;
- distinguish authoritative empty content from content not supplied;
- define explicit deletion intent instead of treating omission as deletion.

Dataset application follows these rules:

- a dataset outside scope is preserved;
- a scoped `REFERENCE` dataset creates or updates external-placeholder
  metadata but does not materialize a body;
- a scoped `CONTENT` dataset applies schema, datapoints, and included
  provenance, with `data=None` meaning authoritative empty only when content
  authority is explicit;
- a scoped explicit deletion can remove a target-owned dataset only after
  binding and sharing checks;
- an incoming real dataset may supersede an external placeholder and rewire
  ports, as current import support already does;
- a target-owned real dataset must not be silently replaced by an incoming
  external placeholder with the same identifier;
- schema-scoped or otherwise shared datasets are never mutated in place by an
  instance apply. Use a target-scoped copy and rewire, or report a conflict.

Applying dataset content updates the current draft dataset, creates the
appropriate immutable dataset revision, refreshes its materialization through
the shared materialization service, and records dataset changes under the same
top-level `instance_export.apply` operation. Existing published instance
revisions retain their pinned dataset revisions.

Dataset content can be substantially larger than the graph. Canonical hashes
must stream over normalized content where practical; planning must avoid
hydrating both full old and new payloads into multiple redundant Python
representations. Performance validation includes a production-sized export,
not only small fixtures.

### 9. Serving uses an effective snapshot

Compilation and persistence use raw source snapshots. Serving a YAML-sourced
instance separately constructs an effective snapshot:

```text
raw YAML snapshot + persisted NodeConfig overlay -> effective snapshot -> runtime
```

Runtime construction and GraphQL must consume the same effective snapshot.
Do not attach a reconciled snapshot while leaving different YAML metadata on
the runtime node. Until native snapshot-to-runtime construction lands, keep
this reconciliation at one explicit adapter boundary.

## CLI and API shape

### CLI

An initial management-command interface can be:

```bash
python manage.py compile_instance_export mainz-bisko \
  --datasets references \
  --output mainz-bisko.export.json

python manage.py apply_instance_export mainz-bisko.export.json \
  --instance mainz-bisko \
  --policy replace-scoped \
  --keep-config-source \
  --dry-run \
  --format human

python manage.py apply_instance_export mainz-bisko.export.json \
  --instance mainz-bisko \
  --policy replace-scoped \
  --keep-config-source \
  --expected-draft-head <uuid>
```

`--format json` emits the complete `InstanceExportApplyPlan`; human format
groups changes by entity and highlights conflicts, removals, and preserved
incoming values. Dry-run performs no writes rather than relying on transaction
rollback.

The existing `sync_instance_to_db` becomes a compatibility wrapper around
compile + plan + apply using `PRESERVE_TARGET + ADOPT_DATABASE`. Its
`--dry-run` uses the same structured planner.

After parity is proven, deprecate `load_nodes.py --update-nodes` in favor of
compile + apply with `REPLACE_SCOPED + KEEP_CURRENT`. Preserve its existing
distinction that updating populated fields is an explicit overwrite operation
during the transition.

### API

Because an `InstanceExport` can contain large dataset bodies, upload it through
a dedicated authenticated, streaming endpoint. The upload validates the
artifact envelope and returns a content-addressed, expiring handle:

```text
POST /v1/instance-exports/
  -> { artifactId, artifactHash, expiresAt }
```

The server stores the artifact outside the GraphQL operation payload. Enforce
compressed and uncompressed size limits, safe decompression, schema-version
validation, per-user quotas, and expiry. The artifact is immutable and keyed
by its canonical digest.

Expose preview and apply control through GraphQL:

```graphql
previewInstanceExportApply(
  instanceId: UUID!
  artifactId: UUID!
  policy: InstanceExportApplyPolicy!
  sourceTransition: SourceTransitionPolicy!
): InstanceExportApplyPreview!

applyInstanceExport(
  instanceId: UUID!
  artifactId: UUID!
  policy: InstanceExportApplyPolicy!
  sourceTransition: SourceTransitionPolicy!
  expectedDraftHead: UUID
  expectedTargetHash: String!
  expectedPlanHash: String!
): InstanceExportApplyPayload!
```

Both operations require instance `change` permission. `KEEP_CURRENT` is valid
for YAML- and DB-sourced instances; `ADOPT_DATABASE` is the explicit
YAML-to-database transition. Applying an artifact never publishes it.

The preview payload exposes the structured plan, target/draft preconditions,
and whether it is applicable. The apply payload returns stale-preview details
when target state changed, otherwise the new draft head and applied summary.

Preview and result use typed GraphQL objects so clients do not need to parse
an opaque diff. Dataset details are paginated or retrieved through a bounded
detail endpoint. Small local management-command workflows can read an export
directly without uploading it; both paths call the same planner.

## Delivery stages

### Stage 1 — InstanceExport artifact and raw YAML compiler

- Define `InstanceSyncArtifact`, `InstanceExportApplyScope`, provenance,
  per-dataset transfer authority, and canonical hashing.
- Add durable dataset UUIDs and unambiguous omitted/reference/content/empty
  dataset states to the export contract.
- Extract `compile_yaml_export()` from `sync_parsed_instance_to_db()`.
- Prove the compiled export's graph contains YAML metadata before DB overlay.
- Export DVC datasets by immutable reference by default, target-owned DB
  datasets out of scope, and selected producer-owned datasets by value.
- Preserve authored UUID > target catalog > deterministic UUIDv3 precedence.
- Add artifact round-trip and deterministic-output tests.

**Gate:** compiling the same merged YAML and dataset sources with the same
instance UUID and identity catalog produces byte-equivalent canonical artifact
content.

### Stage 2 — Entity-aware planner

- Capture the scoped current DB draft as a `TargetExportState` and target
  catalog without unnecessarily loading out-of-scope dataset bodies.
- Implement matching and field-level diffs for instance spec, nodes, ports,
  edges, dataset bindings, dimensions, dataset schemas, datapoints, and
  provenance.
- Add read-only dataset/metric resolution diagnostics.
- Implement `PRESERVE_TARGET` and `REPLACE_SCOPED` resolution.
- Represent `KEEP_CURRENT` and `ADOPT_DATABASE` in the plan and validate
  artifact completeness for adoption.
- Add JSON and human renderers.

**Gate:** plans are deterministic, perform zero writes, and focused fixtures
show creations, updates, removals, preserved values, explicit clears, and
identity/constraint conflicts without positional diff noise.

### Stage 3 — Transactional applier

- Refactor the existing `spec_sync` and dataset-import writers behind
  `apply_instance_export_plan()`.
- Add target locking and head/hash preconditions.
- Record `instance_export.apply` change operations and row-level changes.
- Create dataset revisions, refresh materializations, and preserve published
  revision pins.
- Ensure stale-node handling and placeholder/dataset-port behavior are
  represented in the plan before execution.
- Make cache invalidation explicit.

**Gate:** apply produces exactly the planned desired export state; stale plans
make no writes; any failure rolls back graph rows, dataset state, revisions,
materializations, and the change operation.

### Stage 4 — Management-command adoption

- Add compile and apply commands.
- Route `sync_instance_to_db` through the shared services.
- Replace rollback-only dry-run output with the structured preview.
- Compare `PRESERVE_TARGET` results against current parse-only sync over the
  parse-oracle instance set.
- Compare `REPLACE_SCOPED` metadata results against
  `load_nodes.py --update-nodes --overwrite` on representative YAML models.

**Gate:** existing preservation and database-adoption behavior has no
unintended changes, while the replacement workflow demonstrably applies
YAML-authored names and other scoped metadata without changing
`config_source`.

### Stage 5 — Effective YAML serving

- Build one effective reconciled snapshot for YAML serving.
- Ensure runtime nodes and GraphQL derive metadata from that same state.
- Remove any temporary runtime-node storage of raw YAML sync values.
- Re-run recorded GraphQL queries for YAML- and DB-sourced instances.

**Gate:** YAML-sourced `Node.name`, metric names, node-derived dimension
labels, visibility, color, order, and descriptions consistently reflect the
effective snapshot, while raw compilation still reflects YAML.

### Stage 6 — API preview and apply

- Add typed preview/result GraphQL surfaces and permission checks.
- Add the authenticated streaming artifact-upload endpoint with digest,
  expiry, and size controls.
- Require target hash and plan hash on apply; use draft head where available.
- Return structured stale-preview and validation errors.
- Add request-size, complexity, and audit logging limits.
- Build the model-developer client flow: compile locally, preview remotely,
  confirm, apply.

**Gate:** concurrent Trailhead edits between preview and apply are rejected;
successful application creates one auditable operation and never publishes.

### Stage 7 — Three-way synchronization

- Persist successful source lineage, base artifact, and hashes.
- Implement field-level base/incoming/current classification.
- Return resolvable structured conflicts.
- Add explicit `prefer-current` / `prefer-incoming` conflict resolution only
  at the conflicting paths, not as a global silent fallback.
- Initially classify concurrent dataset-body changes by dataset content hash
  and require an explicit whole-dataset resolution. Defer datapoint-level
  three-way merge until its revision and provenance semantics are specified.

## Validation matrix

At minimum cover:

- authored, catalog-preserved, and deterministically derived node UUIDs;
- identifier rename with and without an authored UUID;
- stable ports and bindings across YAML reordering;
- DB-authored name preserved under `PRESERVE_TARGET`;
- YAML name applied under `REPLACE_SCOPED`;
- YAML-sourced instance remains YAML-sourced under `KEEP_CURRENT`;
- incomplete artifact cannot use `ADOPT_DATABASE`;
- complete artifact deliberately changes source under `ADOPT_DATABASE`;
- admin-only node body preserved because it is outside YAML scope;
- explicit clearing of a scoped optional YAML field;
- node/edge/port creation and stale removal;
- target-owned DB dataset preserved when absent from YAML export scope;
- DVC dataset round-tripped as an immutable external reference;
- owned dataset schema, datapoints, sources, references, and comments applied
  by value;
- authoritative empty dataset distinguished from an omitted body;
- external placeholder promoted to a real target-owned dataset and ports
  rewired;
- shared dataset protected from in-place mutation;
- dataset revision and materialization refreshed while published pins remain
  unchanged;
- unresolved and ambiguous dataset metrics reported during preview;
- no SQL writes during planning;
- stale draft head and stale target export hash rejection;
- apply rollback on a mid-operation error;
- one grouped change operation with accurate row-level entries;
- preview/apply result equality after rebuilding the target export state;
- YAML- and DB-sourced recorded GraphQL parity;
- parse-oracle parity for the compatibility `sync_instance_to_db` policy.

## Documentation changes during implementation

- Update `docs/trailhead/architecture-decisions.md` when the artifact and
  apply-policy contracts land.
- Replace the three-way-sync sketch in `docs/plans/loader-spec-inversion.md`
  with a link to this plan.
- Document the CLI transition and deprecation in the model-building and
  dataset-management command guides.
- Document the preview/apply API beside the existing Trailhead mutation and
  optimistic-concurrency contracts.
