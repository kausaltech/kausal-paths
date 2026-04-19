# Trailhead: Draft / Publish / Revisions — Implementation Plan

## Purpose

Model builders edit an `InstanceConfig` through a series of mutations in the
Trailhead editor. Each mutation may leave the model in a broken, non-computing
state. The public must continue to see a stable, previously-published version
of the model (often the city's "official" emissions results) while editing is
in progress. Users need per-edit undo, named draft checkpoints, and an explicit
publish step to move a draft to live.

This plan is intentionally opinionated about which Wagtail machinery to reuse
and which to bypass. Load-bearing design decisions are stated up front;
subsequent sections detail the implementation.

## Current status (as of April 19, 2026)

**Landed:** phases 1, 2, 2.5 + significant parts of phase 3 + the
read-side of phase 5's history surface.

| Phase | Status | Notes |
|---|---|---|
| 1 — Snapshot schema | ✅ Landed | `InstanceSnapshot`, `InstanceExport { instance, datasets }`, `*Snapshot` Pydantic types with `from_model`, `build_instance_snapshot`, hydrate-from-revision branch on `_create_from_config(source='published')`. Commit `6f5e8915`. |
| 2 — Change tracking | ✅ Landed | `InstanceChangeOperation` + `InstanceModelLogEntry` models, `change_operation()` / `record_change()` / ContextVar carrier, `EditableInstanceChild` superclass applied to `NodeConfig` / `NodeEdge` / `DatasetPort`, `RevisionMixin` added to `Dataset` (kausal_common). Commits `6f5e8915` (+ superclass refactor). |
| 2.5 — PoC on node.create/update/delete | ✅ Landed | All three node mutations wired; cascade delete groups correctly. Commit `6f5e8915`. |
| 3 — Mutation layer refactor | 🟡 Partial | See [Phase 3 status notes](#phase-3-status-notes) below. Done: most edit mutations wired through `change_operation` (edges, dimensions, datapoints, port-level). Open: version-token directive, `@preview` directive, stale-check enforcement, uniform `MutationPayload`. |
| 4 — Resolver split & compute invalidation | ⚪ Not started | — |
| 5 — Publish / revert / undo / named drafts | 🟡 Read-side only | History GraphQL surface landed (`InstanceEditor.changeHistory`, `EditableEntity.changeHistory`). Mutations (publish / revert / undo / save_draft) not yet implemented. |
| 6 — Permissions & migration | ⚪ Not started | — |

### Deltas from the original plan

**`InstanceModelLogEntry` is standalone, not a `ModelLogEntry` subclass.**
The plan described subclassing Wagtail's `ModelLogEntry` via multi-table
inheritance. We dropped that to avoid the extra table-per-insert cost
and the ``LogActionRegistry`` indirection. IMLE mirrors the shape
(content_type + object_id GFK, `action`, `data` JSONField) but is a
freestanding model.

**Change tracking extended to the demo-flow mutations during the smoke
work** (not scheduled in Phase 3's original scope but needed to validate
the chain against `aarhus-c4c`): `edge.create` / `edge.delete` /
`dimension.update` / `dimension.categories.create|update` /
`dimension.category.delete` / `dataset.datapoint.create|update|delete`
are all wired. Commit `9ce1fefa` added them along with the demo smoke
script.

**Port-level mutations landed outside the Phase 3 table:**
`addNodeInputPort` / `addNodeOutputPort` (emits
`node.input_ports.create` / `node.output_ports.create` entries).
`_resolve_or_create_target_port` provides auto-create of matching input
ports on `createEdge` when `toPort` is null — grouped under the
`edge.create` operation so a single undo reverses both the port
addition and the edge.

**Typed `EdgeTransformationInput`** (one-of: `selectCategories` /
`assignCategory` / `flatten`) replaced the opaque `JSON` field on
`CreateEdgeInput`. Mirrors `EdgeTransformationType` on the query side.

**GQL never exposes DB pks** — formalized as a feedback memory
([feedback_gql_identity.md](../../../../../home/jey/.claude/projects/-home-jey-sync-devel-kausal-paths/memory/feedback_gql_identity.md)).
All entity-lookup mutation arguments accept UUID or human-readable
identifier; subscription `InstanceChange.id` uses uuid.

**`import_instance` now syncs `primary_language` / `other_languages`
from the imported spec onto the InstanceConfig row.** Without this,
cloning `aarhus-c4c` (primary=`da`) into a new instance created with
default `en` left ActionGroup TranslatedStrings tagged `da` while the
InstanceConfig claimed `en` — hydrate filtered them all out. Surfaced
by the demo smoke run against the live DB.

**`Dataset.serializable_data` bridged via `paths/dataset_pydantic.py`.**
Dataset gained `RevisionMixin`; the Pydantic snapshot shape is
project-specific so the bridge gates on `IS_PATHS` — Watch doesn't
revision datasets and doesn't need the type.

**`DatasetSnapshot` absorbed `DatasetExport`**, now with full
TranslatedString fields + `data` payload (DataPoints in JSON Table
Schema format). Revisioning the dataset captures its datapoints too,
not just the schema.

**EditableEntity interface (read-side history):** the `target` field
on `InstanceModelLogEntryType` resolves to an interface rather than a
union. `NodeType` / `ActionNodeType` / `NodeEdgeType` /
`DatasetPortType` all implement it and carry `uuid` + `changeHistory`.
The UI can fetch per-entity history without type-narrowing.
`InstanceEditor.changeHistory` returns operations; per-entity
`changeHistory` returns entries.

**Demo smoke command:** `python manage.py trailhead_demo_smoke` walks
the full register-user → create-instance(cads/aarhus-c4c) → add
category → add datapoints → create action node → add input ports →
create transformed edges flow and dumps the resulting change log.
Used to validate the chain end-to-end against a live DB.

### Phase 3 status notes

| Sub-item | Status |
|---|---|
| Change-operation wrapper on mutations | ✅ All edit mutations routed |
| Action-id naming convention (`node.create`, `edge.update`, etc.) | ✅ |
| Entity lookup by UUID / identifier, never pk | ✅ |
| Closed mutation set & inverses enumerated | ✅ Enumerated in plan, inverses not yet implemented (Phase 5 dependency) |
| Typed `EdgeTransformationInput` | ✅ |
| Port mutations (`addNodeInputPort` / `addNodeOutputPort`) | ✅ |
| `draft_head_token` property on `InstanceConfig` | ⚪ Not started |
| `@instance(version: UUID, preview: String)` directive extension | ⚪ Not started |
| Stale-version check with `SELECT FOR UPDATE` | ⚪ Not started |
| Uniform `MutationPayload { ok, ...entity, newHeadToken, invalidatedNodeIds }` | ⚪ Not started (invalidation depends on Phase 4) |
| `apply_snapshot` / `from_serializable_data` classmethod impl | ⚪ Stub only |

### Known open items

- **Perm gating on `EditableEntity.changeHistory`** — currently
  anyone who can see the entity can read its history. The interface is
  the clean place to slot the `change`-perm check against the hosting
  InstanceConfig.
- **`_fetch_entity_history` has no upper bound guard** beyond the
  caller's `limit` argument — fine for now, flag for retention policy
  work later.
- **`dataset.save_revision()` not yet wired into the dataset editor
  mutations** — `DatasetPort.dataset_revision` will only advance when
  the dataset editor begins calling it. Tracking this under the
  `dataset.datapoints.edit` wiring in Phase 3.

---

## Load-bearing decisions

These are settled — later sections assume them.

### Aggregate-level versioning on `InstanceConfig`

- `InstanceConfig` already inherits `RevisionMixin` + `DraftStateMixin` (see
  [nodes/models.py:311](../../../nodes/models.py#L311)). `publish_instance()`
  at line 537 is wired; `revert_to_published()` at line 542 is a stub.
- A `Revision` is the **JSON-serialized whole-model snapshot**, addressed by
  UUID (not pk), so reverts don't care about reassigned integer ids.
- Per-node `DraftStateMixin` is **not** added. A consistent model spans many
  rows (node + edges + params + scenarios + dataset pins) — the unit users
  think about is the instance, so that is the unit we version.

### Draft workspace = the structured tables themselves

- Mutations edit `NodeConfig`, `NodeEdge`, `DatasetPort` rows directly;
  spec-embedded entities (`Scenario`, `ActionGroup`, params) edit
  `InstanceConfig.spec` via `queryset.update()`. Broken intermediate
  states are fine there.
- Publishing **snapshots** current tables + spec → new `Revision`, marks
  it live. `InstanceConfig.live_revision_id` points at the live snapshot.
- Revisions are **coarse**: produced on publish, on explicit "save as named
  draft", and (optionally, later) on periodic safety snapshots. *Never* one
  per mutation.

### Change operations as first-class rows; log entries grouped under them

- **`InstanceChangeOperation`** is a new model — one row per user-facing
  edit. Carries `uuid`, `instance_config` FK, `user` FK (nullable for system
  ops), `action` (dotted string like `node.delete`), `source` (enum:
  `graphql` / `rest` / `admin` / `cli` / `migration`), `created_at`,
  `superseded_by` FK self. Lives in `nodes/` for now; can graduate to
  `kausal_common/` later if Kausal Watch picks it up.
- **`InstanceModelLogEntry`** (subclass of Wagtail's `ModelLogEntry`
  via `UUIDIdentifiedModel`) — one row per **affected ORM row** within an
  operation. FK → `InstanceChangeOperation`. `delete_node` cascading to 7
  edges = 1 operation row + 8 log entry rows.
- Entry payload is **uniform**: `IMLE.data = { action, target_uuid, before, after }`
  where `before`/`after` are the dicts produced by `serializable_data()`
  on the affected row. Wagtail's native `content_type` + `object_id` on
  `BaseLogEntry` provide the GFK for the affected row; `target_uuid` in
  `data` survives row deletion and anchors the undo path.
- **Undo grain is the operation**, not individual entries. When op B
  undoes op A, `A.superseded_by = B`. No mirror entries; the supersession
  link is the history trail.
- **Version token** = `uuid` of the latest `InstanceChangeOperation` for
  the instance. Compared by equality.
- Multi-step revert uses `Revision` restore as the fallback.
- **We bypass `LogActionRegistry`.** Action strings are validated at the
  Pydantic-payload layer. Wagtail calls that would emit their own entries
  (`save_revision`, `publish`) get `log_action=False`; we emit our own
  entries from the change-operation wrapper.

### Editable-child superclass + per-row revisions

Every table that a mutation can edit (`NodeConfig`, `NodeEdge`,
`DatasetPort`) inherits from a single abstract superclass:

```python
class EditableInstanceChild(
    UUIDIdentifiedModel,
    UserModifiableModel,
    RevisionMixin,
    ClusterableModel,
):
    snapshot_model: ClassVar[type[ModelSnapshot]]

    class Meta:
        abstract = True

    def serializable_data(self) -> dict:
        # overrides Wagtail's default; produces snapshot-shaped dict
        return self.snapshot_model.from_model(self).model_dump(mode='json')

    @classmethod
    def from_serializable_data(cls, data: dict, *, ic: InstanceConfig) -> Self:
        # inverse: apply a snapshot dict to create or update a row
        ...
```

Rationale:
- Superclass over mixin — Python mixin typing with `super()` is fragile;
  a proper superclass sidesteps MRO pain.
- Bundles `UUIDIdentifiedModel` / `UserModifiableModel` / `RevisionMixin`
  so all editable children get `.uuid`, `updated_at/by`, and per-row
  Wagtail revisions centrally. The change wrapper calls `save_revision()`
  on each affected row after the ORM write. Storage cost is negligible and
  redundant audit is a feature, not a bug, when something goes wrong.
- Overrides `ClusterableModel.serializable_data()` with a snapshot-shaped
  implementation (Wagtail's default doesn't earn its keep here). Plus
  `from_serializable_data()` as the inverse — the same protocol Wagtail
  uses, just schema-validated.

Spec-embedded entities (`Scenario`, `ActionGroup`, params) are not ORM
rows, so they don't use the superclass. Their mutations edit
`InstanceConfig.spec` via `queryset.update()` and log entries record the
list delta directly.

### Snapshot Pydantic base

```python
class ModelSnapshot(BaseModel):
    @classmethod
    def from_model(cls, obj) -> Self:
        # default: attribute-access into the ORM row. Override when
        # the mapping isn't 1:1 (e.g. FK fields that need resolving).
        return cls.model_validate(obj, from_attributes=True)
```

`NodeSnapshot`, `EdgeSnapshot`, `DatasetPortSnapshot` inherit from this
base. The explicit `from_model()` hook exists because `model_validate(...,
from_attributes=True)` doesn't handle every case — FK resolution and
custom derivations are common enough that every snapshot model has room
to override.

### Resolver split: compute surface vs. authoring surface

- **Compute/read resolvers** (node outputs, edges-as-graph, dimensions,
  computed values) resolve from the hydrated in-memory `Instance`/`Context`,
  not from ORM rows. "Draft vs. published" is a *single* decision at
  `resolve_instance` — hydrate from the latest published revision blob, or
  from current draft tables.
- **Authoring resolvers** (NodeConfig metadata, revision list, log entries,
  permissions, translations, graph layout) resolve from ORM rows. Only
  meaningful for drafts.

### Compute-after-mutation: invalidate, don't compute

- Mutation commits, invalidates cached outputs for topological descendants
  of the change (DAG walk, no actual compute), returns immediately with
  `newHeadToken` and `invalidatedNodeIds`.
- Frontend paints "stale" indicators on those nodes from its local DAG
  knowledge, then fetches updated `contentHash` values for affected nodes
  (reusing `NodeCacheHandler.calculate_hash`), compares against its cache,
  refetches outputs for mismatches only.
- Compute runs lazily when an output is actually asked for. No subscriptions
  needed for v1.

### Optimistic locking via query-level directive

- Every editing mutation carries a version token — the `uuid` of the
  current head `InstanceChangeOperation` for the instance (the "draft
  head"). Server compares by equality; if mismatched, the mutation is
  stale.
- Implemented as an additive arg on the existing **`@instance`** and
  **`@context`** directives (see [paths/schema.py:48-75](../../../paths/schema.py#L48-L75)),
  *not* as an HTTP header. GraphQL conventions reject transport-coupled
  metadata; a directive stays transport-neutral. The same directive also
  carries `preview: "draft" | "published"` (see §4 below).

### Transport-neutral carrier for the active change operation

The active `InstanceChangeOperation` is carried through a
module-level `ContextVar[InstanceChangeOperation | None]`. That's the
ground truth — management commands, Celery tasks, migrations can open a
change operation without any transport plumbing. GraphQL / REST / admin
views *also* attach the operation to their native carrier
(`info.context.operation`, `request.operation`) as typed-access
convenience, but those are pointers to the same object held by the
ContextVar.

The change wrapper (`change_operation(...)` context manager) is the only
place that sets both. Anywhere below the resolver / view boundary —
including deeply nested ORM signals or helpers — uses
`get_current_operation()` which reads the ContextVar.

This keeps the door open for Wagtail-admin and REST-API callers to
trigger mutations through the same audit / undo / revision machinery in
future phases.

### Cascade UX: confirm-and-cascade

- Edges stay as ORM rows (DB-enforced consistency + cheap "what depends on
  X" queries). We do **not** embed edges in node specs.
- `delete_node` with edges: confirm and cascade. Confirmation dialog shows
  the list of affected edges, not just a count.
- All cascade deletions share one `operation_id`; one undo click restores
  everything.

### Revision contents: `InstanceSnapshot` (pinned references only)

The unit of revisioning is `InstanceSnapshot`, reusing and extending the
existing `InstanceExport` / `*Export` machinery at
[nodes/instance_serialization.py](../../../nodes/instance_serialization.py).
Rename existing ref-only types `*Export` → `*Snapshot`; keep `DatasetExport`
for the dataset-body-carrying case:

```python
class InstanceSnapshot(BaseModel):
    """Instance state, dataset references only. Unit of revisioning."""
    schema_version: int = 1
    spec: InstanceSpec          # already contains scenarios, action_groups,
                                # params, pages, normalizations, dimensions
    nodes: list[NodeSnapshot]
    edges: list[EdgeSnapshot]
    dataset_ports: list[DatasetPortSnapshot]

class InstanceExport(BaseModel):
    """Self-contained instance + dataset bodies. Unit of sync/portability."""
    schema_version: int = 1
    instance: InstanceSnapshot
    datasets: list[DatasetExport]
```

- No separate `dataset_pins` list — the pin lives on `DatasetPort` itself
  as a single nullable FK to the dataset's `Revision` (DB-backed and
  DVC-backed alike: DVC pins route through `Dataset.external_ref`, which
  is itself tracked by the dataset's own `Revision` stream).
  `DatasetPortSnapshot` simply serializes the port row including its pin.
- **Dataset references are always pinned** — never floating. Reverting a
  revision restores the pinned refs on the DatasetPort rows; dataset tables
  themselves are never rewound.
  - Separated workflow: dataset editor publishes → model untouched → UI
    surfaces "N datasets have newer versions" → model editor updates refs
    explicitly → republishes.
  - Integrated workflow: editing a dataset *from inside the model editing
    context* auto-updates the pin in the draft. Produces two grouped log
    entries (`edit_dataset` + `update_dataset_ref`) sharing an `operation_id`,
    so undo reverses both.
- Framework linkage is **not** editable once linked, so not part of the
  revertable snapshot.

### State machine: draft / live only, no workflows

- Just `draft` / `live` (+ `has_unpublished_changes`). No scheduled
  publishing, no workflow/approval states. Wagtail's `DraftStateMixin`
  supports this minimal subset.

### Revert is a barrier for undo (v1)

- Undo refuses to cross a `revert` boundary. Alternative (making revert
  itself undoable) would require storing a full pre-revert snapshot in the
  log entry payload, which is heavy. Revisiting later is cheap — we just
  add a new payload variant and remove the barrier check; no data
  migration. Safe to defer.

---

## Phase 1 — `InstanceSnapshot` schema & round-trip

**Goal:** `save_revision()` writes a versioned `InstanceSnapshot`;
`InstanceLoader` can hydrate from it. No UX change yet.

### Changes

1. **Snapshot Pydantic types** at
   [nodes/instance_serialization.py](../../../nodes/instance_serialization.py):
   - Introduce `class ModelSnapshot(BaseModel)` base with a `from_model()`
     classmethod (default impl: `cls.model_validate(obj, from_attributes=True)`;
     subclasses override when mapping isn't 1:1, e.g. FK resolution).
   - `NodeExport` → `NodeSnapshot(ModelSnapshot)`
   - `EdgeExport` → `EdgeSnapshot(ModelSnapshot)`
   - `DatasetPortExport` → `DatasetPortSnapshot(ModelSnapshot)` — gains
     `dataset_revision: int | None` (single FK, no DVC trio; see step 2).
   - `DatasetMetricExport` stays (embedded in `DatasetExport`).
   - `DatasetExport` stays (genuinely carries dataset bodies).
   - New `InstanceSnapshot { schema_version, spec, nodes, edges, dataset_ports }`
     — the revision payload.
   - Refactor existing `InstanceExport` to `{ schema_version, instance, datasets }`
     (composition, not field-flattening; `instance` is the InstanceSnapshot).
     Existing callers of `export_instance` / `import_instance` updated
     accordingly.
   - Drop top-level `scenarios` / `action_groups` — they're already in
     `InstanceSpec` and thus in `instance.spec`.
   - UUID-addressed throughout. `schema_version` enables forward-compat parsing.

2. **`DatasetPort.dataset_revision` FK** (new migration): one nullable FK
   to the dataset's `Revision` model. Covers DB-backed and DVC-backed
   datasets uniformly — DVC datasets surface commit changes through
   `Dataset.external_ref`, which in turn produces a new dataset `Revision`.

3. **`build_instance_snapshot(ic) -> InstanceSnapshot`** — new function.
   Populates `DatasetPortSnapshot.dataset_revision` from the port's current
   pin. Sibling of the existing `export_instance`.

4. **Rework `InstanceConfig.serializable_data()`**
   ([nodes/models.py:521](../../../nodes/models.py#L521)) to emit
   `InstanceSnapshot.model_dump()` under the `model_snapshot` key, alongside
   Wagtail's own revision fields.

5. **`InstanceLoader.from_snapshot(snapshot)`** — thin adapter: `InstanceSnapshot`
   → YAML-equivalent dict (via existing `instance_from_db.py` machinery) →
   `from_dict_config`. Keeps a single hydrate path so dataset lazy resolution
   etc. go through the same code.

6. **Branch `_create_from_config`**
   ([nodes/models.py:547](../../../nodes/models.py#L547)):
   ```python
   if source == 'published':
       snapshot = InstanceSnapshot(**ic.live_revision.content['model_snapshot'])
       return InstanceLoader.from_snapshot(snapshot).instance
   elif config_source == 'database':
       # current path (draft — reads tables directly)
   ```
   Which source is used is controlled by an explicit arg; default remains
   `database` (draft) during phases 1–4 so nothing changes behaviorally
   until phase 6 flips the public default.

### YAML-instance ORM overlay — preserved

For YAML-sourced instances (not revisioned), `NodeConfig` overrides (`name`,
`color`, `order`, `is_visible`, long-form rich-text description, `i18n`)
layer on top of YAML-defined Nodes during hydrate. This overlay behavior
must be preserved — it is the only mechanism for CMS-like edits to YAML
instances. Matrix:

| Source | Base | Overlay |
|---|---|---|
| YAML | `nodes.yaml` etc. | ORM `NodeConfig` fields |
| DB-sourced draft | ORM tables | — (tables are source of truth) |
| DB-sourced published | `InstanceSnapshot` | — (snapshot carries all fields) |

The snapshot's `NodeSnapshot` already records `name/color/order/is_visible/i18n`
so the overlay semantics are inlined at snapshot-build time for DB sources.

### Tests

- Round-trip: `serialize → hydrate → reserialize` produces equal
  snapshots.
- `test_instance` variant that runs against `from_snapshot`.

### Dependencies

None. Lands independent of everything below.

### Note on `Scenario`

`Scenario` is *not* a Django model — it's a Pydantic type embedded in
`InstanceSpec.scenarios` (see [nodes/defs/instance_defs.py:175](../../../nodes/defs/instance_defs.py#L175)).
Scenario CRUD mutations therefore edit `InstanceConfig.spec.scenarios`
(the embedded list) rather than separate rows. Same applies to
`action_groups` ([instance_defs.py:174](../../../nodes/defs/instance_defs.py#L174))
and the other spec-embedded collections. The mutation's log entry's
`before`/`after` payloads carry the list delta.

---

## Phase 2 — Change operations, log entries, editable-child superclass

**Goal:** Lay down the audit + undo foundation without wiring any
mutations to it yet. At the end of this phase we have the data model,
the `change_operation()` context manager, and the `EditableInstanceChild`
superclass applied to `NodeConfig` / `NodeEdge` / `DatasetPort` — but
existing mutations still work unchanged.

### Changes

1. **`EditableInstanceChild` abstract superclass** in `nodes/models.py`:
   ```python
   class EditableInstanceChild(
       UUIDIdentifiedModel,
       UserModifiableModel,
       RevisionMixin,
       ClusterableModel,
   ):
       snapshot_model: ClassVar[type[ModelSnapshot]]

       class Meta:
           abstract = True

       def serializable_data(self) -> dict:
           return self.snapshot_model.from_model(self).model_dump(mode='json')

       @classmethod
       def from_serializable_data(cls, data: dict, *, ic: InstanceConfig) -> Self:
           ...

       def __init_subclass__(cls, **kwargs):
           super().__init_subclass__(**kwargs)
           if not hasattr(cls, 'snapshot_model') and not cls._meta.abstract:
               raise TypeError(f'{cls.__name__} must declare snapshot_model')
   ```

   Apply to `NodeConfig`, `NodeEdge`, `DatasetPort`. Each drops its
   individual inheritance of `UUIDIdentifiedModel` / `UserModifiableModel` /
   `ClusterableModel` / `RevisionMixin` and inherits the superclass instead.
   Set `snapshot_model` to `NodeSnapshot`, `EdgeSnapshot`,
   `DatasetPortSnapshot` respectively.

   `NodeConfig` already has `RevisionMixin`; this unifies `NodeEdge` and
   `DatasetPort` with the same treatment. Per-row Wagtail revisions on
   every editable child are redundant with IMLE but cheap, and "more
   history than less" is the right default when something goes wrong.

2. **`InstanceChangeOperation` model** in `nodes/models.py`:
   ```python
   class InstanceChangeOperation(UUIDIdentifiedModel):
       instance_config = models.ForeignKey(
           InstanceConfig, on_delete=models.CASCADE,
           related_name='change_operations',
       )
       user = models.ForeignKey(
           settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
           null=True, blank=True,
       )
       action = models.CharField(max_length=100)       # e.g. 'node.delete'
       source = models.CharField(                      # trigger transport
           max_length=20,
           choices=[('graphql', 'GraphQL'), ('rest', 'REST'),
                    ('admin', 'Wagtail admin'), ('cli', 'CLI'),
                    ('migration', 'Data migration')],
       )
       created_at = models.DateTimeField(auto_now_add=True)
       superseded_by = models.ForeignKey(
           'self', on_delete=models.SET_NULL,
           null=True, blank=True,
           related_name='supersedes',
       )

       class Meta:
           ordering = ['-created_at']
           indexes = [
               models.Index(fields=['instance_config', '-created_at']),
           ]
   ```

3. **`InstanceModelLogEntry`** (subclass of `ModelLogEntry`) in
   `nodes/models.py`:
   ```python
   class InstanceModelLogEntry(UUIDIdentifiedModel, ModelLogEntry):
       operation = models.ForeignKey(
           InstanceChangeOperation, on_delete=models.CASCADE,
           related_name='log_entries',
       )
       # Wagtail's BaseLogEntry already provides:
       #   - content_type + object_id (GFK to the affected row)
       #   - action (we use dotted strings: 'node.update' etc.)
       #   - data (JSONField)
       # `data` layout: { target_uuid, before, after }
       #   target_uuid survives row deletion; before/after are the dicts
       #   produced by serializable_data() on the affected row.
   ```
   No per-action payload classes. Validation happens when loading: the
   `action` string selects which `ModelSnapshot` subclass validates
   `before` / `after`.

4. **`change_operation()` context manager** in `nodes/change_ops.py`:
   ```python
   _current_op: ContextVar[InstanceChangeOperation | None] = ContextVar(
       'current_change_operation', default=None,
   )

   @contextmanager
   def change_operation(
       ic: InstanceConfig,
       user,
       action: str,
       source: str,
   ) -> Iterator[InstanceChangeOperation]:
       with transaction.atomic():
           # Serializes concurrent mutations against the same instance
           InstanceConfig.objects.select_for_update().filter(pk=ic.pk).first()
           op = InstanceChangeOperation.objects.create(
               instance_config=ic, user=user, action=action, source=source,
           )
           token = _current_op.set(op)
           try:
               yield op
           finally:
               _current_op.reset(token)

   def get_current_operation() -> InstanceChangeOperation:
       op = _current_op.get()
       if op is None:
           raise RuntimeError('No InstanceChangeOperation in context')
       return op
   ```
   Transport layers (phase 3) additionally attach the op to their native
   carrier (`info.context.operation`, `request.operation`) for typed
   access, but the ContextVar is the ground truth.

5. **`record_change()` helper**:
   ```python
   def record_change(
       obj: EditableInstanceChild,
       action: str,
       before: dict | None,
       after: dict | None,
   ) -> None:
       op = get_current_operation()
       InstanceModelLogEntry.objects.create(
           operation=op,
           content_type=ContentType.objects.get_for_model(obj),
           object_id=obj.pk,
           action=action,
           data={
               'target_uuid': str(obj.uuid),
               'before': before,
               'after': after,
           },
       )
       # Per-row Wagtail revision — cheap audit redundancy
       obj.save_revision(user=op.user, log_action=False)
   ```
   Spec-embedded targets (scenarios, action_groups, params) use a
   sibling helper `record_spec_change(ic, action, target_uuid, before, after)`
   since there's no ORM row to hang the GFK on — the entry points at the
   `InstanceConfig` with `target_uuid` identifying the embedded item.

6. **Migrations**: add `InstanceChangeOperation` and `InstanceModelLogEntry`
   tables. Migrate `NodeEdge` + `DatasetPort` onto the superclass (no
   schema change — same columns, different inheritance — but Django may
   generate a no-op migration for the Meta reshuffle; verify).

### Tests

- `change_operation()` creates one operation row and attaches it to the
  ContextVar for the duration of the block.
- `record_change()` on a NodeConfig creates an IMLE with the right shape
  and emits a per-row Wagtail revision.
- Spec-embedded `record_spec_change()` creates an IMLE pointing at the
  InstanceConfig with `target_uuid` set.
- `NodeConfig.serializable_data()` returns a dict that `NodeSnapshot`
  round-trips cleanly.

### Dependencies

Phase 1 (snapshot types). Does *not* touch existing mutation code yet.

---

## Phase 2.5 — Proof of concept (pause for review)

**Goal:** Validate the change-operation + IMLE + superclass pattern on a
small slice before refactoring every mutation.

### Scope

Wire three mutations through `change_operation()`:

- **`node.create`** — baseline create path (IMLE with `before=None`).
- **`node.update`** — baseline update (IMLE with both `before` and
  `after`).
- **`node.delete`** — cascades to `NodeEdge` and `DatasetPort` rows;
  produces 1 operation + (1 + N edges + M ports) IMLE rows, all sharing
  the operation FK.

These three exercise all interesting machinery: create-shape payload,
update-shape payload, delete-shape payload, cascade grouping, per-row
Wagtail revision emission, ContextVar propagation across a nested
ORM-signal teardown.

No undo, no directive, no publish/revert. Just the logging machinery
firing in a realistic mutation flow.

### Success criteria

- Running the three mutations via the existing GraphQL editor endpoint
  produces `InstanceChangeOperation` + `InstanceModelLogEntry` rows with
  the expected shape.
- A manual "restore from IMLE" (scripted in a unit test, not wired to a
  user action) successfully recreates a deleted node + its cascaded
  edges/ports from the `before` payloads.
- Per-row `Revision` rows on NodeConfig / NodeEdge / DatasetPort show
  the mutation history.

### Review gate

Pause here. Evaluate:
- Is the payload shape serviceable for undo?
- Does the ContextVar cleanly propagate across all nested writes
  (including ORM signals from cascade deletes)?
- Does `from_serializable_data()` cleanly reverse `serializable_data()`
  without edge cases we didn't anticipate?
- Does the per-row `RevisionMixin` create noticeable write latency in
  realistic mutation traces?

If all green, proceed to Phase 3 and refactor the rest. If not,
iterate on the shape before scaling it out.

### Dependencies

Phase 2.

---

## Phase 3 — Version token + preview directive; mutation layer refactor

**Goal:** Every editing mutation carries a stale-check token. Existing
`@instance` / `@context` directives gain `preview` and `version` args. All
mutations route through the operation-logging wrapper.

### Changes

1. **`InstanceConfig.draft_head_token: UUID | None`** — computed property,
   the `uuid` of the latest `InstanceChangeOperation` for the instance
   (`None` if none). Exposed on `InstanceEditorFields` as
   `draftHeadToken: UUID`.

2. **Extend `@instance` and `@context` directives**
   ([paths/schema.py:48-75](../../../paths/schema.py#L48-L75)) with two args:
   - `preview: String` — `"draft"` or `"published"` (default `"published"`).
     The editor passes `"draft"`; public queries pass nothing.
   - `version: UUID` — required on editing mutations; ignored on queries.

   `InstanceContextInput` (used by `@context`) gets the same two fields.
   The directive resolver stores them on the request-scoped context, which
   `resolve_instance` then consults.

3. **Stale-token check.** Inside the directive's wrapper, after opening the
   transaction and `SELECT FOR UPDATE` on `InstanceConfig`, compare the
   supplied `version` UUID against `ic.draft_head_token`. On mismatch, raise
   `StaleVersionError` with `currentHeadToken` and `latestOperations`
   (last ~5) in extensions. Equality check (UUIDs aren't ordered).

4. **Directive wrapper opens `change_operation`.** The same wrapper that
   enforces the version check also opens `change_operation(ic, user,
   action, source='graphql')` and attaches the operation to
   `info.context.operation` (the ContextVar is already set by the
   context-manager). Every mutation body then:
   - Calls `record_change(obj, action, before, after)` for each write
     (reads the current operation from the ContextVar).
   - Returns a uniform `MutationPayload { ok, ...entity, newHeadToken, invalidatedNodeIds }`
     (last two populated by phase 4).

5. **Closed mutation set and inverses** (all row-level writes inside one
   `change_operation(...)` block share the same `InstanceChangeOperation`):

   | Action (GraphQL mutation) | Writes | Inverse |
   |---|---|---|
   | `node.create` | Insert `NodeConfig` | Delete by uuid |
   | `node.update` | Update `NodeConfig` fields | Update with prior values |
   | `node.delete` | Delete `NodeConfig` + cascade edges + ports | Recreate node + edges + ports |
   | `edge.create` | Insert `NodeEdge` | Delete by uuid |
   | `edge.update` | Update `NodeEdge` transformations / tags / (future: label) | Update with prior values |
   | `edge.delete` | Delete `NodeEdge` | Recreate `NodeEdge` |
   | `node.dataset_ports.create` | Insert `DatasetPort` | Delete by uuid |
   | `node.dataset_ports.update` | Update `DatasetPort` fields | Update with prior values |
   | `node.dataset_ports.delete` | Delete `DatasetPort` | Recreate `DatasetPort` |
   | `node.dataset_ports.ref_update` | Update `DatasetPort.dataset_revision` (direction-neutral) | Restore prior pin |
   | `scenario.create` | Append to `InstanceSpec.scenarios` | Remove by uuid |
   | `scenario.update` | Update embedded `Scenario` in spec | Restore prior values |
   | `scenario.delete` | Remove from `InstanceSpec.scenarios` | Re-insert at prior index |
   | `action_group.create` / `.update` / `.delete` | Edit `InstanceSpec.action_groups` | Symmetric |
   | `param.update` | Update a param value (global or per-node) | Restore prior value |
   | `dataset.datapoints.edit` | Delegated to `DatasetEditorMutation` + auto `node.dataset_ports.ref_update` per affected port | Revert dataset rev + restore prior pins |
   | `instance.save_draft` | Creates `Revision` (unpublished) with a name | *not undoable* |
   | `instance.publish` | Publish revision | `instance.revert` |
   | `instance.revert` | Restore from snapshot | *not undoable (barrier, v1)* |
   | `instance.undo` | Apply inverse of latest operation; set `superseded_by` | *not undoable (v1)* |

   All scenario / action_group / param operations edit the embedded
   Pydantic lists on `InstanceConfig.spec` via `queryset.update()` to
   sidestep the `ClusterableModel.save()` trap — see
   [docs/trailhead/tools.md](../tools.md#the-clusterablemodel-save-trap).

   Action strings follow Wagtail's dotted-namespace convention. No
   `trailhead.` prefix — "Trailhead" is an ephemeral project codename;
   action ids are persisted data and need to outlive it.

6. **Entity identity = UUID, not pk.** Mutation inputs that currently take
   `id: sb.ID` interpreted as numeric pk need a UUID variant (keep the pk
   form for backwards compat during transition, add uuid as primary).
   `NodeConfig`, `NodeEdge`, `DatasetPort`, `Scenario`, `ActionGroup`
   already inherit `UUIDIdentifiedModel` so no model migration needed.

### Tests

- Two parallel clients, same stale token → second mutation rejected with
  `StaleVersionError`.
- Deleting a node creates exactly one `InstanceChangeOperation` with the
  expected number of IMLE children.

### Dependencies

Phase 2.5 (review of the PoC pattern must have closed green).

---

## Phase 4 — Resolver split & compute invalidation

**Goal:** Separate the compute surface (draft vs. published) from the
authoring surface. Wire up per-node content hashes and descendant
invalidation.

### Changes

1. **`resolve_instance` branch.** Single decision point —
   [paths/schema_context.py](../../../paths/schema_context.py) around line
   287 (`enter_instance_context`) and `InstanceConfig._get_instance`. New
   enum `PreferredInstanceSource { PUBLISHED, DRAFT }`:
   - Default: `PUBLISHED` (fallback to `DRAFT` only if no published
     revision exists, i.e. brand-new instance).
   - `@instance(preview: "draft")` / `@context(... preview: "draft")` →
     `DRAFT`. Requires editor permission; rejected with `PermissionDenied`
     otherwise.
   - `_get_instance(source)` caches per `(identifier, source, draft_head_token_if_draft)`.
     Mutations bump the token → subsequent requests rebuild the Instance.
   - For YAML-sourced instances, `source` is irrelevant — YAML is the base
     and the ORM overlay (NodeConfig overrides) is applied as today.

2. **Resolver split audit:**
   - **Compute/read (from `Instance`/`Context`, works for both sources):**
     `InstanceType.nodes`, `.goals`, `NodeType.metric`, `.outputDataset`,
     `.upstreamNodes`, `.impactMetric`, `.outputMetric`,
     `parameters.value`, `dimensions.categories` — everything that reads
     `root.context.nodes[...]`. Already parameterized by the active
     Instance; no change needed once `_get_instance` is source-aware.
   - **Authoring (ORM, draft-only):** `InstanceEditorFields.edges`,
     `.dataset_ports`, `.datasets`, `.dimensions` (editor variants),
     `.graph_layout`; per-`NodeConfig` `color/order/i18n`; `modelInstance.spec`;
     revisions list; log entries. Already under `InstanceEditorFields` —
     keep the `editor` scoping.
   - **Ambiguous — flagged:** `Instance.action_list_page`,
     `Instance.intro_content`, `Instance.hostname`. These resolve off
     `InstanceConfig` (live CMS content, not the model graph). Leave them
     reading the live row; document that CMS content is **explicitly not**
     under the draft/publish machinery in v1.

3. **Per-node `contentHash` on runtime `NodeType`**
   ([nodes/graphql/types/node.py](../../../nodes/graphql/types/node.py)):
   ```python
   @sb.field
   def content_hash(self, root, info) -> str:
       state = info.context.hashing_state  # cached per request
       return root.cache.calculate_hash(state).hex()
   ```
   This is exactly what the frontend compares against its cache.

4. **Invalidation helper** `nodes/invalidation.py`:
   ```python
   def compute_descendants(ic: InstanceConfig, changed_node_ids: Iterable[str]) -> list[str]:
       edges = NodeEdge.objects.filter(instance=ic).values('from_node_id', 'to_node_id')
       adj = build_adjacency(edges)
       return bfs_descendants(adj, changed_node_ids)
   ```
   Pure structural walk; no compute triggered.

5. **Mutation payload extension.** After each mutation:
   - Compute affected node set: directly changed node + transitive
     descendants for edits that change outputs; only the directly-affected
     node for metadata-only edits (e.g. `update_node(color=...)`).
   - Return `newHeadToken`, `invalidatedNodeIds: list[ID]`.
   - Also invalidate the runtime `Context.cache` for those nodes in the
     current request's Instance. Cross-request invalidation is driven by
     `draft_head_token` changing (instance rebuild on mismatch).

### Tests

- Mutate a formula node → `invalidatedNodeIds` contains all downstream
  outcome nodes.
- Public resolver of a node on a freshly published instance returns
  pre-draft values even after subsequent draft mutations.

### Dependencies

Phase 3 (needs operation logging to bump the token).

---

## Phase 5 — Publish / revert / undo / named drafts

**Goal:** Full state-machine mechanics.

### Changes

1. **`instance.publish(user)`** — mostly exists:
   - Build current `InstanceSnapshot` (phase 1).
   - `ic.save_revision(user=user, log_action=False)` → `Revision` row with
     `content = { ... , model_snapshot: snapshot_dict }`.
   - `ic.publish(revision, user=user, log_action=False)` → sets `live=True`,
     `live_revision_id=revision.pk`, `has_unpublished_changes=False`,
     `last_published_at=now()`.
   - Emit one `InstanceChangeOperation { action: 'instance.publish' }`
     (terminal — recorded but not undoable) with no IMLE children (nothing
     to invert per-row).
   - Invalidate the published compute cache keyed by `InstanceConfig`.
   - Permission: `publish` action (see phase 6).

2. **`instance.save_draft(name)`**:
   - `ic.save_revision(user=user, log_action=False)` + attach label via new
     tiny `RevisionLabel` model `{ revision_fk, label, created_by }`.
     (Side-table vs. stuffing into `Revision.content`: the side-table wins
     on queryability and on not bloating every revision.)
   - `has_unpublished_changes` stays true.
   - Emit operation `instance.save_draft` with label in its action metadata.

3. **`instance.revert(revision_id)`** — diff-based apply, not
   wipe-and-recreate:
   - Atomic, with `SELECT FOR UPDATE` on `InstanceConfig`.
   - `snapshot = InstanceSnapshot(**revision.content['model_snapshot'])`.
   - For each `(uuid → NodeSnapshot)` in `snapshot.nodes`: upsert
     `NodeConfig` by uuid via `from_serializable_data()`. Rows in DB
     whose uuid isn't in the snapshot: delete.
   - Same for `NodeEdge`, `DatasetPort` (including restoring
     `dataset_revision` pins).
   - `InstanceConfig.spec` ← `snapshot.spec` (which includes scenarios,
     action_groups, params, etc.). Written via `queryset.update()` to
     avoid the ClusterableModel trap.
   - Dataset rows themselves are never touched.
   - Emit an `instance.revert` operation with metadata
     `{ from_revision_id, to_revision_id }`. The operation is the undo
     barrier — `instance.undo` walks back up operations and stops at the
     most recent `instance.revert`.
   - Sets `has_unpublished_changes = (revision != live_revision)`.
   - **FK concerns:** `NodeConfig.indicator_node` is self-FK `SET_NULL` —
     if an indicator target got deleted between publish and now, the null
     is accepted. Two-pass apply (rows first, then FKs) resolves the
     "A points at B, B is being recreated" case. Nothing outside the
     instance subgraph holds a hard FK to `NodeConfig` or `NodeEdge`.
     Pages reference nodes by **string identifier** — not revertable;
     may silently break if an identifier disappears. Document as v1
     limitation.

4. **`instance.undo()`**:
   - Atomic, `SELECT FOR UPDATE` on `InstanceConfig`.
   - Find the latest `InstanceChangeOperation` for the instance where
     `action` is in the *undoable set* (not `instance.publish`,
     `instance.revert`, `instance.undo`, `instance.save_draft`) and
     `superseded_by IS NULL`. Refuses if that search crosses an
     `instance.revert` operation (barrier).
   - Fetch all `InstanceModelLogEntry` rows for that operation, ordered
     by `id DESC`.
   - Apply each entry's inverse from the `before` payload:
     - `after is None` (was create) → delete by uuid.
     - `before is None` (was delete) → `Model.from_serializable_data(before)`.
     - both present (was update) → `Model.from_serializable_data(before)`
       for the existing row identified by uuid.
   - Set `operation.superseded_by = <new undo operation>`. *Do not* emit
     mirror IMLE rows — the `superseded_by` link is the history trail,
     which preserves "this is what the user did" vs. "this is what the
     system did to undo it" semantics in the UI.
   - Emit one `instance.undo` operation with metadata
     `{ undone_operation_id }`.
   - Bumps `draft_head_token`.

5. **History GraphQL:**
   - `InstanceEditorFields.history(limit, before)` → list of
     `InstanceChangeOperation` rows ordered by `-created_at`:
     `{ uuid, action, user, createdAt, affectedObjectCount, supersededBy }`.
   - `InstanceEditorFields.revisions` → `Revision` list with named-draft
     labels, `isLive` flag, `createdBy`, `createdAt`.

### Tests

- Publish → revert → state identical to pre-publish.
- Cascade delete + undo → all `1+N` rows restored.
- Two sequential undos → both groups inverted.
- Revert is a barrier: subsequent undo errors with a clear message.
- Published resolver returns old values across a draft mutation.

### Dependencies

Phases 1–4.

---

## Phase 6 — Permissions & migration

**Goal:** Wire roles onto draft/publish actions; migrate existing
DB-sourced instances onto the published-snapshot path.

### Changes

1. **Permission policy extension** on `InstanceConfigPermissionPolicy`
   ([nodes/models.py:167](../../../nodes/models.py#L167)):
   - New action (on top of `view/change/delete`): `publish`.
   - `user_has_perm('publish', obj)` → `is_admin(user, obj) or is_framework_admin(user, obj)`.
   - `revert` piggybacks on `publish`.
   - `save_named_draft` and draft-edit mutations gate on `change`
     (existing reviewer/admin roles).
   - `undo` gates on `change`, plus: the caller owns the latest operation
     group OR is admin. Configurable via
     `settings.TRAILHEAD_UNDO_ALLOW_OTHERS` (default `False`).
   - Viewers see only published. Anonymous public access to published
     instances is preserved via the existing `anon_has_perm('view', obj) + live=True` path.
   - `@instance(preview: "draft")` requires `change` on the target instance.

2. **Migration command** `python manage.py migrate_draft_state_initial`:
   - For every `InstanceConfig` with `config_source='database'`:
     - `build_instance_snapshot(ic)` → `save_revision(user=None, log_action=False)` →
       `publish(revision, user=None, log_action=False)`.
     - Sets `live=True`, `has_unpublished_changes=False`, populates
       `live_revision`, sets `first_published_at = created_at`.
   - YAML-sourced instances: leave `live=False`; publish path irrelevant
     (their compute still loads from YAML).
   - Idempotent (skip if `live_revision` already populated).
   - Transition state: each DB-sourced instance jumps to
     `live + no draft changes`; any subsequent mutation flips
     `has_unpublished_changes=True`. Public behavior byte-identical across
     rollout.

3. **Flip the public default.** `InstanceConfig._get_instance`'s default
   becomes `PUBLISHED` for DB-sourced instances with `live_revision` set.
   Emergency fallback env flag `PATHS_SERVE_PUBLISHED_FROM_SNAPSHOT`
   (default `True`) so we can revert the public path to the draft tables
   if a blob-hydration bug shows up in prod.

### Tests

- Migration on a fresh fixture DB produces one published revision per
  DB-sourced instance.
- Compute output identical before and after migration for every
  test-instance fixture.

### Dependencies

All prior phases; runnable on staging after phase 5 is green.

---

## Cross-cutting concerns

- **Transactions.** Every mutation / undo / publish / revert runs inside
  `transaction.atomic()` with `SELECT FOR UPDATE` on the `InstanceConfig`
  row. Doubles as the lock for the version-token check.
- **Cache layer.** `Context.cache` is per-process; `draft_head_token`
  bumping is the cross-process invalidation signal for other gunicorn
  workers. The existing `CacheablePathsModel` / `cache_invalidated_at`
  field on `InstanceConfig` gets bumped on mutation to purge any external
  caches.
- **Dataset integration workflow.** Integrated path (edit dataset from
  inside model editor): `DatasetEditorMutation` is already reachable from
  `InstanceEditorMutation.dataset_editor`. Nested calls read the active
  operation from the ContextVar that the outer `change_operation(...)`
  set — no `_parent_operation_id` plumbing needed. The nested
  `dataset.datapoints.edit` IMLE and the parent-triggered
  `node.dataset_ports.ref_update` IMLEs all share the same operation FK
  automatically.
- **Backwards compat.** `publish_model_instance` and `revert_model_instance`
  stubs already exist in `editor.py` — names stay, signatures evolve.
  Frontend sees only additive GraphQL changes.

## Open questions

1. **Concurrent sessions of the same user.** Directive rejects stale token
   from *any* source, including the same user's second browser tab. UX
   papercut potential; v1 accepts it. Revisit if real-world friction.
2. **`NodeConfig.indicator_node` across revert.** Two-pass apply (rows
   first, then FKs) handles it. Confirm by testing on a real instance with
   indicator relationships.
3. **Log entry retention.** Infinite growth per instance. No retention
   policy proposed in v1. Flag for a later phase — likely a
   `clear_history(before_revision)` admin action.
4. **Redo.** Out of scope for v1. Enabling later = new log entry type
   `trailhead.redo` + relax the "undo not undoable" rule. No data
   migration.
5. **Dataset "newer versions available" query.** Needs an efficient
   `stale_dataset_pins(ic) -> list[(Dataset, current_pin, newest_revision)]`
   resolver. Straightforward; decide placement
   (`InstanceEditorFields.staleDatasetPins` seems natural).
6. **YAML-sourced instances.** Current `editor.py` already gates mutations
   on `config_source == 'database'`. Assumption: YAML instances never get
   draft/publish. Confirm before phase 6 migration.
