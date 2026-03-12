# Trailhead: Action Authoring Backend Backlog

## Summary

This document lists the backend capabilities needed to support the first
Trailhead action-authoring flow described in
[action-authoring-first-flow.md](/home/jey/sync/devel/kausal-paths/docs/trailhead/action-authoring-first-flow.md).

The target flow is intentionally narrow:

- copy an existing Aarhus-style action
- rename it
- connect it to new precomputed data from Excel
- confirm output wiring
- preview the result
- save it

This is not a backlog for the whole future graph editor. It is a backlog for
the smallest backend slice that makes that flow real.


## Current Baseline

Some important pieces already exist:

- Runtime node and edge specs already exist in `NodeSpec`, `InputPortDef`,
  `OutputPortDef`, and related models.
- Runtime export into DB-backed specs already exists via
  [`nodes/spec_export.py`](/home/jey/sync/devel/kausal-paths/nodes/spec_export.py).
- There is already a Strawberry model-editor schema in
  [`nodes/schema_model_editor.py`](/home/jey/sync/devel/kausal-paths/nodes/schema_model_editor.py)
  that can:
  - fetch a model instance
  - create nodes
  - update nodes
  - create edges
  - mutate scenarios
- The main nodes GraphQL schema already exposes lightweight dataset references,
  bound dataset ports, node specs, and edge bindings in
  [`nodes/schema.py`](/home/jey/sync/devel/kausal-paths/nodes/schema.py).

That means the missing work is not “start a backend from zero”.

The missing work is:

- editor-grade persistence metadata
- dataset read/write APIs
- action-copy workflow APIs
- draft validation and preview
- a save boundary that is more useful than many small low-level mutations


## Principles For The First Slice

- Optimize for the copied-action flow, not generic authoring.
- Prefer high-level workflow mutations over pushing orchestration into the UI.
- Keep validation server-side and structured for friendly UI reporting.
- Support draft computation before persistence.
- Avoid exposing raw YAML-era concepts as the primary contract, even when the
  implementation still maps onto them internally.


## Priority 1: Graph Persistence And Editor Metadata

### 1. Persist node canvas position

Needed because the editor must save and restore graph layout.

Minimum requirement:

- persist `x` / `y` coordinates for each node

Preferred shape:

- store layout metadata in the node spec, not as purely transient frontend state
- ideally reserve space for future UI metadata, even if v1 only uses `x` and `y`

Likely touchpoints:

- `NodeSpec`
- spec export/import path
- GraphQL read/write shape for model editor

### 2. Expose persisted layout in model editor queries

The model instance query returned to the editor should include:

- node id
- node metadata
- ports
- edges
- saved layout coordinates


## Priority 2: Dataset Read APIs For The Editor

The first action-authoring flow depends heavily on dataset discovery and schema
inspection.

### 3. Dataset browser query

Add a query surface for listing candidate datasets for an action.

The UI needs to browse datasets by:

- identifier
- source type
- whether they are likely compatible with the copied action

Minimum payload:

- dataset id / uuid
- identifier
- display label if available
- source reference

### 4. Dataset schema inspection query

Add a query to inspect one dataset in enough detail for the editor.

The UI needs:

- metrics
- dimensions
- dimension categories
- selector-like columns such as `action`
- units
- optionally a small row preview

### 5. Autocomplete/select support queries

Add or formalize query endpoints for editor inputs such as:

- action groups
- datasets
- dataset metrics
- dimensions
- dimension categories
- candidate target nodes
- candidate ports / output metrics

These can be separate queries or filtered views over broader queries, but the
editor needs them as first-class backend support.


## Priority 3: Dataset Mutation APIs

The copied-action flow is blocked unless the user can prepare data from inside
the same product experience.

### 6. Create dataset shell

Support creating a new dedicated dataset for an action when the user does not
want to append to an existing shared table.

### 7. Extend existing dataset schema

Support mutations to:

- add a missing metric
- add a missing dimension category

This is especially important for shared action tables selected by
`action = <selector_value>`.

### 8. Upload / paste data rows

Support creating or replacing dataset rows from the action-authoring workflow.

The backend should accept data shaped like:

- year
- dimensions
- metrics
- optional forecast indicator

The input transport can evolve later; for the first flow the important thing is
that the UI can paste/import Excel-derived rows without leaving the editor.

### 9. Dataset selector/filter binding

Support binding the copied action to a dataset slice, typically:

- dataset = `aarhus/energy_actions`
- selector column = `action`
- selector value = `my_new_action`

This may map onto `InputDatasetDef.filters` internally, but the editor-facing
API should treat it as a clear data-source configuration task.


## Priority 4: Action Copy Workflow

### 10. Copy action as a backend operation

Add a first-class backend operation for:

- source action id
- new action id
- optional new labels and group

The copy operation should duplicate:

- structural metadata
- dimensions
- output metrics
- output edge suggestions

It should not blindly preserve:

- old dataset selector value as final truth

The returned object should be a draft-ready copied action.

### 11. Represent copied edges as suggestions

Copied edge mappings should survive the copy, but not as unquestioned final
state.

The backend should support a state that is effectively:

- copied from source
- editable
- re-validatable after data-source changes


## Priority 5: Draft Validation And Preview

This is the most important behavioral gap.

### 12. Validate a draft action before save

Add a validation endpoint or mutation for draft action state.

Validation should cover:

- missing dataset rows
- missing metrics
- missing categories
- invalid selector config
- metric compatibility
- quantity compatibility
- unit dimensionality mismatch
- dimension mismatch requiring flattening or mapping
- invalid edge tags or target configuration

Validation responses should be structured for UI rendering, with enough detail
to point the user at the failing part.

### 13. Compute draft outputs

Add a preview endpoint that can compute the copied action in draft state before
it is persisted.

Minimum useful preview:

- action output rows

Better preview:

- action output rows
- impact on selected target nodes

### 14. Friendly error reporting

Editor preview failures should not surface as generic tracebacks or opaque
GraphQL errors.

The UI needs error payloads that identify:

- what failed
- where it failed
- whether the failure is blocking or warning-level


## Priority 6: Persistence Boundary

### 15. Save action draft

Add a higher-level save/commit mutation for the edited action.

This mutation should persist:

- action metadata
- data-source binding
- dataset-linked configuration
- output edge mappings
- layout metadata if applicable

The goal is to avoid forcing the UI to coordinate many low-level mutations in a
fragile order.

### 16. Return canonical saved state

After save, the backend should return the action in the same canonical shape
the editor uses for reads.

This avoids client/server drift caused by normalization on save.


## Cross-Cutting Concerns

### 17. Permissions

The backend needs explicit authorization for:

- reading editor model state
- editing node graph state
- editing datasets
- previewing draft computations
- saving structural changes

### 18. Revision / audit trail

Model edits are high-value structural changes. The first slice does not need a
full publishing workflow, but it should not ignore traceability.

Minimum expectation:

- saved edits are attributable
- state changes can be inspected later

### 19. Save/recompute coherence

The system should behave coherently when:

- a copied action changes dataset binding
- edge suggestions are changed
- draft preview is run repeatedly
- saved state is reloaded immediately after commit


## Suggested Sequencing

### Phase 1: unblock editor rendering and basic persistence

1. Persist node `x` / `y`
2. Expose layout in model editor queries
3. Add dataset browser + schema inspection queries

### Phase 2: unblock copied-action data preparation

4. Add dataset schema extension mutations
5. Add dataset row import/paste mutations
6. Add dataset selector/filter binding support

### Phase 3: unblock real authoring workflow

7. Add backend `copyAction` operation
8. Add draft validation
9. Add draft output preview

### Phase 4: make it robust enough for UI integration

10. Add high-level save/commit mutation
11. Improve structured validation and error reporting
12. Tighten permissions and revision behavior


## Aarhus-Specific Scope For The First Backend Slice

The first backend slice should explicitly support the patterns seen in
`aarhus-c4c`, especially:

- shared action datasets filtered by `action`
- output metrics such as `currency`, `energy`, `emissions`, `mileage`, `fraction`
- output wiring into sector cost nodes and existing domain nodes

It does not need to support every possible future node authoring case.


## Out Of Scope For This Backlog

- full blank-from-scratch arbitrary node authoring
- custom compute-class authoring
- generalized formula authoring UX
- fully generic dataset-modeling workflows unrelated to copied actions
- complete publication/review workflow design
