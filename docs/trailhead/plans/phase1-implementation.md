Trailhead Phase 1: Implementation Plan
======================================

This document tracks the current state of the Trailhead Phase 1 work in the
repo, rather than the original greenfield implementation sequence.

Context: Trailhead adds DB-backed model configuration to Kausal Paths. YAML
instances continue to work, and the current DB-backed runtime path still uses a
legacy dict round-trip through `InstanceLoader`.


Current State
-------------

### Landed in the repo

- `InstanceConfig.config_source` exists, with DB-backed loading wired in.
- `InstanceConfig.spec` and `NodeConfig.spec` exist and are validated with
  Pydantic models under `nodes/defs/`.
- `NodeEdge`, `DatasetPort`, `ActionGroup`, and `Scenario` exist as Django
  models and are used by the current DB-backed path.
- `nodes/instance_from_db.py` serializes DB-backed models into the YAML-shaped
  dict that `InstanceLoader` already knows how to consume.
- `test_instance` has `--spec-only` and `--dry-run` for broad validation
  sweeps without updating the saved state file.

### Spec coverage now in place

`InstanceSpec` currently carries at least:

- `uuid`
- `identifier`
- `name`
- language configuration
- `theme_identifier`
- terms, pages, result excels
- impact overviews
- normalizations

`NodeSpec` currently carries at least:

- `uuid`
- `identifier`
- `name`
- `color`
- `order`
- `is_visible`
- type configuration
- goals
- visualizations
- `allow_nulls`
- `node_group`

Action-related spec fields now include:

- `decision_level`
- `group`
- `parent`
- `no_effect_value`

### Runtime/spec splits already done

- `ImpactOverview` is split into:
  - `ImpactOverviewSpec` for persisted/spec data
  - runtime `ImpactOverview` for resolved node references and behavior
- `Normalization` is split into:
  - `NormalizationSpec`
  - `NormalizationQuantitySpec`
  - runtime `Normalization`

### Export / sync / schema work already done

- Runtime export to spec is implemented in `nodes/spec_export.py`.
- `InstanceConfig.save()` and `NodeConfig.save()` sync duplicated identity /
  display fields into `spec`.
- `nodes/schema_spec.py` exposes the spec mirror through GraphQL as JSON and
  scalar fields.
- The DB-backed serializer includes goals, visualizations, impact overviews,
  normalizations, and the duplicated identity/display fields.


What Phase 1 Still Means
------------------------

Phase 1 is no longer about creating the first DB-backed path. That path exists.
The remaining work is about making the spec layer complete and trustworthy
enough that it can replace the legacy dict/YAML compatibility path.

In practice that means:

1. Finish the spec/runtime boundary for the remaining runtime-heavy structures.
2. Validate the new spec models against real YAML instances.
3. Replace the dict round-trip with direct DB-to-runtime construction.


Completed Milestones
--------------------

### 1. Relational storage model

Done for the current Phase 1 scope:

- DB-backed instance and node configs exist.
- Edges and dataset bindings remain relational through `NodeEdge` and
  `DatasetPort`.
- Action groups and scenarios exist as first-class models.

### 2. Pydantic spec layer

Done for the current implemented fields:

- `nodes/defs/` contains the active spec models.
- Goals and visualizations are represented in `NodeSpec`.
- Impact overviews and normalizations are represented in `InstanceSpec`.
- Spec models now use the newer i18n input types and `I18nBaseModel` where
  appropriate.

### 3. Compatibility loading path

Done, but intentionally transitional:

- DB-backed instances can be loaded today.
- The current path still serializes DB state into a YAML-shaped dict and then
  goes through `InstanceLoader`.

### 4. Validation tooling

Done:

- Focused tests cover the newer spec fields in both pure-spec and DB-backed
  paths.
- `python manage.py test_instance --spec-only --dry-run` exists for sweeping
  instance initialization without output comparison or state-file writes.


Open Work
---------

### A. Sweep real instances through the new validation path

Run `python manage.py test_instance --spec-only --dry-run` broadly and fix the
remaining schema mismatches that only appear in real YAML configs.

This is the fastest way to harden the spec models before replacing the loader
compatibility layer.

### B. Finish migration cleanup inside `InstanceLoader`

`InstanceLoader` should own legacy YAML field renames where they still exist,
and the spec models should stay spec-shaped.

This cleanup is partly done already for:

- impact overviews
- normalizations

There are still some remaining compatibility edges to iron out as the old
YAML-shaped assumptions are removed from individual models.

### C. Decide the long-term boundary for graph topology and dataset bindings

For now, keep:

- graph topology in `NodeEdge`
- dataset-node bindings in `DatasetPort`

Open question:

- do these remain relational-only in Trailhead, or do we eventually add spec
  mirrors for fully self-contained JSON snapshots?

This does not need to block the next implementation step.

### D. Replace the dict round-trip with direct DB-to-runtime construction

This is the main remaining engineering milestone.

Target shape:

- `InstanceConfig.create_instance()` or equivalent DB-native builder
- `NodeConfig.create_node(context)` or equivalent builder helper
- direct consumption of `InstanceConfig`, `NodeConfig`, `NodeEdge`, and
  `DatasetPort`
- no YAML-shaped intermediary for DB-backed instances

This work should reuse the spec models and the runtime/spec splits already
landed for impact overviews and normalizations.


Suggested Next Steps
--------------------

### Next step 1: broad validation sweep

Run the spec-only instance initialization path across all or most instances:

    python manage.py test_instance --spec-only --dry-run

Goal:

- surface remaining schema mismatches early
- fix them while the compatibility path is still in place

### Next step 2: introduce the DB-native runtime builder

Start with a minimal path that constructs:

- the runtime `Instance`
- the runtime `Context`
- plain nodes from `NodeConfig.spec`
- then attaches `NodeEdge` and `DatasetPort`

Suggested implementation staging:

1. Create instance/context directly from `InstanceConfig` + `InstanceSpec`.
2. Construct simple nodes directly from `NodeConfig` + `NodeSpec`.
3. Wire edges and dataset ports from relational models.
4. Port actions and remaining special cases.
5. Switch `config_source='database'` away from `nodes/instance_from_db.py`.

### Next step 3: remove transitional pieces only after parity

After the DB-native builder is passing the same validation and test coverage:

- delete `nodes/instance_from_db.py`
- remove DB-path dependence on YAML-shaped dicts
- keep YAML loading as its own explicit compatibility path


Notes
-----

- Rich-text/content fields on nodes are intentionally not duplicated into
  `NodeSpec` yet. Revisit that after the node-graph editor direction is clearer.
- The current direction is to keep backward compatibility behind explicit
  loader/import steps, not inside the common spec models.
