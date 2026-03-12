Trailhead Phase 1: Implementation Plan
=======================================

This plan is for autonomous implementation. Read this file first, then
`docs/trailhead/data-model.md` for the full schema design.

Context: We are adding DB-backed model configuration to Kausal Paths.
YAML-sourced instances continue to work unchanged. DB-sourced instances
("Trailhead") coexist alongside them.

Branch: feat/trailhead (or feat/trailhead/* for sub-features)
Base: main


Phase 1A: Django Models and Migrations
---------------------------------------

**Goal:** Create the new models, add fields to existing models, generate
migrations. No behavior changes — YAML instances still work exactly as before.

### Steps:

1. Add new fields to `InstanceConfig` (nodes/models.py):
   - config_source (CharField, choices: yaml/database, default='yaml')
   - reference_year, minimum_historical_year, maximum_historical_year,
     target_year, model_end_year (IntegerField, nullable)
   - emission_unit (CharField, nullable)
   - dataset_repo_url, dataset_repo_commit, dataset_repo_dvc_remote (nullable)
   - features (JSONField, default=dict)
   - parameters (JSONField, default=list) — global parameter definitions
   - extra (JSONField, default=dict) — catch-all for YAML fields not yet modeled

2. Add new fields to `NodeConfig` (nodes/models.py):
   - node_type (CharField, choices: formula/action, nullable)
   - unit (CharField, nullable)
   - quantity (CharField, nullable)
   - input_ports (JSONField, default=list)
   - output_ports (JSONField, default=list)
   - pipeline (JSONField, nullable)
   - formula (CharField, nullable)
   - action_group (FK to ActionGroup, nullable)
   - decision_level (CharField, nullable)
   - is_outcome (BooleanField, default=False)
   - node_group (CharField, nullable)
   - extra (JSONField, default=dict)

3. Create `NodeEdge` model (new, in nodes/models.py or nodes/edge_models.py):
   - See data-model.md section 2 for full definition
   - instance, from_node, from_port, to_node, to_port, transformations, tags
   - UniqueConstraint on (to_node, to_port)

4. Create `DatasetPort` model (new):
   - See data-model.md section 3
   - instance, node, port_id, dataset, metric
   - UniqueConstraint on (node, port_id)

5. Create `ActionGroup` model (new):
   - See data-model.md section 4
   - instance, identifier, name, color, order, i18n

6. Create `Scenario` model (new):
   - See data-model.md section 8
   - instance, identifier, name, description, kind, all_actions_enabled,
     parameter_overrides (JSONField), i18n

7. Generate migrations:
   - python manage.py makemigrations nodes
   - python manage.py migrate
   - Verify existing tests still pass

### Validation:
- `python -m pytest --reuse-db` passes
- `mypy . | mypy-baseline filter` has no new errors
- `ruff check .` clean
- Existing YAML-based instances load and work as before


Phase 1B: Pydantic Schemas for JSON Fields
-------------------------------------------

**Goal:** Define Pydantic models that validate the JSONField contents.
These are used at the application layer — not at the DB layer.

Look at existing work in `nodes/ser/` for patterns, especially i18n handling.

### Files to create/modify:

1. `nodes/ser/port_schema.py` — InputPortDef, OutputPortDef
2. `nodes/ser/parameter_schema.py` — ParameterDef (shared by global and node-local)
3. `nodes/ser/edge_schema.py` — EdgeTransformation
4. `nodes/ser/scenario_schema.py` — ScenarioParameterOverride

These should be pure Pydantic models with no Django dependencies.
They validate JSON going into and coming out of the JSONFields.


Phase 1C: InstanceLoader.from_db()
------------------------------------

**Goal:** Implement the new loading path that reads relational models and
constructs the runtime Instance + Context + node graph.

### Approach:

The simplest implementation: serialize the relational models to a dict
in the same shape that InstanceLoader already consumes (i.e., the parsed
YAML structure), then feed it through the existing loading code.

This means:
1. Write `serialize_instance_to_dict(instance_config) -> dict`
   that reads all NodeConfig, NodeEdge, DatasetPort, Scenario, ActionGroup
   rows and produces a YAML-equivalent dict.
2. In `InstanceLoader.__init__`, accept this dict the same way it accepts
   parsed YAML data.
3. `InstanceConfig._create_from_config()` checks `config_source` and
   calls the appropriate path.

### Why this approach:
- Reuses all existing loading logic (node class resolution, edge setup,
  dimension setup, etc.)
- Minimizes risk of divergence between YAML and DB loading paths
- Easy to test: import a YAML config to DB, serialize back, compare

### Later optimization:
Once stable, from_db() can construct runtime objects directly from
the relational models without the dict intermediary. But correctness first.


Phase 1D: YAML-to-DB Import Command
-------------------------------------

**Goal:** Management command to import a YAML config into the DB models.

    python manage.py import_yaml_to_db configs/espoo.yaml

### Steps:
1. Load the YAML config using existing InstanceYAMLConfig
2. Create/update InstanceConfig with config_source='database'
3. Create NodeConfig rows for all nodes
4. Create NodeEdge rows for all edges (from input_nodes/output_nodes)
5. Create DatasetPort rows for dataset connections
6. Create ActionGroup rows
7. Create Scenario rows
8. Store anything not explicitly modeled in the `extra` JSONField

### Validation:
- Import a config, then load it via from_db()
- Compare the runtime Instance/Context with one loaded via from_yaml()
- Node graph structure should be identical
- Start with a simple config (e.g., helsinki.yaml), then try a complex one
  (e.g., espoo.yaml)


Phase 1E: Serialization and Publish Snapshot
----------------------------------------------

**Goal:** Implement the serialize/deserialize cycle for publishing.

1. `serialize_instance(instance_config) -> dict` — full model snapshot
2. `deserialize_instance(instance_config, data: dict)` — restore from snapshot
3. Wire up DraftStateMixin on InstanceConfig
4. Add `publish_instance()` method that snapshots and publishes

This builds on 1C's serialization work.


Phase 1F: GraphQL Schema
--------------------------

**Goal:** Queries and mutations for the editor UI.

### Queries:
- modelInstance(id) → full model with nodes, edges, datasets, scenarios
- node(id) → single node with ports, connections, pipeline
- Could also be a single large query that returns the entire graph
  (more efficient for the editor's initial load)

### Mutations:
- createNode, updateNode, deleteNode
- createEdge, deleteEdge
- createDatasetPort, deleteDatasetPort
- updateScenario, createScenario, deleteScenario
- updateInstanceConfig (year params, global params)
- publishInstance
- revertToDraft

Use Strawberry (not Graphene) for new schema — the project is migrating
toward Strawberry.


Important Notes
---------------

- DO NOT modify the YAML loading path. YAML instances must continue
  to work exactly as before.
- All new fields on existing models must be nullable or have defaults,
  so existing data is unaffected by migrations.
- Use `ruff format .` and `ruff check .` before committing.
- Run `python -m pytest --reuse-db` after each significant change.
- Commit frequently with clear messages.
- If stuck on something, leave a TODO comment and move to the next task.
- The `extra` JSONField is the escape hatch — anything from the YAML
  that doesn't have an explicit field goes there.
