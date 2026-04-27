Trailhead: Data Model Design
=============================

This document describes the Django model changes for storing model configurations
in the database. It covers the graph structure (nodes, edges, dataset connections)
and the conventions for what lives in relational fields vs. JSONFields.

Design principles:
- Graph topology is relational (FKs, queryable, referential integrity)
- Node internals are JSON (pipeline operations, port definitions, edge transformations)
- Pydantic validates all JSON structures at the application layer
- New models use clean names; `NodeConfig` keeps its name for now (rename later)


1. NodeConfig Additions
-----------------------

The existing `NodeConfig` model (nodes/models.py) gains fields for DB-sourced
instances. These fields are nullable so that YAML-sourced instances continue
to work with NodeConfig in its current overlay role.

New fields on NodeConfig:

    # What kind of node this is in the graph editor
    node_type = CharField(
        max_length=20,
        choices=[('formula', 'Formula'), ('action', 'Action')],
        null=True, blank=True,
    )

    # Physical unit (pint-compatible string, e.g. "kt CO2/a", "GWh/a")
    unit = CharField(max_length=100, null=True, blank=True)

    # Semantic quantity type (e.g. "emissions", "energy", "cost")
    quantity = CharField(max_length=100, null=True, blank=True)

    # Port definitions — what inputs/outputs this node declares
    # Validated by Pydantic at the application layer.
    # Structure: [{"id": "energy", "quantity": "energy", "unit": "GWh/a",
    #              "dimensions": ["energy_carrier"]}]
    input_ports = JSONField(default=list, blank=True)
    output_ports = JSONField(default=list, blank=True)

    # The declarative operation pipeline (from the pipeline spec).
    # null means this node uses a legacy Python class (YAML-sourced instances).
    # Structure: [{"kind": "multiply", "inputs": [{"port": "a"}, {"port": "b"}]}, ...]
    pipeline = JSONField(null=True, blank=True)

    # For formula nodes: the formula string (compiled to pipeline at runtime)
    formula = CharField(max_length=2000, null=True, blank=True)

    # --- Action-specific fields (null for non-action nodes) ---

    # Action group (e.g. "Renewable energy", "Transport")
    action_group = ForeignKey(
        'ActionGroup', null=True, blank=True,
        on_delete=SET_NULL, related_name='nodes',
    )

    # Decision level: who decides on this action
    decision_level = CharField(
        max_length=20,
        choices=[('nation', 'Nation'), ('region', 'Region'), ('local', 'Local')],
        null=True, blank=True,
    )

    # Is this an outcome node (net emissions, net costs, etc.)
    is_outcome = BooleanField(default=False)

    # Visual grouping label for the graph editor (not a FK, just a string)
    node_group = CharField(max_length=200, null=True, blank=True)

Notes:
- `input_ports` and `output_ports` are JSONFields because ports don't have
  independent lifecycle — they're always edited as part of the node.
- `pipeline` is a JSONField validated against the pipeline spec Pydantic models.
- `formula` is a convenience: the UI shows a formula editor, and the system
  compiles it to a pipeline for execution. Both are stored; pipeline is
  authoritative if both are present.
- The `datasets` M2M through `NodeDataset` is superseded by `DatasetPort`
  for DB-sourced instances (see section 3).


2. NodeEdge
-----------

Represents a connection between two nodes in the computation graph.
Replaces the `input_nodes` / `output_nodes` lists in YAML configs.

    class NodeEdge(UUIDIdentifiedModel, UserModifiableModel):
        """A directed edge in the computation graph."""

        instance = ForeignKey(
            InstanceConfig, on_delete=CASCADE, related_name='edges',
        )
        from_node = ForeignKey(
            NodeConfig, on_delete=CASCADE, related_name='outgoing_edges',
        )
        from_port = CharField(
            max_length=100, default='output',
            help_text="Output port ID on the source node",
        )
        to_node = ForeignKey(
            NodeConfig, on_delete=CASCADE, related_name='incoming_edges',
        )
        to_port = CharField(
            max_length=100,
            help_text="Input port ID on the target node",
        )

        # Structural transformations applied to data flowing through this edge.
        # These reshape data (filter/aggregate dimensions) without changing values.
        # Structure: [{"kind": "flatten", "dimension": "scope"},
        #             {"kind": "select", "dimension": "ghg", "categories": ["co2"]}]
        # Validated by Pydantic.
        transformations = JSONField(default=list, blank=True)

        # Tags modify how the receiving node interprets this edge's data.
        # e.g. ["arithmetic_inverse", "complement", "non_additive"]
        tags = ArrayField(
            CharField(max_length=50), default=list, blank=True,
        )

        class Meta:
            constraints = [
                # Each input port can have at most one incoming edge
                # (a port is fed by either a NodeEdge or a DatasetPort, not both,
                # and not by multiple edges — enforced at application layer for
                # cross-table uniqueness with DatasetPort).
                UniqueConstraint(
                    fields=['to_node', 'to_port'],
                    name='unique_edge_per_input_port',
                ),
            ]


3. DatasetPort
--------------

Connects a dataset metric to a node's input port. This is how data enters
the computation graph from user-provided datasets.

Replaces the `NodeDataset` through-table and the `input_datasets` YAML config
for DB-sourced instances.

    class DatasetPort(UUIDIdentifiedModel, UserModifiableModel):
        """Connects a dataset metric to a node input port."""

        instance = ForeignKey(
            InstanceConfig, on_delete=CASCADE, related_name='dataset_ports',
        )
        node = ForeignKey(
            NodeConfig, on_delete=CASCADE, related_name='dataset_ports',
        )
        port_id = CharField(
            max_length=100,
            help_text="Input port ID on the node (must match a port in node.input_ports)",
        )
        dataset = ForeignKey(
            'datasets.Dataset', on_delete=PROTECT, related_name='node_ports',
        )
        metric = ForeignKey(
            'datasets.DatasetMetric', on_delete=PROTECT, related_name='node_ports',
        )

        class Meta:
            constraints = [
                # Each input port can have at most one dataset connection
                UniqueConstraint(
                    fields=['node', 'port_id'],
                    name='unique_dataset_port_per_input_port',
                ),
            ]

Cross-table uniqueness (a port can't have both a NodeEdge AND a DatasetPort)
is enforced at the application layer via model validation, not DB constraints.


4. ActionGroup
--------------

Groups actions into categories for display in the public UI.
Replaces `action_groups` in YAML configs.

    class ActionGroup(UUIDIdentifiedModel, UserModifiableModel):
        """Category for grouping action nodes (e.g. "Transport", "Buildings")."""

        instance = ForeignKey(
            InstanceConfig, on_delete=CASCADE, related_name='action_groups',
        )
        identifier = IdentifierField(max_length=100)
        name = CharField(max_length=200)
        color = ColorField(max_length=20, blank=True)
        order = PositiveIntegerField(default=0)

        i18n = TranslationField(
            fields=['name'],
            default_language_field='instance__primary_language',
        )

        class Meta:
            unique_together = [('instance', 'identifier')]
            ordering = ['order']


5. InstanceConfig Additions
---------------------------

New fields on `InstanceConfig` for DB-sourced instances:

    # How this instance is configured: YAML file or database
    config_source = CharField(
        max_length=20,
        choices=[('yaml', 'YAML'), ('database', 'Database')],
        default='yaml',
    )

    # Year parameters (currently only in YAML)
    reference_year = IntegerField(null=True, blank=True)
    minimum_historical_year = IntegerField(null=True, blank=True)
    maximum_historical_year = IntegerField(null=True, blank=True)
    target_year = IntegerField(null=True, blank=True)
    model_end_year = IntegerField(null=True, blank=True)

    # Emission unit for the model (pint-compatible, e.g. "kt CO2/a")
    emission_unit = CharField(max_length=100, null=True, blank=True)

    # Dataset repository configuration (for DVC-based data loading)
    # null for DB-sourced instances that don't use DVC
    dataset_repo_url = URLField(null=True, blank=True)
    dataset_repo_commit = CharField(max_length=100, null=True, blank=True)
    dataset_repo_dvc_remote = CharField(max_length=200, null=True, blank=True)

    # Feature flags and UI settings
    features = JSONField(default=dict, blank=True)


6. How It Fits Together
-----------------------

A DB-sourced model instance contains:

    InstanceConfig (config_source='database')
    ├── NodeConfig (node_type='formula', pipeline=[...])
    │   ├── input_ports: [{"id": "energy", ...}, {"id": "factors", ...}]
    │   ├── DatasetPort (port_id='energy', dataset=→Dataset, metric=→DatasetMetric)
    │   ├── DatasetPort (port_id='factors', dataset=→Dataset, metric=→DatasetMetric)
    │   └── outgoing_edges: [NodeEdge → another NodeConfig]
    │
    ├── NodeConfig (node_type='action', action_group=→ActionGroup)
    │   ├── input_ports: [{"id": "reduction_data", ...}]
    │   ├── DatasetPort (port_id='reduction_data', ...)
    │   └── outgoing_edges: [NodeEdge → formula node]
    │
    ├── NodeEdge (from_node=→activity, to_node=→emissions, to_port='energy')
    ├── NodeEdge (from_node=→action, to_node=→activity, to_port='reduction')
    │
    ├── ActionGroup (name='Transport', color='#2196F3')
    ├── ActionGroup (name='Buildings', color='#4CAF50')
    │
    └── Dimension, DimensionCategory (existing models, scoped via DimensionScope)

Loading path:

    InstanceConfig._create_from_config():
        if self.config_source == 'yaml':
            InstanceLoader.from_yaml(config_fn)        # existing path
        elif self.config_source == 'database':
            InstanceLoader.from_db(self)                # new path

    InstanceLoader.from_db() reads Django models and constructs the same
    runtime Instance + Context + node graph that from_yaml() produces.


7. Parameters
-------------

Parameters allow scenario-driven variation without changing model structure.
They are stored as JSONFields — the runtime Context hydrates them into full
Parameter objects for validation and computation.

The rationale: all mutations go through validation against the hydrated runtime
model anyway. The DB is just the serialization layer. If we need to promote
parameters to their own Django model later, the change is localized to the
hydration/serialization code — the computation layer and GraphQL API operate
on runtime objects, not DB models.

### Global parameters (JSONField on InstanceConfig)

Model-wide settings visible in the public UI (sliders, toggles).

    InstanceConfig.parameters = JSONField(default=list, blank=True)

    # Structure (validated by Pydantic):
    [
        {
            "id": "grid_emission_allocation",
            "label": {"en": "Grid emission allocation for GoO", "de": "..."},
            "type": "number",            # number | bool | enum
            "unit": "%",                 # pint-compatible, null for bool/enum
            "value": 80,                 # default value
            "min_value": 0,              # constraints (number only)
            "max_value": 100,
            "step": 5,
            "is_visible": true,          # shown in public UI
            "is_customizable": true      # end users can adjust via slider/toggle
        },
        {
            "id": "biomass_is_emissionless",
            "label": {"en": "Treat biomass as emission-free"},
            "type": "bool",
            "value": false,
            "is_visible": true,
            "is_customizable": false
        }
    ]

### Node-local parameters (JSONField on NodeConfig)

Settings specific to a single node. The `params` field already exists on
NodeConfig; we give it a well-defined Pydantic schema.

    NodeConfig.params = JSONField(null=True, blank=True)

    # Structure:
    [
        {
            "id": "is_enabled",
            "type": "bool",
            "value": true,
            "is_visible": false     # implicit for actions, not shown separately
        },
        {
            "id": "reduction_percentage",
            "label": {"en": "Reduction target", "fi": "Vähennystavoite"},
            "type": "number",
            "unit": "%",
            "value": 30,
            "min_value": 0,
            "max_value": 100,
            "step": 5,
            "is_visible": true,
            "is_customizable": true
        }
    ]

### Pipeline references to parameters

Pipeline operations reference parameters by ID string. The runtime resolves
them from the Context (global) or the owning Node (local).

    # Global parameter reference in a pipeline operation:
    - kind: set_values
      input: {port: grid_emission_factor}
      value: 0
      only_if: {parameter: biomass_is_emissionless}

    # Node-local parameter reference:
    - kind: multiply
      inputs:
        - port: baseline
        - parameter: reduction_percentage

The resolution order: node-local first, then global. This allows a node to
shadow a global parameter with a local override if needed.


8. Scenarios
------------

Scenarios are user-facing entities with names, so they get their own model.
Their parameter overrides are a JSONField — a list of (parameter_id, value)
pairs, optionally scoped to a specific node.

    class Scenario(UUIDIdentifiedModel, UserModifiableModel):
        """A named combination of parameter values."""

        instance = ForeignKey(
            InstanceConfig, on_delete=CASCADE, related_name='scenarios',
        )
        identifier = IdentifierField(max_length=100)
        name = CharField(max_length=200)
        description = TextField(blank=True)

        # Scenario kind determines behavior
        kind = CharField(
            max_length=30,
            choices=[
                ('default', 'Default'),
                ('baseline', 'Baseline'),
                ('progress_tracking', 'Progress tracking'),
            ],
            blank=True,
        )

        # If true, all action nodes have is_enabled=true unless overridden
        all_actions_enabled = BooleanField(default=False)

        # Parameter value overrides for this scenario.
        # Structure:
        # [
        #     {"parameter_id": "grid_emission_allocation", "value": 100},
        #     {"parameter_id": "is_enabled", "node_id": "solar_panels", "value": true},
        #     {"parameter_id": "is_enabled", "node_id": "heat_pumps", "value": false},
        #     {"parameter_id": "reduction_percentage", "node_id": "building_retrofit", "value": 50}
        # ]
        # Global overrides have no node_id. Node-scoped overrides include node_id.
        parameter_overrides = JSONField(default=list, blank=True)

        i18n = TranslationField(
            fields=['name', 'description'],
            default_language_field='instance__primary_language',
        )

        class Meta:
            unique_together = [('instance', 'identifier')]

Notes:
- For MVP, scenarios primarily toggle action nodes on/off (AC-7.3).
  This is modeled as overrides to each action's `is_enabled` parameter.
- `all_actions_enabled` is a convenience: when true, the scenario starts
  with all actions on, and overrides only need to specify which are off.
- Per-scenario slider values (AC-7.6, Future) are already supported by
  the `parameter_overrides` structure — just add number-valued overrides.


9. Versioning: Draft / Publish
------------------------------

### Core idea

The relational models are always the draft. Publishing takes a snapshot.

The editor (react-flow → GraphQL) reads and writes NodeConfig, NodeEdge,
DatasetPort, Scenario, ActionGroup directly. These models have no version
FK — they are simply the current working state.

When the user publishes, the entire model is serialized to a JSON blob and
stored as a Wagtail `Revision` on `InstanceConfig`. The public UI loads
from that published snapshot.

### InstanceConfig changes

Add `DraftStateMixin` to InstanceConfig. This gives us (from Wagtail):

    live                    BooleanField    — is there a published version?
    has_unpublished_changes BooleanField    — has the draft diverged?
    live_revision           FK(Revision)    — the currently published snapshot
    first_published_at      DateTimeField
    last_published_at       DateTimeField

InstanceConfig already inherits from RevisionMixin (indirectly via PathsModel),
so the Revision infrastructure is in place. DraftStateMixin layers the
live/draft state on top.

### Publish workflow

    1. User clicks "Publish" in the editor UI
    2. GraphQL mutation: publishModelInstance(id)
    3. Backend serializes the full model state:
       - All NodeConfig rows for this instance (with their params, ports, pipeline)
       - All NodeEdge rows
       - All DatasetPort rows
       - All Scenario rows (with parameter_overrides)
       - All ActionGroup rows
       - InstanceConfig fields (year params, global parameters, features)
       → single JSON dict
    4. instance_config.save_revision(user=request.user)
       stores the dict in Revision.content
    5. instance_config.publish(revision)
       sets live=True, live_revision=revision, has_unpublished_changes=False

### How the public UI loads data

    InstanceConfig._create_from_config():
        if self.config_source == 'yaml':
            InstanceLoader.from_yaml(config_fn)
        elif self.config_source == 'database':
            if serving_public_ui and self.live_revision:
                snapshot = self.live_revision.content
                InstanceLoader.from_snapshot(snapshot)
            else:
                InstanceLoader.from_db(self)

`from_snapshot()` and `from_db()` both produce the same runtime objects.
`from_db()` can be implemented as: serialize relational models → from_snapshot().

### Draft preview

The public UI can preview unpublished changes by loading from the relational
models instead of the snapshot. This could be a query parameter
(`?preview=draft`) or a separate preview URL that requires editor permissions.

### Revert to published

Deserialize the published snapshot back into the relational models,
overwriting the current draft state. This is the inverse of the publish
operation.

### Per-entity revision history

NodeConfig already has RevisionMixin, giving us per-node revision history
(who changed this node, when, what it looked like before). This is
independent of the model-level publish mechanism — it's audit trail
for the editor, not a publishing gate. We keep it as-is.

### Change tracking (later)

For fine-grained undo beyond publish-point rollback, options to add later:

- Lightweight change log: each GraphQL mutation records
  {entity_type, entity_id, action, diff, user, timestamp}
- django-pghistory for automatic row-level tracking
- Per-entity RevisionMixin snapshots (already in place for NodeConfig)

None of these are needed for MVP. The publish-snapshot model provides
a solid restore point, and per-node RevisionMixin provides audit trail.


10. What's NOT Covered Here
---------------------------

These will be designed separately:

- Impact overviews
- Emission sector grouping mechanism
- Public UI page configuration
- GraphQL mutations for the editor
