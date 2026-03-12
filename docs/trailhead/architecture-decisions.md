# Trailhead Architecture Decisions

This document records the key design decisions made during the Trailhead
refactoring (DB-backed model configuration with visual editor and versioning).

## 1. Single `spec` JSONField per model instead of many columns

**Decision:** Store the computation schema for `InstanceConfig` and `NodeConfig`
in a single `SchemaField` (from `django_pydantic_field`) validated by a Pydantic
model, rather than spreading it across dozens of individual Django model fields.

**Why:**
- The schema is in active flux during Trailhead development. Each column change
  requires a Django migration; reshaping a JSONField requires only a `RunPython`
  migration (or nothing, if the Pydantic model handles both old and new shapes).
- The DB is storage and versioning (via Wagtail revisions), not a query surface.
  Nobody runs `NodeConfig.objects.filter(quantity='energy')` in production.
  The serialization path reconstructs the same dict the Pydantic model already
  represents — round-tripping through 20 columns adds no value.
- The visual editor (Trailhead UI) works with structured objects, not individual
  form fields. A single structured blob maps naturally to a tree editor.
- PostgreSQL can query JSON fields directly (`defs__years__target`), so
  queryability is not sacrificed.

**What stays as Django fields:**
- Identity: `identifier`, `instance` (FK), `name`, i18n fields
- Display/CMS: `color`, `order`, `is_visible`, `body` (StreamField), descriptions
- Relational: `indicator_node` (FK to self)
- Node type: `node_type` (determines Python class — fundamental to what a node *is*)

**What goes into `spec`:**
- `NodeSpec`: type-specific config, ports, output metrics, pipeline, params, `is_outcome`
- `InstanceSpec`: year boundaries, dataset repo, features, params, action groups, scenarios, dimensions

**Schema changes** are handled via `RunPython` migrations that reshape the JSON
data in place — simpler than coordinating `AddField` + `RunPython` + `RemoveField`.

## 2. Namespaced `spec` structure

**Decision:** The `spec` JSONField is namespaced (nested objects), not flat.

**Why:**
- Output-related config (unit, quantity, dimensions) belongs on the output side,
  since we want to support multiple output metrics per node.
- Grouping by concern (years, dataset_repo, ports, params) makes the schema
  self-documenting and easier to evolve independently.
- The Pydantic model provides the structure; nesting inside `spec` mirrors it.

**Example — `NodeSpec`:**
```python
{
    "node_class": "nodes.simple.AdditiveNode",
    "type_config": {"kind": "formula", "formula": "a + b"},
    "input_ports": [...],
    "output_ports": [...],
    "output_metrics": [{"id": "emissions", "unit": "kt/a", "quantity": "emissions"}],
    "pipeline": [...],
    "params": [...],
    "is_outcome": false,
    "extra": {"historical_values": [...], "tags": [...]}
}
```

## 3. Discriminated union for node type config

**Decision:** `NodeSpec.type_config` is a discriminated union
(`FormulaConfig | ActionConfig | SimpleConfig`) keyed on a `kind` field,
while `node_type` stays as a Django field on `NodeConfig`.

**Why:**
- Node types share 90% of their schema (outputs, params, ports). The differences
  are small (`formula` for formula nodes, `decision_level` for actions).
- The `kind` discriminator inside the JSON makes the defs self-contained — you
  can validate the JSON without looking at the Django model.
- `node_type` on the Django model is kept because it determines the Python class
  and is fundamental to the node's identity.

## 4. Parameter classes as Pydantic BaseModel (not dataclasses)

**Decision:** Refactored `Parameter` and all subclasses from `@dataclass` to
`pydantic.BaseModel`.

**Why:**
- `ParameterDef` (Pydantic) and `Parameter` (dataclass) were 80% the same model.
  The Pydantic approach unifies storage schema and runtime object.
- Serialization comes free: `model_dump()` produces the JSON for storage, excluding
  runtime-only fields (`context`, `node`, subscriptions) via `Field(exclude=True)`.
- Strawberry (GraphQL) integration: `strawberry.experimental.pydantic` requires
  `model_fields`, which Pydantic BaseModel provides but Pydantic dataclasses do not.
- Validation on construction catches bad data early.

**Runtime fields** (`context`, `node`, `subscription_nodes`, `subscription_params`)
are marked with `Field(exclude=True, init=False)`. Private state (`_hash`,
`_follows_scenario`) uses `PrivateAttr`.

**The `@parameter` decorator** now just registers the class in the parameter
type registry; it no longer applies `@dataclass`.

## 5. Typed semantic identifiers

**Decision:** Distinct `Annotated[str, ...]` types for different kinds of
identifiers: `NodeIdentifier`, `ParameterLocalId`, `ParameterGlobalId`,
`ActionGroupIdentifier`, `ScenarioIdentifier`, `MetricIdentifier`.

**Why:**
- All share the same regex pattern (`^[a-z0-9_]+$`) but represent different
  domains. Distinct types enable future referential integrity validation
  (e.g., "does a node with this identifier exist in the instance?").
- `ParameterLocalId` vs `ParameterGlobalId`: local IDs are bare (`discount_rate`),
  global IDs can be dotted (`transport_emissions.discount_rate`). The type
  communicates which form is expected at each usage site.
- `MetricIdentifier` allows mixed case (`^[A-Za-z0-9_]+$`) to match the existing
  convention for metric column names (e.g., `Value`, `Reductions`).

## 6. ActionGroup and Scenario moved into InstanceSpec

**Decision:** `ActionGroup` and `Scenario` are defined as Pydantic models
(`ActionGroupDef`, `ScenarioDef`) inside `InstanceSpec`, rather than as
separate Django models.

**Why:**
- They don't have independent lifecycle — they exist only within an instance.
- Storing them in the `spec` JSONField means they're versioned atomically
  with the rest of the instance configuration (via Wagtail revisions).
- Fewer Django models and migrations to maintain.

**`NodeEdge` and `DatasetPort` remain as Django models** because they are
inherently relational (they reference two nodes, or a node + dataset).

## 7. Lazy string preservation in TranslatedString

**Decision:** `TranslatedString` stores the original Django lazy string
(`gettext_lazy` result) in `_lazy_source`, and provides `resolve_languages()`
to populate the `i18n` dict for a set of languages.

**Why:**
- Parameter labels are declared at class definition time using `_('...')`,
  before any `Context` or instance language configuration exists.
- Previously, the lazy string was immediately coerced to `str()`, losing
  the ability to translate into other languages.
- When a parameter is bound to a `Context`, `resolve_languages()` can be
  called with the instance's supported languages to populate all translations.

## 8. Runtime-to-DB export via introspection (not YAML re-parsing)

**Decision:** The `sync_instance_to_db` command loads a YAML instance via the
normal `InstanceLoader` path, then introspects the live runtime objects
(Instance, Node, Context, Scenario, etc.) to build `InstanceSpec` and `NodeSpec`.

**Why:**
- The runtime objects are already validated and fully resolved — no need to
  re-parse YAML dicts and manually map fields.
- Edge dimensions, dataset configs, and parameters are all available in their
  final form on the runtime objects.
- Single source of truth: if the YAML loader works, the data is correct.

**Key files:**
- `nodes/spec_export.py` — builds InstanceSpec/NodeSpec from live objects
- `nodes/instance_from_db.py` — serializes DB specs back to config dicts for InstanceLoader
- `nodes/management/commands/sync_instance_to_db.py` — management command

## 9. NodeSpecExtra: the attic

**Decision:** `NodeSpec.extra` is a `NodeSpecExtra` Pydantic model that holds
legacy config fields we haven't properly modeled yet.

**Why:**
- The InstanceLoader reads many node config fields (`historical_values`,
  `forecast_values`, `input_dataset_processors`, `tags`, etc.) that don't
  yet have proper places in the NodeSpec schema.
- Dumping them into an untyped `dict` loses validation. A typed attic model
  lets us see what's there and plan removal.
- Each field in `NodeSpecExtra` is a candidate for either promotion to a
  proper NodeSpec field or removal when the corresponding YAML-era feature
  is replaced.

**Current contents:**
- `historical_values`, `forecast_values` — create FixedDatasets at load time
- `input_dataset_processors` — e.g. `["LinearInterpolation"]`
- `tags` — node-level tags like `other_node` used for input node filtering
- `other` — catch-all dict for anything else

---

# Known Hacks and Workarounds (to clean up later)

## Dimensions stored as raw dicts in InstanceSpec

`InstanceSpec.dimensions` is `list[dict[str, Any]]` — the raw YAML dimension
config passed through without validation. Dimensions should eventually be
proper Pydantic models, but some dimensions are created dynamically by node
classes at runtime (e.g. `sector` from `HsyNode` with `is_internal=True`),
so the schema needs to account for that distinction.

## Edge transformations format

`NodeEdge.transformations` stores a dict with `from_dimensions` and
`to_dimensions` keys — the same format the InstanceLoader expects. This was
changed from the original flat list of `{kind, dimension, ...}` transforms
because that format conflated source-side and target-side dimension operations.
The current format works but is just pass-through; it should eventually use
proper Pydantic models matching `EdgeTransformation`.

## TranslatedString in InstanceLoader.make_trans_string

`make_trans_string` was patched to handle dict values (from Pydantic's
compact TranslatedString serialization). When a parameter label is serialized
as `{"en": "...", "fi": "..."}`, the original code would store the whole dict
as the value for the default language key, producing broken TranslatedString
objects. The fix detects dict values and constructs TranslatedString directly.

## explanations.py label handling

`_explain_column_filter` in `explanations.py` was patched to handle
`TranslatedString` labels that might be dicts (from the Pydantic serialization
round-trip). The root cause is that some code paths create parameters from
dicts where labels aren't fully converted to TranslatedString. This should
be fixed by ensuring all parameter labels are proper TranslatedString objects
after deserialization.

## Dimensionless unit serialization

`str(pint.Unit('dimensionless'))` returns `''` (empty string). The export
explicitly checks for dimensionless units and stores `'dimensionless'` to
prevent the InstanceLoader from thinking the node has no unit.

## InstanceLoader.generate_nodes_from_emission_sectors

Patched to return early if no emission sectors are configured, rather than
asserting that `emission_unit` is present. DB-backed instances don't use
emission sectors (they have explicit node definitions).

## AnyParameter dynamic union

The `AnyParameter` type (discriminated union of all parameter types) must be
built dynamically from the parameter type registry because action-specific
types like `ShiftParameter` and `ReduceParameter` are registered at import
time. `NodeSpec` and `InstanceSpec` use this type for their `params` field.
The union is built via `params.discover.get_parameter_type_union()` and the
cache must be cleared + models rebuilt after all node modules are imported.

## Scenario parameter overrides

Scenario overrides can set "magic" node parameters like `operations` and
`categories` that change how GenericNode computes. These parameters exist
on `GenericNode.allowed_parameters` but are only created when the node config's
`params` list includes them. The export must include ALL node parameters
(including ReferenceParameters) to ensure scenario overrides can find them.

## ReferenceParameter serialization

`ReferenceParameter` instances are serialized as `{"id": "param_id", "ref": "global_param_id"}`
in the config dict, not as full Pydantic model dumps. The InstanceLoader
reconstructs them from this format.

## One known computation error in Espoo

`road_traffic_emissions` → `road_traffic_mileage` fails with an error during
DB-backed computation. This doesn't affect `net_emissions` output (the values
match YAML exactly) but indicates a missing config field somewhere in the
transport sub-graph. Needs investigation.

---

# Validation status

**First successful DB-backed computation: Espoo `net_emissions` (2024-03-24)**

YAML output:
```
2020: 903.086091 kt/a
2021: 830.813897 kt/a
2022: 897.153434 kt/a
2023: 742.827907 kt/a
2024: 708.86517  kt/a
```

DB output: **identical** (bit-for-bit match)

135 nodes, 176 edges, 3 scenarios, 7 action groups, 4 global params,
19 dimensions exported and loaded successfully from DB.
