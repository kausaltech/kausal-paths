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
- Relational: `action_group` (FK), `indicator_node` (FK to self)
- Node type: `node_type` (determines Python class — fundamental to what a node *is*)

**What goes into `spec`:**
- `NodeSpec`: type-specific config, ports, output metrics, pipeline, params, `is_outcome`
- `InstanceSpec`: year boundaries, dataset repo, features, params, action groups, scenarios

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
    "type_config": {"kind": "formula", "formula": "a + b"},
    "input_ports": [...],
    "output_ports": [...],
    "output_metrics": [{"id": "emissions", "unit": "kt/a", "quantity": "emissions"}],
    "pipeline": [...],
    "params": [...],
    "is_outcome": false
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
