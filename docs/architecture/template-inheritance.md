# Published template inheritance

A database-backed `InstanceConfig` can select one published template revision
through `template_revision`. There is no graph-import model or separate
framework release model. A framework member's selected revision must belong to
its framework's `template_instance`; nested template inheritance is rejected.

## Draft and publication lifecycle

The template's live tables are its editable draft. Dependent instance drafts compose the
selected revision's nodes, bindings, dimensions, and pinned reference data with
their own live nodes, datasets, and input selections. Template draft edits do
not affect them.

`InstanceConfig.publish_instance()` recognizes templates and delegates to
`publish_template_instance()`. It freezes the template graph, declarations, and
reference datasets without moving any municipal draft pin. A municipality
advances separately through `upgrade_template_instance()` or the editor's
`upgradeFrameworkTemplate` mutation, which selects the current published template.

An upgrade retires settings whose nodes or ports disappeared and parameter values
whose declaration disappeared, changed type, or became framework-owned. Other
conflicts do not block the upgrade. A stale or cyclic binding remains stored for repair;
the effective draft omits the invalid binding and exposes `editor.compositionErrors`.
Publication refuses these errors, structural conflicts, and invalid report or
scenario references. Changes to bounds that invalidate a local parameter value
also leave the draft unpublishable.

A dependent instance's publication stores its local authoring inputs: spec,
nodes, ordinary bindings, inherited-node settings, and port binding selections.
It also freezes the selected template revision, a portable content hash, and
local dataset pins. Runtime loading composes these inputs with that exact
revision; neither instance's live draft participates. Relational template pins
protect revisions needed by historical municipal publications. The template hash
covers declarations and dataset content hashes, excluding database-local payload
revision IDs, so it remains stable across imports. Instance publication acquires
the template lock before the dependent instance lock.

`restore_revision()` restores the local model definition and template pin.
`revert_to_published()` uses the current published revision. Nodes absent from the
restored definition become stale; dataset bodies retain their independent draft
state. Older flattened snapshots are adapted back into local authoring inputs at
the restore boundary, without rewriting historical revision content.

Template publication validates calculation structure, but does not require empty
local reporting slots to represent completed local reporting. This is not a
certification assessment.

## Bindings and local settings

Shared node and port UUIDs remain stable across template revisions. An
`InputPortBindingSet` identifies `(instance, node_uuid, port_uuid)` and stores the
complete ordered local replacement for that port:

- no set: inherit the template or local default;
- an empty set: explicitly disconnect the input;
- a populated set: use these dataset-metric or node-output sources.

`NodeInputPortBinding` remains the ordinary local graph storage. Override sets
are generic node models, independent of framework membership. Their typed
serialized sources refer to the selected snapshot's UUIDs rather than mutable
template ORM rows. `InputPortBindingReference` retains referenced datasets,
metrics, and local nodes with protective foreign keys.

On inherited inputs, `InputPortDef.binding_owner == 'instance'` permits local
replacement. Other inputs retain the template's bindings. A locally supplied
emission factor may therefore come from a dataset or from a local
calculation node. Inherited outputs may feed local nodes. The composed graph
must remain acyclic and satisfy the port contracts.

`InstanceConfig.node_settings` retains permitted local goals, layout, and parameter
source links without duplicating shared node definitions. Legacy value selections
are migrated into sparse municipal default-scenario overrides. Parameter ownership
is independent of visitor customization.

## Instance declarations and parameter defaults

A municipal `InstanceModelSpec` contains local declarations and local years.
Its global parameter IDs must be disjoint from the template's IDs. Reports,
pages, impact overviews, normalizations, action groups, scenarios, and shapes
inherit from the pinned revision; local declarations append without shadowing inherited
identities. Reports currently use their translated name as identity; groups use
UUIDs, and the other declarations use their existing identifiers.

There is one `InstanceModelSpec` type for standalone definitions, sparse municipal
definitions, and effective runtime compositions. Municipal lists contain additions;
an empty list adds nothing. Local declarations cannot shadow template identities.
Years remain wholly local. `dataset_repo` must be absent or `None` in a dependent
instance and always comes from the template.

Shapes inherit like the other declarations, keyed by UUID, with one addition (see
[shapes](shapes.md)). A shape the template owns (`owner: framework`) is read-only and cannot
be redeclared. A shape with `owner: instance` is an extension point: conversion and upgrade
give each dependent instance its own record of it under the template's UUID, and composition
uses that record in place of the template's declaration. The record keeps the template's
identifier, dimensions, `inherits` and closedness, cannot be removed, and adds combinations.
Inheritance between shapes is resolved at runtime, so moving the pin brings in the standard's
new combinations without touching the record. A dataset's shape reference follows the pinned
revision, set on activation and upgrade, so it always names a shape the instance declares.

`features`, `terms`, `theme_identifier`, and `sample_size` may be overridden directly.
Serialization preserves explicit field presence, including individual feature and
term fields: omission inherits, while an explicit `False`, `0`, or `None` retains
its meaning. Local scenarios with inherited IDs contain only parameter values and
their declared types; composition retains the template's scenario metadata. A
scenario with a new ID is an ordinary local declaration and must have a name.

Composition returns a separate spec marked by a private runtime flag.
`InstanceSpecField` rejects storing that object in `InstanceConfig.spec`, including
queryset updates and bulk writes. Snapshot serialization remains permitted.
Export/import accepts authoring snapshots rather than composed runtime snapshots.
The flag is intentionally a runtime guard: converting to ordinary JSON and
reconstructing a spec loses it, so persistence paths must always use local inputs.

`InstanceSnapshot.provenance` maps declaration and value paths to their authoring
instance UUID and, for inherited items, template revision and content hash.
Parameter declarations and scenario values have separate origins.
`parameter_value_origin()` follows municipal-default fallback for scenarios that
omit a parameter. Provenance is computed during composition rather than stored
as tags on every item.

`InstanceExport` contains the local authoring snapshot and, for dependent instances,
a nested `template` export of the pinned edition. Its dataset bodies come from
the pinned publication payloads, including frozen external input declarations,
rather than current template draft data. Import verifies template and payload
hashes, reuses a matching edition or installs the bundled edition, and remaps
revision IDs. An installed template belongs to the destination organization.
Existing template drafts are not published or replaced by importing another
edition.

A parameter's `owner` is `framework` or `instance` (the default). The template
owns inherited declarations in both cases; ownership controls whether municipal
administrators can persist values. `is_customizable` controls ephemeral visitor
edits. A framework-owned declaration is injected unchanged, and local values
cannot override it. References preserve ownership; persisting a reference value
edits its target parameter and checks the target's ownership too.

For parameter values, composition applies:

1. The parameter's declared default.
2. A municipal default stored in the local default scenario's `param_values`.
3. An explicit value in the selected scenario.
4. An ephemeral visitor override.

Municipal defaults also become runtime parameter defaults. Thus a weather-corrected
scenario can name only `weather_correction`, retaining the municipality's other
settings. Scenarios no longer automatically capture every customizable parameter.
A scenario's default values contain only deliberate deviations, and changing a
municipal value back to the inherited value removes the local entry.

Use `instanceEditor.setInstanceParameter(parameterId, value, scenarioId, reset)`.
Omit `scenarioId` to edit municipal defaults; `reset: true` removes a local entry.
The mutation requires instance change permission, honors locks and draft versions,
records an audit operation, and validates the value. Editor reads use the effective
spec, while persistence retains only local declarations and overrides.

Formula nodes read `FormulaConfig.formula`; formula actions read
`ActionConfig.formula`. Formulas are calculation definitions, never runtime
parameters or scenario values. YAML adapters still accept the legacy formula
parameter spelling. Migration 0082 converts existing database specs, and snapshot
version 13 adapts old published snapshots without rewriting revision content.
Migration 0083 removes the earlier overrides container and retains historical
template pins; snapshot version 14 introduces sparse authoring snapshots.
Captured default-scenario formulas are removed from live specs even when stale.
Genuinely different named-scenario formulas require explicit migration rather
than being silently discarded. Copied reports are retired by declaration identity even when
their old contents reference removed nodes. Independent local declarations and
parameter values are retained.

## Editor API

The backend computes `Node.isEditable`, `InputPortType.isEditable`, and
`OutputPortType.isEditable` for the displayed graph and current user's
permissions. `NodeMeta.can_edit()` owns the decision for node definitions, port
definitions, and input bindings. GraphQL supplies a request-cached
`NodeEditContext`; the same policy is used by node permissions and binding
mutations. Ownership is relative to the active instance, so being a framework
member does not make its own nodes read-only.

Inherited definitions are read-only. The template's own draft can
be edited even when its nodes retain the old BISKO edit-lock flag.
`InputPortType.bindingsEditable` separately reports whether its sources can be
replaced. Published views are read-only.

For instances with `template_revision`, use
`instanceEditor.setInputPortBindings(nodeId, portId, bindings)`:

- each entry supplies `sourceNodeId` and `sourcePortId`, or `datasetId` and
  `metricId`, plus optional transformations;
- `bindings: []` disconnects; `bindings: null` restores the default;
- new structural conflicts return `ConstraintViolations` without saving.

The per-edge and per-dataset binding mutations remain available for local nodes,
including local ports with overrides. They update the effective selection without
writing through the inherited definition. Template-owned inputs use
`setInputPortBindings` when local selection is allowed. The editor reads the
effective graph, including inherited bindings. Edges may originate from an
inherited output when their target is local.

Dataset schemas and dimensions can belong to a Framework. A schema may be shared
across instances, with at most one dataset per schema in each scope. Dataset
ownership still controls access to values.
`createDataset(input: {schemaId: ...})` reuses a visible schema, `datasetSchemas` lists
available definitions, and `schemaIsEditable` distinguishes schema permissions
from dataset-value permissions. Shared schema and dimension definitions remain
read-only through the local editor.

Dataset editability flags use queryset `Exists` annotations for shared scope and
other datasets using the same schema. `DatasetSchema.objects.for_scope_type()`
encapsulates the generic scope-type filter; Django's existing ContentType cache
handles repeated type resolution. Graph datasets, schema details, and local
override catalogue entries are loaded in batches. Query-count tests cover list
growth for datasets and inherited nodes, including local dataset overrides.

## Explicit BISKO conversion

Provision the framework and quality catalogue with `python -m tools.setup_bisko`.
Conversion requires existing database-backed instances:

```bash
python -m tools.setup_bisko --convert example-bisko --dry-run
python -m tools.setup_bisko --convert example-bisko
```

`--convert` replaces copied shared nodes with inheritance. Framework-owned
bindings always come from the published template; the conversion result counts
copied bindings discarded for this reason. Instance-owned bindings, settings,
and page references are preserved. Repeat the instance flag for multiple
framework instances.

`--prepare-from` explicitly changes the template's input declarations and
category vocabulary using selected migration examples. `--publish` freezes the
template and advances existing dependent drafts. Neither is needed when
adopting an already published template as the authority.

For output-preserving migration of historical national datasets,
`--reference-instance example-bisko` selects that instance's reference-data
edition during publication. It requires `--publish`; using it also advances
existing dependents, so it is not an independent per-city release channel. Do
not use it when the template's own data is authoritative.
Rerun node-output comparisons against the intended reference edition after
conversion.

Quality evidence, data-point grades in the editing API, and certification
criteria/evaluation remain separate work; see [framework quality](framework-quality.md).
