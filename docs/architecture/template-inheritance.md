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
`publish_template_instance()`. Publication locks the template and its dependent
instances, freezes the template graph and reference datasets, and composes and
validates every dependent draft against that revision. Missing targets, cycles,
or newly introduced structural conflicts abort the transaction. Existing draft
conflicts do not prevent an otherwise compatible template update. The template
revision and all dependent draft pointers advance together.

A dependent instance's publication stores the complete effective snapshot, including
its selected template revision and dataset pins. Subsequent template publication
changes that dependent instance's draft only; its existing public snapshot is unchanged.
Dependent instance publication acquires the template lock before the dependent instance lock,
matching the template publisher's lock order.

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

`InstanceConfig.node_settings` retains permitted local goals, layout, and
customizable parameter selections without duplicating shared node definitions.

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
python -m tools.setup_bisko --prepare-from example-bisko --dry-run
python -m tools.setup_bisko --prepare-from example-bisko --publish --convert example-bisko
```

`--prepare-from` explicitly reconciles the template's input declarations and
category vocabulary with the selected migration examples. `--publish` freezes
the template and advances existing dependent drafts. `--convert` replaces copied
shared nodes with inheritance while preserving local bindings, settings, and
page references. Repeat the instance flags for multiple framework instances.

For output-preserving migration of historical national datasets,
`--reference-instance example-bisko` selects that instance's reference-data
edition during publication. It requires `--publish`; using it also advances
existing dependents, so it is not an independent per-city release channel.
Rerun node-output comparisons against the intended reference edition after
conversion.

Quality evidence, data-point grades in the editing API, and certification
criteria/evaluation remain separate work; see [framework quality](framework-quality.md).
