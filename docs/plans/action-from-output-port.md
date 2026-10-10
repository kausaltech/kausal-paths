# Creating an action from an output port

*Produced by Claude Opus 5.5 on 2026-10-08.*
*Version 2 produced by Claude Opus 5.5 on 2026-10-09.*
*Responsible: Jouni Tuomisto.*

Status: agreed with Juha 2026-10-09, with one change, added in version 2: before any
dataset is created, the modeller chooses for each effect whether its numbers come from a
new dataset, an existing dataset or an existing node (*Where an effect's numbers come
from*). Built 2026-10-09–10, backend and UI; see *As built* for where the result differs
from the design.

It is written against `feat/data-studio-backend`, because the two pieces it builds on,
action hooks and node-owned datasets, exist only there, and it is built on a branch from
that branch (`80e2b6d8`). The table below is the starting point; the build added both
missing prerequisites.

## Summary

A modeller clicks an output port of an existing node and chooses *New action
acting on this*. The port is the template: the new action's effect has the port's
quantity, unit, dimensions and categories. The wizard creates a
`simple.AdditiveAction` that acts on the node through a hook, and gives it a dataset
of its own with exactly that shape. Its start year holds zeros, and every other cell
is left for the modeller to fill. The modeller then only enters numbers, usually one
target year, and the years in between are interpolated.

An action with several effects (energy, emissions, cost) is built the same way: each
further effect is another output port picked on the canvas, and each becomes one
output port, one hook and one metric of the action.

A new dataset is the default, not the only source. For each effect the modeller can
instead connect an existing instance dataset or the output of an existing node, when
the numbers already exist in the model.

Nothing here is new machinery. The wizard is the first client of two things the
branch already plans: hooks in the model editor (`docs/architecture/action-hooks.md`,
*Not built yet*) and the editor mutation that creates a port's own data
(`docs/plans/node-owned-datasets.md`, step 2).

## What the modeller does

1. **Click an output port** on the canvas and choose *New action acting on this*. The
   wizard opens with that port as the first effect.
2. **Effects.** Each effect shows its target node and port, quantity, unit,
   dimensions and categories. The modeller can:
   - untick categories the action does not touch;
   - choose *Add another effect*, then click another output port on the canvas.
3. **Data source.** For each effect, the wizard asks where its numbers come from:
   - **New dataset** (the default): the action gets a dataset of its own, as
     described below;
   - **Existing dataset**: pick an instance dataset and one of its metrics;
   - **Existing node**: click an output port on the canvas, as when picking an effect.

   Only shape-compatible sources are offered (see below). The choice is made here,
   before anything is created, so that no dataset is created only to be replaced.
4. **Identity.** Name (the identifier is derived from it), action group (inline
   creation exists, AC-4.3), and the start year, which only new datasets use.
5. **Create.** One mutation creates everything (below). The wizard closes. If any
   effect has a new dataset, the action's dataset opens in the existing dataset grid
   (`DatasetEditor` / `DatasetDataGrid` in kausal-paths-ui); otherwise the action is
   shown selected on the canvas.
6. **Enter numbers** (new datasets only). The modeller adds the years they have estimates for with the
   grid's *Add years*. This is often just a target year, such as 2030 or 2045. They
   type the change, negative for a reduction. CSV/Excel import works as for any
   dataset.

An example with two effects: *Replace oil boilers with heat pumps*. Effect 1 is
picked from the output port of building final energy use (dimensions: energy carrier),
with categories narrowed to oil and electricity. In 2035 it is −40 GWh/a for oil and
+12 GWh/a for electricity. Effect 2 is picked from the output port of building
heating costs, at −1.5 M€/a. If a cost model already computes the saving in a node,
effect 2 connects that node instead, and only effect 1 gets a new dataset.

## What `feat/data-studio-backend` had at the start

| Piece | State on the branch | Where |
| --- | --- | --- |
| Hooks: an action acts on a node's output | Runtime done, verified on a real model; YAML `output_nodes: [{hook: true}]` | `nodes/hooks.py`, `ActionHookDef` in `nodes/defs/node_defs.py`, `docs/architecture/action-hooks.md` |
| Hooks in GraphQL and the editor | **Not built.** The planned gesture is to drag from an action's output port onto another node's output port | `action-hooks.md`, *Not built yet* |
| Explicit `hooks:` in YAML | Agreed 2026-09-23, not started | `docs/plans/yaml-ports-and-hooks.md` |
| Node-owned datasets (scope = `NodeConfig`, no identifier, deleted with the node) | Step 1 done | `docs/plans/node-owned-datasets.md` |
| Editor mutation that creates a port's own dataset from the port's shape and binds it | **Planned, step 2, not built** | same, *Step 2* |
| Output port effective shape (dimensions, categories, quantity, unit) | Done | `OutputPortType.effectiveShape`, `nodes/graphql/types/spec.py` |
| `createNode` for `AdditiveAction`: input ports paired 1:1 with output ports | Done | `_paired_action_input_ports`, `nodes/graphql/editor.py` |
| `bindDataset` with transformations | Done | `nodes/graphql/bindings.py` |
| `interpolate` / `extend` / `backfill` binding ops | Done | `InterpolateOp`, `ExtendOp`, `BackfillOp` in `nodes/defs/transform_def.py` |
| Blank cells for a year, shaped like an existing year | Done (`ensure_empty_year`, used by `addInventoryYear`) | `datasets/year_slots.py` |
| Dataset grid, *Add years*, import (UI) | Done | kausal-paths-ui `components/model-editor/datasets/` |
| `createEdge`: an edge from a node's output port into a given input port, with transformations | Done | `create_edge`, `CreateEdgeInput` in `nodes/graphql/editor.py` |
| Clickable output ports on the canvas (UI) | Not built: plain React Flow source handles | kausal-paths-ui `ElkNode.tsx` |

So three things are missing: hooks in GraphQL, the step-2 dataset mutation, and the
UI. The wizard needs all three and adds only a thin orchestration on top.

## Design

### The action acts on the node through a hook, not through an input edge

The action's effects are hooks on the template ports, not edges into the target's
input ports:

- The target's own calculation does not change. This matters most for
  framework-owned nodes, which a municipality cannot edit; that problem is what hooks
  were built for.
- Every hook on a node sees the same un-hooked value, so effects add up and each
  action's impact is exactly its own output.
- The "template" idea and the hook rule are the same thing: a hook's contribution
  must have the target port's unit and dimensions (`hook_contribution`). The wizard
  derives the shape from the port so that this holds by construction.

A consequence to state in the UI: **hooks only act after the last historical
year.** An action never moves a historical balance.

### Where an effect's numbers come from

Every effect is the same on the action's side: one output port, its paired input port,
and one hook. What differs is what feeds the paired input port.

| Source | What feeds the input port | Created by the wizard |
| --- | --- | --- |
| New dataset | a binding to a metric of an action-owned dataset | the dataset, its start-year zeros, the binding |
| Existing dataset | a binding to the chosen metric of an instance dataset | the binding |
| Existing node | an edge from the chosen output port | the edge |

So the two new choices reuse the two connections the editor already has, `bindDataset`
and `createEdge`; nothing about the hook changes.

**Compatibility is the hook's rule.** A source is offered only if its shape can be the
hook's contribution: a compatible unit (the hook converts with `ensure_unit`) and the
same dimensions as the target port. Categories may be a subset, with the same meaning
as narrowing (left-out categories contribute zero). A source with an extra dimension,
or missing one, would need a sum or a broadcast on the binding. That is possible, but
v1 does not offer it: the picker leaves such sources out, and says why if the modeller
asks for one. For a node, the shape is its output port's `effectiveShape`; for a
dataset metric, the schema's dimensions and the metric's unit.

**Which datasets.** Instance datasets only. Another node's owned dataset is internal to
that node (decision 10 of the node-owned datasets plan), and binding it is refused by
`bindDataset` anyway. If the numbers live in another node's dataset, connect that node.

**The source must hold changes, not levels.** A hook adds its contribution to the
target. Connecting a dataset or node that holds a level (a scenario's total energy use,
say) adds the whole level. The data-source step states this next to the choice; the
wizard cannot check it.

**No cycles.** A node source must not depend on the hooked target, or the action would
feed on its own effect. The constraint check must refuse this. Whether the existing
cycle detection sees the hook as an edge is not yet verified; if it does not, that is
part of the hooks-in-GraphQL work.

### Shape inheritance

This applies to effects with a new dataset; for the other two, the source supplies
the shape, within the compatibility rule above.

Each effect copies, from the template port's `effectiveShape` (the solver's answer,
not the declared `dimensions`, which can be empty):

| Property | Inherited | Can the modeller change it? |
| --- | --- | --- |
| Quantity | yes | no |
| Unit | yes | no, in v1 (any compatible unit would work, since the hook converts with `ensure_unit`) |
| Dimensions | yes, all of them | **no**: a contribution missing one of the target's dimensions is an error in `hook_contribution` |
| Categories | yes, all of them | yes, narrowing only: a category the action leaves out contributes zero, because `apply_hooks` joins outer and keeps the base value where the contribution is null |

These are the semantics of ordinary addition: missing categories are zeros, and
missing dimensions are an error unless an explicit operation is added (a hook's
`transformations`). The wizard never produces a missing dimension, so v1 needs no
such operation.

### The action node

- Class `simple.AdditiveAction`, with one output port per effect, each copying the
  template's quantity, unit and dimensions. Its input ports are paired automatically,
  as `createNode` already does.
- `ActionConfig.hooks` holds one `ActionHookDef` per effect:
  `node` = the template's node, `port` = the template port, `from_port` = the
  matching output port of the action, `transformations` = empty.
- The hooks on the branch have so far been exercised with `GenericAction` and
  `FormulaAction`. `AdditiveAction` renames each input's metric to its output port's
  `column_id`, and the hook selects by that `column_id`, so it should work, but it
  needs a test with two output ports and two hooks.

### The dataset

Only effects with a new dataset get one. If no effect does, the action has no dataset.

- **Owned by the action** (`Dataset.scope` = the action's `NodeConfig`), with no
  identifier. It is created, copied, exported and deleted with the action, and it
  never shows up in the instance's dataset list.
- **One metric per effect.** The metric label is the target's name and port label,
  and the unit and quantity are the target's. If the label is not unique within the
  dataset, the editor asks for a different label, as decision 13 of the node-owned
  datasets plan requires.
- **One dataset per distinct dimension set** among the new-dataset effects. When all of them have the same
  dimensions, which is the common case, there is one dataset with several metrics.
  When they differ, there is one dataset per dimension set. A union of the
  dimensions would leave cells that mean nothing and would need sum-over operations
  on the binding. Because owned datasets carry no identifier, several of them on one
  action do not collide.
- **Cells.** Only the start year is created, holding an explicit `0` for every
  (metric × kept category) combination. Nothing else is pre-created: we don't know
  which years the modeller has numbers for, and often there is only one target year.
  The grid's *Add years* creates the next year's blank cells, shaped like the
  start year (`ensure_empty_year` with the dataset as its own prototype).
- **Start year default:** the instance's `maximum_historical_year`, so the change
  ramps up from zero over the first forecast years. The modeller can set a later year
  for an action that starts later.

### The binding

Each paired input port fed by a dataset, new or existing, is bound to its metric with
these transformations:

- **`interpolate` on, set explicitly.** Actions rarely have estimates for every
  year. `AdditiveAction` inherits `interpolates_input_datasets_by_default = False`,
  so the flag has to be written into the binding, not left to the class.
- **`extend` on.** Without it, the contribution is null after the last entered year,
  and the outer join in `apply_hooks` then returns the base value: the action's
  effect would vanish after its target year. With `extend`, a measure that reaches
  −40 GWh/a in 2035 stays at −40 GWh/a until the model end year. **This needs
  agreement** (see the open questions below).
- **No `backfill`.** Before the start year there is no contribution, which is
  correct.

For an existing dataset the wizard writes no start-year zero (it does not write into a
dataset it does not own), so the effect starts at the dataset's first year rather than
ramping up from zero. The data-source step says so. The transformations can be changed
afterwards like those of any binding.

An input port fed by a node gets an edge with no transformations: a node's output
already covers every model year.

### Sign convention

Reductions are negative, and the stored value is what the modeller typed. The grid
header says "change in <unit>; negative = decrease". The wizard does not flip signs:
older models with mixed practices show what that costs.

### The mutation

A new mutation on `InstanceEditorMutation` runs all the steps in one transaction,
so a failure leaves nothing behind:

```graphql
createActionFromPorts(input: {
  name: String!
  identifier: String          # derived from name when omitted
  group: ID                   # action group
  startYear: Int              # default: maximum_historical_year
  effects: [{
    target: { nodeUuid: UUID!, portId: UUID }     # portId optional for single-output nodes
    label: String             # output-port (and new metric) label; default from the target
    source: {                 # @oneOf; omitted = newDataset with no narrowing
      newDataset: { categories: [{ dimension: UUID!, categories: [UUID!]! }] }
      dataset: { datasetId: ID!, metricId: ID }   # metricId optional for one-metric datasets
      node: { nodeUuid: UUID!, portId: UUID }
    }
  }!]!
}): AnyNodeType | ConstraintViolationsType
```

Steps:

1. Resolve each effect's target port and read its effective shape. If the target
   is not computable, so that its shape is unknown, refuse with the reason.
2. Resolve each existing source and check it against the compatibility rule. The
   picker already filtered, so a failure here means the model changed meanwhile;
   refuse with the reason.
3. Group the new-dataset effects by dimension set.
4. Create the action node (`createNode` internals) with one output port per effect
   and with `ActionConfig.hooks`.
5. For each dimension set, create an owned dataset using the step-2 primitive:
   schema from the ports, one metric per effect.
6. Write the start-year zeros.
7. Bind each dataset-fed input port to its metric (new or existing) with
   `interpolate` and `extend`, and create an edge into each node-fed input port
   (`createEdge` internals).
8. Run the constraint check that `bindDataset` runs (`LocalBindingEditor.add`, which
   returns `ConstraintViolationsType`), which must include the cycle check for node
   sources. On a violation, roll back and return it.
9. Record the change operations so that reverting a deletion of the action
   (decision 8 of the node-owned datasets plan) can rebuild it, datasets included.

Steps 5 and 7 are exactly the step-2 mutation, so that mutation should be built
first as a reusable function, and `createActionFromPorts` should call it. The same
applies to hooks: an `ActionHookInput` on `ActionConfigInput` (create and update)
comes first, and the wizard mutation uses it.

Why one server-side mutation rather than chaining the existing ones from the client:
chaining needs five round trips (`createDataset`, `createNode`, `bindDataset` ×n,
`createDataPoints`, plus hooks) and leaves half-built actions on failure. It would
also make the client rebuild the port pairing, column naming and effective-shape
rules, which belong to the backend (design principle 6).

## Frontend (kausal-paths-ui)

1. **Clickable output ports.** In `ElkNode.tsx`, give source handles a click or
   context menu with *New action acting on this*.
2. **Fetch `effectiveShape` on output ports** in `queries.ts`, which today fetches
   it only for input ports.
3. **Wizard drawer** with the four steps above. *Add another effect* puts the
   canvas into a pick mode where the next output-port click adds an effect, with
   Esc to cancel. *Existing node* in the data-source step uses the same pick mode,
   with incompatible ports dimmed.
4. **Dataset picker** for *Existing dataset*: instance datasets with their metrics,
   incompatible ones left out. This needs a query that answers compatibility for a
   given target port, so that the client does not reimplement the rule.
5. **After creation,** open the new dataset (if any) in `DatasetEditor` with the sign hint
   in the header, then show the action selected on the canvas, with its hooks drawn
   as port-to-port edges.

The hook-drawing gesture planned in `action-hooks.md` (drag from an action's output
port onto a node's output port) is the same operation for an action that already
exists. The two should share the `ActionHookInput` mutation and the edge rendering.

## Out of scope for v1

- **Relative effects.** Absolute changes are often hard to estimate, and relative
  ones easier: "−20 % of oil heating by 2035". These need their own rules, and the
  main difference is that **missing dimensions are fine**: a factor without the
  carrier dimension applies to every carrier. The branch already has the mechanism, a
  `FormulaAction` that reads the target's un-hooked value (`base_as` in the YAML
  plan), with formula `target * (factor - 1)`. A later wizard variant would create
  that action with a factor dataset, using dimensions chosen by the modeller rather
  than inherited. Note that hooks make two −20 % changes on the same cell remove
  40 %, not 36 %.
- **Shared action tables** (one wide dataset with an `action` column, filtered per
  action), which the older first-flow document (`docs/trailhead/action-authoring-first-flow.md`)
  starts from. Owned datasets replace that pattern.
- **Copy an existing action** (the same first-flow document). It fits the same
  mutation, with an existing action's ports as the template instead of the target's,
  and should reuse it rather than get its own.
- **Changing the shape later** (adding a dimension or an effect to an existing
  action). The `addOutputPort` and hook mutations cover it by hand for now.
- **Historical impact** of a realised measure (`action-hooks.md`, *Not built yet*).

## Open questions

1. **`extend` by default?** Proposed yes: a measure's effect normally persists after
   its target year. The alternative is to require the modeller to enter the model
   end year explicitly.
2. **`AdditiveAction` or `GenericAction`?** `AdditiveAction` is what the editor
   already creates and pairs ports for. `GenericAction` is what the hook
   verification used. One test settles whether `AdditiveAction` works with hooks.
3. **Framework restrictions.** `action-hooks.md` plans `action_hooks: false` on
   nodes where an action makes no sense, such as emission aggregates. The wizard
   should hide the menu item on such ports once that exists.
4. **Shift-type effects** that must sum to zero across a dimension (the oil →
   electricity example has different units after efficiency, so it does not). A
   later validation rule, not part of the wizard.
5. **Mixed sources in one effect?** For example, a new dataset for the years the
   modeller estimates and a node for the rest. Not in v1: one source per effect.
6. **Which instances get it?** The editor refuses yaml-sourced instances, so this
   is database-sourced only, as is the rest of the editor.

## As built (2026-10-09–10)

Branch `feat/action-from-output-port` in kausal-paths (from `feat/data-studio-backend` at
`80e2b6d8`) and in kausal-paths-ui (from `main`). Where it differs from the design above:

- **One dimension set per action.** The runtime gives a node one set of output
  dimensions for all its outputs (`Node.validate_dims`), so effects with different
  dimensions cannot share an action. The wizard refuses them and asks for a separate
  action; *one dataset per distinct dimension set* never arises. The heat-pump example's
  cost effect (no carrier dimension) is therefore a second action. Lifting this needs
  per-port dimensions at runtime, which the planned implicit dimensions would bring.
- **Hooks are edited by their own mutations**, `addActionHook` and `deleteActionHook`,
  not through `ActionConfigInput`. A config update keeps the hooks; before this, it
  silently dropped them.
- **`interpolate`, `backfill` and `extend` are now in the GraphQL transformation
  vocabulary.** They existed at runtime only, so the UI could neither set nor read them.
- **The draft graph counts hooks as edges**, so its cycle check sees them. It did not.
- **One rule for the metric behind an output port**, `Node.output_metric_for_port`. A
  single-output node's metric is named with the runtime default column whatever its
  port's `column_id` says, and three places matched by `column_id` instead: hook loading
  (a hook on such a node failed to resolve), `AdditiveAction` (an action whose single
  port has its own identifier failed to compute) and the output preview (*Metric for
  column None not found* on a node synced from YAML).
- Added: `InstanceEditor.effectSourceCandidates(target)` for the picker,
  `InstanceEditor.hooks` for drawing them, and `NodeEditor.createPortDataset` (step 2 of
  the node-owned datasets plan).
- `InstanceEditor.dataset(id)` now finds node-owned datasets, so the dataset editor can
  open the action's own dataset (lists still leave them out).

**UI** (kausal-paths-ui, branch `feat/action-from-output-port` from `main`, uncommitted):
*New action acting on this* on a node's context menu, on an output port's (right-click
the dot), and on each port row of the details panel. The wizard is a panel on the right
of the canvas, so effects and source nodes can be picked on the graph while it is open;
in pick mode nodes that cannot be picked are faded. After creation it opens the new
dataset in the dataset editor, or focuses the action when no effect has one. Hooks are
drawn as dashed edges into the output port they act on, outside the ELK layout. The
binding editor can now carry `interpolate`, `backfill` and `extend` through a rewrite.

The UI on `main` queries `DatasetValidationViolation.requirementGroup`, which
`feat/data-studio-backend` removed with the combination rules, so every dataset page failed
against that branch. This branch puts the field back, deprecated and always null, until the
UI stops asking for it. Still open: migration `nodes.0085_dataset_snapshot_v2` fails on
databases whose stored snapshots still carry `validation.combinations`.

Not built: removing hooks when the node they act on is deleted (the action then fails to
initialize).

## Implementation order

1. Hooks in GraphQL: `ActionHookInput` on create/update, hooks exposed on
   `ActionConfig`, the action→target relation drawn on the canvas.
2. Node-owned datasets step 2: the "create a port's own dataset" primitive and
   mutation.
3. A test: an `AdditiveAction` with two output ports, two owned-dataset metrics,
   two hooks; it should show interpolation between a start-year zero and one target
   year, `extend` to the end year, a narrowed category contributing zero, and no
   change in historical years. A second case feeds one effect from an instance
   dataset and the other from a node, and a third shows that a node source depending
   on the hooked target is refused.
4. `createActionFromPorts`.
5. UI: clickable output ports, `effectiveShape` on output ports, the wizard, and
   the hand-off to the dataset grid.
