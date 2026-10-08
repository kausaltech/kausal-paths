# Creating an action from an output port

*Produced by Claude Opus 5.5 on 2026-10-08.*
*Responsible: Jouni Tuomisto.*

Status: proposal, not yet agreed with Juha. It is written against
`feat/data-studio-backend` as of `da413585` (2026-10-07), because the two pieces it
builds on, action hooks and node-owned datasets, exist only there. Build it on a
branch from that branch, not from `main`.

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
3. **Identity.** Name (the identifier is derived from it), action group (inline
   creation exists, AC-4.3), and the start year.
4. **Create.** One mutation creates everything (below). The wizard closes, and the
   action's dataset opens in the existing dataset grid
   (`DatasetEditor` / `DatasetDataGrid` in kausal-paths-ui).
5. **Enter numbers.** The modeller adds the years they have estimates for with the
   grid's *Add years*. This is often just a target year, such as 2030 or 2045. They
   type the change, negative for a reduction. CSV/Excel import works as for any
   dataset.

An example with two effects: *Replace oil boilers with heat pumps*. Effect 1 is
picked from the output port of building final energy use (dimensions: energy carrier),
with categories narrowed to oil and electricity. In 2035 it is −40 GWh/a for oil and
+12 GWh/a for electricity. Effect 2 is picked from the output port of building
heating costs, at −1.5 M€/a.

## What `feat/data-studio-backend` already has

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

### Shape inheritance

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

- **Owned by the action** (`Dataset.scope` = the action's `NodeConfig`), with no
  identifier. It is created, copied, exported and deleted with the action, and it
  never shows up in the instance's dataset list.
- **One metric per effect.** The metric label is the target's name and port label,
  and the unit and quantity are the target's. If the label is not unique within the
  dataset, the editor asks for a different label, as decision 13 of the node-owned
  datasets plan requires.
- **One dataset per distinct dimension set.** When all effects have the same
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

Each paired input port is bound to its metric with these transformations:

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
    categories: [{ dimension: UUID!, categories: [UUID!]! }]   # optional narrowing
    label: String             # metric / output-port label; default from the target
  }!]!
}): AnyNodeType | ConstraintViolationsType
```

Steps:

1. Resolve each effect's target port and read its effective shape. If the target
   is not computable, so that its shape is unknown, refuse with the reason.
2. Group the effects by dimension set.
3. Create the action node (`createNode` internals) with one output port per effect
   and with `ActionConfig.hooks`.
4. For each dimension set, create an owned dataset using the step-2 primitive:
   schema from the ports, one metric per effect.
5. Write the start-year zeros.
6. Bind each paired input port to its metric with `interpolate` and `extend`.
7. Run the constraint check that `bindDataset` runs (`LocalBindingEditor.add`, which
   returns `ConstraintViolationsType`). On a violation, roll back and return it.
8. Record the change operations so that reverting a deletion of the action
   (decision 8 of the node-owned datasets plan) can rebuild it, datasets included.

Steps 4 and 6 are exactly the step-2 mutation, so that mutation should be built
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
3. **Wizard drawer** with the three steps above. *Add another effect* puts the
   canvas into a pick mode where the next output-port click adds an effect, with
   Esc to cancel.
4. **After creation,** open the new dataset in `DatasetEditor` with the sign hint
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
5. **Which instances get it?** The editor refuses yaml-sourced instances, so this
   is database-sourced only, as is the rest of the editor.

## Implementation order

1. Hooks in GraphQL: `ActionHookInput` on create/update, hooks exposed on
   `ActionConfig`, the action→target relation drawn on the canvas.
2. Node-owned datasets step 2: the "create a port's own dataset" primitive and
   mutation.
3. A test: an `AdditiveAction` with two output ports, two owned-dataset metrics,
   two hooks; it should show interpolation between a start-year zero and one target
   year, `extend` to the end year, a narrowed category contributing zero, and no
   change in historical years.
4. `createActionFromPorts`.
5. UI: clickable output ports, `effectiveShape` on output ports, the wizard, and
   the hand-off to the dataset grid.
