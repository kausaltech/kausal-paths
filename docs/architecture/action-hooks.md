# Actions acting on nodes (hooks)

An action can *act on* any node. The node computes exactly as it would
without the action, and the action's output is added to that result before it
reaches the node's consumers. In the code the connection is a **hook**; in the
UI and in modeller-facing text it is "the action acts on the node".

The problem this solves: a municipality inheriting a framework template (BISKO)
wants to connect its own measures to nodes the framework owns. Adding an input
to a framework node changes the framework's calculation, and would need an
explicit port on every node a measure might plausibly affect. A hook changes
neither: the framework node's own calculation stays intact, and the hook
belongs to the action, so the instance does not need to own the node.

## Semantics

In graph terms, a hooked output behaves like an extra additive node spliced
into the target's outgoing edges:

```
target.out ──► Σ ◄── action.out
               Σ ──► (the target's consumers)
```

At runtime there is no such node: `Node._get_output_pl()` applies the hooks
after `compute()`, see `nodes/hooks.py`. The rules:

1. **Additive only.** A contribution is an amount in the target's unit and
   dimensions, or a shift between categories that sums to zero. Multiplicative
   effects are not supported: with several actions on one cell they compound,
   and the cross term belongs to no action, so no action's impact is its own.
2. **Every hook on a node sees the same un-hooked value.** Hooks do not chain.
   Effects add up, their order does not matter, and each action's effect is
   exactly its own output.
3. **Balance years are never touched.** Contributions enter only years after
   the instance's last historical year (`maximum_historical_year`, or the
   target's last non-forecast year when that is unset). An action's historical
   values are the realised part of the measure. They are kept, not treated as
   an error, for the historical-impact view below.
4. **Relative effects are explicit in the action.** An action may read the
   un-hooked value of a node it acts on (`Node.get_base_output_pl()`) and turn
   a relative effect into the amount it adds. `formula.FormulaAction` does this
   by name: in its formula, the identifier of the node it acts on stands for
   that node's un-hooked value, e.g.
   `district_heating_emissions_predominant_variant * (factor - 1)`. Because of
   rule 2, two −20 % changes remove 40 % of the un-hooked value, not 36 %.

A hook is part of the action's spec (`ActionConfig.hooks`, see `ActionHookDef`),
never a binding on the target: the target's `input_nodes`, input ports and
bindings are unchanged. For graph purposes it counts as an edge from the action
to the target: `context.node_graph` contains it (cycle detection, downstream
traversal, impact paths), `get_upstream_nodes()` follows it, and a node's cache
hash includes the hashes of the actions acting on it, so toggling an action
invalidates the target and everything downstream.

A hooked node has two hashes and two cache entries: its own (base) output, and
its effective output (base plus hooks). An action reading the base hashes
against the base hash, the effective hash includes the action, so the two never
recurse; modifying the target invalidates the actions reading it. Reading the
base adds no edge to `node_graph`, so it is not a cycle. Disabled actions
contribute nothing.

## YAML

An entry in the action's `output_nodes` marked `hook: true`. The dimension
options are those of an ordinary edge; `tags` are rejected.

```yaml
- id: kwp_waermewende
  type: simple.GenericAction
  unit: MWh/a
  output_nodes:
  - id: final_energy_use
    hook: true
    to_metric: energy        # only when the target has several outputs

- id: zp_fw_erdgas_heizoel_ausstieg
  type: formula.FormulaAction
  quantity: emissions
  unit: kt_co2e/a
  input_datasets:
  - id: mainz/zielpfad_fernwaerme   # a path of factors, 1 = no change
    tags: [factor]
  params:
  - id: formula
    value: district_heating_emissions_predominant_variant * (interpolate(factor) - 1)
  output_nodes:
  - id: district_heating_emissions_predominant_variant
    hook: true
```

For an action with several output metrics, `metrics: [one_metric]` names the
one that acts.

## Verified on Mainz

Rewiring `kwp_waermewende` from an input of `final_energy_use` to a hook gives
identical `final_energy_use` and `net_emissions` (to floating-point precision)
with the action enabled and disabled; the action's impact on `net_emissions`
is unchanged. Rewriting the multiplier action `zp_fw_erdgas_heizoel_ausstieg`
as the `FormulaAction` above is equally exact, target, `net_emissions` and
impact alike. The formula's `interpolate(factor)` does what `GenericAction`
did implicitly: it fills the years between the path's support years. The
verification ran with the dataset flag `interpolate: true`, which uses the same
implementation. Where the `zielpfad_*` scenarios
enable two multiplier paths on the same cell, they compound today; as hooks
they add, and the overlap becomes visible.

## In the model editor

`ActionConfigType.hooks` exposes an action's hooks, and `InstanceEditor.hooks`
lists every hook in the instance with both ports resolved, for drawing them on
the canvas. `NodeEditor.addActionHook` and `deleteActionHook` edit them.
`addActionHook` resolves omitted ports to the node's only output, and refuses a
hook from a node that is not an action, on a port that does not exist, the same
hook twice, one whose shapes cannot add up (different dimensions, or a unit
that does not convert), and one that closes a loop. The shape rule is
`incompatibility()` in `nodes/action_effects.py`: the solver's shape where it
has one, the port's declaration where it does not.

`ActionConfigInput` carries no hooks, so updating an action's config keeps
them. The draft `InstanceGraph` counts a hook as an edge from the action to the
node it acts on (`InstanceGraph.hook_edges`), as the runtime `node_graph` does.

`createActionFromPorts` builds a whole action from the ports it acts on; see
`docs/plans/action-from-output-port.md`.

## Not built yet

- **Drawing a hook by hand on the canvas.** The editor shows hooks (dashed edges
  into the output port acted on) and creates them through the action wizard, but
  the planned gesture for an existing action, dragging from its output port onto
  another node's output port, is not built. The mutations exist.
- **Deleting a node that actions act on** leaves their hooks dangling; the
  action then fails to initialize with `Invalid hook`. The delete should remove
  them, or refuse.
- **Framework restrictions.** The template should be able to mark nodes where
  an action makes no sense (`action_hooks: false`: emission aggregates,
  data-quality and presence nodes). Shift-type outputs such as
  `transport_energy_shares` could declare that contributions must sum to zero
  across a dimension.
- **Historical impact.** The measured history already contains a realised
  measure, so its impact is the difference to a counterfactual history without
  it: `observed − contribution`, the mirror image of the forecast
  (`baseline + contribution`). One hook serves both: its future part is added
  to form the with-action path, and its past part is subtracted to form the
  without-action history, which never replaces the balance. Before this can be
  built, calibrated nodes need marking. A per-unit intensity back-calculated
  from observed totals ("measured street-lighting electricity / number of
  luminaires") must be held at its observed value in the counterfactual run.
  Otherwise it re-fits to the same measured total and the impact vanishes.
  The terminology is measurement and verification (IPMVP): avoided energy use
  against an adjusted baseline; deemed vs measured, gross vs net savings.
