# Scenario mutations in the model editor

*Produced by Claude Opus 5.5 on 2026-11-06.*
*Responsible: Jouni Tuomisto.*

Status: **proposal, for review before any code is written.** Nothing here is implemented.
The open questions in the last section need answers first.

## Goal

Expert users should be able to change everything about a scenario from the model
editor: create and delete it, rename it, change its description, kind,
`all_actions_enabled` and `is_selectable`, and set or clear the parameter values it
holds. Today they can do none of this. Scenarios are edited only in YAML, or by hand in
`InstanceConfig.spec`.

## Where things stand

### On `main`

`src/nodes/graphql/editor.py` declares `createScenario`, `updateScenario` and
`deleteScenario`, with `CreateScenarioInput` and `UpdateScenarioInput`. All three raise
`Scenario mutations not yet implemented`. The inputs are shaped differently from the rest
of the editor:

- they take `instance_id` / `scenario_id` arguments instead of hanging off
  `instanceEditor(instanceId)`;
- `CreateScenarioInput` has no `description`;
- `kind` is a free `str`, not the `ScenarioKind` enum;
- `UpdateScenarioInput.parameter_overrides` is untyped JSON.

### On `feat/data-studio-backend`

Checked at `7d97b622` (30 October 2026). The branch is 122 commits ahead of `main` and 54
behind. The three scenario stubs are unchanged there, but the branch already has most of
the parameter half of this work:

- **`setInstanceParameter(parameterId, value, scenarioId, reset)`** on the instance
  editor. It calls `nodes.parameter_values.set_scenario_parameter`, which:
  - looks the parameter up in the *effective* spec, with the template merged in;
  - follows a `ReferenceParameter` to its target;
  - validates the value with `parameter.clean()`;
  - refuses framework-owned parameters on template-derived instances;
  - refuses while the instance is locked;
  - stores only real deviations: a value equal to the inherited one, or `reset=True`,
    removes the entry instead of writing it.

  On template-derived instances it also records `parameter_types`. Leaving out
  `scenarioId` means the default scenario. The change is recorded as
  `instance.parameter.set`.
- **Scenario semantics changed.** `Scenario.activate()` restores every parameter to its
  configured value before applying the scenario's own. The loader no longer copies
  customizable parameters into the default scenario. See
  [`docs/architecture/scenarios.md`](../architecture/scenarios.md) on the branch.
- **Template-derived instances keep scenarios sparse.** `compose_instance_spec`
  (`src/nodes/template_spec.py`) treats a local scenario whose id matches a template
  scenario as a `param_values` override only: its name, kind and flags are ignored.
  `InstanceModelSpec.local_scenario()` creates such override entries with an empty name.
  A local scenario with a new id is added as a scenario of its own, and must have a name.
- **Publish-time check.** `validate_spec_references` refuses to publish a spec whose
  scenarios name a missing parameter.

So for these instances `spec.scenarios` is not "the list of scenarios". It mixes real local
scenarios with override entries for template ones. Any scenario mutation has to work out
which kind of entry it is editing.

## Proposal

### 1. Build on the branch, not on `main`

The rules these mutations must enforce are exactly what the branch rewrites: which
parameters may be set, what an omitted parameter means, and how template scenarios
combine with local ones. Written on `main`, they would encode the old rules and then
conflict in `editor.py`. Either the branch merges first, or this work branches from it.

### 2. Parameters: keep `setInstanceParameter`, drop `parameter_overrides`

`setInstanceParameter` already does what a typed set/unset pair would do, `reset` included.
Remove `parameter_overrides` from `UpdateScenarioInput` rather than give it a second,
untyped path that skips `clean()` and the ownership check.

Two small additions:

- On instances that are not template-derived, `set_scenario_parameter` uses
  `next(s for s in spec.scenarios if s.id == scenario.id)` with no default. A scenario
  that only exists in the effective snapshot would raise `StopIteration`, not a validation
  error. That cannot happen without templates today, but it is cheap to make explicit.
- A query listing each scenario's effective values together with where each value comes
  from (template, local default, this scenario). The editor needs this to show what
  `reset` would revert to.

### 3. Scenario CRUD under `instanceEditor`

Move the three mutations onto the instance editor type, next to `setInstanceParameter`
and the action-group mutations, and copy the action-group pattern:

- run inside `gql_change_operation`;
- `refresh_from_db(fields=['spec'])`;
- edit a copy of the list;
- persist it;
- `record_change` with before and after snapshots;
- invalidate the draft runtime.

Like every editor change, it goes to the draft and is public only after the instance is
published.

| Mutation | Arguments | Rules |
|---|---|---|
| `createScenario` | `identifier`, `name`, `description?`, `kind?` (enum), `allActionsEnabled`, `isSelectable`, `copyFrom?` | Identifier must be valid and not used by any scenario in the effective spec. Kind cannot be `DEFAULT` (there is exactly one) or `CUSTOM` (that is the visitor's session scenario). `copyFrom` copies another scenario's effective `param_values`, the usual way to start a new scenario. |
| `updateScenario` | `scenarioId`, then `name`, `description`, `kind`, `allActionsEnabled`, `isSelectable`, each a `Maybe` | Kind cannot change to or from `DEFAULT`, or to `CUSTOM`. On a template-derived instance, a template scenario cannot be changed (see open question 2). |
| `deleteScenario` | `scenarioId` | Refused for the default scenario and for template scenarios. Refused while anything refers to the identifier (see section 4). |

The mutations take the identifier as `scenarioId`, because scenarios have no UUID.

### 4. The identifier is fixed after creation

Unlike `ActionGroup`, a scenario has no UUID, so its identifier is its identity, and several
things refer to it by name:

- the base scenario a visitor's custom scenario was branched from, kept in their session;
- scenario-specific values declared on actions (`instance_loader.py`, the per-scenario
  parameter values);
- `self.scenarios['baseline']` in `Context` (`src/nodes/context.py`), which assumes that
  identifier exists;
- template override entries, which match by id.

The proposal is to refuse renaming the identifier for now and let users change `name`
freely. Adding a `uuid`, as `ActionGroup` has, would make renaming possible later. That is
a spec migration, not part of this work.

For `deleteScenario`, the reference check covers the action declarations and the
`baseline` identifier. Session references cannot be checked, and do not need to be:
`CustomScenario.resolve_base()` already falls back to the default scenario, with a
warning, when the stored base no longer exists.

### 5. Tests

Using `PathsTestClient`, on both a plain database-sourced instance and a template-derived
one:

- create → update → delete round trip, with the change log entries;
- `copyFrom` copies effective values, including template ones;
- refusals:
  - a duplicate identifier;
  - creating with kind `DEFAULT` or `CUSTOM`;
  - deleting the default scenario;
  - deleting a scenario that an action still refers to;
  - editing a template scenario;
  - any change while the instance is locked;
- after each change, the draft runtime's scenario list matches, and activating a new
  scenario applies its values;
- `setInstanceParameter` on a scenario created through `createScenario`.

## Open questions

1. **When does `feat/data-studio-backend` merge?** That decides whether this branches from
   it or waits.
2. **Template scenarios on a derived instance:** can a municipality rename one, change its
   flags or hide it (`isSelectable: false`), or only override its parameter values? The
   merge currently ignores everything except `param_values`. Allowing more means extending
   the override entry and `compose_instance_spec`.
3. **Was an API already planned for the stubs?** If the stub signatures reflect a
   frontend contract, the move under `instanceEditor` needs agreeing with the UI side.
4. **`actual_historical_years`:** should the editor expose it, or is it framework territory?
5. **Formula storage,** raised by the new `setFormula` editor mutation. A node created by
   the editor with kind `FORMULA` keeps its formula in `FormulaConfig.formula`, which the
   runtime never reads. The runtime reads the `formula` parameter. `setFormula` currently
   writes both. Which one should be canonical? `FormulaConfig` fits better if a formula is
   node structure rather than a value scenarios may set.
