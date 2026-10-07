# Scenarios, and what the custom scenario is a diff against

*Produced by Claude Opus 5.0 on 2026-09-18.*
*Responsible: Jouni Tuomisto.*

A scenario is a set of parameter values. Activating one restores configured parameter
defaults, then walks its `param_values` and sets each named parameter to the value
the scenario gives it (`Scenario.activate`,
`src/nodes/scenario.py`). **A scenario sets only what it names.**

Scenario activation restores configured defaults, including municipal defaults, so
omitted parameters cannot retain a deviation from the previously active scenario. The rule that
follows is worth stating exactly, because the intuition that values "carry over" is
wrong:

* A parameter **no scenario names** is its config default in every scenario. Switching
  scenarios cannot change it, and nothing unexpected happens.
* A parameter **some scenarios name and others do not** is its config default in the ones
  that do not. Still nothing unexpected on a plain scenario switch.
* The default scenario names only deliberate deviations from configured defaults.
  The loader does not copy customizable parameters into its `param_values`. A
  parameter such as `weather_correction` can therefore remain at its declared
  default without an entry in the default scenario.
* **The one place two scenarios' values meet inside a single request is the custom
  scenario**, which activates a base and then applies a diff. That is where a parameter
  differing *between* scenarios could produce a value belonging to a scenario the visitor
  is not looking at — and it did, until the base stopped being fixed to the default. On
  `mainz-bisko` the parameter that differs is `selected_number`: `baseline` 0, `default`
  0, `knsv_szenario_2` 1.

So the thing to get right is not "name everything everywhere", it is: **a parameter whose
value differs between scenarios must be named by each scenario that needs a non-default
value**, and the custom scenario's base has to be the one the visitor branched from.

## The custom scenario

`CustomScenario` is the visitor's own scenario. It holds no values of its own: it
activates another scenario and then applies the overrides stored in the session
(`SessionStorage`, `src/params/storage.py`), so it is a **diff plus a base**.

Two mechanics follow from that, and both have bitten:

### The base is the scenario the user branched from

`custom_base` in the session records which scenario the stored overrides are a deviation
*from*, and `CustomScenario.resolve_base()` reads it. `set_parameter`
(`src/params/schema.py`) maintains it with one rule:

* editing a parameter while a **named** scenario is active starts a new branch — the
  previous overrides are cleared and the base becomes that scenario;
* editing while the custom scenario is already active adds to the branch in progress.

Before 18 September 2026 the base was bound once to the default scenario, and the
overrides were never cleared. That produced two failures, both reproducible and both now
covered by `src/params/tests/test_custom_scenario_base.py`:

* Turn action X off in the default scenario, switch to the baseline, turn action Y on.
  The result was *every* action enabled except X, because the diff still held X and was
  still being applied to the default. It is now the baseline plus Y.
* Touch any action while a non-default scenario is active, and every parameter the diff
  did not name silently reverted to the default scenario's value. On `mainz-bisko` this
  re-based all seventeen `selected_number` parameters from Szenario 2 to Szenario 1:
  turning a measure **off** lowered Scope 1+2 by 957 t, because the hidden variant
  switch outweighed the measure. Emissions now move in the direction the user's action
  implies.

`base_scenario` remains on the class as the fallback for a session that has not branched
yet, and for a stored base id a later config no longer has.

Because the branch rule triggers on an edit, it depends on `setParameter` only ever
being issued for a deliberate user edit. In `kausal-paths-ui` that holds: the live call
sites are `ParameterWidget.tsx` (slider, number input, switch) and
`NodeDetailsPanel.tsx` (the model editor's action switch), both from change handlers,
with no effect-driven calls. A component that fired `setParameter` on mount would
silently discard a saved branch, so keep it that way.

### A hidden parameter is a scenario's property, not a node's

`is_customizable: false` makes `setParameter` refuse ephemeral visitor edits
(`src/params/schema.py`). It does not govern municipal administrator edits: the
separate `owner` field controls persisted local values. No parameter is automatically
folded into the default scenario. A scenario sets only what it **names**; omitted
parameters use configured defaults, including municipal defaults composed from
local default-scenario overrides. See [template inheritance](template-inheritance.md).

So a parameter that selects between published variants — the case this was built for is
`selected_number` with `select_variant` — has to be named by every scenario that needs a
value other than the config default, and is worth naming in the others too. On
`mainz-bisko` only `knsv_szenario_2` strictly needs its entries, since the config default
already matches what `baseline` and `default` want; listing all three is a statement of
intent rather than load-bearing, and it stops the model depending on the node default and
the scenario's intent continuing to coincide. See [`action-design.md`](action-design.md),
*A worked example*.

### Per-operation overrides are the custom scenario without the session

`InstanceType.model(scenario:, parameters:, normalizer:)` builds a runtime of its own for
that field. It exists for clients that hold no session, such as a server loader with a
bearer token, and for showing two settings side by side in one response (Bundesmix and
Lokalmix under two aliases).

It is deliberately *not* a way of setting parameters outside scenarios (see below).
`ModelOverrides.storage_for` (`src/params/overrides.py`) starts from the visitor's stored
settings and applies the overrides by the same branch rule as `setParameter`, into an
in-memory `InstanceDataStorage` that is never written back. So an overridden parameter
puts the runtime in the custom scenario, based on the scenario that was active or the one
named by `scenario:`. `activeScenario` under that `model` says so, as the selector would.

Two rules keep the runtimes apart, and both are there because breaking them fails
silently:

* **A field takes its runtime from its root.** `pass_context` passes `root.context`, and
  objects that leave the runtime (action groups read from a spec, visualizations) carry
  a reference back to it. Only entry points with no root, such as top-level query fields,
  mutations and pages, ask the request for the operation's plain runtime.
* **An overridden runtime is never ambient.** It is entered with `ambient=False`, so
  `InstanceConfig.get_instance()` keeps returning the plain runtime for the rest of the
  request.

## Considered and rejected: parameters outside scenarios

Discussed in mid-2026 and **deliberately not implemented**. Recorded so it
is not re-opened without a new reason.

The idea was that some parameters do not really belong to a scenario. Weather correction
is the example: you may want to look at the default scenario with the correction and
without it, and in both cases it is arguably still the default scenario. A parameter like
that could sit outside the scenario system and be toggled without changing which scenario
is active.

The objection is the side effect. In a BISKO model a visitor could switch from the default
scenario to the baseline without noticing that weather correction has been on the whole
time, and every figure they then look at is **not BISKO-conformant** while the scenario
selector says nothing about it. A result that depends on state the scenario selector does
not show is the same failure the custom-scenario base fixed above, and a parameter outside
scenarios would make it the intended design rather than a bug.

Conclusion: there is no strong case for it. Do not implement it unless an important use
case turns up that this reasoning does not cover.

### Why it cannot leak: the context is rebuilt every request

The objection above is about a *design* that does not exist, and it is worth saying why
the runtime does not produce the same effect by accident.

`InstanceConfig.enter_instance_context` calls `_initialize_instance` on every request
(`src/nodes/models.py`); the `_pytest_instances` reuse path is test-only. Each request
builds a fresh `Instance`, `Context` and parameter set. Scenario
activation also explicitly restores configured defaults before applying the
scenario, so the loader's initial default-scenario activation cannot leak into
another scenario. A parameter value
therefore cannot survive a scenario switch in memory.

Checked against `mainz-bisko`, where `weather_correction` is a customizable global
parameter that no scenario names in YAML:

| step | `weather_correction` |
|---|---|
| default scenario | `False`, its config default |
| visitor switches it on | `True`, and the selector says *Custom* |
| visitor selects any scenario | `False` — fresh context, and no scenario names it |

So what a parameter outside scenarios would introduce is genuinely new, not a formalisation
of something already happening.

**Switching versus evaluating.** Activation is a switch: it restores configured
defaults first, so a long-lived context behaves like a fresh request on each
activation. `Scenario.override()` is an evaluation: it activates the scenario the same
way, so another scenario's deviations do not carry into it, but by default it then
reapplies the visitor's own edits (customized parameters) that the scenario does not
name -- which is what an action's impact against the baseline needs.
`override(isolated=True)` leaves them out, for views that must show the scenario as
configured, such as a dashboard card. Either way it saves and restores every
parameter, including the ones the scenario omits. Session
branching still needs request-level tests because its overrides live in session
storage.

## What is not built yet

* ~~The base is not exposed.~~ Added 18 Sep 2026: `ScenarioType.baseScenario` returns the
  scenario the custom scenario is a diff against, and null for every other scenario, while
  `ScenarioType.customizedParameters` lists the parameter ids it overrides. Together they
  are what a "how does my scenario differ from the one I branched from" view needs. The
  frontend does not read either field yet.

  `parameterOverrides` stays empty for the custom scenario, because a visitor's overrides
  live in the session rather than in `param_values`; its description says so.
* **There is no explicit "save the current scenario as my custom scenario".** The branch
  rule makes the common case right without one. If it is added, it belongs in a mutation
  rather than a pseudo-scenario in the dropdown: the custom scenario is already a member
  of `context.scenarios`, so anything else added there becomes activatable and storable
  as the active scenario id.
