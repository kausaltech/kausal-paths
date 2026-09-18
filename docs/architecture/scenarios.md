# Scenarios, and what the custom scenario is a diff against

*Produced by Claude Opus 5.0 on 2026-09-18.*
*Responsible: Jouni Tuomisto.*

A scenario is a set of parameter values. Activating one walks its `param_values` and
resets each named parameter to the value the scenario gives it (`Scenario.activate`,
`src/nodes/scenario.py`). **A scenario sets only what it names.**

That is safe rather than dangerous, because the context is rebuilt on every request (see
below), so each request starts from the parameters' config defaults. The rule that
follows is worth stating exactly, because the intuition that values "carry over" is
wrong:

* A parameter **no scenario names** is its config default in every scenario. Switching
  scenarios cannot change it, and nothing unexpected happens.
* A parameter **some scenarios name and others do not** is its config default in the ones
  that do not. Still nothing unexpected on a plain scenario switch.
* The loader folds every *customizable* parameter into the default scenario
  (`src/nodes/instance_loader.py`), capturing the value it has at load time — i.e. the
  config default. So for the default scenario, "named" and "not named" are the same
  value. On `mainz-bisko`, `weather_correction` is named by the default scenario with
  `False`, which is exactly its config default.
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

`is_customizable: false` does two things: `setParameter` refuses the parameter
(`src/params/schema.py`), and the loader stops folding its value into the default
scenario (`src/nodes/instance_loader.py`). The second is easy to miss, and it means a
scenario can only set what it **names**: a non-customizable parameter no scenario names
sits at its config default in every scenario, and no scenario can move it.

So a parameter that selects between published variants — the case this was built for is
`selected_number` with `select_variant` — has to be named by every scenario that needs a
value other than the config default, and is worth naming in the others too. On
`mainz-bisko` only `knsv_szenario_2` strictly needs its entries, since the config default
already matches what `baseline` and `default` want; listing all three is a statement of
intent rather than load-bearing, and it stops the model depending on the node default and
the scenario's intent continuing to coincide. See [`action-design.md`](action-design.md),
*A worked example*.

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
(`src/nodes/models.py`); the `_pytest_instances` reuse path is test-only. So each request
builds a fresh `Instance`, `Context` and parameter set, with every parameter at its
**config default**, and then activates the scenario the session names. A parameter value
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

**A word of warning about testing this.** `tools/debug_instance` holds a single
long-lived context, so calling `ctx.activate_scenario(...)` several times in one `-c`
script models something the web app never does: values set by an earlier activation
persist into the next one. A sequence of scenario switches written that way will show
leaks that do not exist in the product. Anything about cross-request behaviour has to be
checked through separate GraphQL requests -- which is what
`src/params/tests/test_custom_scenario_base.py` does.

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
