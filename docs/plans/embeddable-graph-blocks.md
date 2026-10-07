# Embeddable graph blocks: impact overviews and node graphs

*Produced by Claude Opus 5.5 on 2026-10-07.*
*Responsible: Jouni Tuomisto.*

Status: **design, not implemented.** This document fixes the vocabulary and the
contract before any code is written. Implementation is expected to proceed in the
phases of §9.

## 1. Goal

Two kinds of graph exist today only as parts of fixed pages:

- **Impact overviews** — the cost-effectiveness, cost-benefit, return-on-investment,
  simple-effect, stacked-raw-impact and wedge graphs on the action list page.
- **Node graphs** — the main graph on a node's own page, and the one inside an outcome
  card.

We want both as **Wagtail StreamField blocks**, so that an editor can place a specific
graph on a Paths dashboard page or static page and, later, on any Kausal Watch page that
accepts such blocks. A placed graph must show the same thing to every visitor and on
every site, unless the editor has deliberately chosen to let some aspect follow the
visitor.

That last requirement is the hard part. Most of what decides a graph's content today is
not stored anywhere: it is visitor state held in React, in Apollo reactive variables, or
in the server-side session. §3 lists it. A block has to turn every one of those into
either a stored value or an explicit "follows the visitor".

## 2. What exists today

### 2.1 Backend

- `ImpactOverviewSpec` (`src/nodes/defs/action_def.py`) defines an overview: `graph_type`,
  `effect_node_id`, `cost_node_id`, units, cut-points, dimension ids and labels. The
  runtime object is `ImpactOverview` (`src/nodes/actions/action.py`).
- GraphQL exposes `impactOverviews` (list) and `impactOverview(id)` (one)
  (`src/nodes/graphql/operations.py`), typed by `ImpactOverviewType`
  (`src/nodes/graphql/types/impact.py`). Both compute in the **request's active
  scenario and normalization**, i.e. the visitor's session state.
- `ImpactOverview.calculate_iter` skips actions that are not enabled. Which actions
  appear in an overview therefore depends on the active scenario.
- `NodeType.metricDim(withScenarios, includeScenarioKinds)` returns the node's
  `DimensionalMetric`, again in the session's scenario and normalization.
- `DashboardCardBlock` (`src/pages/blocks.py`) is the only existing block that carries a
  graph. It computes its data **inside grapple resolvers on the block**, so the data
  arrives with the page query. Its `ActionImpactBlock` child is the precedent this design
  follows for scenarios: the editor picks a scenario (`admin_instance_scenario_choices`
  excludes the custom scenario), and the resolver computes under
  `scenario.override(set_active=True, isolated=True)`, so the chart does not move when
  the visitor edits parameters.
- `Scenario.override(isolated=True)` (`src/nodes/scenario.py`) activates the default
  scenario first and then the named one, and restores every touched parameter
  afterwards. It is the mechanism for "this scenario, as a fresh request computes it".

### 2.2 Paths UI (`kausal-paths-ui`)

- Action list page: `ActionListPage.tsx` holds the state, `useActionListData.ts` loads
  and **post-processes** the data (cumulative sums over the year range, cost-benefit
  totals, wedge shares, sorting), and `ActionListGraphView.tsx` dispatches on
  `graphType` to `EfficiencyGraph`, `CostBenefitAnalysis`, `ReturnOnInvestment`,
  `SimpleEffect`, `StackedRawImpact`, `WedgeDiagram`, falling back to
  `ActionsComparison`.
- Node graph: `DimensionalNodeVisualisation.tsx` wraps the shared
  `kausal_common/src/components/paths/NodeGraph.tsx`. The node page shows it with
  controls; the outcome card (`OutcomeNodeContent.tsx`) without.
- Generic StreamField rendering (`src/components/common/StreamField.tsx`) handles only
  rich text, text, card list and framework landing blocks. The dashboard renders its
  cards itself (`DashboardPage.tsx`).

### 2.3 Kausal Watch

- Watch UI talks to Paths through its own Apollo client, with a per-operation context
  `{ uri: '/api/graphql-paths', headers: { x-paths-instance-identifier } }`. The proxy
  route forwards the `paths_api_*` cookies, which carry the Paths session (scenario,
  parameters, normalization).
- The Paths instance comes from `Plan.kausal_paths_instance_uuid`, which holds the
  instance **identifier** despite its name.
- Existing Paths-backed blocks in the Watch backend: `PathsOutcomeBlock`
  (`actions/blocks/paths_content.py`) and `PathsNodeSummaryBlock`
  (`actions/blocks/category_page_layout.py`). Both store Paths node identifiers as
  **free text** (`CharBlock`), validated nowhere.
- A page with Paths blocks turns on Watch's Paths settings panel (scenario, goal,
  parameters), so the visitor can change the session state that Paths graphs read.

## 3. What decides a graph's content today

### 3.1 Impact overview on the action list page

| Input | Held in | Visitor can change |
|---|---|---|
| Selected overview | React state `userSelectedOverviewId` | yes |
| List or graph view | React state `listType` | yes |
| Action group filter | React state `actionGroup` | yes |
| Only active actions | React state `showOnlyActiveActions` | yes |
| Sort key, direction | React state, default from `ActionListPage.default_sort_order` | yes |
| Year range | Apollo var `yearRangeVar` | yes |
| Scenario | session (`ActivateScenario` mutation), mirrored in `activeScenarioVar` | yes |
| Goal | Apollo var `activeGoalVar` | yes |
| Normalization | session (`SetNormalizer` mutation) | yes |
| Only municipal actions | `ActionListPage.show_only_municipal_actions` | no |
| Accumulated effects | `instance.features.showAccumulatedEffects` | no |

There is no annual/cumulative toggle; the graph type decides that.

### 3.2 Node graph

| Input | Held in | Visitor can change |
|---|---|---|
| Grouping dimension, category filter | React state `sliceConfig`, default from `DimensionalMetric.getDefaultSliceConfig(activeGoal)` | yes |
| Goal | `activeGoalVar` | yes |
| Year range | `yearRangeVar` | yes |
| Scenario | session | yes |
| Normalization | session | yes |
| Chart type | not settable: `stackable ? 'bar' : 'line'` in `NodeGraph` | no |
| Baseline line | `instance.features.baselineVisibleInGraphs` and `site.baselineName` | no |
| Progress-tracking series | existence of a progress-tracking scenario | no |

`NodeGraph` already accepts a `chartType` prop; nothing passes it.

## 4. Principles

1. **One definition, owned by Paths.** A graph is defined by a typed specification
   that Paths owns and validates (§5). Blocks are editors for that specification, not
   the definition itself. Watch stores the same specification; it does not invent its
   own field set.
2. **Every input is either pinned or declared as following the visitor.** No input is
   left to whatever happens to be in the session. Pinned is the default.
3. **Data is fetched, not embedded.** The block stores only the definition. The
   frontend fetches data from a stateless, parameterised Paths query (§6). This is the
   one approach that serves both Paths and Watch pages; computing data in grapple
   resolvers, as `DashboardCardBlock` does, works only for pages served by Paths.
4. **Rendering lives in `kausal_common`.** The graph components and the
   post-processing now in `useActionListData.ts` move into the shared UI package, so
   Paths and Watch render a block with the same code.
5. **References are stable or validated.** A block must not silently break when a node
   is renamed or an overview's derived id changes (§5.4).

## 5. The specifications

Both specifications are Pydantic models in Paths, e.g. in a new
`src/nodes/defs/chart_specs.py`. They share a context part.

### 5.1 Shared: `ChartContextSpec`

| Field | Type | Meaning |
|---|---|---|
| `scenario` | `ScenarioRef` | A scenario id, or the sentinel `follow_visitor`. Default: the instance's default scenario. Never the custom scenario: it is each visitor's own, so a block pinned to it would show something different to everyone. |
| `goal` | `GoalRef \| None` | An instance goal id (as in `instance.goals[].id`), `follow_visitor`, or `None` for the instance's default goal. |
| `normalization` | `NormalizationRef \| None` | A normalization id from `context.normalizations`, `follow_visitor`, or `None` for no normalization. |
| `start_year` | `int \| None` | `None` means the instance's reference year (or minimum historical year if it has none). |
| `end_year` | `int \| None` | `None` means the instance's target year. |

The year range is pinned only. Letting it follow the visitor would make one block
answer different questions on different visits, and the year range is precisely what
turns a series into a total in the cumulative graph types.

`follow_visitor` exists for scenario, goal and normalization because Watch shows its
Paths settings panel on any page with Paths content (§2.3): a graph that ignores the
scenario the visitor just picked in that panel reads as a bug. The editor chooses per
block.

### 5.2 `ImpactOverviewChartSpec`

| Field | Type | Meaning |
|---|---|---|
| `context` | `ChartContextSpec` | §5.1 |
| `impact_overview` | `str` | The overview's id. See §5.4 on stability. |
| `view` | `graph \| list` | Graph, or the tabular action list with overview columns. |
| `action_group` | `str \| None` | Restrict to one action group. |
| `only_enabled_actions` | `bool` | Default `true`. |
| `only_municipal_actions` | `bool` | Default `false`. |
| `actions` | `list[str] \| None` | Explicit subset of action ids, in display order; overrides sort. Optional, for a block that is about a few named actions. |
| `sort` | `SortKey \| None` | Only meaningful for ranking graph types; validated against `graph_type`. `None` uses the graph type's default (cost-benefit forces cumulative efficiency, descending). |
| `sort_ascending` | `bool` | |

Fields that a given `graph_type` does not use are rejected by a model validator, in
the same way `ImpactOverviewSpec.validate_graph_type_fields` already rejects fields per
graph type. A spec that validates is a spec the renderer can draw.

### 5.3 `NodeChartSpec`

| Field | Type | Meaning |
|---|---|---|
| `context` | `ChartContextSpec` | §5.1 |
| `node` | `NodeRef` | The node (§5.4). |
| `metric` | `str \| None` | Output metric id; required only when the node has more than one output metric. |
| `group_by` | `str \| None` | Dimension id to stack/group by. `None` with no filters means the total. |
| `filters` | `list[CategoryFilter]` | `{dimension: str, categories: list[str]}` per filtered dimension. Groups are allowed where the dimension has them. |
| `chart_type` | `bar \| line \| area \| None` | `None` keeps today's rule (`stackable ? bar : line`). |
| `show_goal` | `bool` | Default `true` when the node has goals. |
| `show_baseline` | `bool` | Default follows `instance.features.baselineVisibleInGraphs`. |
| `show_total_line` | `bool` | Default follows today's rule (stackable and has negative values). |
| `show_progress_tracking` | `bool` | Default `false`. |
| `compare_scenarios` | `list[str]` | Extra scenarios drawn as lines, as `metricDim(withScenarios)` already supports. |

`group_by` and `filters` replace `sliceConfig`. They are validated against the node's
**output dimensions** at save time, so a stale category is caught in the editor rather
than producing an empty chart.

### 5.4 References

| Reference | Today | Stable? | Proposal |
|---|---|---|---|
| Node, in Paths | `NodeChooserBlock` → `NodeConfig` FK | yes, survives renames | keep |
| Node, in Watch | free-text node identifier | no | store `NodeConfig.uuid`; resolve with a new `node(uuid:)` lookup, or accept either |
| Impact overview | `spec.id`, defaulting to `graph_type:effect_node:cost_node` | no: renaming either node, or changing the graph type, changes the id | make `id` required for any overview a block references, and validate uniqueness; longer term give overviews a UUID in the DB spec |
| Scenario, goal, normalization, dimension, category | string ids | as stable as the model config | validate at save; report broken references (§8.4) |

## 6. Backend API

### 6.1 Stateless parameterised queries

Add context arguments to the two entry points:

```graphql
impactOverview(id: ID!, context: ChartContextInput): ImpactOverview
node(id: ID!) { metricDim(context: ChartContextInput, ...): DimensionalMetric }
```

with

```graphql
input ChartContextInput {
  scenario: ID          # absent = follow the session
  goal: ID
  normalization: ID     # explicit null vs absent must be distinguishable
  startYear: Int
  endYear: Int
}
```

Resolution:

- A pinned `scenario` computes under `scenario.override(set_active=True, isolated=True)`,
  exactly as `DashboardCardBlock.scenario_action_impacts` does. This also fixes which
  actions are "enabled" for `calculate_iter`.
- A pinned `normalization` is applied for the duration of the resolver and restored
  afterwards. Today normalization is only ever set through `SetNormalizer` and the
  setting storage; a context manager analogous to `Scenario.override` is needed
  (`Context.override_normalization(norm)`), so the resolver cannot leak it into the
  rest of the request.
- Absent fields fall back to the session, which is what `follow_visitor` compiles to.

The pinned path must not depend on the session cookie at all. That is what lets a Watch
page cache and render the graph server-side, and what keeps one visitor's edits from
reaching a graph another editor pinned.

### 6.2 Caching

Isolated computation is a full set of model runs per pinned scenario. The cache key
must include everything in `ChartContextInput` that changes the result. Check that
scenario isolation and normalization are already part of the node cache hash before
relying on it; global parameters and their cache interaction were reworked recently
(`59db4288`). A dashboard with several blocks pinned to the same scenario should reuse
one computation.

### 6.3 Validation endpoint

Watch's admin cannot import Paths models, so Paths exposes validation:

```graphql
validateChartSpec(kind: ImpactOverview | Node, spec: JSON!): [ChartSpecError!]!
```

returning field-addressed errors (`context.scenario: unknown scenario 'x'`). The Paths
blocks call the same Pydantic validation directly.

### 6.4 Chooser data

For editors outside Paths, one query returns everything a chart editor needs to offer
choices: scenarios (excluding custom), goals, normalizations, impact overviews with
their graph types, and — for a given node — its output metrics and dimensions with
categories. Paths' own blocks use the same data through callable `ChoiceBlock` choices,
like `admin_instance_scenario_choices`.

## 7. Blocks

### 7.1 In Paths

- `ImpactOverviewBlock` and `NodeGraphBlock` in `src/pages/blocks.py`, each a
  `StructBlock` whose fields map one-to-one onto the spec, plus presentation fields:
  `title`, `caption` (rich text), `interactive` (bool).
- `clean()` builds the spec and runs its validation, raising
  `StructBlockValidationError` with field-addressed errors.
- GraphQL exposes the block's fields and a `spec` field (the serialised spec). It does
  **not** compute data: the frontend calls §6.1 with the spec's context.
- Available on `StaticPage.body` and `DashboardPage` (as a sibling of `card` in
  `dashboard_cards`, or a new body field), later on `InstanceRootPage`.

`interactive` controls only whether the graph shows its own controls (slice selector,
year range, chart tabs). With controls, the pinned values are the **initial** state and
the visitor's changes stay local to that block; they never write to the session or to
the global reactive variables. Without, the graph is static.

### 7.2 In Watch

- `PathsImpactOverviewBlock` and `PathsNodeGraphBlock`, storing the spec as a
  `JSONField`-backed block (or a `StructBlock` with the same field names) and the Paths
  instance identifier, defaulting to the plan's.
- Editing uses a custom block widget that loads chooser data from §6.4 and validates via
  §6.3 on save. Free-text ids, as in the existing Watch blocks, are not acceptable for
  the new ones.
- The existing `PathsOutcomeBlock` and `PathsNodeSummaryBlock` stay as they are; they
  can migrate to the new spec later.

### 7.3 Alternative considered: stored chart definitions in Paths

Instead of inline specs, Paths could keep named chart definitions as rows (a
`ChartDefinition` model with a UUID, owned by an instance, with a permission policy),
and blocks in both systems would reference one by UUID. That gives a single place to
edit a graph used on several sites, survives renames by construction, and spares Watch
any chooser UI beyond "pick a chart". It costs a new model, a permission policy, an
admin UI, and a two-step editing workflow.

**Recommendation:** start with inline specs and a remote chooser. Move to stored
definitions if editors start re-creating the same graph on several pages or sites, at
which point the inline spec becomes the stored row's payload and nothing else changes.

## 8. Rendering

### 8.1 Shared components in `kausal_common`

- `ImpactOverviewChart({ spec, instance })`: fetches `impactOverview(id, context)`,
  applies the post-processing now in `useActionListData.ts` (year-range sums,
  cost-benefit totals, wedge shares, filtering, sorting), and dispatches on
  `graphType`. The dispatch now in `ActionListGraphView.tsx` moves with it, and the
  action list page becomes a consumer of the shared component.
- `NodeChart({ spec, instance })`: fetches `metricDim(context)` and goals, and renders
  `NodeGraph` with `chartType`, slice and toggles taken from the spec.
- Both take the Apollo client context as a prop, so Paths uses its own endpoint and
  Watch uses `/api/graphql-paths` with the instance header.

### 8.2 Dispatch

- Paths UI: `StreamField.tsx` gains two cases; `DashboardPage.tsx` renders them beside
  cards.
- Watch UI: `StreamField.tsx` gains two cases. A block with any `follow_visitor` field
  counts as Paths content for `hasPathsContent`, so the settings panel appears; a fully
  pinned block does not need it.

### 8.3 Pinned values must be visible to the reader

A graph pinned to a scenario that differs from the visitor's active one must say so in
its caption ("Scenario: Climate plan"), as must a pinned normalization ("per
inhabitant"). Otherwise the same page shows two graphs of the same node with different
numbers and no explanation.

### 8.4 Broken references

If validation fails at render time (a node removed, a category renamed), the component
renders a visible placeholder naming the problem to editors in preview, and hides the
block for public visitors. It never falls back to a default slice or scenario: a wrong
graph is worse than a missing one.

## 9. Phases

1. **Stable overview ids.** Make `id` required for referenced overviews; add a check
   that lists overviews with derived ids. No user-visible change.
2. **Specs and validation.** `ChartContextSpec`, `ImpactOverviewChartSpec`,
   `NodeChartSpec` with validators and tests against an instance with dimensions,
   goals, normalizations and several scenarios.
3. **Parameterised queries.** `ChartContextInput` on `impactOverview` and `metricDim`,
   the normalization override, and tests that a pinned query is unaffected by session
   state (edit a parameter in the session, query pinned, compare).
4. **Paths blocks** on `StaticPage` and `DashboardPage`, with `clean()` validation.
5. **Shared rendering** in `kausal_common`; move the action list page onto it.
6. **Watch**: chooser and validation queries (§6.3–6.4), the two blocks, the custom
   editor widget, the Watch UI dispatch.

Phases 1–3 are backend only and are useful on their own: the parameterised queries
also let the action list page and the node page drop their dependence on session state
for any graph that does not need it.

## 10. Open questions

- **Instance per block in Watch.** A plan has one Paths instance today. Should a block
  be able to name another (e.g. a regional instance next to a city's)? The spec allows
  it; the Watch proxy and cookie handling would need checking.
- **Published revision vs draft.** Public visitors are served the instance's live
  revision; editors previewing a Watch page see what? The block's validation should run
  against the revision the page will be read with.
- **Overview identity in the DB spec.** Whether `ImpactOverviewSpec` should carry a
  UUID, as nodes and datasets do, or a required human-readable id is enough.
- **Normalization and goals.** A goal implies a dimension slice in some instances
  (`getDefaultSliceConfig(activeGoal)`). When both `goal` and `group_by`/`filters` are
  pinned and disagree, which wins? Proposal: the explicit slice wins, and validation
  warns.
- **Year range in non-timeline overview types.** For the summary types the year range
  is part of the computation, not just the axis. Confirm that moving the summation
  server-side, or into the shared component, reproduces today's numbers exactly before
  switching the action list page over.
