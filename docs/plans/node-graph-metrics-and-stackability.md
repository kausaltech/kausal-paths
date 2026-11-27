# Node graph: multiple metrics and stackability

*Produced by Claude Opus 5.5 on 2026-11-15.*
*Version 2 produced by Claude Opus 5.5 on 2026-11-27.*
*Responsible: Jouni Tuomisto.*

A plan for a new version of the node graph that (1) shows a node's several output
metrics, either as tabs or side by side, and (2) decides, dimension by dimension, whether
the values may be summed. It is built as a **copy** of the current components and is
first used from `ParameterPanelBlock`. The node detail page, `OutcomeNodeContent`, the
model editor and Watch keep the current components until the copy has been tried on real
models.

**Version 2 (2026-11-27)** records a design discussion. In short:

- Tabs are no longer the only layout for a multi-metric node; graphs side by side are
  tried too, and the choice is made after seeing both (§4).
- Dimensions come in three kinds (common, node, scenario), and scenario is not a common
  dimension. The dimension-level veto (`stackable: false` on a dimension) is therefore
  kept as a **planned possibility, not built**, until a case appears that cannot be
  handled without it. Overlapping categories ("young people", "women") are a data
  problem, to be fixed with exhaustive categories in explicit dimensions (§2.2).
- The quantity-kind and metric layers stand, but must be built on the quantity registry
  in `configs/quantities/` and `src/nodes/quantities.py` (§2.6).
- Weighted sums stay in the plan as a concept. The two current use cases are handled by
  model nodes and need no new feature (§2.7).
- The visitor gets no stackability or reduction controls. They choose what they choose
  today: the dimension to break down by and the categories to filter (§3.1).

Related documents:
[`embeddable-graph-blocks.md`](embeddable-graph-blocks.md) (the `NodeChartSpec` this
plan should stay compatible with) and
[`../architecture/metric-dataframe.md`](../architecture/metric-dataframe.md)
§*Quantity kind vs aggregation behavior* (which already calls a single stackable flag
"too coarse").


## 1. Current state

### 1.1 Multi-metric nodes draw nothing

`Node.metricDim` calls `metric_from_node` (`src/nodes/metric_gen.py:465`), which calls
`node.get_default_output_metric()` (`src/nodes/node.py:1218`). With more than one output
metric and none named `default`, that raises; the resolver catches the exception, sends it
to Sentry and returns `null`, and the page shows no graph. The field has no argument for
choosing a metric.

What already exists:

- `from_node_output_metric(node, m, …, port=…)` (`metric_gen.py:397`) computes any metric.
- `DimensionalMetric.from_output_port` (`metric.py:455`), exposed as
  `OutputPortType.output`, is used by the model editor. It has no scenarios, no input
  nodes and returns `null` for action nodes.
- `metric_gen.py:479` has a FIXME: goals are not filtered to the chosen metric.
- The `DimensionalMetric` id is `port.id if port else node.id` (`metric_gen.py:430`). Two
  metrics of one node fetched without a port would therefore share an id, and Apollo
  would merge them in its cache.

### 1.2 Stackability is one flag per metric, and the chart ignores it when summing

- Backend: `stackable = m.quantity in STACKABLE_QUANTITIES` (`metric_gen.py:420`), plus a
  special case making `mix` non-stackable on action nodes. The quantity registry
  (`configs/quantities/quantity_kinds.yaml`) has its own `is_stackable`, exposed in GraphQL
  and unused by the UI. In it `mix` and `grouped_mix` are stackable and `fraction` is not.
- `NodeGraph.tsx:150`: stackable → bars (or the caller's chart type), otherwise lines. The
  total line and total row appear only when stackable.
- `sliceBy` (`kausal_common/src/utils/paths/metric/slicing.ts:51`) **sums every dimension
  that is not the break-down dimension**, and `flatten` sums all of them, whatever the
  metric's stackability. For a non-stackable metric with two or more dimensions, each line is
  a sum of, say, emission factors over the other dimension. It is right only when the
  visitor's filters happen to leave exactly one category in every other dimension.
- The older `Metric.from_node` (`metric.py:106`) refuses (returns `None`) in the same
  situation instead of summing; the dimensional path has no such guard.
- Two dimensions already avoid the problem by being filtered away before slicing: the
  scenario dimension (filtered per scenario in `parse.ts`) and the uncertainty dimension
  (filtered to `median` in `from_node_output_metric`).
- `DimensionSumRule` (`src/datasets/validation_rules.py:94`) checks that a dataset sums to
  a target over **one** dimension within each remaining group. It is the closest prior art
  for the share invariant in §2.4.

### 1.3 Three kinds of dimension

`MetricDimension.kind` (`DimensionKind`, `src/nodes/metric.py:239`) is one of:

| Kind | Where it comes from | Example |
|---|---|---|
| `common` | An instance dimension, declared in the model | `energy_carrier`, `vehicle_type` |
| `node` | Made by a node for its own output | The implicit *Sector* dimension of `AdditiveNode`, one category per input node |
| `scenario` | Made by the query when it asks for outputs in several scenarios | Baseline vs. active scenario |

A scenario dimension is therefore not a model dimension at all, and the chart can treat
it by its kind: its categories are alternative worlds and are never summed. That removes
the main case the dimension veto in version 1 was meant for.


## 2. The stackability model

### 2.1 Terms

- A metric is **stackable over dimension D** when summing its values over D's categories
  gives a meaningful value of the same metric. On a chart that means the categories of D
  can be drawn as stacked bars or areas, and a total line is meaningful.
- Stackability is a property of the **pair** (metric, dimension), not of either alone.
  The metric's quantity gives a default and the metric states the exceptions.

### 2.2 Layers

| Layer | Field | Status | Effect |
|---|---|---|---|
| Dimension kind | `DimensionKind` | Exists | `scenario` is never stackable. `common` and `node` dimensions follow the layers below. |
| Quantity kind | `aggregation: sum \| share \| none` | To build | Default for every metric of that quantity. `sum`: stackable over every dimension (emissions, energy, mileage, cost). `none`: stackable over none (emission factor, unit price, per capita). `share`: stackable over a declared set, within which the values sum to 1 (fraction, mix, grouped mix). |
| Metric (output port) | `stackable_dimensions: list[str] \| None` | To build | Overrides the quantity default in **either direction**. Required for a `share` metric with more than one dimension. |
| Dimension | `stackable: bool = true` | **Planned, not built** | An absolute veto that no metric could override. Kept in reserve until a case appears that the other layers cannot handle (see below). |

The metric override has to work in both directions because the quantity alone is wrong in
both directions:

- Emission factor by greenhouse gas: the CO₂e factors of the gases add up to the total
  factor, so an `aggregation: none` quantity is stackable over `ghg`.
- Per-capita emissions by sector: the denominator is shared, so they add up to the total
  per-capita value.
- Emissions by a vehicle-type dimension that has an "all vehicles" category alongside the
  parts is not stackable over it, although emissions are `sum`. That is a data problem to
  fix in the dataset; the override only lets a graph be right meanwhile.

**Why the dimension veto is deferred.** Version 1 proposed it for three cases. Scenario is
covered by its kind (§1.3). The uncertainty dimension is filtered to `median` in the
backend today and can stay so. The third case, overlapping stakeholder groups such as
"young people" and "women" as categories of one dimension, is a data problem: the
categories should be exhaustive and mutually exclusive, in explicit dimensions
(`age: young` × `gender: female`, …), and then they stack. A flag that makes such a
dimension safe to display would also make it easier to keep. So the veto is built only
when a case turns up that is genuinely non-additive *and* cannot be restructured; the
field name stays reserved, and nothing below should prevent adding it.

Resolution, done once in the backend and sent to the client per dimension:

```python
def is_stackable(metric, dim) -> bool:
    if dim.kind == DimensionKind.SCENARIO:
        return False
    # (planned) if not dim.stackable: return False
    if metric.stackable_dimensions is not None:  # metric says so explicitly
        return dim.id in metric.stackable_dimensions
    return quantity(metric).aggregation == 'sum'
    # 'share' without a declaration: see 2.5
```

A `node` dimension such as *Sector* follows the metric like any other: the sectors of an
`AdditiveNode` are its inputs, which the node itself sums, so they are stackable whenever
the metric is `sum`.

### 2.3 Examples

| Metric (quantity) | Dimensions | Stackable over | Not stackable over |
|---|---|---|---|
| emissions (sum) | vehicle_type, energy_carrier | both | — |
| emissions (sum) | sector (node kind) | sector | — |
| emission_factor (none) | vehicle_type | — | vehicle_type |
| emission_factor (none), declared `[ghg]` | ghg, vehicle_type | ghg | vehicle_type |
| fuel_fraction (share), case A: declared `[energy_carrier, production_year]` | energy_carrier, production_year, vehicle_type | energy_carrier, production_year | vehicle_type |
| fuel_fraction (share), case B: declared `[energy_carrier]` | energy_carrier, production_year, vehicle_type | energy_carrier | production_year, vehicle_type |
| any | scenario (scenario kind) | — | scenario |

### 2.4 Share metrics

A `share` metric with stackable set **S** and remaining dimensions **G** satisfies:

> for each year and each combination of categories of G, the values summed over all
> categories of S equal 1 (within a tolerance).

It follows that:

- Summing over any **subset** of S is valid and gives the marginal shares. In case A,
  summing over `production_year` gives each vehicle type's energy-carrier mix, still
  totalling 100 %.
- Filtering an S dimension to one category leaves parts that sum to that category's share.
  In case A, filtering `production_year = 2020` shows the energy carriers summing to the
  share of 2020 vehicles in the fleet of that vehicle type.
- A G dimension can never be summed: n vehicle types summed give n × 100 %. It has to be
  filtered to one category or reduced by a mean (§3.2).

Case A and case B differ only in where the 1 is, and that is exactly what
`stackable_dimensions` records.

### 2.5 Share metrics with no declaration

Most mixes have a single dimension, and for them S is obviously that dimension. So:

- One dimension: S defaults to it, no declaration needed.
- More than one: report a model problem ("share metric does not declare which dimensions
  sum to 100 %") and fall back to **S = ∅**: lines, no stacking. A wrong stack is worse
  than an unstacked chart, because it looks right.

The data can tell which case it is. A check (§5.3) finds the smallest S for which the
invariant holds and proposes the declaration, so the modeller confirms rather than works
it out.

### 2.6 Building on the quantity registry

Quantity kinds now live in `configs/quantities/quantity_kinds.yaml`, loaded by
`src/nodes/quantities.py` into `QuantityKind` objects with boolean classes
(`is_stackable`, `is_activity`, `is_factor`, `is_unit_price`, `is_scalar_identity`) and,
for factors, a `numerator`. The product algebra (`derive_product_quantity`) already uses
these classes. `aggregation` belongs in the same place, as one more field on
`QuantityKind`, and not in a separate table.

Things to settle when it is built:

- **Default from the existing classes.** `aggregation` can default to `sum` for
  `is_stackable` kinds and `none` for `is_factor` and `is_unit_price` kinds, so that only
  the share kinds and the exceptions need an explicit value. `mix`, `grouped_mix` and
  `fraction` become `share`; at present `mix` and `grouped_mix` are `is_stackable` and
  `fraction` is not, which is exactly the inconsistency `share` resolves.
- **Kinds that fit no class.** `ratio`, `probability`, `rate`, `per_capita`,
  `concentration`, `temperature`, `quality_of_data` and others carry none of the flags.
  They default to `none`; each should be reviewed once, because the default decides how
  every metric of that kind is drawn.
- **Two lists of stackable quantities.** `STACKABLE_QUANTITIES` in `nodes/constants.py`
  is a hand-written set beside the registry's `is_stackable`. On 2026-11-27 they disagreed
  only on `value_weight`, which the registry wrongly called stackable; that was corrected
  the same day, and the two now agree. The computation
  (`generic.py` split operations, `Metric.from_node`, `debug.py`) uses the constant; the
  chart should use the registry. Deriving the constant from the registry is a small,
  separate clean-up that this plan would benefit from but does not require.
- **The node's quantity vs the metric's quantity.** The resolution reads the metric's
  quantity, which for a multi-metric node can differ between metrics (an action with a
  `currency` metric and an `emissions` metric). Check that every output metric carries its
  own quantity, including DB-sourced output ports.

### 2.7 Weighted sums

Aggregating a non-stackable quantity over a dimension often means a weighted mean: the
values weighted by an extensive quantity over the same categories, then summed. It is
logically part of the aggregation question, so the concept stays in the plan, but the two
use cases known today are both handled inside the model, with no new feature:

- **BISKO data quality.** `weighted_data_quality_by_sector` and
  `weighted_data_quality_by_energy_carrier` (`configs/modules/bisko-base/model.yaml`) sum
  the energy use over one dimension and weight the quality grades carried as qualifiers
  with it (`qualifier(sum_dim(energy, energy_carrier), bisko_quality.score)`).
- **Average emission factor.** A new node takes energy use and emission factors split by
  the dimension to be aggregated, and computes the weighted sum: emissions summed, divided
  by energy summed, as a `FormulaNode`.

Both make the weighted result an ordinary node, visible in the graph, with its own
description, and computed once rather than by every chart. That is the preferred way while
it works. The chart-side `weighted_mean` (§3.4) is built only when a use case appears that
the model-side way cannot serve well, most likely one where the visitor's filters change
the set being averaged and a node per filter state is not practical.


## 3. What the chart does with each dimension

### 3.1 Roles

Each dimension of the metric is, in one view, either:

- **broken down by** (the "break down by" control),
- **filtered** to one or more categories, or
- **free** (neither).

**What the visitor controls.** Exactly what they control today: which dimension is broken
down by, and which categories are filtered. Stackability, the reduction policy and the
chart type that follows from them are decided by the model and the page editor, never by
the visitor; the chart has no control for them. The visitor's choices change what the
policies are applied *to* (a filter can leave one category, so nothing needs reducing),
not the policies themselves.

| Role | Stackable | Non-stackable |
|---|---|---|
| Broken down by | Stacked bars or areas; total line when there are negatives | One line per category; no total, no stacking |
| Filtered, one category | That category | That category |
| Filtered, several categories | Summed over the selection | Reduced over the selection by the dimension's policy |
| Free | Summed (today's behaviour) | Reduced by the dimension's policy |

The chart type follows from the dimension being broken down by, not from the metric:
`emission_factor` by `ghg` can be stacked bars, while the same metric by `vehicle_type`
must be lines.

### 3.2 Reduction policies for non-stackable dimensions

There is no single right answer, so the policy is a setting, to be tried on real nodes:

| Policy | Behaviour | Shown to the visitor as |
|---|---|---|
| `first` | Filtered automatically to the dimension's default category (first, or one the modeller names) | A filter chip that is visibly set, e.g. "Vehicle type: cars", so it is not a silent choice |
| `mean` | Equal-weight mean over the categories (or the selected ones) that have values | "Average over vehicle types (unweighted)" |
| `weighted_mean` | Mean weighted by another node's output (§3.4); deferred, see §2.7 | "Average over vehicle types, weighted by mileage" |
| `require_choice` | Empty graph with a prompt to choose a category | "Choose a vehicle type to see this graph" |

The policy can be set at three levels, the most specific winning:

1. **Quantity default**, as a starting point: `mean` for the G dimensions of `share`
   metrics, `first` for `none` metrics (an unweighted mean of emission factors is more
   misleading than a mean of mixes, which at least still sums to 100 %).
2. **Metric**: `reduction: {vehicle_type: first}` on the output metric.
3. **Block**: an override on the `ParameterPanelBlock` graph, set by the page editor, so
   the same node can be shown with different policies side by side while they are
   compared. It is not shown to the visitor as a control (§3.1).

### 3.3 Order of operations

Filter → **sum** the free stackable dimensions within each group → **reduce** the free
non-stackable dimensions → break down.

Summing before reducing matters when data are missing. In a share metric, a missing cell
usually means 0. Averaging first, over the vehicle types that happen to have a value in
that cell, would inflate it; summing first gives each vehicle type a complete composition
totalling 100 %, and the mean of complete compositions totals 100 % too. With complete
data both orders agree.

The mean is taken per year over the groups with at least one value, so a vehicle type
with no data at all does not pull the mean towards zero.

### 3.4 Weighted mean in the chart (deferred)

Built only if a use case calls for it (§2.7). The design, for when it does: a weight is an
extensive node output whose dimensions are a subset of G, e.g. vehicle mileage by vehicle
type. The metric declares it:

```yaml
output_metrics:
  fuel_fraction:
    quantity: fraction
    unit: '%'
    stackable_dimensions: [energy_carrier, production_year]
    reduction:
      vehicle_type: weighted_mean
    weights:
      vehicle_type: vehicle_mileage   # node id; its output must have vehicle_type and no S dims
```

The backend sends the weight series with the metric (values per G category and year,
same scenarios). The client normalises the weights over the selected categories, so the
visitor can still filter freely without another request. Computing the weighted view in
the backend instead would need one request per filter state.

### 3.5 Other parts of the chart

- **Total line and total row**: only when the dimension being broken down by is
  stackable. For a share with no filter on S, the total is 100 % and could be drawn as a
  reference line instead.
- **Table and download** follow the same reduction, and the header says which (e.g.
  "unweighted mean over vehicle types").
- **Goals** are already matched to a category choice; a goal on a non-stackable dimension
  continues to require that category to be filtered.
- **Scenario dimension**: never stackable, by its kind (§1.3). Its categories are drawn
  as separate lines or separate series, as today.
- **Uncertainty dimension**: stays filtered to `median` in the backend, as today. If the
  chart later shows uncertainty bands, that is a display of its own, not stacking.


## 4. Multi-metric nodes

The typical multi-metric node is an action with two metrics (for example its effect on
emissions and its cost), but there are others, such as a node with an activity and a
factor, or a node whose metrics are the same quantity under different accounting.

Two layouts are built and compared on real nodes before either becomes the default:

| Layout | Strengths | Weaknesses |
|---|---|---|
| **Tabs** | Compact; one graph at a time, full width; scales to many metrics | The second metric is hidden, so a visitor may never see the cost of an action |
| **Side by side** | Both metrics visible at once, which suits an action's effect and cost; the comparison is the point | Half the width each; on a phone they stack vertically; awkward beyond two or three metrics |

No decision now. The block chooses the layout (§5.4), so both can be shown on the same
page and judged.

### 4.1 Backend

- `NodeType.outputMetrics: [NodeOutputMetricType!]!` with `id`, `label`, `quantity`,
  `unit`, `isDefault`. Cheap: no computation. The ids should be the output port
  identifiers, since ports are where the Trailhead spec is going; for YAML nodes they
  default to the metric's column id. (To verify during implementation: that every runtime
  node, YAML- and DB-sourced, has a port per metric.)
- `metricDim(metric: ID)`: optional. Absent → today's behaviour, so old clients are
  unaffected. Unknown id → GraphQL error rather than `null`.
- Always pass the port to `from_node_output_metric`, so the returned metric's id is unique
  per metric (§1.1), and so the label is the metric's own.
- Resolve the FIXME at `metric_gen.py:479`: return only the goals for the chosen metric.
- A multi-metric node with no default and no argument should return `null` without
  sending an exception to Sentry; it is an expected state, not an error.

The side-by-side layout fetches every metric at once. The metrics of one node come from
one computation of that node, so the extra cost is in serialisation, not in computing;
the query can ask for them with aliases on `metricDim(metric: …)`.

### 4.2 Frontend

Common to both layouts:

- The slice configuration (break-down dimension, filters) is kept per metric: metrics can
  have different dimensions, so one configuration cannot be shared.
- Each metric has its own unit, stackability per dimension and goals.
- Label: the port label, falling back to the quantity label.
- A node with one metric looks exactly as today in either layout.

Tabs:

- Tabs above the graph. The first tab is the `isDefault` metric, or the first one.
- Each tab fetches its metric when first opened, not all metrics up front.
- Switching back to a tab restores its slice configuration.

Side by side:

- One graph per metric, in a row that wraps to a column at narrow widths.
- Each graph has its own controls. A shared control for a dimension present in all the
  metrics (e.g. filtering both graphs to one sector) is worth trying, but is not needed
  for the first version.
- Same y-axis scales are **not** shared: the metrics have different units.


## 5. Changes by repository

### 5.1 kausal-paths (backend)

| Where | Change |
|---|---|
| `configs/quantities/quantity_kinds.yaml`, `nodes/quantities.py` | `aggregation: sum \| share \| none` on `QuantityKind`, defaulting from the existing classes (§2.6). `fraction`, `mix`, `grouped_mix` → `share`. `is_stackable` and `STACKABLE_QUANTITIES` stay as they are: the computation (split operations, action checks) uses them, and changing them is out of scope. |
| `NodeMetric`, `OutputPortDef` | `stackable_dimensions: list[str] \| None`, `reduction: dict[str, ReductionPolicy] \| None`. YAML: on `output_metrics` entries and, for single-metric nodes, at node level. `weights` only if §3.4 is built. |
| `metric.py` `MetricDimension` and GraphQL type | `stackable: bool` (resolved per §2.2, using `kind`), `reduction: ReductionPolicy`, `defaultCategory`. |
| `DimensionalMetric` and GraphQL type | `aggregation`. Keep `stackable` with its current meaning for the old components. |
| `nodes/graphql/types/node.py` | `outputMetrics`; `metric` argument on `metricDim`. |
| `pages/blocks.py` `ParameterPanelBlock` | `metric` and `layout` on the `graphs` entries (§5.4). |
| `nodes/dimensions.py` `Dimension` | **Not now.** `stackable: bool = True` only if the veto is ever needed (§2.2). |

### 5.2 kausal_common (shared with Watch)

**Status (2026-11-15):** the copies exist, unchanged apart from names and imports, in
paths-ui `src/components/general/node-metric-chart/` (`NodeMetricChart.tsx`,
`NodeMetricChartGraph.tsx`) rather than in the `kausal_common` submodule, so that the
experiment needs no submodule commits. They are drawn by `ParameterPanelBlock`'s new
`graphs` field (GraphQL: `nodeGraphs`). Each graph has a node and, optionally, a
break-down dimension and category filters. These are defaults: the visitor can still
change them, and they are a first version of `group_by` and `filters` in §5.4. They move
to `kausal_common` when promoted.

New files only, so neither the node page nor Watch changes:

- `components/paths/NodeMetricChart.tsx`: copy of `DimensionalNodeVisualisation.tsx`,
  with both multi-metric layouts and the role logic.
- `components/paths/NodeMetricChartGraph.tsx`: copy of `NodeGraph.tsx`, with the chart type
  taken from the dimension being broken down by.
- `utils/paths/metric/aggregation.ts`: the filter → sum → reduce → break down pipeline
  (§3.3). `sliceBy` and `flatten` stay untouched.
- `ParsedMetric` gains optional fields (`dim.stackable`, `dim.reduction`); existing code
  ignores them.

### 5.3 Validation

- Spec time: `stackable_dimensions` and `reduction` keys must be output dimensions; a
  multi-dimension `share` metric without `stackable_dimensions` is a problem (§2.5).
- Compute time: the share invariant (§2.4), reported as a model problem so that
  `test_instance` records it. Generalise `DimensionSumRule` to a list of dimensions and
  reuse its check rather than writing a second one.
- A read-only command (e.g. `check_share_metrics <instance>`) that, for each share metric,
  finds the smallest S satisfying the invariant and prints the `stackable_dimensions` line
  to add. That turns 796 existing `fraction`/`mix` declarations in `configs/` into a
  review, not a manual audit.
- Not a validation, but a data-quality report worth having: dimensions whose categories
  look overlapping (an "all"/"total" category beside parts, or a sum over the categories
  that exceeds a sibling total). These are the cases §2.2 says to fix in the data.

### 5.4 ParameterPanelBlock graphs

```
graphs: list of
  node                 NodeChooserBlock
  metric               optional; one metric only. Absent on a multi-metric node: all metrics
  layout               tabs | side_by_side; used when several metrics are shown
  group_by             optional dimension id
  reduction_override   optional {dimension: policy}, for trying policies out
```

`node`, `metric` and `group_by` are named as in `NodeChartSpec`
([`embeddable-graph-blocks.md`](embeddable-graph-blocks.md) §5.3), so the block can later
move onto that spec without a data migration; `layout` and `reduction_override` should be
added to `NodeChartSpec` too. After the panel changes a parameter, the graphs refetch
their metrics.


## 6. Phases

1. **Multi-metric nodes** (independent of stackability). Backend `outputMetrics` and the
   `metric` argument, unique metric ids, goals per metric. Frontend copies with both
   layouts. `metric` and `layout` on `ParameterPanelBlock` graphs. Tests: GraphQL test for
   a multi-metric node (each metric, unknown id, no argument); component tests for tab
   switching keeping per-tab slice state, and for the side-by-side layout keeping
   independent slice state per graph.
2. **Stackability in the backend.** Quantity `aggregation` on the registry, metric
   `stackable_dimensions` and `reduction`, resolved flags in the payload (using dimension
   kind), spec-time validation, the share check and `check_share_metrics`. Tests: the
   table in §2.3 as unit tests of the resolution; the invariant check on case A and case B
   data, including a case that fails it.
3. **Aggregation in the copy.** Roles, sum-then-reduce, policies `first`, `mean` and
   `require_choice`, the block-level override, labels saying what was reduced. Tests in
   `aggregation.ts` with case A and case B fixtures: each policy, filters on S and on G,
   missing cells (§3.3), and that the stacked total is 100 % where it should be.
4. **Trial and promotion.** Put the copy on real nodes through `ParameterPanelBlock`,
   compare the two multi-metric layouts and the reduction policies, choose the defaults
   per quantity class. Then switch the node page to the new component, and later Watch,
   and delete the old components once nothing uses them.

Deferred, each built only when a case needs it:

- **Dimension veto** (`Dimension.stackable`, §2.2).
- **Chart-side weighted mean** (§3.4), when the model-side way (§2.7) does not serve.

Each phase leaves production unchanged until phase 4.


## 7. Open questions

- **Multi-metric layout.** Tabs, side by side, or chosen per block (§4). Settled in
  phase 4 by looking at both on real action nodes.
- **Default policies.** §3.2 proposes `mean` for share metrics and `first` for `none`
  metrics; phase 4 is meant to settle this on real nodes.
- **`grouped_mix`.** It is in the quantity registry and nothing else in `src/` or
  `configs/` uses it. Its name suggests exactly case B, a mix normalised within groups. If
  so, it becomes redundant once `stackable_dimensions` exists.
- **When the dimension veto is needed.** What would a genuinely non-additive dimension
  look like that cannot be restructured into exhaustive categories? None is known.
- **Weighted aggregation use cases.** Beyond the two in §2.7, are there cases where a
  weighted mean must follow the visitor's filters?
- **Computation.** The same per-metric stackability could later replace the
  quantity-only checks in `generic.py` split operations. Out of scope here, but the field
  names should not prevent it.

Resolved in version 2:

- Overlapping categories inside a dimension (an "all" category beside its parts, or
  overlapping groups) are a data error to fix in the dataset, not something to model
  (§2.2).
- Visitor control: the visitor cannot change stackability or the reduction policy. They
  choose the break-down dimension and the filtered categories, as now (§3.1).
