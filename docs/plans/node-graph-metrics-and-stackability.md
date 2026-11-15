# Node graph: metric tabs and stackability

*Produced by Claude Opus 5.5 on 2026-11-15.*
*Responsible: Jouni Tuomisto.*

A plan for a new version of the node graph that (1) lets the visitor choose between a
node's output metrics, and (2) decides, dimension by dimension, whether the values may be
summed. It is built as a **copy** of the current components and is first used from
`ParameterPanelBlock`. The node detail page, `OutcomeNodeContent`, the model editor and
Watch keep the current components until the copy has been tried on real models.

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

- Backend: `stackable = m.quantity in STACKABLE_QUANTITIES` (`metric_gen.py:418`), plus a
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
- Two dimension kinds already avoid the problem by being filtered away before slicing:
  the scenario pseudo-dimension (`DimensionKind.SCENARIO`, filtered per scenario in
  `parse.ts`) and the uncertainty dimension (filtered to `median` in
  `from_node_output_metric`). They are the two cases the general rule below has to cover
  without special code.
- `DimensionSumRule` (`src/datasets/validation_rules.py:94`) checks that a dataset sums to
  a target over **one** dimension within each remaining group. It is the closest prior art
  for the share invariant in §2.4.


## 2. The stackability model

### 2.1 Terms

- A metric is **stackable over dimension D** when summing its values over D's categories
  gives a meaningful value of the same metric. On a chart that means the categories of D
  can be drawn as stacked bars or areas, and a total line is meaningful.
- Stackability is a property of the **pair** (metric, dimension), not of either alone.
  The metric's quantity gives a default, the dimension can rule it out, and the metric can
  state the exceptions.

### 2.2 Three layers

| Layer | Field | Effect |
|---|---|---|
| Dimension | `stackable: bool = true` | `false` is an absolute veto. Used where categories are alternative worlds or overlap: scenario, uncertainty iteration, overlapping stakeholder groups (young people, women). No metric can make such a dimension stackable. |
| Quantity kind | `aggregation: sum \| share \| none` | Default for every metric of that quantity. `sum`: stackable over every stackable dimension (emissions, energy, mileage, cost). `none`: stackable over none (emission factor, unit price, per capita). `share`: stackable over a declared set, within which the values sum to 1 (fraction, mix, grouped mix). |
| Metric (output port) | `stackable_dimensions: list[str] \| None` | Overrides the quantity default in **either direction**, within the dimension veto. Required for a `share` metric with more than one dimension. |

The metric override has to work in both directions because the quantity alone is wrong in
both directions:

- Emission factor by greenhouse gas: the CO₂e factors of the gases add up to the total
  factor, so an `aggregation: none` quantity is stackable over `ghg`.
- Per-capita emissions by sector: the denominator is shared, so they add up to the total
  per-capita value.
- Emissions by a vehicle-type dimension that has an "all vehicles" category alongside the
  parts is not stackable over it, although emissions are `sum`. (That is a data problem
  rather than something to model, but the override lets a graph be right meanwhile.)

Resolution, done once in the backend and sent to the client per dimension:

```python
def is_stackable(metric, dim) -> bool:
    if not dim.stackable:                       # dimension veto
        return False
    if metric.stackable_dimensions is not None: # metric says so explicitly
        return dim.id in metric.stackable_dimensions
    return quantity(metric).aggregation == 'sum'
    # 'share' without a declaration: see 2.5
```

### 2.3 Examples

| Metric (quantity) | Dimensions | Stackable over | Not stackable over |
|---|---|---|---|
| emissions (sum) | vehicle_type, energy_carrier | both | — |
| emission_factor (none) | vehicle_type | — | vehicle_type |
| emission_factor (none), declared `[ghg]` | ghg, vehicle_type | ghg | vehicle_type |
| fuel_fraction (share), case A: declared `[energy_carrier, production_year]` | energy_carrier, production_year, vehicle_type | energy_carrier, production_year | vehicle_type |
| fuel_fraction (share), case B: declared `[energy_carrier]` | energy_carrier, production_year, vehicle_type | energy_carrier | production_year, vehicle_type |
| any | scenario | — | scenario (dimension veto) |

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

- One stackable dimension: S defaults to it, no declaration needed.
- More than one: report a model problem ("share metric does not declare which dimensions
  sum to 100 %") and fall back to **S = ∅**: lines, no stacking. A wrong stack is worse
  than an unstacked chart, because it looks right.

The data can tell which case it is. A check (§5.3) finds the smallest S for which the
invariant holds and proposes the declaration, so the modeller confirms rather than works
it out.


## 3. What the chart does with each dimension

### 3.1 Roles

Each dimension of the metric is, in one view, either:

- **broken down by** (the "break down by" control),
- **filtered** to one or more categories, or
- **free** (neither).

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

There is no single right answer (your points 6–8), so the policy is a setting, to be
tried on real nodes:

| Policy | Behaviour | Shown to the visitor as |
|---|---|---|
| `first` | Filtered automatically to the dimension's default category (first, or one the modeller names) | A filter chip that is visibly set, e.g. "Vehicle type: cars", so it is not a silent choice |
| `mean` | Equal-weight mean over the categories (or the selected ones) that have values | "Average over vehicle types (unweighted)" |
| `weighted_mean` | Mean weighted by another node's output (§3.4) | "Average over vehicle types, weighted by mileage" |
| `require_choice` | Empty graph with a prompt to choose a category | "Choose a vehicle type to see this graph" |

The policy can be set at three levels, the most specific winning:

1. **Quantity default**, as a starting point: `mean` for the G dimensions of `share`
   metrics, `first` for `none` metrics (an unweighted mean of emission factors is more
   misleading than a mean of mixes, which at least still sums to 100 %).
2. **Metric**: `reduction: {vehicle_type: weighted_mean}` on the output metric.
3. **Block**: an override on the `ParameterPanelBlock` graph, so the same node can be
   shown with different policies side by side while you compare them.

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

### 3.4 Weighted mean (later phase)

A weight is an extensive node output whose dimensions are a subset of G, e.g. vehicle
mileage by vehicle type. The metric declares it:

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
- **Scenario and uncertainty** dimensions become ordinary dimensions with
  `stackable: false` and policy `first` (active scenario; `median`). Their current special
  handling can stay in the old component; the copy should not need it.


## 4. Multi-metric tabs

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

### 4.2 Frontend

- Tabs above the graph only when the node has more than one output metric. The first tab
  is the `isDefault` metric, or the first one.
- Each tab fetches its metric when first opened, not all metrics up front.
- The slice configuration is kept per tab: metrics can have different dimensions, so one
  configuration cannot be shared. Switching back restores the previous one.
- Each tab has its own unit, stackability per dimension and goals.
- Tab label: the port label, falling back to the quantity label.


## 5. Changes by repository

### 5.1 kausal-paths (backend)

| Where | Change |
|---|---|
| `nodes/dimensions.py` `Dimension` | `stackable: bool = True`. In YAML and in `spec.dimensions` for DB-sourced instances. |
| `configs/quantities/quantity_kinds.yaml`, `nodes/quantities.py` | `aggregation: sum \| share \| none`, defaulting from `is_stackable`. `fraction`, `mix`, `grouped_mix` → `share`. `is_stackable` and `STACKABLE_QUANTITIES` stay as they are: the computation (split operations, action checks) uses them, and changing them is out of scope. |
| `NodeMetric`, `OutputPortDef` | `stackable_dimensions: list[str] \| None`, `reduction: dict[str, ReductionPolicy] \| None`, later `weights`. YAML: on `output_metrics` entries and, for single-metric nodes, at node level. |
| `metric.py` `MetricDimension` and GraphQL type | `stackable: bool` (resolved per §2.2), `reduction: ReductionPolicy`, `defaultCategory`. |
| `DimensionalMetric` and GraphQL type | `aggregation`, `weights` (later). Keep `stackable` with its current meaning for the old components. |
| `nodes/graphql/types/node.py` | `outputMetrics`; `metric` argument on `metricDim`. |
| `pages/blocks.py` `ParameterPanelBlock` | Optional `graphs` list (§5.4). |

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
  with tabs and the role logic.
- `components/paths/NodeMetricChartGraph.tsx`: copy of `NodeGraph.tsx`, with the chart type
  taken from the dimension being broken down by.
- `utils/paths/metric/aggregation.ts`: the filter → sum → reduce → break down pipeline
  (§3.3). `sliceBy` and `flatten` stay untouched.
- `ParsedMetric` gains optional fields (`dim.stackable`, `dim.reduction`, `weights`);
  existing code ignores them.

### 5.3 Validation

- Spec time: `stackable_dimensions` and `reduction` keys must be output dimensions;
  `stackable_dimensions` may not include a dimension with `stackable: false`; a multi-
  dimension `share` metric without `stackable_dimensions` is a problem (§2.5); `weights`
  must name a node whose dimensions are a subset of G.
- Compute time: the share invariant (§2.4), reported as a model problem so that
  `test_instance` records it. Generalise `DimensionSumRule` to a list of dimensions and
  reuse its check rather than writing a second one.
- A read-only command (e.g. `check_share_metrics <instance>`) that, for each share metric,
  finds the smallest S satisfying the invariant and prints the `stackable_dimensions` line
  to add. That turns 796 existing `fraction`/`mix` declarations in `configs/` into a
  review, not a manual audit.

### 5.4 ParameterPanelBlock graphs

```
graphs: list of
  node                 NodeChooserBlock
  metric               optional; required when the node has several metrics
  group_by             optional dimension id
  reduction_override   optional {dimension: policy}, for trying policies out
```

The first three fields are named as in `NodeChartSpec`
([`embeddable-graph-blocks.md`](embeddable-graph-blocks.md) §5.3), so the block can later
move onto that spec without a data migration; `reduction_override` should be added to
`NodeChartSpec` too. After the panel changes a parameter, the graphs refetch their metric.


## 6. Phases

1. **Metric tabs** (independent of stackability). Backend `outputMetrics` and the `metric`
   argument, unique metric ids, goals per metric. Frontend copies with tabs only. The
   `graphs` field on `ParameterPanelBlock`. Tests: GraphQL test for a multi-metric node
   (each metric, unknown id, no argument); a component test for tab switching keeping
   per-tab slice state.
2. **Stackability in the backend.** Dimension flag, quantity `aggregation`, metric
   `stackable_dimensions` and `reduction`, resolved flags in the payload, spec-time
   validation, the share check and `check_share_metrics`. Tests: the table in §2.3 as unit
   tests of the resolution; the invariant check on case A and case B data, including a
   case that fails it.
3. **Aggregation in the copy.** Roles, sum-then-reduce, policies `first`, `mean` and
   `require_choice`, the block-level override, labels saying what was reduced. Tests in
   `aggregation.ts` with case A and case B fixtures: each policy, filters on S and on G,
   missing cells (§3.3), and that the stacked total is 100 % where it should be.
4. **Weighted mean.** `weights` declaration, payload, client-side normalisation.
5. **Trial and promotion.** Put the copy on real nodes through `ParameterPanelBlock`,
   compare policies, choose the defaults per quantity class. Then switch the node page to
   the new component, and later Watch, and delete the old components once nothing uses
   them.

Each phase leaves production unchanged until phase 5.


## 7. Open questions

- **Default policies.** §3.2 proposes `mean` for share metrics and `first` for `none`
  metrics; phase 5 is meant to settle this on real nodes.
- **`grouped_mix`.** It is in the quantity registry and nothing else in `src/` or
  `configs/` uses it. Its name suggests exactly case B, a mix normalised within groups. If
  so, it becomes redundant once `stackable_dimensions` exists.
- **Overlapping categories inside a partition dimension** (an "all" category beside its
  parts): model it with the override, or treat it as a data error to fix in the dataset?
- **Visitor control.** Should the visitor be able to switch the reduction policy (e.g.
  "average" vs "choose one"), or is it the modeller's choice only?
- **Computation.** The same per-dimension stackability could later replace the
  quantity-only checks in `generic.py` split operations. Out of scope here, but the field
  names should not prevent it.
