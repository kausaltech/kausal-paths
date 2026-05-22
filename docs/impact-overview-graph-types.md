# Impact Overview Graph Types

`ImpactOverview` objects describe how to visualise the collective impact of
climate actions on a particular effect node (e.g. net emissions, total costs).
Each overview has a `graphType` that determines both which backend computation
to run and how the UI should render the result.

This document focuses on the three graph types that produce **time-series**
output: `simple_effect`, `wedge_diagram`, and `stacked_raw_impact`. The
remaining types (`cost_benefit`, `cost_efficiency`, `return_on_investment`,
`benefit_cost_ratio`, `value_of_information`) produce per-action summary
scalars and are not covered here.

---

## Common concepts

### Unit handling

Time-series graph types keep values as **rates** (e.g. `kt CO2e/a`, `M€/a`).
The `indicatorUnit` field tells you what unit the values are already expressed
in — no client-side conversion needed.

Summary graph types (not covered here) integrate over time and express values
as totals (e.g. `kt CO2e`, `M€`). Do not mix the two.

### Action impact

An action's *impact* on an effect node is always the **difference** between
the scenario where that action is active and the scenario where it is not.
This means:

- Business-as-usual costs cancel out — only the *incremental* cost or saving
  attributable to the action appears.
- A positive impact value means the action *increases* the quantity (e.g.
  adds cost). A negative value means it *reduces* it (e.g. saves emissions or
  money).

### `DimensionalMetric`

Most numeric payloads are returned as a `DimensionalMetric`. Relevant fields:

| Field | Type | Meaning |
|-------|------|---------|
| `years` | `[Int!]!` | Calendar years covered |
| `values` | `[Float!]!` | One value per year, in the same order as `years` |
| `unit` | `UnitType` | Unit of the values |
| `forecastFrom` | `Int` | First year that is a forecast (not historical); `null` if all years are historical |
| `stackable` | `Boolean!` | Whether the series can meaningfully be stacked with others |
| `dimensions` | `[MetricDimension!]!` | Non-year dimensions (empty for scalar time series) |

---

## `simple_effect`

### Purpose

Show the per-action impact on the effect node as a single scalar per action
(summed or averaged across some time window). Useful for ranking actions by
total impact.

### Data source: `actions`

```graphql
{
  impactOverviews {
    id
    graphType
    indicatorUnit { short }
    actions {
      action { id name }
      effectDim {
        years
        values
        unit { short }
        stackable
        forecastFrom
      }
    }
  }
}
```

Each entry in `actions` represents one enabled action. `effectDim.values`
contains the action's impact per year. The unit matches `indicatorUnit`.

### Rendering suggestion

Bar chart, one bar per action, showing the time-integrated total or the value
at a target year. The `wedge` field is `null` for this graph type.

---

## `stacked_raw_impact`

### Purpose

Show action impacts as a **stacked time series** — one band per action, stacked
from zero, covering the full planning horizon. Useful for budget planning:
"do we have enough money in each individual year?"

Because impacts are differences (see above), the stack starts at zero — there
is no baseline floor to show. Implementation costs appear as positive values;
savings (e.g. reduced energy spend) appear as negative values, shrinking the
stack.

### Data source: `actions`

Same query as `simple_effect`:

```graphql
{
  impactOverviews {
    id
    graphType
    indicatorUnit { short }
    actions {
      action { id name }
      effectDim {
        years
        values
        unit { short }
        stackable
        forecastFrom
      }
    }
  }
}
```

`effectDim.stackable` is `true` for all action series. `values` are already
in `indicatorUnit` (a rate, e.g. `M€/a`).

### Rendering suggestion

Stacked area or bar chart over time. Stack all action bands from zero. Bands
with negative values should extend downward. A node-goal line (fetched
separately from the node's goal field) can be overlaid as the budget ceiling.

The `wedge` field is `null` for this graph type.

---

## `wedge_diagram`

### Purpose

Show how the gap between the **current scenario** and the **baseline** (all
actions off) is filled by individual actions — the classic "wedge" view.
Each action's band is scaled so the bands together span exactly from the
current scenario curve to the baseline curve.

This is useful for communicating "which actions get us from here to there and
by how much" without implying that one action is independent of the others.

### Math

Let:
- `x₀(t)` = current scenario value at year `t`
- `x(t)` = baseline (all actions off) value at year `t`
- `raw_i(t)` = raw impact of action `i` at year `t` (computed against the
  all-off baseline)

The multiplier `m(t) = (x(t) − x₀(t)) / Σ raw_i(t)` is applied uniformly
to all actions so that:

```
x₀(t) + Σ (raw_i(t) × m(t)) = x(t)
```

The multiplier is typically negative (when actions reduce the quantity), which
makes each action's adjusted impact positive when stacked on top of `x₀`.

### Data source: `wedge`

```graphql
{
  impactOverviews {
    id
    graphType
    indicatorUnit { short }
    wedge {
      id
      label
      isScenario
      metric {
        years
        values
        unit { short }
        stackable
        forecastFrom
      }
    }
  }
}
```

### `wedge` entries

The list is ordered as follows:

| Position | `id` | `isScenario` | `stackable` | Meaning |
|----------|------|--------------|-------------|---------|
| First | `current_scenario` | `true` | `false` | Current scenario — the floor of the wedge |
| Second | `baseline_scenario` | `true` | `false` | Baseline (all actions off) — the ceiling of the wedge |
| Remaining | action id | `false` | `true` | Scaled action impacts |

`metric.values` for scenario entries are **absolute** values of the effect
node output. `metric.values` for action entries are **scaled incremental**
values (positive, to be stacked on top of `current_scenario`).

### Rendering suggestion

1. Draw the `current_scenario` line as the bottom boundary.
2. Draw the `baseline_scenario` line as the top boundary.
3. Stack action bands (all `isScenario == false`) upward from the
   `current_scenario` line. The top of the stack should reach
   `baseline_scenario` in every year (by construction).
4. Use `forecastFrom` to visually distinguish historical from forecast years
   (e.g. solid vs dashed border, or reduced opacity).

The `actions` field returns an empty list for `wedge_diagram` overviews.

---

## Fetching all overviews in one query

All fields can be queried together. Use `graphType` to select the rendering
path on the client:

```graphql
{
  impactOverviews {
    id
    graphType
    indicatorUnit { short }

    # Used by: simple_effect, stacked_raw_impact
    actions {
      action { id name }
      effectDim {
        years
        values
        unit { short }
        stackable
        forecastFrom
      }
    }

    # Used by: wedge_diagram
    wedge {
      id
      label
      isScenario
      metric {
        years
        values
        unit { short }
        stackable
        forecastFrom
      }
    }
  }
}
```

For graph types that do not use a field, `actions` returns `[]` and `wedge`
returns `null`.

---

## Instance selection

The GraphQL endpoint is instance-scoped. Pass the instance identifier as a
request header:

```
x-paths-instance-identifier: <instance-id>
```

In the GraphQL Playground, add this under the **Headers** tab (bottom left).
