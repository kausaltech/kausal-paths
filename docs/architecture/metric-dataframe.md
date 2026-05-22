# MetricDataFrame

This document sketches a replacement for the current `PathsDataFrame`
subclassing approach.

The goal is not to design a complete future data model up front. The goal
is to define the smallest wrapper that lets us:

- stop relying on Polars `DataFrame` inheritance
- describe the semantic role of columns explicitly
- support per-metric qualifier data
- support additional dimensions such as Monte Carlo iteration or action ID
- move toward time resolutions beyond just yearly data


## Why Replace the Current Approach

The current system stores semantic information partly in the Polars
subclass itself:

- primary-key columns
- dimension columns
- metric units
- a few reserved columns such as `Forecast`

This has worked, but inheritance from `pl.DataFrame` is discouraged and may
be brittle across Polars upgrades. We should move the semantic layer out of
the Polars object and into a wrapper that owns:

- the raw `pl.DataFrame`
- a compact description of what kinds of columns the frame contains


## Core Idea

Use a thin wrapper:

```python
class MetricDataFrame:
    df: pl.DataFrame
    columns: DataFrameColumns
```

`df` is the physical table.

`columns` is the semantic description of the columns in that table.

The name `columns` is intentional: Polars already uses `schema` for
column-name-to-dtype mapping, so we should avoid introducing another
schema-like term for a different concept.


## `DataFrameColumns`

`DataFrameColumns` should be the single semantic container carried by
`MetricDataFrame`.

Its job is to answer:

- which columns are part of the key
- which columns are dimensions
- which columns are metrics
- which metric columns have qualifier columns
- which dimensions are temporal or otherwise special

A minimal shape:

```python
class DataFrameColumns:
    keys: tuple[str, ...]
    dimensions: dict[str, DimensionColumn]
    metrics: dict[str, MetricColumn]
```

This should stay small. If more views are needed, they should usually be
derived from these three fields rather than stored separately.


## Dimension Columns

Dimension columns are columns that form part of datapoint identity.

Examples:

- `Year`
- `Sector`
- `iteration`
- `action_id`

A minimal dimension description:

```python
class DimensionColumn:
    column: str
    kind: DimensionKind
    time_resolution: TimeResolution | None = None
```

Suggested dimension kinds:

- `structural`
- `temporal`
- `ensemble`
- `decomposition`

Meaning:

- `structural`: ordinary dimensions such as sector, fuel, municipality
- `temporal`: time axes such as year, month, quarter
- `ensemble`: Monte Carlo iteration or other sampled execution axes
- `decomposition`: dimensions such as `action_id` that split a value into
  contributions


## Time As a Dimension

`Year` should be treated as a temporal dimension, not as a completely
special one-off concept.

That means:

- `Year` can still be the current temporal column in practice
- the wrapper can expose convenience helpers for the common yearly case
- the underlying model does not assume that yearly resolution is the only
  possible one

A temporal dimension may therefore carry `time_resolution`, for example:

- `year`
- `quarter`
- `month`

This is enough to let us move beyond yearly data later without redesigning
the whole wrapper.


## Metric Columns

Metric columns are value-bearing columns with units and, optionally,
qualifier columns.

A minimal description:

```python
class MetricColumn:
    column: str
    unit: Unit
    quantity_kind_id: str | None = None
    qualifier_column: str | None = None
```

Examples:

- `Energy`
- `EmissionFactor`
- `Cost`

If a metric has associated qualifier data, it should point to a separate
column that stores it.

`quantity_kind_id` is optional.

This is intentional: unit is an internal computational invariant, but
quantity kind is a semantic declaration that may be unknown or not worth
declaring for intermediate metrics created inside a node body. We care about
quantity kind especially at node and dataset boundaries, but internal working
columns should not be forced to carry it if the meaning is temporary or
context-dependent.


## Qualifier Columns

Qualifier columns describe properties of a metric value at a datapoint.

Examples:

- forecast status
- interpolation status
- data quality score
- provenance or derivation details

For multi-metric frames, qualifiers should be per-metric rather than
implicitly row-level. For example:

- `Energy`
- `Energy__qual`
- `EmissionFactor`
- `EmissionFactor__qual`

This avoids ambiguity when one metric in a row is interpolated and another
is not.

The concrete storage inside a qualifier column is left open for now. A
`Struct` dtype is a plausible option because it keeps related qualifier
fields together without creating many top-level columns.

At this stage, the important design choice is not the exact storage type
but the explicit pairing:

- a metric column may have one qualifier column
- that qualifier column belongs to that metric


## Dimensions vs Qualifiers

Dimensions and qualifiers solve different problems and should not be merged
into one generic “metadata” mechanism.

Dimensions:

- distinguish many datapoints along an axis
- become part of the key
- usually pass through normal computations

Qualifiers:

- describe one metric value at one datapoint
- are not part of the key by default
- need explicit merge and reduction rules

Examples:

- `iteration` is an ensemble dimension
- `action_id` is a decomposition dimension
- `Forecast` belongs in a qualifier
- interpolation status belongs in a qualifier


## Aggregation and Collapse

We should distinguish two kinds of behavior:

### 1. Dimension collapse policy

Dimensions usually pass through. The important question is what happens
when an operation wants to collapse one.

Examples:

- structural dimensions may be collapsed by ordinary summation
- ensemble dimensions such as Monte Carlo iteration should not be collapsed
  by ordinary summation; they need statistical reducers
- decomposition dimensions such as `action_id` should only be collapsed by
  explicit attribution/decomposition-aware operations

### 2. Qualifier merge and reduction rules

Qualifier fields need their own field-specific behavior.

Examples:

- a boolean `forecast` field might reduce with `any`
- a `forecast_share` field might reduce numerically
- a quality score may reduce with a conservative rule such as `min`

These rules belong to qualifier semantics, not to dimension semantics.


## Design Constraints

This wrapper should stay intentionally modest.

We should avoid:

- inventing a large generic metadata framework up front
- storing many parallel semantic registries that can drift out of sync
- treating every possible special case as a new top-level abstraction
- forcing a complete narrow-only or wide-only redesign now

We should prefer:

- one wrapper around `pl.DataFrame`
- one semantic container, `columns`
- explicit metric-to-qualifier pairing
- temporal dimensions instead of hard-coding yearly assumptions
- gradual migration from the current implementation


## Practical Migration Direction

The likely migration path is:

1. Introduce `MetricDataFrame(df, columns)`.
2. Move current semantic information from the subclass into `columns`.
3. Keep compatibility helpers for common current operations.
4. Introduce per-metric qualifier columns where needed.
5. Add dimension kinds for temporal, ensemble, and decomposition axes as
   real use cases appear.

This gives us a path away from Polars inheritance without committing to
more machinery than we currently need.


## For Future Consideration

The discussions around quantity semantics exposed some real needs that are
important, but should not be part of the initial `MetricDataFrame` core.

### Quantity kind vs aggregation behavior

`quantity_kind_id` should remain a semantic classification of what a metric
measures. It should not be expected to fully determine how that metric may be
collapsed over dimensions.

Examples:

- `emissions` are often directly additive
- `emission_factor` is usually not directly additive
- `fraction` or `mix` may be meaningful to sum over one dimension within a
  partition, but meaningless to sum over another dimension

This suggests that “stackable” is too coarse as a universal boolean. It may
still be useful as a default hint in the quantity registry, but not as the
full semantics of a metric inside a concrete dataframe.

### Weighted aggregation

Some non-additive metrics are still aggregatable when a weighting basis is
known.

Example:

- building-heating emission factors may be aggregated over heating types as a
  weighted mean, using energy shares as weights

This is different from direct additivity. It is better thought of as a
higher-level aggregation rule than as part of the minimal MDF contract.

### Decomposition and attribution

There are also cases where we want not only an aggregate value, but a way to
explain or visualize the contribution of components to it.

Examples:

- decomposing a weighted-mean emission factor into contributions
- tracking action contributions against a baseline

These concerns appear related to:

- decomposition dimensions such as `action_id`
- explicit aggregation or attribution logic
- visualization-oriented projections

They should be treated as future semantic layers around MDF, not as required
fields on every metric inside node-internal computations.

### Working rule for MDF additions

A field belongs in the MDF core only if generic dataframe operations can
preserve it mechanically without understanding domain intent.

This rule is why the current proposal includes:

- column roles
- units
- optional quantity-kind references
- optional qualifier-column references

and does not yet include:

- stackability policies
- weighted-aggregation definitions
- decomposition semantics
- attribution rules
