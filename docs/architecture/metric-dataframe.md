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

Forecast status becoming a qualifier has one immediate consequence for
dataset input pipelines: the `set_forecast_from` operation *sets a
qualifier field on a metric*. It does not add a `Forecast` column.

For multi-metric frames, qualifiers should be per-metric rather than
implicitly row-level. For example:

- `Energy`
- `Energy__qual`
- `EmissionFactor`
- `EmissionFactor__qual`

This avoids ambiguity when one metric in a row is interpolated and another
is not.

The first implementation uses a Polars `Struct` column. Related fields
stay together without creating many top-level columns.

At this stage, the important design choice is not the exact storage type
but the explicit pairing:

- a metric column may have one qualifier column
- that qualifier column belongs to that metric

### First implementation in PathsDataFrame

Commit `735203b7` introduced qualifier columns before the wrapper migration.
The pairing is currently by name (`<metric>__qual`), with three flat fields:
`quality` (score), `graded` (assessment coverage), and `supplied` (source
presence). Rename and drop carry the pairing; projections preserve it only
when explicitly requested through `PathsDataFrame.qualified()`.

Bindings attach grades from the evidence-owned `quality_of` projection or a
declared dataset default. The two ifeu transport datasets declare BISKO B.
Generic sums weight grades by value magnitude; products retain the sole
qualifier or conservatively combine both; choices carry the selected value's
qualifier. Fill operations rejoin qualifiers by key and mark created cells.
`prefer_by_year` can use the source-presence flag rather than a separate
availability input. Caches can store the structs without a separate metadata
registry; pandas conversion deliberately omits them.

This first version reduces `supplied` with OR for sums and erases the grade
of all filled cells. Neither rule is the final contract below. Qualifiers
were not exposed through dimensional GraphQL metrics in this commit, and
the BISKO quality and availability reporting nodes were retained.

### Runtime qualifier catalog

`Context.qualifiers` resolves the catalog once per runtime context. It contains
always-on built-ins (`reported` initially), plus the quality schemes of the
instance's framework, including framework templates. Instances without a
framework have only the built-ins. No customized qualifier selection is
persisted in `InstanceModelSpec` at this stage.

Scheme fields are named `<framework.identifier>_<scheme.identifier>`: the
BISKO framework's scheme is named `quality`, so its field is `bisko_quality`.
A data migration renames the previous `bisko` scheme without changing scheme
or level UUIDs. Historical payload readers accept its previous name explicitly.

The catalog groups versions of a scheme into one named field, retaining each
version's exact grade UUIDs and scores. The most recently created scheme version
supplies dataset defaults; version strings are labels, not assumed to sort
numerically or lexically. Frameworks and scheme families never share a global
identifier-only grade lookup. Catalog names, propagation mechanisms, versions
and grade scores participate in cache identity. Reconstruct the runtime context
when the framework vocabulary changes.

Fixed propagation mechanisms and typed payloads live in `common.qualifiers`;
`frameworks.qualifiers` projects the ORM schemes into the catalog. A context
constructs one Polars struct dtype from those definitions. Generic dataframe
operations resolve the fixed mechanisms from the typed payloads and preserve
all named fields, without accessing the ORM or requiring parallel node chains.

Database and published-payload readers attach assessments from evidence by
cell key and grade identity. The legacy numeric `quality_of` projection remains
for existing calculations; it must not assign a score to an unrelated scheme.
Evidence currently stores one quality level per data point. The catalog can
carry several schemes, but this change does not add multiple simultaneous
evidence assessments for the same source cell.

### Covered assessments

A quantitative assessment and its coverage are one entity. The covered-assessment shape is:

```text
Energy__qual:
  bisko_quality:
    score: 0.75
    coverage: 0.60
  reported: true
```

`bisko_quality.score` is the mean score within the assessed portion;
`bisko_quality.coverage` is the fraction of the declared weighting basis assessed.
An assessed source cell has coverage 1, including a BISKO D cell whose score
is zero. An explicitly unassessed cell has coverage 0 and no score. A missing
assessment record means the assessment is unknown. Zero coverage implies no
score, and an undefined weighting denominator yields an undefined assessment.

Another quantitative qualifier can use the same covered-score shape with
its own coverage. The coverages need not describe the same contributions.
Shared arithmetic does not impose a universal propagation rule: each
assessment must state its weighting basis and combination semantics.

For positive additive energy flows, let `w_i = abs(value_i)`, `g_i` be coverage
and `q_i` score. Reduce the pair atomically:

```text
coverage = sum(w_i * g_i) / sum(w_i)
score = sum(w_i * g_i * q_i) / sum(w_i * g_i)
```

An unassessed contribution participates in the total weight. Thus 60 MWh
graded A plus 40 MWh unassessed has score 1 and coverage 0.6. `score * coverage`
is a possible whole-balance reporting convention, not another assigned grade.
All-zero groups have no energy-weighted assessment; do not substitute a
cell-count average. Signed corrections can cancel values and make magnitude
weighting depend on grouping; their grading policy must be explicit.

Exact constants and unit conversions preserve assessments. A product whose
assessment follows one designated input carries that input's pair. When both
inputs require assessment, a combination policy is needed: the minimum of
two coverage fractions is not generally their joint assessed coverage. The
existing conservative product rule is provisional, not proof of coverage.

### Reported data

Use `reported`, a nullable boolean, to exercise boolean qualifier propagation:

> True when all contributing source cells contain reported values; false when
> any required contribution was filled, interpolated or extended; null when
> this is unknown.

Reported zero is true. Calculations from reported inputs remain true; exact
constants and unit conversions are neutral. Sums and data-dependent products
use three-valued AND: false dominates, otherwise unknown remains unknown.
An absent side of an outer join contributes nothing and is neutral, whereas
a present value with unknown reporting status contributes unknown. Choices
carry the selected value's flag.

This does not identify who reported the data: provider defaults can be
reported too. Nor does it certify that all required cells exist. Mandatory
category/year grids must still be checked before reduction.

Transport source selection asks whether *any* original cell was reported in a
year. Evaluate that question before aggregation, or pass explicit coverage
from that boundary. An AND-reduced reporting flag cannot recover it. Preserve
the existing per-year source choices during migration.

### Derivation and assessment are independent

BISKO grades describe data origin, including estimates derived from regional
primary data. The Methodenpapier (July 2024, section 3.4) does not prescribe
blanket grade erasure for interpolation. The Klimaschutz-Planer handbook's
chimney-sweep section explicitly recommends interpolation between observations
collected every two or three years.

An approved derivation can produce an assessed value with `reported=false`.
Its method must explicitly preserve, replace or invalidate the assessment;
neither retaining A nor erasing every grade is a universal rule. Linear
interpolation now interpolates coverage and `score * coverage` using
actual year distances, then divides to recover the assessed score. It preserves
matching endpoint assessments and leaves entirely unassessed endpoints
unassessed. Backfilling and constant extension copy the endpoint assessment.
All created cells have `reported=false`. Structural zero-fill and unsupported
extrapolation do not invent assessments. Each category is handled independently.

### Current implementation after the qualifier refinement

The refinement implements named covered assessments such as
`bisko_quality: {score, coverage}` and nullable
`reported`, three-valued AND for reporting status, and undefined energy-weighted
assessments for all-zero groups. `quality(x)` and `graded(x)` read the sole
assessment's nested
score and coverage; when several schemes are available, give the field name
explicitly, e.g. `quality(x, 'bisko_quality')`. `reported(x)` replaces
`supplied(x)`. Existing formula names
for quality and coverage remain stable. The legacy `make(supplied=...)`
construction keyword is accepted temporarily; the stored shape is always new.

A single indexing helper serializes dimensional values and qualifiers together;
GraphQL exposes one object per qualifier: a `BooleanQualifierType` with a
`values` array, or a `CoveredScoreQualifierType` with `scores` and `coverage`
arrays. Each array aligns exactly with the metric's flattened `values` index;
construction rejects mismatched lengths and duplicate qualifier identifiers.
Each object carries its catalog `identifier` and an `id` derived as
`<DimensionalMetric.id>:<identifier>`, inheriting the metric's identity contract.
No framework-specific fields are hard-coded in the schema. Missing cube cells
produce null elements. Round trips represent these as unknown struct fields;
the distinction between an absent struct and an entirely unknown struct is
not exposed. Coverage zero remains distinct from unknown coverage. Data Studio
reads computed assessments from final energy, shows coverage beside grades,
and uses it in weighted scores and improvement ranking. Source provenance still
comes from entry evidence, and grade-distribution buckets for blended scores
remain approximate. Cache format/semantics versions invalidate old structs.

The wrapper migration, further product grading policies, and
removal of BISKO reporting nodes are subsequent steps, not implemented by this
refinement. Products keep the provisional rule described above. Required grids
and source-route reports remain
in the BISKO graph while those contracts are migrated.

### GraphQL, Data Studio and BISKO migration

Expose typed qualifiers alongside `DimensionalMetric.values`, aligned to the
same dimension/year index. Missing combinations rendered as zero retain
unknown qualifiers. Preserve that alignment in every metric serialization
path, including inputs and visualizations.

Data Studio should read computed quality from the final-energy values and
display both score and coverage, while data entry continues to edit evidence.
Do not round a blended score into an assigned class or count the unassessed
portion as fully graded. Exact A/B/C/D distributions require more information
than a mean and coverage; provenance/source-route reporting also needs more
than `reported`.

Replace duplicate BISKO quality chains with projections of the energy
qualifiers after comparing the grading of entered inventory versus corrected
results. Migrate reporting consumers before deleting their node IDs. Replace
availability plumbing with input-cell qualifiers only when required grids,
reported zeros, wholly absent categories, and per-year transport choices
remain covered. Energy-weighted assessment coverage cannot replace a check
that every mandatory input has a grade, especially for zero-valued cells.

Before rolling out the Data Studio consumer, refresh saved graph/catalog
metadata through the normal configuration publishing workflow. Older saved
instances can lack both `default_quality` and `quality_of` metadata even when
the current module YAML declares them. Local verification encountered a saved
configuration with no assessed final-energy scores while its current YAML
configuration did carry assessments. Absent metadata must not be repaired by
guessing grades in the UI.

`python -m tools.setup_bisko --dry-run` previews the BISKO repair; omit
`--dry-run` to apply it. Setup reads declared defaults from the current BISKO
YAML, merges them into matching template dataset metadata, and publishes a
corrected template revision when its released catalog differs or a member's
pin is stale. The existing publisher advances dependent drafts atomically and
checks for new constraint conflicts. Repeated setup retains the revision when
those declarations and pins already match. Explicit `--publish` still requests
a new full release. Unpinned instances remain standalone, and already published
municipal balance snapshots remain historical snapshots.


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


## Ports and MDF

A node input or output port carries **exactly one metric**. So the value
at a port is an MDF with exactly one entry in `columns.metrics`, that
metric's qualifier column if it has one, and its dimension columns.

This matters for the qualifier design: the ambiguity that motivates
per-metric qualifiers ("one metric in this row is interpolated, another is
not") cannot arise at a port boundary. It arises only inside node bodies,
where a frame may legitimately carry several metrics at once. Per-metric
qualifiers are therefore what makes node-internal frames well-defined,
while port boundaries stay trivially unambiguous.

See [`dimension-constraints.md`](dimension-constraints.md) for the port
and binding model this refers to.


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

The same split governs dimension requirements: node dimension signatures
and port requirements range over **structural** dimensions only. Temporal
axes are always present, and ensemble and decomposition axes are
transparent by construction. See
[`dimension-constraints.md`](dimension-constraints.md).

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
