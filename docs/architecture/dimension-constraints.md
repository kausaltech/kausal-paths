# Dimension Constraints and Transformations

## Overview

Data in Paths flows downstream through edges. Dimensional shape
requirements flow upstream: an outcome node declares which dimensions
it expects, and those requirements propagate back through the graph,
modified by each node's signature and each edge's transformations.

This document captures the vocabulary and the design direction.


## The vocabulary

### Node dimension signature

Every node (or pipeline step) has a dimension signature with four facets:

| Facet | Meaning | Example |
|---|---|---|
| **requires** | Dimensions the node must receive | GWP node requires `ghg_species` |
| **consumes** | Subset of `requires` that the node removes from its output | GWP consumes `ghg_species` (flattens into CO2e) |
| **produces** | New dimensions the node adds to its output | A disaggregation node might produce `building_class` |
| **transparent** | Whether extra dimensions pass through untouched | Most nodes are transparent; outcome nodes are not |

`consumes` is always a subset of `requires`. `produces` is disjoint
from `requires`.

For datasets, only `produces` applies — a dataset declares the shape
of what it emits.

For outcome nodes, `requires` is set explicitly by the user (e.g.
"I want `sector` in the final output").


### Dimension transformations on edges

Edge transformations are adapters between an upstream `produces` and a
downstream `requires`. They reshape data as it flows across an edge.

The core operations:

| Operation | What it does |
|---|---|
| **FilterDimension** | Keep or exclude specific categories within a dimension. Optionally flatten (sum over) the dimension afterward. |
| **AssignDimension** | Tag every row with a fixed category in a new dimension. Adds a dimension that didn't exist upstream. |

These two operations cover every edge transformation pattern that
actually reshapes data, and they are the same operations used in dataset
input pipelines.

Dataset pipelines add data-prep operations on top (renaming, null
handling, temporal limits, qualifier setting) that mostly don't apply to
edges, because node outputs are already well-formed. "Mostly" is
deliberate: null handling and temporal limits are meaningful on both
sides. See [Unification with dataset transforms](#unification-with-dataset-transforms)
for how that is modelled.

Metric selection is **not** a transformation. A binding names the single
metric it carries (a `DatasetMetric` for dataset bindings, an output port
for edges); the op pipeline only reshapes what that selection produced.


### `FlattenTransformation` is not a flatten

The current `FlattenTransformation` type is a misnomer and must not be
folded into `FilterDimension(flatten=True)`.

It is only ever produced from a bare `to_dimensions` entry (one with no
`categories`), and `_get_output_for_target()` skips to-dimension entries
that have no categories. Its only runtime effect is that its dimension id
joins the set asserted against the edge output. In other words it is a
**shape declaration about the consuming port**, not an operation.

Real flattening on an edge is `SelectCategoriesTransformation(categories=[],
flatten=True)`, produced from a bare `from_dimensions` entry.

The target state is therefore:

- the shape declaration moves to the port (`InputPortDef`), where
  constraint propagation can compute or validate it
- `FlattenTransformation` disappears from the op vocabulary rather than
  merging into another op


### How constraint propagation works

1. Start at outcome nodes. Their output ports declare a concrete
   dimension set.

2. Walk upstream. At each node, the node's signature determines what
   each input port requires:
   - Additive: every input must match the output dims (after edge
     transforms)
   - Multiplicative: output dims = union of input dims, so each input
     covers its own subset
   - GWP-style: output dims = input dims minus `consumes`

3. Edge transformations modify the requirement as it crosses:
   - A `FilterDimension(flatten=True)` means the upstream node must
     *have* that dimension, even though the downstream port doesn't
     require it after flattening.
   - An `AssignDimension` means the upstream node does *not* need to
     have that dimension — it's added in transit.

4. The propagated requirement at an input port is a function of:
   downstream shape + node signature + edge transforms.


### Structural dimensions only

Signature facets (`requires` / `consumes` / `produces` / `transparent`)
and collapse policy apply to **structural** dimensions only.

- **Temporal** axes are always present and are not part of
  requires/produces bookkeeping. A node does not "require `Year`".
- **Ensemble** and **decomposition** axes (Monte Carlo iteration,
  `action_id`) are transparent by construction. They are never consumed
  implicitly, and collapsing them requires an explicit reducer, not
  summation.

See [`metric-dataframe.md`](metric-dataframe.md) for the dimension kinds
this rule refers to. A consequence: `FilterDimension(flatten=True)` must
refuse non-structural dimensions rather than silently summing them.


### Where the declarations live

| What | Where | Static or computed? |
|---|---|---|
| Node class dimension rules | Node class or pipeline definition | Static (per class/pipeline) |
| Outcome node required dims | `InputPortDef.required_dimensions` | Static (user-configured) |
| Port shape declarations (ex-`FlattenTransformation`) | `InputPortDef` | Static (user-configured) |
| Binding transformations | `PortBindingDef.transformations` on the **consuming** port | Static (user-configured) |
| Input port effective requirement | Computed from downstream | Computed at validation/editor time |
| Dataset produced dims | Dataset schema | Static |


### Authored vs computed declarations

Stored dimension fields on ports are **authored** data. Propagation
results are **computed** and must not be written back over them.

This is not yet true in the code: `_export_input_ports()` fills in
`required_dimensions` *and* `supported_dimensions` on every port from
observed runtime dimensions, which stores computed data in authored
fields — the drifting-registry failure that
[`metric-dataframe.md`](metric-dataframe.md) warns against. The same
applies to `DatasetPortSpec.output_dimensions`, a manual override of what
the dataset schema plus the op pipeline should derive.

Rules going forward:

- `InputPortDef.required_dimensions` is authored, and meaningful for
  outcome nodes and explicit shape declarations. Everywhere else the
  requirement is computed.
- Propagation results are exposed as a separate derived field (for
  example `effectiveRequiredDimensions` in GraphQL), never by overwriting
  the authored one.
- No mutation may accept computed dimension sets as input.
  `DatasetPortSpec.output_dimensions` stays read-only and is retired once
  schema + ops can derive it.


### Transformations attach to the consuming port

Propagation walks upstream *through input ports*, so it needs
transformations attached to the binding at the consuming port. The code
does not do this yet: edge transformations execute on the **producing**
node, in `_get_output_for_target()`, keyed by consumer identity. Dataset
filters execute inside dataset loading. Neither runs at the port.

Propagation also needs port identity that is authored and stable. Ports
are currently derived at export time and their ids are hashed from
`(node, direction, key)`, so they are regenerated on every sync and
nothing downstream can attach computed state to them.


## Unification with dataset transforms

The dimension-aware subset of dataset transforms and edge transforms are
the same operations. Today's encodings map like this:

```
Edge: SelectCategoriesTransformation  ≡  Dataset: FilterDimensionDatasetTransformOp
Edge: AssignCategoryTransformation    ≡  Dataset: FilterDimensionDatasetTransformOp(assign_category=...)
Edge: FlattenTransformation           ≡  (nothing — it is a port shape declaration)
```

The second line describes a **legacy encoding** that is being retired:
assignment is its own operation (`AssignDimension`), not a field on a
dimension filter.

### Decision: one op type, not two layers

There is **one** op union, `PortTransformOp`, covering dimension ops and
data-prep ops together. Applicability is a property of each op, validated
against the binding kind that carries it — not a second type hierarchy.

Rationale: two unions means two GraphQL input types and two executors,
which is the duplication this unification exists to remove. And the
dimension/data-prep line does not fall cleanly between edges and datasets
anyway — null handling and temporal limits are meaningful on both sides.

The ops:

| Op | Applies to | Notes |
|---|---|---|
| `filter_dimension` | edge, dataset | Categories or groups; optional exclude and flatten. Refuses non-structural dimensions when flattening. |
| `assign_dimension` | edge, dataset | Was `assign_category` on the dataset dimension filter. |
| `drop_nulls` | edge, dataset | |
| `filter_temporal` | edge, dataset | Currently the yearly specialization (`min_year` / `max_year`). |
| `filter_column` | dataset | Legacy, pre-dimension column filtering. |
| `rename_column` | dataset | Legacy wide-DVC column labels. |
| `rename_item` | dataset | Category value remapping. |
| `set_forecast_from` | dataset | Sets the forecast **qualifier**; see `metric-dataframe.md`. |
| `interpolate` | dataset | Fills gaps and sets the interpolation qualifier. |

There is deliberately no `select_column` op: metric selection is the
binding's source reference, not a transformation.


## Current state and next steps

**Done:**
- `PortTransformOp` is the one vocabulary; dataset bindings execute it as a
  pipeline, and `PortBindingDef` carries `transformations` for both kinds,
  always presented in the current vocabulary.
- `NodeEdge.transformations` stores the unified kinds: `select_categories` and
  `assign_category` were migrated to `filter_dimension` / `assign_dimension`
  (`0054`), sync emits the new kinds, and the legacy kinds survive only as
  tolerated input. `flatten` remains stored as the placeholder for a port
  shape declaration.
- Editing over GraphQL: `bindingEditor` resolves both kinds;
  `updateDatasetBinding` / `updateEdgeBinding` / `deleteBinding`, plus
  `bindDataset` and `createEdge`. Each kind has its own `oneOf` input type,
  so applicability is the input type's field list, introspectable by the
  editor UI; `createEdge` still accepts the deprecated legacy fields.
- Ports may carry an optional human-readable `identifier`.

**Next, in dependency order.** The first three are prerequisites for
propagation, not independent work:

1. Give ports authored, stable identity that survives a sync, and make
   `instance_from_db` emit port wiring that the loader consumes. Until
   then port ids are cosmetic. Related: unify `NodeEdge` and `DatasetPort`
   into one `NodeInputPortBinding` table. That is where a binding's ordering
   within a `multi` port can live — ordering has to be shared, because a port
   may hold both an edge and a dataset — and it retires the
   `(node, dataset_index)` grouping that stands in for binding identity today.
2. Move the ex-`FlattenTransformation` shape declarations onto
   `InputPortDef` and retire the `flatten` kind.
3. Rewrite `_get_output_for_node()` and `_get_output_for_target()` to
   consume the op pipeline at the consuming port instead of the legacy
   `from_dimensions` / `to_dimensions` dicts on the producer. Until then,
   `_transforms_to_config()` in `instance_from_db.py` is the seam that
   translates stored transformations back into those dicts, and it bounds
   what an edge can execute — which is why `EdgeTransformationInput` accepts
   only the dimension-reshaping kinds for now. This step widens it with
   `drop_nulls` / `filter_temporal` / `ensure_unit` (additive, non-breaking).
4. Add node dimension signatures (requires/consumes/produces/transparent)
   to node classes or pipeline definitions.
5. Implement upstream constraint propagation for the editor's validation
   and port compatibility checks, exposing results as derived fields.

Removed from this list: "remove the `side` field from edge
transformations" — there is no such field in `edge_def.py`.
