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

These two operations cover all current edge transformation patterns and
are the same operations used in dataset input pipelines
(`FilterDimensionDatasetTransformOp`, `assign_category`).

Dataset pipelines add data-prep operations on top (column selection,
renaming, null handling, year limits) that don't apply to edges because
node outputs are already well-formed.


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


### Where the declarations live

| What | Where | Static or computed? |
|---|---|---|
| Node class dimension rules | Node class or pipeline definition | Static (per class/pipeline) |
| Outcome node required dims | `InputPortDef.required_dimensions` | Static (user-configured) |
| Edge transformations | `NodeEdge.transformations` (`list[EdgeTransformation]`) | Static (user-configured) |
| Input port effective requirement | Computed from downstream | Computed at validation/editor time |
| Dataset produced dims | Dataset schema | Static |


## Unification with dataset transforms

The dimension-aware subset of dataset transforms and edge transforms
are the same operations:

```
Edge: SelectCategoriesTransformation  ≡  Dataset: FilterDimensionDatasetTransformOp
Edge: AssignCategoryTransformation    ≡  Dataset: FilterDimensionDatasetTransformOp(assign_category=...)
```

A future refactoring could unify these into a shared
`DimensionTransformOp` type, with dataset pipelines adding
data-cleaning ops (column selection, renaming, null handling, year
limits) as a separate layer.


## Current state and next steps

**Done:**
- `NodeEdge.transformations` uses structured Pydantic types
  (`SelectCategoriesTransformation`, `AssignCategoryTransformation`)
  stored via `SchemaField`.
- Export and import round-trip through these types.
- `edge_def.py` defines the transformation types.

**Next:**
- Remove the `side` field from edge transformations — the operations
  are semantically the same regardless of from/to context.
- Rewrite `_get_output_for_node()` and `_get_output_for_target()` to
  consume the transformation pipeline directly instead of the legacy
  `from_dimensions` / `to_dimensions` dicts.
- Add node dimension signatures (requires/consumes/produces/transparent)
  to node classes or pipeline definitions.
- Implement upstream constraint propagation for the editor's validation
  and port compatibility checks.
- Unify dimension operations between edge and dataset transforms.
