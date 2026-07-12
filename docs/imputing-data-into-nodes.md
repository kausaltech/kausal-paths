# Imputing Data into Nodes

## Overview

Most node classes already support an *implicit* additive input: wire a
node in as an input edge (optionally tagged), and its output gets added
into the target's own computed result, without touching the target
node's own configuration. This is how actions typically attach to a
target node — the edge is defined on the input node's side, and the
operation is addition.

`impute` is a second kind of implicit input, alongside that additive
one. Instead of adding on top of the node's own computed output, an
`impute`-tagged input **replaces** values in the result — value by
value, wherever the tagged input has data. It is meant for cases like
"use this generic/regional default, but substitute local values
wherever we have them": a node computes its normal output, and one or
more `impute`-tagged inputs then overlay their own values onto that
result, falling back to the node's own value wherever the tagged input
has none.

This is the same idea already used by `CoalesceNode`'s `'primary'`/
`'secondary'` tags (`nodes/generic.py`), generalized so it works as an
implicit input on any of the four node classes below, without requiring
a dedicated node class.


## Semantics

An `impute`-tagged input node is combined with the node's own computed
output via an **outer join on dimensions**, followed by a **coalesce**:

- The two dataframes must have the exact same set of dimensions
  (`dim_ids`). This is checked explicitly and raises a clear
  `NodeError` on mismatch — imputing does not introduce or drop
  dimensions, unlike `add_from_incoming_dims`/`add_to_existing_dims`.
- For each row (year × dimension categories), the tagged input's value
  wins if it has one; the node's own computed value is used only where
  the tagged input has none.
- If more than one node is tagged `impute` on the same target, they are
  applied in the order the edges were added, each one taking priority
  over everything applied before it (including the node's own value).

This is implemented once, on the base `Node` class, as
`Node.impute_nodes_pl(df, nodes)` (`nodes/node.py`), which folds
`node_df.paths.coalesce_df(df, how='outer')` over the tagged nodes in
turn. `coalesce_df` (`common/polars_ext.py`) is the same primitive
`CoalesceNode` already uses.


## Using it

Tag the input edge (or the input node itself) with `impute`:

```yaml
- id: local_emission_factor
  name: Local emission factor
  type: formula.FormulaNode
  ...

- id: default_emission_factor
  name: Default emission factor
  type: formula.FormulaNode
  formula: 'regional_default * some_multiplier'
  input_nodes:
  - id: regional_default
  - id: some_multiplier
  - id: local_emission_factor
    tags: [impute]
```

Here `default_emission_factor` computes its formula as normal from
`regional_default` and `some_multiplier`, and the result is then
overlaid with whatever years/categories `local_emission_factor`
provides — everywhere else, the formula's own result stands.

### FormulaNode

`impute`-tagged inputs are not referenced in the formula string; they
are applied after it is evaluated, as the last step. (They are also
excluded from the "unused inputs get added" backward-compatibility
fallback that otherwise applies to any input node not referenced by
name in the formula — an `impute`-tagged input is *never* implicitly
added, only imputed.)

### GenericNode

`impute` is a regular operation, dispatched like any other entry in the
`operations` parameter — it is **not** applied automatically, only
when listed:

```yaml
params:
  operations: get_single_dataset,add_to_existing_dims,impute
```

The operation (`_operation_impute` in `nodes/generic.py`) requires at
least one input node tagged `impute` and raises otherwise.
`impute`-tagged inputs are also excluded from `GenericNode`'s default
add/multiply classification, so they don't additionally get summed or
multiplied in if `operations` also includes `add`/`multiply`.

### AdditiveNode

`impute`-tagged inputs are excluded from the addition (the same way
`non_additive`-tagged inputs are) and applied as the final step, after
all of `AdditiveNode`'s own post-processing (`drop_nans`,
`scale_by_reference_category`/`_year`, `get_shares`, etc.).

### MultiplicativeNode

`impute`-tagged inputs take no part in the multiplication/addition
classification and are applied as the final step, after
`replace_nans`.


## Dimension requirements

Unlike the additive `add_to_existing_dims`/`add_from_incoming_dims`
tags, `impute` does not reshape dimensions. The tagged input's output
must have exactly the same dimensions as the node's own computed
result (same `dim_ids`, in any order) — no more, no fewer. If a model
needs to impute values that are broken down differently than the
target (e.g. a coarser or finer category split), that redistribution
has to happen upstream of the `impute`-tagged node, not as part of the
impute step itself.


## Relationship to other coalesce-style mechanisms

- **`CoalesceNode`** (`nodes/generic.py`) predates `impute` and solves
  the same problem, but only for a dedicated node class, and only with
  exactly one `'primary'`- or `'secondary'`-tagged input (the tag
  choosing which side of the join wins). `impute` is meant to replace
  this pattern for new models; `CoalesceNode` is kept for existing
  models that already use it.
- **`add_to_existing_dims`/`add_from_incoming_dims`** are the additive
  counterpart described in the Overview: same "implicit input" idea,
  but the operation is addition (with `fill_null(0)`) rather than
  coalesce, and — for `add_from_incoming_dims` — the input can bring in
  dimensions the target doesn't already have.
