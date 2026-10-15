# Imputing Data into Nodes

*Produced by Claude, last revised 2026-10-14.*
*Section on `prefer_by_cell` added by Claude Opus 5.5 on 2026-10-14.*
*Responsible: Jouni Tuomisto.*

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
- **`prefer_by_year`** chooses a *source* per year rather than merging
  values per row. See below — it is the mechanism to reach for when
  "use local data where we have it" means whole reporting years, not
  individual cells.
- **`prefer_by_cell`** chooses a source per input field: the field a
  city entered comes from its data, every other field from the default.
  It is what an input form pre-filled with defaults needs; see the
  section after `prefer_by_year`.


## `prefer_by_year`: choosing a source per year, not per row

`impute` merges two frames cell by cell. `prefer_by_year`
(`common/polars_ext.py`, and the formula function of the same name)
chooses between two *sources* one year at a time:

```yaml
- id: vehicle_kilometers
  type: formula.FormulaNode
  input_nodes:
  - id: vehicle_kilometers_own
    tags: [own]
  - id: vehicle_kilometers_default
    tags: [default]
  - id: vehicle_kilometers_own_availability
    tags: [coverage]
  params:
  - id: formula
    value: prefer_by_year(own, default, coverage)
```

A year the city has supplied is served **entirely** from `own`; every
other year **entirely** from `default`.

### When to use which

Use `impute` when the two inputs are alternative estimates of the same
quantity and the best available value should win cell by cell.

Use `prefer_by_year` when the two inputs are *different data sources*
and a result that silently mixes them within one reporting year would
be wrong or unauditable. The motivating case is a city moving from
national default statistics to its own data collection: the switch
happens in a particular year, and a combination the city left empty
inside a year it *did* report is a gap in the city's data — something
to report as missing, not to backfill from the default. Backfilling
would produce a figure that is partly one source and partly the other
with nothing on screen to say which.

### The `coverage` argument, and why it is not optional in practice

Pass `coverage` whenever the preferred frame comes from a dataset
binding. It is tempting to let the function work out which years are
covered from the preferred frame's own values, and the two-argument
form does exactly that — but by the time a node's output reaches the
formula it has been through `empty_to_zero`, `interpolate`, `backfill`
and `extend`, so its values no longer say which years the source
actually contained.

This matters most in the case the mechanism exists for. A node reading
an empty template needs `empty_to_zero` to produce a dimensioned frame
at all, and those zeros are indistinguishable from reported zeros — so
an *unfilled* template would claim to cover every year and suppress the
default entirely. Note that a genuinely reported zero **is** data and
does win its year; that is BISKO's rule and the function honours it,
which is precisely why fabricated zeros cannot be tolerated.

A `DataAvailabilityNode` reading the same dataset is the honest answer,
because it inspects the dataset before any of that post-processing
happens. Using it has a second benefit worth keeping: it is normally
already the node displayed as the data-availability report, so what
drives the result and what is reported as available cannot drift apart.

The two-argument form remains correct for a frame whose gaps are still
genuine nulls — typically one computed from other nodes rather than
read from a binding.

### Details

- Dimensions must match between the two value frames, as with `impute`.
- Units are reconciled to the preferred frame's unit.
- A frame that covers no years yields the fallback unchanged.
- Rows left null by the choice are dropped, so a combination with no
  value behaves like one that was never in the data.
- Unlike `select_port`, which evaluates only the branch it selects,
  `prefer_by_year` evaluates both. Converting a `select_port` to it
  will surface any latent error in the previously-dead branch.


## `prefer_by_cell`: an input form pre-filled with defaults

`prefer_by_cell(own, default, coverage)` makes the same choice as
`prefer_by_year`, but per cell: a year together with the categories of
`coverage`'s dimensions. A cell the coverage marks comes from `own`,
every other cell from `default`. This is the model of a form whose
fields come pre-filled with a default, where an entry overwrites that
one field and leaves the rest alone.

The BISKO module switched its transport inputs from `prefer_by_year` to
`prefer_by_cell` in October 2026. The certifier's reading of the
standard is the form: defaults are *voreingestellte Werte*, and default
and local values mix within a year. Under the per-year choice, one
entered mileage cell switched the whole year to the city's data and
every empty cell to zero, which removed almost all of road transport
from that year's balance.

```yaml
- id: vehicle_kilometers
  type: formula.FormulaNode
  input_nodes:
  - id: vehicle_kilometers_own
    tags: [own]
  - id: vehicle_kilometers_default
    tags: [default]
  - id: vehicle_kilometers_own_availability
    tags: [coverage]
  params:
  - id: formula
    value: prefer_by_cell(own, default, coverage)
```

The `coverage` argument is required, for the reason given for
`prefer_by_year`: an own-data frame that has been zero-filled cannot
tell an entered zero from an empty field. An entered zero is an entry
and wins its cell.

`coverage` may have fewer dimensions than the data. Coverage by vehicle
class and fuel then covers that class and fuel on every road category.

### Choosing which to use

Use `prefer_by_year` when a reporting year should come wholly from one
source, for example when the two sources are alternative *methods*
(the three district heating routes in BISKO). Use `prefer_by_cell` when
each field is a separate entry that can come from either source.

### `split_categories`: a measured total that contains its parts

A measured figure often contains categories the result reports apart:
diesel sales include the biodiesel blend. Replacing only the default's
fossil diesel cell with it would leave the default's biodiesel standing
and count the blend twice.

```text
split_categories(measured, reference, 'energy_carrier',
  'diesel: diesel + biodiesel, petrol: petrol + bio_petrol')
```

splits each aggregate row into its members in the proportions of
`reference` (here the default calculation for the same vehicle class
and year). The first member is the fallback: where the reference has
no positive total, the whole value stays there, and it always keeps a
row. An aggregate is **not** split where the data already gives another
member with a value: a city entering diesel and biodiesel separately has
entered fossil diesel. `reference` may lack dimensions of the data; the
BISKO road split uses mileage by vehicle class and road category to
spread energy that is also broken down by fuel.

