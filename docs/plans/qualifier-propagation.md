# Qualifier Propagation: One Rule per Meta Column

*Produced by Claude Opus 5.5 on 2026-11-19.*
*Responsible: Jouni Tuomisto.*

## The idea

A node's frame holds three kinds of column, and each kind needs its own rule when frames are
combined:

- **Values** follow whatever operation the node is designed to do: a sum, a product, a choice.
- **Forecast** is combined with OR when frames are joined.
- **Qualifiers** (`<metric>__qual`, see `common.qualifiers`) have no universal treatment. A
  BISKO quality grade is combined as a mean weighted by the value column. Another qualifier, or
  another meta column, may need a different rule.

So the machinery should be generic: each meta column declares the aggregation designed for it,
and every operation applies that declaration without knowing what the column means. This
follows design principle 6: the qualifier owns its semantics, and the operations do not
reconstruct them.

## What exists

Most of the machinery is in place (`src/common/qualifiers.py`):

- A runtime `QualifierCatalog` of `QualifierDefinition`s, resolved once per context from the
  instance's frameworks (`frameworks/qualifiers.py`). An instance without a framework gets only
  the built-in `reported` qualifier.
- Each definition names a `Propagation`:
  - `REPORTED`: any/all flags.
  - `COVERED_SCORE`: a score with coverage, combined as a mean weighted by |value|. This is the
    BISKO rule.
  - `SOURCES`: a union of provenance.
- One combiner per operation: `combine_sum`, `reduce_sum` (sum over a dimension),
  `combine_product`, `choose`, `remove_subset` and `carry_over` (fills). The `PathsExt`
  operations call them.

Since 2026-11-19, every node class that combines inputs in polars carries its inputs'
qualifiers into its output:

- `PathsDataFrame.select` keeps the qualifier of every metric it passes through unchanged or
  only renamed, as `rename` and `drop` already did. Before this, a hand-written projection
  dropped it, and that is how `AdditiveNode2` and `MultiplicativeNode2` lost every dataset's
  qualifier.
- `src/nodes/tests/test_qualifier_conformance.py` checks this for `AdditiveNode2`,
  `MultiplicativeNode2`, `GenericNode` and `FormulaNode`. A new class that combines inputs
  belongs in that test.
- The v1 `AdditiveNode` and `MultiplicativeNode` are marked as expected failures. They compute
  in pandas, and the pandas conversion leaves qualifiers out by design. They are to be retired,
  not fixed (`additive-multiplicative-modernization.md`).

## What remains

### 1. Strategies instead of a closed enum (deferred)

Each combiner dispatches on `Propagation` with its own `if`/`elif`, so a new kind of meta column
means editing six functions. The generic form:

- A propagation strategy is a class with one method per operation kind: sum, reduction over a
  dimension, product, choice, subtraction of a subset, and fill.
- Strategies are registered by name, and a `QualifierDefinition` names its strategy.
- Each combiner loops over the catalog's definitions and delegates to their strategies.

A new meta column is then one class, with nothing to change elsewhere. This is not urgent: the
three existing kinds cover BISKO 2.0. It waits until a fourth kind is needed or the code is
touched for another reason.

### 2. The product rule has no authoritative basis

When both factors of a product are assessed, `combine_product` takes the lower score and the
lower coverage. BISKO assesses the aggregated quality of end-energy data, not of emissions, so
it defines no rule for a product such as energy times an emission factor, and no other source
does either.

**The minimum is not based on any outside source, nor on reasoning; it is an intuitive first
choice.** It stays until something better comes up. The docstring of `combine_product` and
`docs/architecture/metric-dataframe.md` say the same.

A better rule would have to settle at least what the weight of each factor is in a product. A
value-weighted mean, as for sums, has no obvious weight there.

### 3. Forecast as a qualifier (deferred to a quiet period)

`docs/architecture/metric-dataframe.md` plans for forecast status to become a qualifier field,
reduced with `any`. That would make it uniform with the other meta columns. The `Forecast`
column is read in hundreds of places, though. The migration has to be done when nothing else is
going on in the codebase, or it will drown in merge conflicts.

## Sequence

1. Done: projections keep qualifiers, and the conformance test exists.
2. Now: make sure BISKO 2.0 (`configs/modules/bisko-base/`) works properly with qualifiers
   carried through every v2 node.
3. Later: strategy classes (§1), and a better-founded product rule if one turns up (§2).
4. In a quiet period: Forecast as a qualifier (§3).
