# Diagnosing categories a product drops

*Produced by Claude Opus 5.5 on 2026-11-21.*
*Responsible: Jouni Tuomisto.*

A product of two frames keeps only the rows both have. That is right for a factor: an emission
factor for a fuel nobody uses should vanish. It is wrong for an activity: fuel with no emission
factor vanishes too, and its emissions with it, with no error and no trace in the result. This
plan is for making the second case visible: a node says which of its inputs may not lose rows, and
the model reports it when they do.

## 1. Where rows are dropped

- `multiply_with_dims` and `divide_with_dims` (`common/polars_ext.py`) join with `how='inner'`.
  `MultiplicativeNode2`, `GenericNode`'s multiply operation and the `*` and `/` of formulas all
  use them.
- A row is lost when its categories on the dimensions the operands share, or its year, have no
  partner.

Dropping is often deliberate:

- **Factors.** A factor table covers more categories than any one city uses.
- **Years.** Whether a year has data decides the route a value takes. The district heating split
  exists only in the years its temperatures were entered, and its product with the fuel
  emissions keeps only those years, which is how a year without them falls back to another BISKO
  route. The same pattern recurs wherever `prefer_by_year` chooses a source.
- **Selections.** The `partial_factor` tag multiplies by a factor defined for some categories and
  deletes the others on purpose.

So the rule cannot be "a product may not drop rows". It has to name the operand whose rows must
survive, and it has to leave whole missing years alone until the existing models have been
studied (§4).

## 2. Three layers

1. **A check at compute time.** After a node computes, every row of a marked input must have a
   counterpart in the output, on the dimensions the input and output share. A row that has none
   is an error naming the categories and years. This catches the case for every node class,
   whatever route its computation takes, and it works on the actual data.
2. **Inference from the quantity.** In a product, the extensive operand (energy, emissions,
   mass, distance, counts) must keep its rows, and the intensive one (emission factor, share,
   fraction, price) may lose them. That marks most inputs without a tag, and a tag then declares
   only the intended drops. Quantities are plain strings today, so this needs a list of which
   kinds are extensive. Existing models surely rely on drops, so it starts as a warning counted
   across all instances with `test_instance`, and becomes an error only once the count is known.
3. **A diagnostic in the constraint solver, for the editor.** `ProductShapeRule`
   (`nodes/constraints/rules.py`) would list the operands whose categories must survive, and the
   solver would report a product that drops some wherever the categories on both sides are known:
   from shapes, from category filters, and from the observed profiles of database datasets. The
   solver does not know the categories of a DVC source (`docs/architecture/dimension-constraints.md`),
   so this cannot replace the check at compute time. It is the same kind of diagnostic as
   `disjoint_category_filter`. To be designed with the solver's author.

## 3. Layer 1, as built on 2026-11-21

The tag `keep_categories` on an input binding, a node edge or a dataset binding:

```yaml
input_nodes:
- id: district_heating_energy_use
  tags: [fuel, keep_categories]
```

After the node computes (`Node._check_kept_categories`, called from `_compute_validated`), each
binding with the tag is read again as the node received it, and:

- the rows that count are those with a value other than null and zero: a zero row carries nothing,
  and empty templates read with `empty_to_zero` are full of them;
- they are compared with the output's rows on the year and on the dimensions the input and output
  share; a dimension the node sums away is not compared;
- a year in which the output has no rows at all is not compared, because a missing year is the
  deliberate fallback of §1;
- a row with no counterpart whose value is not null is an error (`NodeError`) listing the missing
  category combinations and their years.

Nothing else changes: no node class, no formula function and no existing model uses the tag, so
no result changes until a model adds it. The first user is the plant-level district heating
route in `bisko-base`, where a fuel without a generation factor or heat without a heat-output
factor must stop the calculation rather than vanish.

## 4. Open before layers 2 and 3

- **Which models drop deliberately, and how.** Add a warning mode to the check, run every
  instance with every product input marked, and sort the drops into factors (fine), years
  (route choice), selections (`partial_factor`) and errors. That count decides whether whole
  missing years can ever be compared, and what layer 2 can turn into an error.
- **The extensive/intensive list** for layer 2, with the quantity strings in use.
- **The solver's category facts** for layer 3, and whether formula nodes compile product rules
  from their formulas.
- **Severity.** Layer 1 fails the node. Whether a dropped row should instead mark the node
  degraded, as `allow_no_match` does, depends on whether a city can fix the cause itself.
