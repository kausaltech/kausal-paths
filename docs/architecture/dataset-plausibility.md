# Dataset plausibility findings

`DatasetMetricPlausibilityRange` stores an advisory reference band for cells of
one dataset metric. It is scoped to a framework or to one instance. This is
Paths-owned because the shared dataset models also serve Kausal Watch. A
framework range works for BISKO, NZC, or another framework using the same
dataset machinery.

Plausibility findings are warnings. They never block editing or publication,
and they are calculated when queried, so changes to population projections or
reference ranges take effect without refreshing a dataset materialization.
Validation is evaluated first: a cell with a validation violation gets no
plausibility finding, and neither does a sum that includes one.

## What a range checks

A range combines four independent choices.

**Selection.** `selection` maps dimension UUIDs to lists of category UUIDs.
A listed dimension is restricted to its listed categories; an unlisted one is
unrestricted. UUIDs rather than identifiers keep a range valid when a category
or dataset column is renamed.

**Aggregation.**

- `cell` checks each selected cell on its own.
- `sum` checks the sum of the selected cells, once per year. A sum must list
  its categories for **every** dimension of the metric's schema, so that a new
  category, or a total category such as "Gesamt", cannot silently join it.
  A sum is *incomplete* when any selected cell is an empty (null) data point.
  An incomplete sum only grows as cells are filled in, so it is only reported
  for exceeding the upper bound; a low incomplete sum means "not finished yet".

**Denominator.** `none`, or `population`: the observed population of the
instance's organization in the same year. A missing observation leaves the
check unevaluated.

**Reference.**

- `absolute` compares the value (per denominator) with the bounds.
- `previous_year` compares the ratio of the value to the same selection's value
  in the latest earlier year within `max_gap_years`. Such a range has no
  denominator and a positive lower bound. For a sum, both years must be
  complete, because a ratio between partial sums says nothing about either year.
  A category swap between years keeps a sum's total, so swaps are caught by
  previous-year ranges on single cells, while sums catch shifts in the total.

Bounds carry no unit of their own. An absolute bound is in the metric's unit,
divided by the denominator's unit (`MWh/cap` for an `MWh` metric with a
population denominator); a previous-year bound is a dimensionless ratio. The
model derives it as `bound_unit`. Changing a metric's unit therefore changes
what its ranges mean, and their bounds must be rescaled with it.

## Sources

Each range points to a `PlausibilitySource`, which holds what a set of ranges
was derived from: name, URL, the source data revision, and the method.
`is_example` marks local demonstration data, which must not be presented as an
empirical benchmark. A source that is not an example requires a URL.

## GraphQL

The `DatasetFinding` interface supplies the same cell locator for
`DatasetValidationViolation` and `DatasetPlausibilityFinding`.

- `dataset.plausibilityRanges` lists the applicable ranges with their
  `selection` (a flat list of dimension/category pairs, where a dimension may
  repeat), `aggregation`, `denominator`, `reference`, derived `unit` and
  `source`.
- `dataset.plausibilityFindings` lists current outliers. For a sum, `coordinates`
  holds the dimensions its selection fixes to a single category, `selection`
  every selected pair, `componentCount` how many cells were summed, and
  `complete` whether any was empty. `normalized` is the quantity compared with
  the bounds; for a previous-year range, `referenceYear` and `referenceValue`
  say what it was compared with.
- The instance editor exposes `datasetPlausibilityFindings` across its
  currently bound datasets.
- Each `dataPoint.plausibilityRanges` entry gives `lower` and `upper` in the
  point's metric unit, plus a `reference` to the original range. Only `cell`
  ranges appear there. An entry keeps its reference but has null bounds when
  population (or, for a previous-year range, an earlier value) is missing; the
  UI should not assess the cell against it. Null-valued data points receive
  ranges too, so the editor can show guidance before a value is entered.

## Local UI fixture

After migrating the database, run:

```bash
python manage.py seed_plausibility_example bisko-12060005 kommune/endenergieverbrauch --metric Value --year 2023
```

The command requires `DEBUG`, chooses a valid positive cell, and creates an
instance-scoped `ui-example` range, with an `is_example` source, that
deliberately yields one finding. The bounds are derived from that cell solely
for UI development. Rerunning it updates the same range. Remove the example
row when no longer needed.
