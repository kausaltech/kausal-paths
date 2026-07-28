# Making a Paths Model Match an Existing Emissions Inventory

**Problem.** A city has a locally-commissioned greenhouse gas inventory — a BEI,
a consultant's audit, a national statistics release — and will only trust a
Paths model if the model reproduces that inventory. The inventory is authoritative
to them regardless of how it was built, and often regardless of whether it is
right.

**This document** is the general method for getting there: how to compare a model
against an inventory in enough detail that every difference can be named, and how
to close each difference with the right mechanism.

**Relationship to [`nzp-city-inventory-compliance-pattern.md`](nzp-city-inventory-compliance-pattern.md):**
that document classifies differences into three buckets (emission factor,
activity, boundary) and prescribes the *model mechanism* for each — dataset
override, observed-anchor ratio correction, named residual. It answers "what do I
do about this gap?". This document answers the question that comes first: "what
exactly is the gap?" — and adds the machinery needed when the city wants the
inventory reproduced including its mistakes.

---

## 1. The four steps

1. **Detailed comparison** — build a line-level ledger of both sides.
2. **Match the emission factors** to the inventory's values (bucket A).
3. **Match the activities** to the inventory's values (bucket B).
4. **Name the category discrepancies** — explicit categories for what genuinely
   differs in scope or boundary (bucket C).

The order matters. Factors are cheap swaps and shift both the inventory match and
the action impacts coherently. Activities need the observed-anchor machinery and
are where bias gets propagated if done carelessly. Only what survives both is a
real boundary difference. So the comparison in step 1 must produce, for every
line, enough information to say which of the three it is — which means it must
carry **activity and emission factor separately, not just emissions**.

## 2. Why emissions-only comparison fails

The natural first artifact is a sector-level table of inventory emissions against
model emissions. It is not sufficient, for two reasons.

**It cannot discriminate.** An emissions gap of −15% tells you nothing about
whether the factor or the quantity is responsible, which is exactly what steps 2
and 3 need to know.

**Totals hide offsetting errors.** Sector-level differences routinely cancel.
A model and an inventory can agree within a few percent on the grand total while
individual sectors are off by tens of percent in opposite directions — in which
case reconciling on totals declares success on a model that is wrong everywhere.
Always compute the sum of *absolute* sector differences alongside the net, and
quote both.

**Corollary: the comparison must be re-runnable, not hand-copied.** A markdown
table of model outputs goes stale the moment someone edits the model, and stale
reconciliations are worse than none because they are trusted. Script both sides.

## 3. Prerequisite decisions

Settle these before extracting anything; each one changes what gets extracted.

- **Comparison year.** Usually the model's reference year. Check whether the
  inventory's figure for that year is *measured* or *projected from an earlier
  base* — if projected, note that matching it means matching a scaled older
  number, and check whether the model even has the earlier year (if not, the
  question of which to match dissolves).
- **Scope frame.** Which scopes are in the target total, per sector. Get this
  from the inventory's own detail sheets, not from its headline: a headline
  often omits scope 3 for one sector while the detail reports it. Picking the
  wrong row can put a whole sector's worth of emissions into the "gap".
- **GWP set.** Confirm the inventory's CH₄ and N₂O factors and their vintage.
  A mismatch is a systematic multiplier on every CH₄/N₂O-dominated sector.
  Often verifiable by reverse-engineering (see §5.3) rather than by asking.
- **Acceptance test.** Which model node's value must equal which inventory
  figure, and to what tolerance. Without a named target list, steps 2–4 have no
  stopping condition.
- **How far the model's dimensions must be extended.** In a city-specific
  config this is nearly free — added categories affect no other city, and
  anything generally useful can be promoted to the shared framework file later.
  So let the comparison ask for the precision it needs rather than economising.
  Expect to need this wherever the model carries a lumped "other" bucket that
  the inventory reports as several sectors.

## 4. The ledger

One tidy CSV, both sides in the same schema:

```
side, sector, subsector, category, carrier, scope, ghg, year,
activity_value, activity_unit, ef_value, ef_unit, emissions_ktco2e,
status, source_ref, note
```

- **`source_ref` is non-negotiable.** For the inventory side, `workbook!sheet!cell`;
  for the model side, the node id. Steps 2 and 3 need to know exactly which cell
  or node a target value came from, and much of the diagnosis lives in the prose
  cells surrounding the numbers.
- **`activity_value` and `ef_value` may be null.** Some inventory sectors are
  top-down: a prior-year figure uplifted by a national trend, with no activity
  basis at all. That is a structural fact to record, not a gap to fill — such
  sectors can only ever be matched at the emissions level.
- **`status`** distinguishes `reported` (part of the inventory total) from
  `alternative` (computed by the consultant but not used) and `context`
  (reference figures the inventory explicitly excludes). Consultant workbooks
  frequently contain more than one estimate of the same quantity; the ledger must
  record which one was published and keep the others visible.
- **Derive the EF rather than reading it** wherever activity and emissions are
  both available. The derived factor is what the inventory *actually applied*,
  which is not always what its factor table says.

## 5. Step 1a — the inventory-side ledger

### 5.1 Where to look

Consultant workbooks are usually better structured than they appear. Look for:
an index sheet describing every numbered sheet; a sheet 1 "overall summary"
carrying energy demand *and* emissions together; per-category breakdown sheets;
and a conversion-factors sheet. Between the energy summary and the emissions
summary you generally get a full category × carrier matrix for both, which is
exactly the ledger's shape.

Non-energy sectors use the same schema with a different activity unit — tonnes of
waste, head of livestock, hectares, population for per-capita factors.

### 5.2 Validations to build into the extractor

These double as the inventory-internal consistency check the pattern doc asks for
before treating the inventory as ground truth:

- `emissions == activity × EF` per row
- per-category sheets sum to their summary sheet; summaries sum to the headline
- implied factors reconcile against the workbook's own factor table
- the same quantity reported in two places agrees

Separate **genuine inventory inconsistencies** from **extraction bugs** in the
output. A check that fails because the workbook disagrees with itself is a
finding; a check that fails because the script mis-read a cell is a defect. Tag
them differently and don't let the former fail the run.

### 5.3 Reverse-engineering the applied factor

This is the single most productive technique in the whole exercise. Divide
reported emissions by reported activity, per cell, and compare the result across
categories and against the factor table. It routinely settles methodology
questions that would otherwise need a data request, and it localises errors to
individual cells.

What it finds, in rough order of frequency:

- **A factor set that is uniform where you expected variation** (or vice versa),
  telling you how the consultant actually weighted a blend.
- **One cell using the wrong fuel's factor** — a copy/paste error, identifiable
  because the implied factor exactly equals another carrier's.
- **The baseline and the projections using different factor sets**, which means
  the glidepath is not consistent with the inventory it starts from.
- **The GWP set**, confirmable to four significant figures from any single
  CH₄ line.

### 5.4 Pitfalls

- **Spreadsheet total rows.** Pivot sheets end with a `Grand Total` row, often
  followed by the same value restated in other units. All of these have numeric
  cells in the data column. Filter on a per-record key (an id column) rather than
  on "is this cell a number", and validate summed totals against the workbook's
  own headline — a factor-of-two error in an implied EF is the symptom.
- **Two competing estimates of the same sector.** Where a workbook contains both
  a bottom-up and a top-down estimate, the published total may take the *level*
  from one and the *breakdown* from the other. The reported emissions then cannot
  be reproduced as activity × EF from any activity figure in the workbook, and no
  amount of model work will make both match. Record both, and say so.
- **Category labels that differ between sheets** for the same category. Match by
  position and assert the ordering.
- **Granularity floor.** A category that is 90%+ one bucket cannot support a
  breakdown comparison however much the city wants one. Establish the real floor
  early and set expectations.

## 6. Step 1b — the model-side ledger

A Paths model is activity × EF throughout, so this is a matter of naming the
triples. For each emitting chain, record (activity node, EF node, emission node)
and dump the comparison year's values with full dimensions.

Implement as a re-runnable script on the same instance-loading path as
`load_nodes.py` / `tools/debug_instance.py`. It will be run after every change in
steps 2–4, so its output must be current by construction.

Where the model has a lumped node fed by a single city-data column, say so in the
ledger rather than inventing a decomposition. An opaque bucket is a finding: it
tells you which dimension extension step 4 will need.

## 7. Step 1c — the crosswalk

A hand-authored, reviewable CSV:

```
inventory_sector, inventory_subsector, inventory_category,
model_sector, model_node, relation, confidence, note
```

with `relation` one of `1:1`, `n:1`, `1:n`, `inventory_only`, `model_only`.

Keep it a file for comment, not logic inside a script. It is the one place where
judgement rather than arithmetic decides the answer, it is the blocking artifact
for everything downstream, and it becomes step 4's explicit discrepancy
categories. Draft it early and get it reviewed.

## 8. Step 1d — gap decomposition

For each crosswalk group, compute the activity ratio, the EF ratio and the
emissions delta, and decompose:

> ΔE = ΔEF · A_inventory + EF_model · ΔA + interaction

**Decompose within a carrier, never across.** This is the single easiest way to
get a badly wrong answer. If the two sides have different carrier mixes, blending
them into one activity × EF pair manufactures a factor difference that does not
exist: the "EF gap" is really the mix difference, and an equal and opposite
"activity gap" appears to cancel it. Split by carrier first, and only then
decompose. The same applies to any other dimension along which the factor varies
a lot — heating type, vehicle mode, waste treatment.

**Check for a basis mismatch before calling anything a factor difference.** If a
sector's activity ratio and its inverse factor ratio are the same number, the two
sides are measuring different quantities — useful heat delivered versus fuel
consumed, tonnes treated versus tonnes collected, gross versus net. The emissions
will match while the activities disagree by tens of percent. That is not
agreement, and it matters: any action expressed as a fraction of the activity
lands on the wrong base. Record it as a basis difference, not as an offsetting
A/B pair.

A quick diagnostic before the full decomposition: compare total activity and
implied average EF separately, **per carrier**. If the activity ratio explains
most of the gap it is mostly bucket B; if the factor ratio does, mostly bucket A;
if each explains all of it in opposite directions, suspect a basis mismatch.

Each line then gets a bucket and a **named fix target** — the dataset to override
for A, the node chain to anchor for B, the residual category to create for C.
Rank by magnitude so steps 2–4 get a worklist in priority order rather than a
wall of differences.

## 9. Artifacts and the stopping condition

- the ledger CSV, both sides, one schema
- the reviewed crosswalk
- a reconciliation report with the waterfall from model total to inventory total
- an **error register**, if the city wants the inventory's errors reproduced (§10)
- a **status check** that reruns everything and prints match/no-match per
  acceptance-test line

That last artifact is what makes progress through steps 2–4 measurable rather
than asserted, and it is the reason the whole pipeline has to be scripted.

## 10. When the city wants the inventory's errors reproduced

Cities sometimes require the *main* model to carry the inventory's numbers even
where those numbers are demonstrably wrong, because the model's output has to be
the number they report. Offering a correct main model plus a side-module that
reproduces the inventory for display is a reasonable proposal, and it can be
declined — the city may want one set of numbers, theirs, everywhere.

Take the requirement at face value and make it reversible through documentation.

**Do not model errors as alternatives.** A boolean parameter or parallel branch
selecting between "as published" and "correct" builds a structure whose whole
purpose is to become obsolete: once the consultant fixes the inventory, the
switch and both branches are dead weight that still has to be understood before
it can be removed. Toggles are for genuine alternatives someone may want to keep
comparing over time. An error is not one.

**Do keep an error register** — one entry per adopted discrepancy:

| field | content |
|---|---|
| what the inventory did | the method as implemented, with `source_ref` |
| what is defensible | the correct treatment, with reasoning |
| delta | signed, in the reporting unit, at the comparison year |
| implemented at | node / dataset / column in the city config |
| how to flip | the specific value change that restores the correct treatment |
| status | adopted / disputed / awaiting the consultant |

Plus the rationale in the affected node's description, and the correct number as a
comment on the affected data point, so it is present where someone would look for
it.

The accepted cost is that the corrected value is not computable — the reconciliation
reads it from the register, not from the model. That is the right trade: the
register is a document that is *supposed* to be deleted line by line as fixes
land, whereas model structure is not.

**Distinguish errors from disputes in the register.** "The consultant applied a
methane GWP to combusted methane" is an error with a known correct value.
"The consultant assumed complete gas capture" is a disputed methodological choice
where neither side's figure is credible and the answer needs the city's own
operational data. Both belong in the register; only the first has a `how to flip`
that can be executed unilaterally.

## 11. The consequence to state plainly

Inventory demand estimates are frequently benchmark-synthetic — floor area ×
generic per-m² constants — rather than measured, and often well above metered
reality. Anchoring the model's *levels* to such an estimate propagates the bias
into every action impact, cost and capacity figure the model produces, in
proportion.

The pattern doc's recommended split (inventory-facing view separate from
model-facing basis) avoids this. Where a city declines that split, the
consequence should be stated plainly in whatever is published alongside the
model, and the ledger should retain the model-native activity figures so the size
of the effect stays visible and recoverable.

---

## Checklist

- [ ] Prerequisite decisions settled: year, scope frame, GWP, acceptance test,
      dimension extensions (§3)
- [ ] Inventory-side ledger extracted with `source_ref` per row, validations
      built in, inventory inconsistencies tagged separately from extraction bugs
- [ ] Implied factors reverse-engineered and compared against the workbook's own
      factor table
- [ ] Model-side ledger scripted on the same schema, re-runnable
- [ ] Crosswalk drafted as a reviewable file and actually reviewed
- [ ] Gaps decomposed into A/B/C with a named fix target each, ranked by size
- [ ] Net *and* gross sector differences both quoted
- [ ] Error register started if the city wants errors reproduced; no toggles
- [ ] Status check runs and reports per acceptance-test line
- [ ] Benchmark-synthetic-demand consequence documented if levels were anchored
