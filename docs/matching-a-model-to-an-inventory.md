# Making a Paths Model Match an Existing Emissions Inventory

**Problem.** A city has a locally-commissioned greenhouse gas inventory — a BEI,
a consultant's audit, a national statistics release — and will only trust a
Paths model if the model reproduces that inventory. The inventory is authoritative
to them regardless of how it was built, and often regardless of whether it is
right.

**This document** answers "what exactly is the gap?": how to compare a model
against an inventory in enough detail that every difference can be named, ranked,
and assigned a fix — plus the machinery needed when the city wants the inventory
reproduced including its mistakes (§10).

**Scope.** This applies when a model **already exists** — typically a framework
instance — and has to be reconciled against a separately-produced inventory. Two
independent estimates, and the work is explaining the difference between them.

| You have | Read |
|---|---|
| A model that must reproduce a city inventory | **This document** |
| A diagnosed gap in an NZP-based model, needing a mechanism | [`nzp-city-inventory-compliance-pattern.md`](nzp-city-inventory-compliance-pattern.md) — three buckets (factor, activity, boundary) and the model mechanism for each |
| City workbooks and no model at all | [`building-a-model-from-city-workbooks.md`](building-a-model-from-city-workbooks.md) — the greenfield build, and the shared technique for reading a consultant workbook (§3) |

The greenfield case is a different problem, not a subset of this one: there is only
one estimate, so nothing to decompose, and the failure mode is a model that
matches the headline while being wrong underneath rather than one that visibly
disagrees.

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
  Often verifiable by reverse-engineering the applied factor rather than by
  asking — see §5.
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

**Reading the workbook is documented once, elsewhere.** Workbook anatomy, the
extractor validations, reverse-engineering the applied factor, and the traps —
placeholder zeros, one ledger row per applied factor, structure that changes
between years, restated rows, spreadsheet total rows, competing estimates of the
same sector, the granularity floor — are the same job whether you are reading a
workbook to compare it against a model or to build one from it. See
[`building-a-model-from-city-workbooks.md`](building-a-model-from-city-workbooks.md)
§3 (reading the ledger) and §6.1 (what the extractor must validate).

Two of those points carry extra weight here. **Reverse-engineering the applied
factor** does double duty in a comparison: it produces §4's `ef_value` column
*and* the inventory-internal consistency check the pattern doc requires before the
inventory is treated as ground truth. And **separating inventory inconsistencies
from extraction bugs** decides who owns each failure — conflating them either
buries real findings or blocks your work on someone else's error.

Three things are specific to the comparison case:

- **The unpublished alternative is often the explanation.** When building a model
  from a workbook you take the published estimate and move on; when explaining a
  gap, the estimate the consultant computed and discarded frequently *is* the
  model's number. That is what §4's `status` column is for.
- **Extract at the granularity the crosswalk needs**, which may be finer than the
  model currently carries — see the dimension-extension prerequisite in §3.
- **Capture the prose.** Assumption text, method notes and caveat cells are where
  most of the diagnosis lives, and they are what turn a numeric difference into a
  named one.

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

Two things make the status check worth having rather than decorative:

- **Mark each line `required` or `blocked`.** A `required` line is expected to
  match and a failure is a regression that fails the run; a `blocked` line is a
  known open gap, reported with its size but not enforced. Each answered question
  flips one line from blocked to required, and from then on the check protects
  the result. Without this split the check is either all-red (useless) or has to
  be read by a human every time (forgettable).
- **Key the lines on sector/subsector, not scope.** Scope is the obvious choice
  and usually will not work: a model loses scope wherever an emission node has no
  scope dimension, because the assignment happens on the edge into the total; and
  inventories often report combined `scope 1+2` figures for some sectors. Pick a
  key that is stable on both sides.

Test the check by breaking something on purpose. A status check that has never
gone red is not known to work.

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
