# Building a Paths Model from a City's Own Workbooks

**Problem.** A city has its greenhouse gas inventory in Excel — usually a
consultant's workbook, often with a companion forecast-and-scenario tool — and no
Paths model exists. The task is to build one that reproduces the inventory
exactly, projects it the way the city projects it, and can carry actions.

**This document** is the end-to-end method: what to read before extracting
anything, how to write an extractor that stays honest, how to shape the data, how
to build the graph from it, and how to know when it is right.

**Relationship to the two companion documents.** They start from a *different
artifact*, and that is the whole distinction:

| You have | Read |
|---|---|
| City workbooks, no model | **This document** |
| A framework model (NZC/GPC) that must reproduce a city inventory | [`matching-a-model-to-an-inventory.md`](matching-a-model-to-an-inventory.md) |
| A diagnosed gap in an NZP-based model, needing a mechanism | [`nzp-city-inventory-compliance-pattern.md`](nzp-city-inventory-compliance-pattern.md) |

The difference matters more than it looks. In the framework case there are two
independent estimates and the work is reconciling them. Here the inventory *is*
the model's source, so there is no gap to decompose — and the failure mode is not
disagreement but **silent agreement**: a model that matches the headline while
being wrong underneath.

Both this document and `matching-…` need the same workbook-reading technique.
It lives here, in §3, because it is load-bearing here; that document links to it.

**Worked example.** Everything below comes from one real build, of a city referred
to throughout as **Test City**. The figures, defects and near-misses are that
city's actual ones — it is a real inventory produced by real consultants, not a
constructed example — but the city and its consultants are not named, because a
list of the problems found in someone's inventory is not ours to publish. Ask
internally for which build it was if you need to read the config alongside this.

**Prerequisites**, not restated here: the CSV upload format
([`dataset-csv-format.md`](dataset-csv-format.md)), the upload commands
([`dataset-management-commands.md`](dataset-management-commands.md)), and general
node/YAML design ([`paths-purpose-and-design.md`](paths-purpose-and-design.md)).

---

## 1. Decide the shape before you start

**Build from workbooks only when a framework does not fit.** If the city's
inventory maps onto GPC or NZC, instantiate the framework and reconcile — you
inherit the action mechanics, the shared maintenance, and cross-city
comparability. Build from workbooks when the city has a rich inventory of its own
and forcing it into a framework would lose most of it.

What you need before starting:

- **The inventory workbook**, ideally with a per-record ledger sheet.
- **The forecast/scenario workbook, if one exists.** This is worth more than it
  looks: it usually carries the growth drivers, the projection, the policy list
  and the targets — that is the entire forecast side of the model, already
  quantified by people the city trusts.
- **A named contact who can answer questions.** You will generate a list. Some
  answers change the model.

Then settle, before extracting:

- **Which years the model covers**, and whether the earliest is real (see §3.1).
- **The reporting unit.** Use `t_co2e`, not `t`: the figures are CO₂-equivalent,
  not the physical mass of the gases, and `t_co2e` is a distinct pint dimension so
  the two cannot be silently added.
- **Whether to convert units.** Prefer not to. Keeping every cell in the unit the
  inventory used means `emissions = activity × EF` reproduces the published figure
  exactly and any cell in the CSV is directly comparable with the workbook cell it
  came from. That is worth more during review than tidy magnitudes. Add missing
  units to `nodes/units.py` rather than converting — and check what you add
  (redefining a pint unit silently rewrites its aliases; `short_ton` meant a tonne
  in this codebase until that was found).

## 2. The extractor is the deliverable

Not the CSV. The CSV is a build artifact that will be regenerated many times —
five uploads in the Test City build, each after a defect was found.

**Requirements:**

- **Re-runnable and self-validating.** It checks its output against the
  workbook's own published figures and exits non-zero on failure, so a stale or
  mis-read extraction fails loudly instead of producing a plausible file.
- **Every emitted value traceable.** A `source_ref` per row
  (`workbook!sheet!row`), and per-year sources where the workbook cites a
  different data file per inventory year — one source cell for the whole series
  would attribute a 2008 value to the 2023 file.
- **A findings register**, distinguishing *inventory-internal issues* from
  *extraction defects*. A check that fails because the workbook disagrees with
  itself is a finding, printed on every run; a check that fails because the script
  mis-read a cell is a defect and fails the build. Test City ended with 19
  findings, of which 7 were new discoveries made while building the model.
- **Alongside it, a data-assessment record** — a README next to the data saying
  what the data is, what it says, what is wrong with it, and what was extracted.
  This is the artifact the city's contact reads, and the one that stops the next
  person re-deriving everything.

**Never patch a value you think is wrong.** Keep it as the inventory has it and
attach the finding. The exception is writing an explicit zero to stop a bad value
propagating (§4.3) — which is not a patch, because the zero is what the workbook's
own totals say.

> **Findings register or error register?** Two different questions, and both may
> apply. A **findings register** (here) records what is wrong with the *source
> data*, so the next reader does not re-derive it — every entry is a thing to
> raise with the city. An **error register**
> ([`matching-a-model-to-an-inventory.md`](matching-a-model-to-an-inventory.md)
> §10) records the discrepancies the *model deliberately reproduces* because the
> city needs its published numbers, with a `how to flip` for each. Use that one as
> soon as the city asks the model to carry a figure you have established is wrong —
> and never as a toggle or a parallel branch, which builds structure whose whole
> purpose is to become obsolete.

## 3. Reading the ledger

### 3.0 Workbook anatomy

Consultant workbooks are usually better structured than they appear. Look for:

- an **index sheet** describing every numbered sheet;
- a **per-record ledger** ("Database", "All") with one row per activity entry —
  where this exists, extraction is largely read-and-rename, and Test City's had
  19,233 rows already carrying activity, factor, emissions and a source reference;
- an **overall summary** sheet carrying energy demand *and* emissions together;
- **per-category breakdown** sheets;
- a **conversion-factors / emission-factors** sheet.

Between the energy summary and the emissions summary you generally get a full
category × carrier matrix for both. Non-energy sectors use the same shape with a
different activity unit — tonnes of waste, head of livestock, hectares, or
population where the factor is per-capita.

The rest of this section is the traps. Each one below cost real time and produced
a wrong model that looked right. They apply equally when the workbook is being
read for comparison against an existing model rather than to build one.

### 3.1 Check whether the earliest year is real

A target baseline is often a back-cast rather than an inventory. Test City's
2007 column — the year all its targets are set against — is every 2008 line
multiplied by 0.983860, the 2007/2008 service-population ratio. It carries no
sectoral information of its own.

Test it: divide each line of the earliest year by the corresponding line of the
next, and see whether you get the same constant every time. If you do, say so
prominently. A model reproducing that year reproduces the next year's structure
at a different level, and the city's target is expressed against a number nobody
measured.

### 3.2 Reverse-engineer the applied factor

**The single most productive technique in the exercise.** Divide reported
emissions by reported activity, per row, and compare the result across categories
and against the workbook's own factor table.

It routinely settles methodology questions that would otherwise need a data
request, and it localises errors to individual cells. What it finds, in rough
order of frequency:

- **The tabulated factor is not the applied factor.** Four instances in Test
  City: natural gas T&D (tabulated 9% below applied), transit electricity
  (14.3% below, and exactly 7/6 of it), transit diesel, and renewable diesel with
  no published factor at all. Carry both — the tabulated series for the forecast,
  a `(derived)` series for the inventory match — and register the discrepancy.
- **A factor set uniform where you expected variation**, or vice versa, telling
  you how the consultant actually weighted a blend.
- **One cell using the wrong fuel's factor**, identifiable because the implied
  factor exactly equals another carrier's.
- **The GWP set**, confirmable to four significant figures from one CH₄ line.
- **A cell that is not what its column says it is.** Test City's 2023
  managed-soil row implies exactly 1.000 t_co2e/acre, because 2022's *emissions*
  figure had been pasted into the activity cell. An implied factor of exactly 1
  is always worth a second look.

### 3.3 One row per applied factor, not per quantity

A ledger row is often **(activity, one applied factor)**, so the same physical
quantity repeats once per factor applied to it. Summing the matching rows — the
obvious way to aggregate — multiplies the quantity.

Test City's waste tonnage appears six times: landfill methane, landfill
transport and landfill stored carbon; compost fugitive, compost transport and
compost soil carbon. Summing gave **twice** the real tonnage on one branch and a
waste sector three times too large.

Read the quantity from **one** designated row and validate that the others agree.
That check is worth keeping: it is what tells you if a future workbook stops
repeating them, or renames one so a stream silently disappears.

### 3.4 A zero is not always a zero, and a blank is not always a blank

Two symmetric traps, both about how the workbook fills cells it has no data for.

**Placeholder zeros.** A forecast workbook spanning 2007–2050 may be populated
only for inventory years and the projection, with a literal `0` everywhere else.
Test City: all 106 activity rows, all 162 emission rows, 35 of 68 factor rows.
Read as real, a factor of zero collapses that year's emissions to nothing *and*
drags the interpolation on both sides of it.

Detect the populated years from the sheet rather than hard-coding them — any year
where some activity row is non-zero — and blank the rest. Keep zeros in populated
years: those are real assertions (a utility genuinely assigned a zero grid factor,
poultry with no enteric fermentation).

**Meaningful blanks.** See §4.3 — this one cannot be fixed by reading alone.

### 3.5 The workbook's structure changes between years

Inventories get refined, and the refinement is rarely backfilled:

- **A category splits.** Test City reported one lumped "other/mixed" waste
  material before 2019 and nine materials from 2019.
- **A process splits.** Livestock had one combined factor per animal until 2022,
  parked in the enteric-fermentation row with the manure row blank, then separate
  factors from 2022. Read naively, the pre-2022 years get 2022's manure factor
  added on top of an already-combined figure.
- **A dimension appears.** Waste tonnage is split residential/commercial for the
  last two inventory years only. A sector-resolved series would cover two of eight
  years and overlap the unsplit one wherever that gets carried forward — so
  aggregate it away unless the split covers enough years to drive something.
- **A row is restated.** The same quantity appears as a per-utility detail row and
  a county summary row, and the labels do not distinguish them. Test City's BAU
  electricity series came out **zero in every year** because the extractor matched
  the first row with the right label, which belonged to a utility outside the
  county — hiding 4.4 Mt/a. Prefer the sheet's own summary rows, and check that
  the leaves sum to the published total for every year.

### 3.6 Group on normalised labels, not the workbook's spellings

Where the workbook spells one category several ways (`Jet Fuel`/`Jet fuel`, two
spellings of an airport), fold them to one label — and fold them **on the way in
too**, in whatever loop groups source rows.

Iterating raw spellings emits one series per spelling, all landing on the same
series identity. That stays invisible for as long as the spellings' years happen
not to overlap, and then fails as a hard duplicate error at upload. Assert that no
two emitted rows share an identity.

### 3.7 Other pitfalls

- **Spreadsheet total rows.** Pivot sheets end with `Grand Total`, often followed
  by the same value in other units. Filter on a per-record key, not on "is this
  cell a number".
- **Two competing estimates of the same sector.** The published total may take the
  *level* from one and the *breakdown* from the other, in which case reported
  emissions cannot be reproduced as activity × EF from any activity figure in the
  workbook. Record both and say so.
- **Granularity floor.** A category that is 90% one bucket cannot support a
  breakdown however much the city wants one. Establish the floor early.

## 4. Shaping the data

### 4.1 Wide or long

A CSV is entirely one or the other — `convert_to_standard_format` decides on
`'Year' in df.columns` for the whole file, before `split_by_dataset` runs. Split
the extraction into two files by a single criterion: **does provenance vary
*within* a series?**

Long, if a source or note is true of some years only — a per-year source file, a
note about one year's data quality, a "held flat from here" note on a factor's
projected years. Wide otherwise, where a series reads as one row.

### 4.2 Units the model cannot fix later

`ensure_unit` converts; it cannot add a dimension. So if head of livestock (a
stock, `pcs`) times a factor must yield an emission *rate*, the annualisation has
to be on the factor — `t_co2e/pcs/a`, emitted that way. Getting this wrong is a
hard failure at compute time, not a silent one, but it costs an upload round-trip.

Rates versus stocks: tree cover *loss* is a rate (`acre/a` against
`t_co2e/acre`); managed soil *area* is a stock (`acre` against `t_co2e/acre/a`).

### 4.3 Write the zeros the model needs

**`Node.get_cleaned_dataset` runs `_add_missing_years`, which back-fills leading
gaps with the first observed value** and carries the last value forward. So a
series the inventory only starts reporting in 2019 gets that value pushed back
across all of history, and nothing warns you.

This was the largest single source of error in the Test City build. Unfixed it
put 205,000 short tons of landfilled paper into 2008, 388 million electric
vehicle-miles into 2008, and 5.0 Mt into 2020 — the last from a single blank
light-duty gasoline row being bridged 2019→2022 across a year that already held
those miles under another vehicle type.

**When the inventory reports a year but not a series, the series is zero for that
year, not unknown.** Confirm against the workbook's own totals, write the zero
explicitly, and attach a comment saying the extractor wrote it and why. Then
assert that every activity series covers every reporting year, because it
regresses invisibly.

The same technique terminates a value that should not propagate. Where a series
exists in one year only and looks like an artifact, keep it where the inventory
put it — deleting a value the source contains hides the error — and write an
explicit zero at the next reporting year so it cannot extend. Make the guard
self-retiring (`if the series still ends at that year`), so the entry lapses when
the source is fixed.

## 5. Building the graph

### 5.1 Mirror the inventory's own reporting structure

Emission sectors should be the sectors the inventory publishes, including ones a
framework model would fold elsewhere — Test City reports refrigerants and
sequestration as sectors in their own right. Then the model's sector breakdown is
directly comparable with the city's table, and an action attaches to a sector
rather than bypassing the breakdown.

### 5.2 One chain per source: activity × factor = emissions

Build the arithmetic as an explicit causal chain even where the workbook computes
it in one cell, so an action can act on either side. Division needs
`formula.FormulaNode` — `GenericNode` has no divide operation. The factor node
needs `tags: [non_additive]` to land in the multiply group when its unit is
compatible with the target's.

Where one quantity carries several factors, that is several emission nodes fed by
the same activity node — not one node with a blended factor. Test City's waste
tonnage feeds four: landfill disposal, compost fugitive, transport, and stored
carbon as a negative. Only all four together give the published figures.

### 5.3 Sources with no activity basis

Some sectors are reported as emissions only — a national inventory scaled by
population, a model output, a reported facility total. Test City: 13% of gross.

Read the reported series directly and say so in the node description. Only an
emissions-level action can act on such a source, and that is a structural fact to
record rather than a gap to fill with an invented activity.

### 5.4 Growth drivers

If the forecast workbook tags each activity with a growth driver, that is the
model's forecast. Attach the driver as an input node with
`tags: [ratio_to_max_hist_year]`: it returns 1.0 for every year up to
`maximum_historical_year`, so it cannot disturb the inventory and shapes only the
forecast — which makes it safe to add to a chain already validated against
history.

Where the driver varies *within* a node — Test City's building energy follows
population for residential demand and employment for commercial and industrial —
it cannot be expressed in the config: a `to_dimensions` edge accepts only one
category. Resolve it per category in the dataset and read it as one dimensioned
driver node. Assert that everything else uses the default driver, so a workbook
that changes one is caught rather than discovered.

### 5.5 Actions from published reduction curves

If the workbook publishes a per-policy reduction series, wire each one as an
action subtracting its curve from the sector it acts on (`multiplier` as a
list-form param with a unit, not an inline scalar). This reproduces the city's
published trajectory exactly and makes the whole policy set visible.

State plainly what it does not do: the actions have **no mechanism**. Toggling a
zero-emission-vehicle standard subtracts a fixed series rather than shifting the
fleet's fuel mix, and the wedges do not interact — switching two on subtracts both
in full even where they act on the same emissions. That is a legitimate first
step, and worth agreeing with the city as such, but it is not the endpoint. Decide
which curves deserve real mechanisms after the model is visible.

Give each action a defined baseline in the last inventory year, writing the zero
if the workbook leaves it blank, so the forecast branches from a known point.

## 6. Validation

### 6.1 In the extractor

Every check below exists because the thing it catches had already gone wrong
silently in the Test City build. That is the standard to aim for: a check per
defect found.

- extracted totals equal the published totals, every year
- `activity × EF` reproduces reported emissions, per source
- component quantities sum to the workbook's own subtotal rows
  (`Total landfill tonnage`, `Total Emissions`, …), every year
- rows that share a quantity agree on it (§3.3)
- every series covers every reporting year (§4.3)
- no two rows share a series identity (§3.6)
- no metric name collides with a dimension column of its own dataset
- every row has a value, a source, and no empty comment fragment

### 6.2 In the model

**Check every year and every sector, not the reference year.** Test City's 2023
matched exactly while five separate defects hid in other years — a 24% error in
2020, and 0.3% errors in four more years that turned out to be three unrelated
causes. Matching the reference year is the weakest possible evidence, because the
reference year is what everyone tunes against.

Then compare against the workbook's forecast, per scenario. The most useful
diagnostic in the whole exercise:

> **A discrepancy that is identical across scenarios has one cause, and it is in
> the baseline. A discrepancy that changes when actions are toggled is in the
> actions.**

Test City's model sits 320,411 t below the published 2030 no-action figure and
320,410 t below the published federal-and-state figure. That the two agree to
1 t is what proved the 25 wedges were exactly right and localised the whole
remaining difference to one cause in the projection — a workbook discontinuity
where the forecast switches to a different fuel-economy basis at the first
projected year, stepping one category up 80% with no change in activity.

Cross-check internal identities too: baseline minus all-actions should equal the
sum of the published reduction curves.

## 7. What to hand back

- **The findings register**, with each entry's consequence.
- **Open questions, ranked by impact on the model** — not by discovery order. The
  city's contact has limited attention; spend it on the answer that moves the
  model most.
- **What is not built, and why**, separated from what is broken. "The model cannot
  compute 2007 because activity data starts in 2008, and 2007 is the target
  baseline" is a finding for the city, not a defect to hide.
- **Where the model knowingly departs from the published numbers**, with the size
  of the departure. A documented 1.15% offset with a named cause is a result; an
  undocumented one is a bug.

---

## Checklist

- [ ] Framework fit assessed; building from workbooks is the right call (§1)
- [ ] Forecast/scenario workbook obtained, not just the inventory (§1)
- [ ] Reporting unit is `t_co2e`; source units kept, additions to `units.py`
      checked for alias damage (§1)
- [ ] Earliest year tested for being a back-cast (§3.1)
- [ ] Implied factors reverse-engineered against the workbook's factor table;
      tabulated-vs-applied differences carried as separate series (§3.2)
- [ ] Ledger rows understood as (activity, applied factor); shared quantities read
      once and cross-checked (§3.3)
- [ ] Placeholder zeros in unpopulated years blanked; real zeros kept (§3.4)
- [ ] Structure changes between years handled: category splits, process splits,
      late dimensions, restated rows (§3.5)
- [ ] Extraction loops group on normalised labels; series identities asserted
      unique (§3.6)
- [ ] Wide/long split decided on whether provenance varies within a series (§4.1)
- [ ] Stock-versus-rate units resolved on the factor side (§4.2)
- [ ] Explicit zeros written for unreported years; coverage asserted (§4.3)
- [ ] Emission sectors mirror the inventory's own reporting sectors (§5.1)
- [ ] Every source is activity × factor where an activity basis exists; sources
      without one documented as emissions-only (§5.2, §5.3)
- [ ] Growth drivers attached with `ratio_to_max_hist_year` (§5.4)
- [ ] Published reduction curves wired as actions, with the no-mechanism
      limitation agreed with the city (§5.5)
- [ ] Extractor validates against published totals and its own invariants (§6.1)
- [ ] Model verified for **every** year and sector, and per scenario (§6.2)
- [ ] Findings register, ranked open questions, and known departures handed back
      (§7)
