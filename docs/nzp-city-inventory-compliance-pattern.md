# Reconciling a City's Own Emissions Inventory Against the NZP Model

**Audience:** NetZero Consortium — a reusable pattern for any city that needs its
NZP-based model to reproduce a locally-commissioned emissions inventory (a BEI,
a consultant's audit, a national statistics office release) while retaining full
NZP action mechanics.

**Origin:** developed while reconciling Cork's KPMG building emissions inventory
(548 kt/a, 2023) against the NZC model's Arup/SEAI-derived estimate (418 kt/a) —
a 31% gap that turned out to be almost entirely methodological, not missing
emissions.

**Design goal:** every step below is expressed as datasets, dataset overrides, or
node wiring inside the *city's own config file*, using operations already present
in the shared model (`nzc.yaml` or equivalent). None of it requires adding
dimensions, node types, or shared chains to the base model. This is what lets a
city bring its own inventory without forking the model the consortium maintains.

---

## Step 1 — Decompose the gap before touching anything

A gap between a city's inventory and the model's estimate is rarely one thing. Treat
it as a waterfall with named components, and classify each component into exactly
one of three buckets before deciding on a fix:

| Bucket | What it means | Example (Cork) |
|---|---|---|
| **A. Emission factor (EF) difference** | Same activity, different factor applied to it (fuel EF vintage, grid EF, GWP set) | Grid electricity EF: KPMG 254 g/kWh vs model's SEAI figure |
| **B. Activity/demand difference** | Same physical scope, different estimation method for the quantity | KPMG: floor area × fixed benchmark constants; model: national statistics scaled to city |
| **C. Boundary/scope difference** | The two inventories don't cover the same thing | Industrial process energy, public lighting, public buildings counted once/twice/never |

Diagnostic shortcut: compare total activity (energy) and implied average EF
(emissions ÷ energy) separately. If the energy ratio between city inventory and
model explains most of the gap, it's mostly bucket B; if the EF ratio does, it's
mostly bucket A.

**Do not** resolve an undiagnosed gap with a single blanket scaling factor. A city
inventory's demand estimate is often benchmark-synthetic (floor area × generic
per-m² factors) rather than measured; pinning the model's *levels* to it without
this diagnosis silently inflates every action's absolute impact and every
downstream cost/capacity number by the same margin, and makes the disagreement
undiscoverable later.

## Step 2 — Route each bucket to its own mechanism

**Bucket A (EF difference) → dataset override.**
Replace the relevant EF dataset(s) in the city config (`allow_override: true`).
This is the only bucket where a straight swap is correct on its own — it moves
both the inventory match and the action impacts coherently, since actions
computed against a wrong EF were wrong for the same reason the inventory was.

**Bucket B (activity difference) → observed-anchor + baseline-ratio correction.**
This is the pattern already present in the base model (see box below). Three
nodes:
1. A historical node reading the city's own activity statistic.
2. A ratio node: `city_statistic / model_output_at_baseline_scenario`, extended
   across the forecast so it's a constant multiplier, not just a historical
   pin. Computing the denominator specifically at the **baseline scenario** is
   the load-bearing detail — it means the ratio is fixed regardless of which
   actions are switched on, so action deltas pass through undivided in every
   forecast year.
3. A correction node that multiplies the model's native activity by that ratio,
   gated by a boolean parameter so the correction can be switched off for
   debugging or for audiences who want the uncorrected model view.

**Bucket C (boundary difference) → an explicit named residual.**
Where the city inventory includes something the model genuinely does not
represent (or vice versa), add a separate node for exactly that quantity,
feeding the relevant total with `inventory_only, extend_values` (pinned to the
known historical figure, held flat or trended explicitly — never silently
absorbed into a scaling factor). This keeps the boundary difference visible and
auditable rather than hidden inside a corrected total.

> **Reference implementation in the base model:** `nzc.yaml`'s electricity
> chain already does this for bucket B —
> `electricity_consumption_historical` (city statistic) →
> `electricity_consumption_ratio` (`extend_all(inventory_only(observed)) /
> output_with_scenario(modelled, 'baseline')`) → applied to
> `total_electricity_consumption_uncorr` via unit-based auto-multiplication
> (a dimensionless ratio is unit-incompatible with an energy quantity, so
> `GenericNode` routes it into the multiply group automatically — no explicit
> tag required) → `total_electricity_consumption_corr`, gated by a
> `statistical_correction` boolean parameter. New instances of this pattern
> (e.g. for buildings) should copy this structure, with their own boolean
> parameter if independent toggling is wanted.

## Step 3 — Keep the inventory-facing view and the model-facing view separate

A city typically needs two things that are easy to conflate:

- **An inventory-reproduction view**: numbers that match the commissioned
  inventory exactly, broken down the way that inventory presents them (e.g. by
  building type), for reporting continuity.
- **A model-facing basis**: the quantity that actions, costs, and capacity
  calculations actually run on.

After Step 2, these can be the same thing (the corrected/anchored node) if the
city wants full inventory compliance to flow through to the whole model — that
was Cork's requirement. If a city instead wants provenance-preserving
*comparison* (see two-hypothesis design note below), keep them as two separate
outcome nodes rather than introducing a shared "which-basis" dimension into the
model's core chain. Two nodes displayed side by side achieves visual comparison
with no propagation cost to the rest of the model or to other cities; a shared
dimension threaded through a chain does not resolve until some node collapses
it, and every node it passes through must carry it, in every city that inherits
that chain — including cities with no need for the comparison at all.

## Step 4 — Check the resulting number against the inventory's own internal logic

Before presenting a corrected figure, verify the city's own inventory dataset is
internally consistent: row/column sums reconcile, `emissions = activity × EF`
holds per category, and any single-value average (e.g. a flat EF applied across
several sub-categories) is understood as *derived from* a weighting scheme (e.g.
a certificate-count fuel-share survey) rather than itself a primary measurement.
Reverse-engineering a supplied EF or share by checking whether it reproduces a
known formula is a fast way to both validate the data and answer open
methodology questions (in Cork's case, this is how the applied grid EF and the
county-vs-CIBSE fuel-split question were both resolved without needing a new
data request).

## Checklist for a new city adopting this pattern

- [ ] Decompose the reported gap into EF / activity / boundary components; get
      the two figures (city inventory total, model total) down to the same
      activity × EF structure before deciding on a fix.
- [ ] For EF differences: override the relevant dataset(s) in the city config.
- [ ] For activity differences: build historical → ratio (at baseline
      scenario) → correction, gated by a boolean parameter.
- [ ] For boundary differences: add a named residual node, not a scaling
      adjustment.
- [ ] Confirm no shared dimension or node type needed to be added to the base
      model; if one genuinely is, raise that explicitly as a consortium-level
      ask rather than forking the shared model silently.
- [ ] Validate the city's own inventory dataset for internal consistency before
      treating it as ground truth (sum checks, implied-factor checks,
      plausibility against independent metered data where available).
- [ ] Decide, and document, which basis (city-inventory-anchored or
      model-native) feeds the goals/targets and any cross-city comparison —
      this is a policy decision for the city, not something a correction factor
      should decide implicitly.
