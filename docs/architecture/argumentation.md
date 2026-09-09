# Argumentation: helping a coordinator answer objections

*Produced by Claude Opus 5.0 on 2026-09-08.*
*Version 2 produced by Claude Opus 5.0 on 2026-09-09.*
*Responsible: Jouni Tuomisto.*

## 1. The use case

A city sustainability coordinator wants to advance a climate action. In the
council committee, in the local paper, in the department meeting, they meet
objections: it costs too much; our city is too small to matter; climate is not
our business; those are not our emissions; this hurts poor households.

The coordinator's goal is **not to convert the objector.** Reasoning rarely
moves someone whose position is fixed, and a coordinator who visibly tries to
win that duel looks partisan. The goal is to make the objection **stop working
on the third parties in the room** — the undecided councillors, the journalist,
the department head who has not formed a view.

That target changes what the model must produce. Against a committed opponent
you would need a knock-down proof. Against an undecided audience you need
something much more achievable and much more robust:

1. The objection **reconstructed faithfully**, in the objector's own terms, at
   its strongest.
2. An honest statement of **the context in which that reasoning is sound**,
   granted sincerely.
3. The **specific premise that fails here**, with the number or structure that
   shows it.
4. What would have to be true for the objection to hold after all.

An audience that watches an objection get granted its best case and *still* not
survive contact with the model draws its own conclusion. That is far more
durable than being told who is right. It also protects the coordinator: the
same document that refutes weak objections **concedes the strong ones**, which
is what makes the refutations credible rather than promotional.

## 2. Why a systematic taxonomy, and where it comes from

The first draft of this design classified objections into four types
(empirical / aggregation / jurisdictional / value) derived from three examples
that came up in conversation. That is not a taxonomy, it is a sample. This
section replaces it with a classification taken from the argumentation
literature, which turns out to disentangle **three independent axes** that the
four-type list had collapsed into one.

Keeping them separate matters practically: each axis answers a different
question for the coordinator.

### Axis A — Locus of disagreement: *what is actually in dispute*

Classical **stasis theory** (Hermagoras, transmitted through Cicero's *De
Inventione*; the modern counterpart is the "difference of opinion" in
pragma-dialectics) holds that any dispute settles at one of four points. This
axis determines **whether the model can speak to the objection at all**, so it
is the primary classification in the register.

| Stasis | Question in dispute | Can the model speak? |
|---|---|---|
| **Conjecture** (existence/magnitude) | Is the effect real? How large? | Yes — this is what the model is for |
| **Definition** (boundary/category) | What counts as *ours*? | Yes — inventory scope and accounting convention |
| **Quality** (evaluation) | Is it good? Worth the cost? | Partly — it can price and rank, not adjudicate |
| **Place** (*translatio*: forum/authority) | Is this our decision to make? | Weakly — it can show which instruments the council holds |

Misidentifying the locus is the most expensive mistake available, and the repo
already contains an instance of it. `OBJ-G1: Drop in the ocean` in
`configs/modules/transportation/congestion_charge.yaml` is typed as an
"Empirical objection about scale and impact". It is not. Its scale premise —
that one city is a rounding error on the global total — is **true**, and it is
the one part of the objection a coordinator cannot win. The error lies in the
inference from a small share to no obligation, which is a part–whole error. A
coordinator sent to dispute the premise loses; one sent to dispute the
inference wins easily. Same objection, opposite outcomes, decided entirely by
the classification.

### Axis B — Inference scheme: *how the reasoning is built, and therefore how it fails*

Walton, Reed & Macagno (2008), *Argumentation Schemes*, catalogues roughly
sixty presumptive reasoning patterns, and — this is the part that makes it
directly usable here — attaches to each one a set of **critical questions**:
the standard probes that a given scheme must survive. Those critical questions
are exactly the "where does this hold / what fails here" content the register
needs. We do not have to invent that field; for most objections it can be
filled by instantiating the scheme's published critical questions against the
climate context.

Schemes that recur in municipal climate deliberation:

| Scheme | Typical wording | The critical question that usually decides it |
|---|---|---|
| **Part–whole / division** | "our share is negligible" | Does the property of the whole distribute over the parts? Do parts aggregate linearly? |
| **Practical reasoning (means–goal)** | "this is the wrong way to get there" | Is there a better means? Are the side effects accounted? |
| **Argument from consequences** | "it will damage the local economy" | How likely is the consequence? Is the magnitude quantified or asserted? |
| **Popular practice** | "no comparable city is doing this" | Is the practice actually general? Does generality make it right? |
| **Analogy / precedent** | "city X tried it and it failed" | Are the cases alike in the respect that matters? |
| **Expert opinion** | "our engineers say it cannot be done" | Is the expert in the right field? Is the claim within it? |
| **Sunk cost / waste** | "we just rebuilt that road" | Does past expenditure bear on future choice? |
| **Slippery slope** | "charges now, car bans later" | Is there an actual mechanism linking the steps? |

### Axis C — Delay function: *what the objection is doing rhetorically*

Lamb et al. (2020), "Discourses of climate delay", is an empirical taxonomy of
the arguments that accept climate science and still counsel inaction. It is the
closest published match to a city coordinator's actual inbox, and it is the
axis that tells them **which tactic to use**. Four families:

- **Redirect responsibility** — whataboutism, the free-rider excuse,
  individualism. ("Our city is too small"; "China emits more".)
- **Push non-transformative solutions** — technological optimism,
  all-of-the-above, no sticks only carrots. ("Wait for better technology.")
- **Emphasise the downsides** — policy perfectionism, appeal to the well-being
  of the poor, appeal to social justice. ("This is regressive.")
- **Surrender** — change is impossible, doomism. ("Too late anyway.")

Two features make this axis worth carrying. First, the *appeal to social
justice* family is the one where a coordinator most often gets a genuinely
serious objection dressed in delay clothing, and cannot tell the difference
without doing the distributional arithmetic — which is precisely what
[`value_profiles.md`](value_profiles.md) already specifies. Second, naming the
family lets the register be cross-checked against a published catalogue rather
than against our own imagination, which is how we find out what we have
missed.

Coan et al. (2021) provides the sibling taxonomy for outright contrarian
claims (five super-claims, twenty-seven sub-claims: warming is not happening,
humans are not the cause, impacts are not bad, solutions will not work,
science is unreliable). A municipal coordinator mostly does not face these —
the argument in a council chamber is nearly always about *delay*, not
*denial* — so the register scopes to Lamb and cites Coan as the boundary.

### Axis D (not a taxonomy) — model role

Cutting across all three: what the model can actually contribute. Four values,
and the fourth is load-bearing.

- **`refutable`** — the model computes something that defeats the inference.
- **`boundable`** — the objection is legitimate; the model computes its range,
  and the case is argued on that range. (`OBJ-CC13` displacement already does
  this correctly, via the `hypotheses` dimension.)
- **`value_only`** — no computation can settle it. The model can show what a
  given value weighting implies, and then stop.
- **`out_of_scope`** — the model has nothing to say. Recording this is not a
  failure; a register that claims an answer to everything is not believed
  about anything.

## 3. The register schema

The fields below are a **Toulmin layout** (Toulmin 1958): *claim* and *grounds*
are the objector's, the *warrant* is the inference they license, *backing* is
the context that licenses the warrant, and *rebuttal* is the condition under
which it fails. Naming it explicitly keeps the fields from drifting into
free-form advocacy.

| Field | Content | Why it is there |
|---|---|---|
| **Claim** | The conclusion, in the objector's words | |
| **Grounds** | The facts they rely on — **steelmanned** | A straw man is noticed by the audience, and the coordinator then loses on credibility rather than substance. This field deserves the most care of any in the register. |
| **Warrant** | The inference rule that gets from grounds to claim, stated as a general rule | Isolating the rule as its own object is what makes the reductio in §4 possible |
| **Locus / Scheme / Delay family** | Axes A, B, C | Determines model role and tactic |
| **Where this holds** | A class of problems where the warrant is sound, with an example the objector would accept | Granted sincerely. This is the field that persuades third parties, and the one the first draft of the schema lacked entirely. |
| **What fails here** | The specific premise that breaks, and the node or number showing it | |
| **Consistency test** | What else follows if the warrant is kept and applied generally | §4 |
| **What would vindicate it** | The falsifiable version | Converts "you are wrong" into "here is what you would need to show". Either it becomes a modelled hypothesis, or it cannot — and that itself is the finding. |
| **Model role** | Axis D | Tells the coordinator whether to reach for a number at all |

## 4. The reductio pattern

For objections whose locus is **conjecture** but whose scheme is **part–whole**
— the "too small to matter" family — the effective move is not to dispute the
premise. Concede it entirely, and instead apply the objector's own warrant
consistently.

The warrant behind "our city is too small" is a general rule: *an actor may
disregard its emissions when its share of the global total falls below some
threshold.* Let the objector set the threshold. Then compute how much of world
emissions that rule exempts.

Because the objector chose "city" as the unit of analysis, and because cities
collectively account for the large majority of global emissions while almost
every individual city falls below any plausible threshold, the rule exempts
substantially everything. The conclusion is not that the objector is wrong
about their city. It is that the rule they used cannot be held by everyone who
is equally entitled to hold it.

Two properties recommend this over the counter-assertion currently in the
module (`ARG-18: Cities represent significant emissions`, which states the
70–80% figure in prose and computes nothing):

- **The objector supplies the disputed number**, so there is nothing to argue
  about. Any threshold they are willing to defend in public produces the
  result.
- **It concedes the premise**, which removes the coordinator's weakest ground
  from the field before the exchange starts.

Why the warrant fails specifically for greenhouse gases: the well-mixed
long-lived gases have effects that are, over the relevant range, close to
linear in emitted mass and additive across sources, and the resulting warming
tracks *cumulative* emissions. There is no threshold below which a tonne stops
contributing. The warrant is sound for threshold-type harms — a single vehicle
on an empty rural road genuinely does not affect local air quality, because
that harm has a concentration threshold and does not accumulate — and it is
this disanalogy, not the size of the city, that decides the objection.

This is the same physical premise as the equal-tonne argument: **a tonne's
effect does not depend on who emitted it or where.** In the model that premise
is carried not by a node but by the *absence* of one — the city's emissions
enter the global total with a coefficient of exactly one, and there is no
parameter anywhere on that path. One premise, two rhetorical uses: it defeats
"we are too small" and it grounds "no one is exempt".

### 4.1 What the sweep shows: the objection has no defensible threshold

Sweeping `negligibility_threshold` over `equalia` (which emits 0.0088% of the
global total) gives the following. `blameless units` is
`exempt_partition_count`; `we exempt?` is whether this city itself falls below
the line it just drew.

| threshold | blameless units | largest exempt emitter (Gt/a) | we exempt? |
|---|---|---|---|
| 5% | 20 | 2.65 | yes |
| 1% | 100 | 0.53 | yes |
| 0.1% | 1 000 | 0.053 | yes |
| 0.02% | 5 000 | 0.0106 | yes |
| 0.01% | 10 000 | 0.0053 | yes |
| 0.005% | 20 000 | 0.0027 | **no** |
| 0.001% | 100 000 | 0.0005 | **no** |

The objection is caught in a pincer, and **the crossover sits exactly at the
city's own share**:

- Set the threshold high enough that this city is exempt, and the world divides
  into at most a few thousand individually blameless units which together are
  all of global emissions.
- Set it low enough to avoid that, and this city is no longer exempt — so the
  objection no longer applies to the actor making it.

There is no value of the threshold at which the objection both applies to this
city and does not license everyone. That is a stronger result than the
consistency test alone, it was not visible before the model was run, and it
follows from a single number the objector supplies.

The temporal twin is worth noting, because it is the same error rotated:
**surrender / doomism** ("it is too late, so why bother") applies a
threshold rule to *time* rather than to *actors*, and fails for the same
reason — cumulative emissions have no point past which an additional tonne
stops mattering.

## 5. What the model must not attempt

The **`value_only`** cases are where this apparatus can do real damage if
overreached. "Climate is not our business, we have other priorities" contains
two separable claims:

- A **jurisdictional** claim (locus: place) — not our mandate. Factual, and
  checkable: the actions in a Paths model *are* the council's instruments, so
  the summed impact of all available actions against `net_emissions` yields how
  much of the inventory the council can actually move. That is an answer from
  the model's own structure rather than a claim about statute.
- A **value** claim (locus: quality) — we care about other things more. Not
  refutable, and the model should not pretend otherwise.

The correct treatment of the second is to let it be held: set
`value_stewardship` to zero, and show the priority ranking that results.
"Here is the plan implied by placing no weight on future generations — adopt it
if you are willing to defend it" is stronger than any refutation, and it is
honest. A coordinator who appears to hold a machine that proves opponents
immoral loses the room. This is caveat **C6** of
[`value_profiles.md`](value_profiles.md) — the tool informs deliberation, it
does not replace it — and the register is where that caveat is either kept or
broken.

## 6. Implementation in the platform

Argument nodes are an existing pattern: `quantity: argument`, no numeric
contribution, wired to the computational node they bear on.

**Argument nodes are excluded from arithmetic structurally.** A node whose
quantity is `argument` is dropped from every arithmetic operation before any
unit or dimension test runs, so an objection can be attached directly to the
node whose number it disputes, whatever dimensions that node carries, with no
tags of any kind. The rule lives in `nodes.operands.is_non_computational`.

This replaced a mechanism that neutralised argument nodes *by arithmetic*: the
`ignore_content` tag rewrote the input frame to the target's no-effect value and
then let it be summed, which is a no-op only because adding zero is. Addition
was doing work semantically unrelated to it, and it leaked in two directions.
The frame still had to satisfy the target's dimension check, so an objection
could not be attached to a dimensioned node at all — and when it was, the
failure landed on the *target*, taking `net_emissions` and everything
downstream with it. And every operation that was not addition needed its own
patch: `FormulaNode` carried a `quantity == 'argument'` special case, the
`GenericNode` family checked the tag in `resolve_input_nodes`, and the
port-binding path checked nothing, which is where the dimension error came from.

The structural rule replaces all of that. Where it is enforced:

| Path | Chokepoint |
|---|---|
| `GenericNode` family | `operands.resolve_input_nodes` |
| `AdditiveNode` / `MultiplicativeNode` port bindings | `Node.iter_computational_input_bindings` |
| Legacy helpers | `Node._add_nodes_impl`, `multiply_nodes_pl`, `impute_nodes_pl` |

The split that keeps this honest: **arithmetic paths resolve inputs through
`iter_computational_input_bindings`; structure-facing callers keep using
`iter_input_bindings` and see everything.** The argument edge stays in the
graph, the editor and the explanations — it is excluded from computation, not
from the model.

Consequences worth knowing:

- `ignore_content` is no longer needed on an argument edge, and the tag retains
  its old meaning for non-argument inputs, which was left alone deliberately.
- Removing a zero from a sum changes floating-point associativity. Across
  `equalia`'s 108 nodes exactly one value moved, in the sixteenth significant
  figure (`1162845307.764707` to `...764706`); shapes, dimensions, year ranges
  and null counts were identical. Expect last-digit churn, not real change.

One remaining deviation from the `congestion_charge.yaml` precedent, and one
kept:

- **`generic.ConstantNode` with `constant: 0`, not `generic.GenericNode` with
  `input_datasets: [transportation/zeros]`.** The older argument nodes pull a
  DVC dataset purely to have an input; a constant needs none, which keeps the
  module portable.
- **Objection parameters are `simple.ValueAction`, not node parameters.** A node
  parameter marked `is_customizable` does not reach a site visitor; only action
  parameters render in the public UI.
- **Never let an action's output be a divisor.** `ValueAction.no_effect_value` is
  0 and the baseline scenario is every action's off state, so division by it
  yields `Inf` in BAU and `-Inf` in Impact. Clamp in a dedicated node and bound
  the parameter above zero. See §7.

The catalogue lives in `configs/modules/moral_argumentation/objections.yaml`
and is included per instance. Objections are **documented, never toggled** —
there is no switch that turns an objection off, in the same spirit as the error
register: the register's value is that it records the state of the argument,
and a register that can be silenced records nothing.

## 7. Computational properties of the argument types

This is a **fourth axis**, independent of the three in §2, and it is the one
that predicts implementation cost. Classifying an objection by what the model
must be able to *do* to answer it turns out to sort the register quite
differently from any of the literature axes.

| Argument type | What the model needs | New data? | Output type |
|---|---|---|---|
| Part–whole ("too small") | one threshold parameter, one scalar total | none | a **count**, constant in time |
| Definitional ("not ours") | the same inventory re-aggregated under two conventions | none | two totals |
| Timing / doomism | a **cumulative** operator over the timeline | none | integrated stock |
| Distributive ("regressive") | a stakeholder/income dimension on *cost and benefit* nodes | none, but heavy plumbing | distribution + floor test |
| Value ("not our business") | a parameter sweep re-running the whole graph | none | a **ranking** (permutation) |
| Empirical (displacement, leakage) | a hypotheses dimension and a data range | yes | bounded time series |

Four observations, all of which emerged from building it rather than from
planning it:

**The most conclusive argument is the cheapest.** The part–whole reductio needs
no data at all — not a distribution of emitters, not an inventory, nothing
beyond a threshold the objector sets and a global total for context. This is
because the result is a property of the *warrant*, not an empirical claim about
the world. My first draft made it depend on a global emitter distribution, which
was a design error: it made the strongest argument in the register the only one
that could not be computed. Arguments whose force is structural should be
implemented structurally.

**Its output is not a time series, and Paths is a time-series engine.**
`exempt_partition_count` is 1 000 in every year from 2010 to 2050. The entire
temporal machinery — forecast flags, interpolation, scenario years — contributes
nothing, and the node's graph is a flat line whose only informative feature is
its height. Several argument types share this: what is wanted is a scalar, a
count, or a ranking, and the platform's native output shape fits none of them
well. The `value_only` case is the sharpest: its answer is a *permutation* of
the action list, for which there is no native representation at all
(`impact_overview` is the closest thing).

**The expensive one is the one that matters most politically.** "This is
regressive" is the objection a coordinator is least able to wave away and most
likely to face, and it is the only row in the table needing structural work: a
stakeholder or income dimension carried through the cost and benefit nodes.
`value_profiles.md` specifies the welfare functions that consume such a
dimension, but nothing in the repo produces one. That gap — not the missing
emitter data — is the real blocker on the register being useful in a committee
room.

**An action must never supply a divisor.** This is the sharpest form of the
"not a time series" problem, and it shipped as a live bug. `ValueAction` has
`no_effect_value = 0.0` hardcoded, and the baseline scenario *is* the off state
of every action, so an action-supplied parameter is guaranteed to be zero in
BAU. `exempt_partition_count` divides by it: BAU read `Inf`, and Impact —
computed against that baseline — read `-Inf`. Neither appears in a terminal run,
because `load_nodes.py` computes the default scenario unless asked otherwise;
both appear immediately in the UI, which always shows BAU and Impact alongside.

The fix is `negligibility_threshold_effective`, which clamps the slider to a
small positive floor, so all four downstream nodes are finite in both scenarios.
The action's `min_value` is also lifted off zero, so an *enabled* slider cannot
reach the singularity either — the clamp handles the off state, the bound
handles the on state, and both are needed.

But the clamp only removes the infinity; it does not make the reading meaningful.
**There is no business-as-usual value of a rhetorical premise.** The negligibility
rule is either invoked or it is not, so the counterfactual that BAU and Impact
are computed against does not exist, and those two columns should be ignored on
every node downstream of the threshold. This is the category error in §7's second
observation showing up as a number on a screen: a scenario engine was asked to
hold a premise, and it answered with a difference against a baseline that has no
interpretation. The node descriptions say so; the platform has no way to suppress
the columns.

Two general rules follow, and they are worth applying beyond this module:

1. If a user-settable quantity is a divisor, clamp it in a node of its own, and
   bound the parameter away from zero as well.
2. Test every argument node in the **baseline** scenario, not only the default.
   The terminal is the wrong place to look for this class of bug.

**Sweeping a node-local parameter is not possible from the CLI.**
`load_nodes.py --param` resolves only against `context.global_parameters`
(`src/nodes/context.py:582`), so an action's `constant` cannot be swept with it
even though `--list-params` prints it. The sweep in §4.1 was produced by a
script that loads the config through `InstanceLoader.from_yaml` and calls
`get_parameter('constant').set(t)` per run. Since the whole point of these nodes
is that the objector moves the slider, a sweep is the natural way to study them,
and this is a tooling gap worth closing.

## 8. Objection catalogue

Classification for the objections a municipal coordinator actually meets. Rows
marked ● are implemented in `objections.yaml`; the rest are the backlog, listed
so that the gaps are visible rather than forgotten.

| | Objection | Locus | Scheme | Delay family | Model role |
|---|---|---|---|---|---|
| ● | Too small to matter ("drop in the ocean") | conjecture | part–whole | redirect responsibility | refutable |
| ● | Not our emissions (scope / ownership / consumption) | definition | — | redirect responsibility | refutable |
| ● | Regressive — hurts poor households | quality | consequences | emphasise downsides | boundable |
| ● | Climate is not our business | place + quality | practical reasoning | redirect responsibility | value_only (+ refutable part) |
| | Costs too much / opportunity cost | quality | practical reasoning | emphasise downsides | boundable |
| | Cancelled by the ETS cap (waterbed effect) | conjecture | consequences | redirect responsibility | boundable |
| | Wait for better technology | quality | practical reasoning | non-transformative | boundable |
| | Emissions just relocate (leakage) | conjecture | consequences | redirect responsibility | boundable |
| | This measure is flawed, so no measure | quality | practical reasoning | emphasise downsides | refutable (structural) |
| | Too late anyway (doomism) | conjecture | part–whole (in time) | surrender | refutable |
| | Nobody voted for this | place | — | — | out_of_scope |
| | The modelling is biased | — | expert opinion | — | out_of_scope |
| | City X tried it and it failed | conjecture | analogy | non-transformative | boundable |
| | We just built it (stranded asset) | quality | sunk cost | emphasise downsides | refutable |
| | Starts here, ends in a car ban | conjecture | slippery slope | emphasise downsides | out_of_scope |

Three of these deserve a note, because they are the ones the first draft of
this design missed and they are the ones with the most machinery already
waiting for them:

- **"Not our emissions"** is probably the single most common objection in city
  inventory work, and the platform is saturated with the relevant apparatus —
  BISKO, the GPC scopes, and
  [`matching-a-model-to-an-inventory.md`](../matching-a-model-to-an-inventory.md).
  It is a *definitional* dispute, which means it is settled by naming the
  accounting convention and showing the same total under both conventions, not
  by arguing about magnitude.
- **"Regressive"** already exists in the repo as `OBJ-CC1`, and
  [`value_profiles.md`](value_profiles.md)'s floor-constrained and Atkinson
  nodes are built precisely to answer it — currently with no objection wired to
  them. This is the largest unexploited connection in the codebase: a
  coordinator facing "this hurts poor households" can answer with a
  floor-constrained welfare computation and a stated floor, rather than a
  reassurance.
- **"Cancelled by the ETS cap"** is technically serious and sometimes correct.
  Granting it where it holds is what buys the register the credibility that
  makes the refutations land.

## 9. Open items

**Resolved by building and running it** (recorded because the first draft of
this file asserted otherwise):

- `quantity: argument` nodes render normally everywhere in the UI. Confirmed.
- The reductio formula computes. An earlier draft specified a `sum_where()`
  function that does not exist in the platform at all; the version in the module
  uses only `sum_dim` and dataframe comparison, both real.
- The module is included in `configs/equalia.yaml` and all eleven of its nodes
  compute, with `net_emissions` and `total_utility` unchanged.
- `FixedDataset` / `historical_values` is deprecated and is not an option for
  inlining data; it is also year-indexed only, so it could not have carried a
  dimensioned distribution anyway.

- Argument nodes no longer need `ignore_content`, and no longer have to match
  the target's dimensions: exclusion from arithmetic is structural (§6).
  `obj_not_our_emissions` is now attached to `net_emissions` itself, which is
  the number it disputes. Verified against `equalia` at HEAD: 108 node outputs,
  no change beyond one sixteenth-significant-figure float difference.

**Still open:**

- **An emitter distribution, if the empirical strengthening is wanted.** The
  structural reductio in §4 needs none, so this is now an enhancement rather
  than a blocker. What it would add: "what share of world emissions is actually
  held by actors below the threshold", alongside the structural "what share
  *could* be". Candidate sources are EDGAR (JRC) or the Global Carbon Project
  for countries, and Climate TRACE or the CDP–ICLEI unified reporting dataset
  for cities. **City-level data is what the argument actually needs**, because
  the objection chose the city as its unit of analysis; a country-level table
  would answer a different reductio than the one at issue. No figures have been
  invented in the meantime.
- **`global_total_emissions` carries an unsourced default** (53 Gt CO2e/a) as a
  user-settable constant, with its label saying so. It needs a citation before
  the module is shown to anyone outside the team, since it is the denominator of
  `our_share_of_global`.
- **BAU and Impact are meaningless on the reductio nodes**, and the platform
  cannot suppress them per node (`baseline_visible_in_graphs` is instance-wide).
  The descriptions warn the reader instead, which is weaker than not showing a
  number that has no interpretation.
- **`ignore_content` still zeroes rather than drops for non-argument inputs**,
  and for a *factor* that means the product becomes zero rather than being left
  unchanged. `resolve_input_nodes` drops such an input while `_add_nodes_impl`
  zeroes it, so the two paths already disagree. Out of scope here — argument
  nodes no longer depend on the tag — but it is a live inconsistency.

  The redundant half of the tag's use has now been removed: of the 70 edges
  carrying `ignore_content`, the **31 whose source was a `quantity: argument`
  node** were deleted (`congestion_charge.yaml` 21, `forestry-fi.yaml` 8,
  `finland-syke.yaml` 2), verified as a no-op against `dinspec`, `equalia`,
  `finland-syke` and `forestry-fi` — 532 node outputs, no schema, row-count or
  value difference, and no new failures.

  The **39 remaining uses are a different mechanism wearing the same tag**:
  "show this edge in the graph but contribute nothing", on sources carrying real
  quantities that `quantity: argument` cannot absorb — `budget.yaml`'s
  `collect_*` nodes (deleted), `dut_transport_actions.yaml`'s `sink_node` (deleted), and the
  forestry utility chains (`forestry/greentransition.yaml` 9,
  `greentransition.yaml` 8, `dut_transport_actions.yaml` 7, `budget.yaml` 6,
  `forestry/economy.yaml` 4, `forestry-fi.yaml` 4,
  `dinspec/car_to_bike_shift.yaml` 1). Retiring the tag altogether means
  deciding what those become — the leading candidate is an edge-scoped
  non-computational role, dropped at `resolve_input_nodes` /
  `iter_computational_input_bindings` the way `is_non_computational` drops a
  node, which would also retire `_ignore_content`'s dependency on the target's
  output metric and its `FIXME`.
- **No CLI sweep for node-local parameters** (§7).
- **The distributive dimension does not exist.** No cost or benefit node in the
  repo is disaggregated by stakeholder or income, so `obj_regressive_burden`
  currently has a description and no computation behind it. This is the largest
  gap between the register as designed and the register as usable.
- A **briefing document** view of the register may serve a coordinator better
  than node pages; deferred, since the immediate interest is in how the
  arguments behave computationally.

## 10. References

### Argumentation theory

Toulmin, S. E. (1958). *The Uses of Argument*. Cambridge University Press.
ISBN: 978-0521534833. https://www.cambridge.org/9780521534833

Walton, D., Reed, C., & Macagno, F. (2008). *Argumentation Schemes*.
Cambridge University Press. ISBN: 978-0521723749.
https://www.cambridge.org/9780521723749

van Eemeren, F. H., & Grootendorst, R. (2004). *A Systematic Theory of
Argumentation: The Pragma-Dialectical Approach*. Cambridge University Press.
ISBN: 978-0521830751. https://www.cambridge.org/9780521830751

Perelman, C., & Olbrechts-Tyteca, L. (1969). *The New Rhetoric: A Treatise on
Argumentation*. University of Notre Dame Press. ISBN: 978-0268004460.

Fairclough, I., & Fairclough, N. (2012). *Political Discourse Analysis: A
Method for Advanced Students*. Routledge. ISBN: 978-0415499231.

### Climate argument taxonomies

Lamb, W. F., Mattioli, G., Levi, S., Roberts, J. T., Capstick, S., Creutzig, F.,
Minx, J. C., Müller-Hansen, F., Culhane, T., & Steinberger, J. K. (2020).
Discourses of climate delay. *Global Sustainability*, 3, e17.
https://doi.org/10.1017/sus.2020.13

Coan, T. G., Boussalis, C., Cook, J., & Nanko, M. O. (2021).
Computer-assisted classification of contrarian claims about climate change.
*Scientific Reports*, 11, 22320. https://doi.org/10.1038/s41598-021-01714-4

### In this repository

- [`value_profiles.md`](value_profiles.md) — welfare and moral scoring functions
- [`../../configs/modules/moral_argumentation/README.md`](../../configs/modules/moral_argumentation/README.md) — values, duties, evaluation nodes
- [`../matching-a-model-to-an-inventory.md`](../matching-a-model-to-an-inventory.md) — inventory scope and accounting conventions
