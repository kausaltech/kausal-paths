# Objections to the Fair-Share Argument

*Produced by Claude Opus 5.5 on 2026-10-07.*
*Responsible: Jouni Tuomisto.*

The carbon budget (`README.md`) answers *how much* a city owes. A city that presents it will hear
objections that ask something else: whether a local tonne changes anything, whose job it is,
whether the duty holds while others do nothing, and whether the city can afford it. This
document takes them seriously in the same way the budget does. For each objection it states the
**strongest form** an honest sceptic would defend, finds a **premise most people accept** that
answers it, and makes whatever is genuinely disputed a **setting**, so that a reader who rejects
the default can see what follows from their own reading.

It is a deduction, not a weighing. `moral_argumentation/value_weights.yaml` weighs values
against each other; here, each objection is granted as far as it can be defended, and the model
shows what is still owed.

## The claim under test

The claim is: *the city should cut at least what its scenario cuts against Business as Usual.*
The model calls that cut `fair_share_planned_reduction` (per year) and
`fair_share_planned_reduction_cumulative` (summed from 2020). An objection can attack the claim
in three ways:

| Kind | Example | What it changes |
|---|---|---|
| It lowers what is owed | "It's the federal government's job" | The duty |
| It lowers what a local tonne is worth | "Too small to matter", "the cap makes it pointless" | The effect |
| It makes the duty conditional | "Only if others act" | Whether the duty applies |

**The fair-share leg is a ratio.** The cut the budget requires against Business as Usual is
*BAU − budget*; the scenario delivers *BAU − scenario*. The first is at least the second exactly
when the scenario still uses up or overdraws the budget, i.e. when
`fair_share_city_remaining_budget` is not positive in that scenario. So "the planned cut is owed
in full" and "the scenario does not leave the city within its share" are the same statement.
`fair_share_planned_reduction_owed_share` states it as the part of the planned cut that is owed,
capped at 100 %, which a dashboard card can draw against a 100 % goal. At
generous readings of the premises a scenario can cut more than fairness requires, and the rest of
its cut has to be justified by another argument; objection 1 supplies one that does not depend
on the budget at all.

## Three legs, any one of which suffices

Most objections are not about fairness, so the budget alone cannot meet them. The module rests
the claim on three independent arguments, each from its own premise:

1. **Fairness** (the budget, `carbon_budget.yaml`) answers *how much*.
2. **Avoiding harm** (objection 1 below) answers *why at all, whatever others do*.
3. **Self-interest** (energy costs, CO₂ price, health) answers *why it pays*. Not built in this
   module; an instance such as `mainz-bisko` already has `energy_spending` and
   `co2_price_paid`.

A sceptic who wants to escape the conclusion has to reject all three premises, not one.

## Objection 1: "We are too small to matter"

**Strongest form.** The city emits a few thousandths of a per cent of world emissions. Its cut
cannot be measured in the global temperature, so it achieves nothing.

**Premise.** *Whoever foreseeably harms others needs a justification, even when the harm is
small and shared among many.* Warming grows with cumulative emissions, roughly linearly, so each
tonne carries its share of the damage. Being small makes the damage small, not zero.

**What the model computes.**

* `objection_climate_cost_rate`: the damage per tonne, by year of emission, from the German
  Environment Agency's *Methodenkonvention 4.0* (February 2026, Table 1, EUR 2025): 935 €/t for
  2020, 990 for 2025, 1,050 for 2030 and 1,240 for 2050 at a 0 % pure rate of time preference;
  310, 345, 375 and 485 at 1 %. Interpolated linearly between the tabulated years, as the
  handbook recommends.
* `objection_climate_damage_caused`: the city's balance times that rate, per year.
* `objection_climate_damage_avoided`: the scenario's cut times that rate, summed from 2020.

**Disputed, so a setting.** How much less future generations count
(`fair_share_pure_time_preference`, 0 % or 1 %). The agency's main value is 0 %: equal weight.

**Why it answers the objection.** The argument does not depend on what others do. If they do
not act, it gets warmer, and the damage per tonne tends to rise. A measure that costs less per
tonne than the damage it avoids is justified on this premise even if the city acts alone.

**Limits.** The agency's rates use equity weighting (a euro of damage to a poor household counts
more), which a sceptic may reject; the 1 % rate is the concession. The model has no investment
costs, so it shows the benefit side of the comparison only. Each year's damage is the present
value at emission, and the sum over years is not discounted further.

## Objection 2: "Emissions trading makes it pointless"

**Strongest form.** Power, district heating and industry are under the EU emissions trading
system (EU ETS 1), and from 2028 heating and road fuels are too (EU ETS 2). The cap fixes total
emissions. Whatever the city saves frees an allowance that someone else uses, so emissions only
move: the "waterbed". In the technical sense, this is the strongest objection.

**Premise.** *A measure counts by what it changes in the world.* This cuts both ways, and the
model applies it both ways: it removes the part of the cut that a cap releases again, and keeps
the part that no cap releases.

**What the model computes** (`emissions_trading.yaml`, BISKO dimensions required).

* `objection_emissions_by_trading_system`: the balance split into EU ETS 1, EU ETS 2 (from 2028),
  the German national CO₂ price under the BEHG (2021–2027, a fixed price or price corridor that
  limits no quantity) and uncapped.
* `objection_waterbed_released_ets1` / `_ets2`: the part a cap releases, by the waterbed share.
* `objection_effective_emissions`: what is left, the emissions that depend on the city's
  choices. The level is an auxiliary measure; the difference between scenarios is the point.
* `objection_effective_reduction(_cumulative)`: the scenario's cut that lowers world emissions.
* `objection_effective_reduction_share`: the same as a share of the planned cut, against a 100 %
  goal (no caps).
* `objection_climate_damage_avoided_effective`: objections 1 and 2 together.
* `objection_trading_coverage_check`: the share of the balance the data assigns to a system.
  It must be 100 %, or part of the balance is missing from the test.

**The assignment grants the objection where the facts are unclear.** Industrial fuels and
district heating count wholly as EU ETS 1, although smaller installations and waste incineration
are outside it, and the upstream energy supply in each BISKO factor counts as capped, although
much of it happens abroad under no cap. A conclusion that survives this assignment survives a
more accurate one.

**Disputed, so settings.** The waterbed share per system (`fair_share_waterbed_ets1`,
`fair_share_waterbed_ets2`). 100 % is the textbook result for a binding cap, and the default an
instance should start from: it is the reading most favourable to the objection. Three facts argue
for less:

* the market stability reserve cancels surplus allowances, so part of a saving lasts;
* EU ETS 2 issues extra allowances when prices are high, and every saving reduces that issue;
* caps are political: they are tightened where abatement succeeds and loosened where it gets
  expensive. The start of EU ETS 2 was put off from 2027 to 2028 to damp prices.

**What remains if the objection is granted in full.** Under a firm cap, local measures decide
not *how much* is emitted but *who* cuts. The fair share then applies to the use of the cap: an
emission beyond the city's share is paid for with an allowance that others must cut to free.
That is premise 8 again, a duty to compensate. And a tonne saved under a cap is still worth the
allowance price to whoever no longer has to cut it; it is just not climate damage avoided. So
the objection does not dissolve the duty; it changes its form from "cut" to "cut or pay for the
cut elsewhere".

Before 2028 the objection is weak for heating and road fuels: a saving under the national fixed
price frees nothing for anyone else.

### The consequence: whoever raises objection 2 must want a stricter cap

The objection has a premise the objector rarely states: *the cap alone decides how much is
emitted.* Combined with the premise almost everyone holds, *climate change is a problem that must
be solved*, it yields a conclusion: **support a stricter cap**, because on the objector's own
premises it is the only lever. A stricter cap raises the allowance price, and the local measures
then pay off financially as well, through the CO₂ costs they avoid.

Whoever rejects the conclusion must give up one of the two premises: either trading does not work
as a waterbed (and the local cut counts directly), or climate change is not a problem. The
apparent third ways lead back into the argument:

| Way out | Where it leads |
|---|---|
| "The cap is already strict enough" | Then the cut must happen somewhere; the waterbed only decides who delivers it, which is the fair share's question. A cap on a path to zero cannot be met without real abatement. |
| "Use another instrument, not trading" | A disagreement about the tool. The same stringency is still required. |
| "A stricter EU cap moves industry to China" | The waterbed one level up. Followed through, nothing in Europe matters, which is objection 3, with its own test. |
| "Cut only as far as it pays" | The cost-benefit position, and it needs a stricter cap most of all: see below. |
| Hold both premises and still oppose a stricter cap | Consistent only as free riding, which the fair share's premise 6 rejects. |

**What the model computes.** `objection_carbon_price_share_of_damage`: the CO₂ price on heating
and road fuels (the instance's `co2_price_fuels`) divided by the climate cost rate, against a
100 % goal. For Mainz it is about 5 % today and 6 % from 2028 at the default reading; 13–17 % at a
1 % time preference; and 67 % in 2030 even at the highest ETS 2 price the settings allow
(250 €/t) with the 1 % rate. Whoever weighs costs against benefits must therefore find the price,
and with it the cap, far too lax. The EU ETS 1 price has stayed below about 100 €/t, which tells
the same story for power and industry.

## Objection 3: "Only once the others act"

**Strongest form.** Why should the city cut while China, the United States or India do not?
Acting alone is being exploited.

**Premise.** The budget's premise 6 (no exemption) rejects the condition outright, and that is
the premise most people hold. But the objection is taken at its word: **grant the condition,
then check whether it is met.** Whoever cooperates conditionally owes the action once enough
others act, and gets to say what "enough" and "act" mean.

**What the model computes.**

* `objection_cooperation_<criterion>`: the share of world emissions from countries that act, by
  three criteria, from the weakest to the strictest. One node per criterion rather than one with a
  dimension, because the criteria cover different years and a dimensional metric fills a missing
  cell with zero, which would read as "0 % acting":
  * *net-zero target adopted* (Net Zero Stocktake 2025): 87 % in 2024, at least 74 % in 2025
    after the United States abandoned its target;
  * *carbon price in force* (World Bank, State and Trends of Carbon Pricing): 21.5 % in 2021 to
    28 % in 2025, at very different prices;
  * *emissions clearly falling*: countries whose five-year mean is at least 10 % below their
    highest so far, computed from the Global Carbon Project's country series. About a third of
    world emissions since 2015. The United States, Japan and the large EU members are in it;
    China and India are not.
* `objection_cooperation_met_<criterion>`: whether the coverage reaches the objector's threshold
  (`fair_share_cooperation_threshold`), per criterion.

**The honest result.** At a 50 % threshold the condition is met by pledges and not by deeds.
Whoever demands falling emissions from a majority of the world before acting is, today, not
bound by this argument. That person is still bound by objection 1, which does not depend on
others, and by premise 6, if they accept it.

## Objections not yet modelled

Each has a premise that answers it and something the model could compute. Listed so that the
tool's coverage is visible, roughly in order of how often a city will hear them.

| Objection | Premise that answers it | What to compute |
|---|---|---|
| "It's the federal or EU job" | Ought implies can (premise 8): the duty reaches the city's own levers | Split the scenario's cut into city-controlled, city-influenced and national (e.g. the grid factor) |
| "We can't afford it; schools first" | Public money is justified when benefits exceed costs | Cost per tonne of each measure against the climate cost rate and the CO₂ price |
| "It hits the poor" | Burdens should not fall on those least able to carry them | A constraint on *how*, not *whether*: costs by income group (needs data the model lacks) |
| "Cheaper to offset abroad" | Premise 8: compensation is legitimate for what remains | Overshoot × offset price, with an integrity discount |
| "Wait for better technology" | Premise 2: delay uses up the budget | The annual reduction rate needed if action starts now versus five years later |
| "Our emissions aren't ours" (commuters, national grid) | A fair accounting boundary is fixed in advance | The balance under a consumption-based boundary |
| "We have already done enough" | Premises 2 and 4: past cuts are already counted | Nothing new; the budget handles it |
| "Spend it on adaptation" | Premise 1 / non-harm: adaptation does not remove harm done to others | Text |
| "The state shouldn't dictate how I heat" | Intertemporal freedom (Federal Constitutional Court, 2021) | Text |

## Including the module

```yaml
include:
- file: modules/fair_share/carbon_budget.yaml
  allow_override: false
- file: modules/fair_share/objections.yaml
  allow_override: false
- file: modules/fair_share/emissions_trading.yaml   # German BISKO instances only
  allow_override: false
```

`objections.yaml` needs `carbon_budget.yaml`'s requirements, plus a scenario with the id
`baseline` and the global parameters `fair_share_pure_time_preference` and
`fair_share_cooperation_threshold`. `emissions_trading.yaml` needs `net_emissions` with the BISKO
dimensions `sector` and `energy_carrier` and the global parameters `fair_share_waterbed_ets1` and
`fair_share_waterbed_ets2`. `configs/mainz-dev.yaml` has a complete block. As with the budget,
each parameter's label carries the explanation, because the settings panel shows no descriptions.

The data is `fair_share/climate_cost_rate`, `fair_share/cooperation_coverage` and
`fair_share/emissions_trading_coverage`, built by
`modules/fair-share/scripts/create_objections_data.py` in paths-data, whose docstring records the
source choices.

The reductions are measured against the scenario `baseline` with `output_with_scenario`, which
keeps the visitor's settings, so a slider moves the comparison too.
