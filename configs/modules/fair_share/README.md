# Fair share of the global CO₂ budget

*Produced by Claude Opus 5.5 on 2026-10-06.*
*Responsible: Jouni Tuomisto.*

A module that asks whether a city stays within its fair share of what the world may still
emit, and states the value judgements that answer depends on as premises the reader can
change. It turns "should we do this?" into "if you accept these premises, then...".

The module reads the city's emissions and computes budgets. Nothing in it feeds back into
the balance, the scenarios or any action's impact.

## The argument

The conclusion is a deduction: *if you accept premises 1–8, the city's emissions from 2020 on
must stay under a budget B. A scenario that exceeds B is not enough, and a city whose
scenario without measures already exceeds B owes the measures.* Each premise is phrased so
that most people accept it; where people genuinely disagree, the disagreement is a global
parameter.

| # | Premise | What is disputed → parameter | Node |
|---|---|---|---|
| 1 | **Limit.** Dangerous warming is a serious harm; the Paris limit marks where it begins. | Warming limit (1.5–2.0 °C) and likelihood of keeping to it (50–83 %), which together give the global remaining budget (IPCC AR6, Table SPM.2) | `fair_share_global_budget` |
| 2 | **Cumulative.** The climate responds to the total emitted, so the budget binds, not a target year. | — (physics; also the reasoning of the German Federal Constitutional Court, 2021) | `fair_share_city_cumulative_emissions` |
| 3 | **Equal claim.** The atmosphere is a commons; no person has a stronger claim than another. | Weight of current emissions in the allocation (grandfathering), 0–100 % | `fair_share_national_budget_share` |
| 4 | **Responsibility.** Whoever has used more than their share has less left. | Start year of historical responsibility, 1850–2020 | `fair_share_national_overuse` |
| 5 | **Capability.** The richer should carry more. | — (text: why an overshoot stays a duty) | `fair_share_city_overshoot` |
| 6 | **No exemption.** Being small is no excuse. | — | `fair_share_city_budget` |
| 7 | **Subsidiarity.** A national share passes to cities by a fair key. | The instance's population key, and what its balance counts as its emissions | `fair_share_city_budget`, `fair_share_city_cumulative_emissions` |
| 8 | **Ought implies can.** A duty reaches only as far as the feasible; what cannot be avoided becomes a duty to compensate. | — (text) | `fair_share_city_overshoot` |

Conditional cooperation ("we act if others do") is deliberately not a premise. It is the one
position many people explicitly reject, and premise 6 is its negation. The
`moral_argumentation` module represents it as a value with a weight, which is the right place
for a view to be weighed rather than deduced.

### The arithmetic

With *B* the global budget from 1 January 2020, *p* and *e* the country's shares of world
population and fossil CO₂ in 2019, *g* the grandfathering weight, and *H*, *E* the country's
actual and equal-per-capita emissions from the start year to 2019:

    national budget = (g·e + (1−g)·p) · B  −  (1−g) · (H − E)
    city budget     = national budget · city population share
    remaining       = city budget − Σ city emissions from 2020

`fair_share_city_remaining_budget` is the conclusion; `fair_share_city_budget_used` states it
as a percentage (100 % = used up), which is what a dashboard card can draw, since its bars
cannot show a negative remaining budget. Set such a card's `year` to the model's end: at the
target year a scenario can look within budget a year before it overdraws it.

The overuse *H − E* is deducted only from the per-capita part: grandfathering *is* the view
that past emissions confer an entitlement, so it would be incoherent to charge for them there.
*E* sums each year's population share times that year's world emissions, because shares move
(Germany: 2.6 % of the world's people in 1850, 1.1 % in 2019).

### Arguing *a fortiori*

The result is strongest when it holds under the most generous reading a sceptic would still
defend, because then everyone who accepts the premises at all has to accept the conclusion,
and a stricter reading only strengthens it. For Germany (OWID / Global Carbon Budget 2025,
2019 shares 1.07 % of population and 1.91 % of fossil CO₂) the national budget from 2020 is:

| Reading | Germany from 2020 |
|---|---|
| 1.5 °C, 67 % | 4.3 Gt |
| 1.5 °C, 50 % | 5.3 Gt |
| **1.7 °C, 67 % (default)** | **7.5 Gt** |
| 1.7 °C, 50 % | 9.1 Gt |
| 2.0 °C, 67 % | 12.3 Gt |
| 2.0 °C, 50 % (not "well below 2 °C") | 14.4 Gt |
| 1.7 °C, 67 %, full grandfathering | 13.4 Gt |
| 1.7 °C, 67 %, responsibility since 1990 | −7.9 Gt |

Germany emitted 3.2 Gt of fossil CO₂ in 2020–2024. The default is the row of the IPCC table
nearest to the basis the Federal Constitutional Court accepted (the German Advisory Council
on the Environment's budget for 1.75 °C at 67 %), with an equal claim per person and no past
counted. It is a middle reading, not the most generous one.

## Including the module

```yaml
include:
- file: modules/fair_share/carbon_budget.yaml
  allow_override: false
```

An include carries nodes, not parameters, so the instance must also provide:

1. **Global parameters** `country_code` (ISO 3166-1 alpha-2, e.g. `DE`),
   `fair_share_temperature_limit` (1.5–2.0, step 0.1), `fair_share_likelihood` (%, 50–83,
   step 1), `fair_share_grandfathering_weight` (%, 0–100) and
   `fair_share_responsibility_start_year` (1850–2020, step 10). Numbers between the steps
   have no row in the data and empty the node. `configs/mainz-dev.yaml` has a complete block.
   **Put the explanation in the label.** The settings panel shows labels and not
   descriptions, so each label names its premise, asks its question and says what the ends
   of the scale mean.
2. **`net_emissions`**, in whatever dimensions the instance has; the module sums them with
   `sum_dim(emissions)`. Which emissions are "the city's" is part of premise 7. An instance
   that should count something else overrides `fair_share_city_cumulative_emissions` and says
   in its description what it counts.
3. **`fair_share_key_population`** — the city's share of the national population, %, from
   one base year and held, so no scenario moves the city's own obligation.

The data is `fair_share/global_co2_budget`, `fair_share/national_co2_key` and
`fair_share/national_co2_responsibility` in the dataset repository, built by
`modules/fair-share/scripts/create_fair_share_budget.py` in paths-data. The national
datasets carry Germany only; another country needs its rows added there first.

`fair_share_met` is the duty node that `moral_argumentation/value_weights.yaml` expects, so an
instance that includes both can weigh fairness against other values.

## Known simplifications

* **The budget is CO₂; a municipal balance is usually CO₂e** and may include the upstream
  energy supply. Counting it against a CO₂ budget errs against the city.
* **Fossil shares applied to a budget that includes land-use CO₂.** Consistent on the
  historical side (fossil against fossil); approximate on the budget side.
* **Interpolated budgets.** IPCC tabulates 1.5, 1.7 and 2.0 °C at 17/33/50/67/83 %. The rows
  between are linear interpolations and say so in their comment.
* **AR6 is generous.** Later estimates of the remaining budget are markedly lower, so every
  reading here errs in the city's favour.
* **Capability (premise 5) does not change the budget.** It argues that an overshoot must be
  compensated rather than excused; pricing that compensation is a further value judgement the
  module leaves out.
* **Within the country, only a population key.** A key by current emissions would compare a
  municipal balance with a national inventory, which count differently.
