# Vegetarian catering (library action with its argument)

*Produced by Claude Opus 5.5 on 2026-11-14.*
*Responsible: Jouni Tuomisto.*

**Status (15 Nov 2026):** built, datasets uploaded, and in `mainz-dev` in place of Mainz's own
catering action, reproducing it (sections 7 and 9); dataset repository commit `85c0ce7`. Not yet
on any page.

A library action for any city that accounts for the meals it serves or buys: canteens, schools,
day-care centres, events. It shifts hot meals from meat and mixed dishes to vegetarian ones. It
is also the first library action that carries its own argument, as
[`docs/argumentation-framework.md`](../../../docs/argumentation-framework.md) describes. Its
factual disputes are hypotheses, its value disputes are value frameworks, and its objections
are argument nodes, so a city that takes the action gets the debate with it.

The action is small in tonnes and large in controversy. In 2013 a proposed weekly meat-free day in
public canteens ("Veggie Day") became one of the defining issues of the German federal election
campaign. That combination makes it a good teaching case: the numbers settle less than the
debate suggests, and the values settle more.

## 1. Two packages

The action comes in two packages with the same target share of vegetarian hot meals:

| Package | What changes | Choice kept? |
|---|---|---|
| **A: vegetarian-only days** | On some days, or for some lines, meat is not offered at all | No |
| **B: vegetarian by default** | The vegetarian dish is the standard; a meat dish is available on request or on a second line | Yes |

Package B is the compensation case of `argumentation-framework.md` section 4. It aims to keep
most of the effect while removing the grievance that drives the opposition. Whether it does keep
the effect is a factual dispute (hypothesis H2).

## 2. The causal chain

```
target share of vegetarian hot meals
  × package effectiveness (H2)
  → hot meals shifted from meat and mixed to vegetarian
  → emissions of the meals served, through the emission factor per meal type (H1)
```

The meals shifted are taken from the meat and the mixed categories in proportion to their size,
and the total number of meals stays the same, as in Mainz's existing
`knsv_vegetarische_verpflegung`. The action writes signed meal counts into
`municipal_items_purchased` of [`modules/gpc/municipal_balance.yaml`](../gpc/municipal_balance.yaml),
whose `emission_source` categories (`meal_hot_meat_or_fish`, `meal_hot_mixed`,
`meal_hot_vegetarian`) are the hook. A city that uses the municipal balance module can take the
action without other changes.

The arithmetic happens in the model, from the city's meal counts and its target path, rather
than in a producer script per city. That keeps the action generic.

Each step is its own node with a one-line formula, so the chain can be read in the graph:

| Node | Formula | Mainz, 2035 |
|---|---|---|
| `vegetarian_catering_hot_meals` | the city's meal data (`gpc/municipal_balance_activity`), hot meals only | 1.41 million |
| `vegetarian_catering_vegetarian_share` | vegetarian meals / all hot meals | 51.1 % |
| `vegetarian_catering_share_gap` | max(target − share, 0) | 28.9 points |
| `vegetarian_catering_conversion_rate` | gap / (1 − share): the share of meat and mixed dishes that becomes vegetarian | 59.1 % |
| `vegetarian_catering_meals_moved` | rate × meat and mixed dishes | 14,045 + 393,114 |
| `vegetarian_catering_meal_shift` | the moved dishes added to vegetarian, minus where they left | ±407,159 |
| `vegetarian_catering` (action) | `select_port(condition, constant, 1) * shift` | |

The action *acts on* `municipal_items_purchased` (a hook,
[`docs/architecture/action-hooks.md`](../../../docs/architecture/action-hooks.md)), so the city's
node and its inputs stay untouched. With package A it adds the whole shift; with package B, the
share of it that `vegetarian_catering_default_effect` sets.

The chain starts from the city's meal data rather than from `municipal_items_purchased`, because
reading the node the action acts on would be a cycle. So the action does not see a city measure
that has already shifted meals, and the two must not be on in the same scenario: in `mainz-dev`,
`knsv_vegetarische_verpflegung` is off wherever this action is on.

Two global parameters, registered in `params/global_params.py`, choose what goes into the
balance, and both appear in the scenario settings:

* `vegetarian_catering_keep_meat_dish` (bool): package B if true, A if false. The instance sets
  the default; `mainz-dev` uses A, which is what the KNSV report's measure describes.
* `vegetarian_catering_default_effect` (%, default 9.3): how much of A's effect B achieves. The
  two published estimates are 58 % (field trials) and 9.3 % (at scale); the slider allows any
  value between 0 and 100 %.

They reach the action through its `condition` and `constant` parameters with `ref:`. A choice
parameter (`type: dimension_category`) would fit the meaning better, but the UI has no widget
for one yet.

## 3. Factual hypotheses

| | Hypothesis | Categories | Source |
|---|---|---|---|
| **H1** | Emission factor per meal type | `ifeu_2020` (library default), `city_balance` (the city's own balance factors) | ifeu (2020), *Ökologische Fußabdrücke von Lebensmitteln und Gerichten in Deutschland*, Table 6: eight example dishes. Default factors are the means: meat or fish 1.7 kg CO₂e per meal (range 0.6–2.6), vegetarian and vegan 0.775 kg (range 0.6–1.0) |
| **H2** | Effectiveness of package B, as a fraction of package A's effect | `field_trials` 0.58, `at_scale` 0.09 | Field trials: Boronowsky et al. (2022), three RCTs at catered campus events. Under a meat default 18.4 % chose plant-based, under a plant-based default 65.5 %, so the default closed 58 % of the gap that removing meat would close. At scale: DellaVigna & Linos (2022, *Econometrica*), nudges run by government units have about a sixth of the effect of published trials (1.4 vs 8.7 percentage points) |

Package A's effectiveness is 1 by construction: the meat is not on offer. The objection that
diners then eat elsewhere is noted as an argument, not a hypothesis. A twelve-week Swiss university canteen trial found no fall in meals sold, but it added vegetarian dishes rather than removing meat, so it bears on package B more directly than on A.

H1 matters more than it looks. Mainz, the first city, has two earlier assessments of its own
administration that value the same meals differently. One uses 1.44 kg (vegetarian) and 1.95 kg
(mixed and meat or fish). The other uses 3.5 kg, 5.0 kg (mixed) and 2.44 kg (meat or fish: lower
than mixed, unexplained). The difference per mixed meal shifted is 0.51 kg in one and 1.5 kg in
the other, so the city's own sources disagree about the action's effect by a factor of three. The
library default gives 0.93 kg.

**Mixed meals** take the meat-or-fish factor. The category is not defined by the city that
reports it; the likeliest reading is a line that serves meat on some days and fish on others.

**Not a hypothesis: home-diet spillover.** Canteen habits may carry over to meals at home. The
evidence is one qualitative workplace study (Verfuerth et al. 2021), one agent-based simulation,
and a review that found no sustained change beyond the intervention for many interventions. That
is not enough for a number, so it is an argument node (A3) that states the evidence. It becomes a
hypothesis when a field study measures it.

## 4. Value frameworks

Each package is judged against each value framework, side by side, as in
[`modules/fair_share/value_frameworks.yaml`](../fair_share/value_frameworks.yaml). These nodes
compare the packages and never feed the balance.

| | Value framework | Accepts a package when | Chain |
|---|---|---|---|
| **W1** | Every tonne counts | it lowers emissions under every factual hypothesis | long: action → meals → emissions |
| **W2** | Choice as a limit | a meat dish stays available every day | none: a judgement about the act itself |

Expected verdicts:
* Package A: W1 accepts, W2 rejects, so A is **negotiable**, and the dispute is about choice.
* Package B: both accept, so B is **robust**. That holds as long as H2 gives it any effect at all,
  which both categories do. B buys that robustness with effect: 58 % of A's in field trials, 9 %
  at scale.

That is the point of the example: the compensation turns a negotiable action into a robust one.
A value framework on cost (cost per tonne against the CO₂ price) is left for later, because no
citable source for the cost difference between meat and vegetarian canteen meals has been found
yet.

## 5. Argument nodes

| | Argument | Kind | Linked to |
|---|---|---|---|
| A1 | "The city has no business deciding what people eat." | Value about the act (zero-length chain). Consistency test: the current menu is also the city's choice, so the judgement applies to the status quo as well. | W2 |
| A2 | "It is pure symbolism." | Implicit theory of change. As a factual claim it says the direct effect is negligible, and the model shows that, as a share of the city's total, it is. As a value claim it says that examples set by the city do not matter, which is a separate dispute. | the effect node |
| A3 | "Canteen habits spill over to home." | Factual, evidence weak (section 3) | the effect node |
| A4 | Health | Less red and processed meat is a health benefit. Argument only, not quantified (decided 14 Nov 2026). Source to be added: the German Nutrition Society's recommendation for meat intake. | the action |
| A5 | Popularity | Descriptive, not a criterion. Infratest dimap for *Welt am Sonntag*, 2013, as reported secondhand: 61 % against a weekly meat-free canteen day, 36 % in favour. A second report gives 69 % against and 39 % in favour, which sum to more than 100; 69 is most likely a misprint of 61. Enters a dataset only once the original is found. This concerns package A; no poll on package B has been found. | the action |

## 6. Data

A producer script in paths-data, `modules/vegetarian-catering/scripts/create_vegetarian_catering_data.py`,
writes the generic datasets (written 14 Nov 2026, not yet uploaded):

| Dataset | Content | Dimensions |
|---|---|---|
| `vegetarian_catering/meal_emission_factors` | H1 library default: ifeu 2020, kg CO₂e per hot meal | `emission_source`, `meal_factor_set` |
| `vegetarian_catering/package_effectiveness` | H2: fraction of package A's effect | `catering_package`, `default_effect_hypothesis` |
| `vegetarian_catering/catering_packages` | `keeps_choice`: 1 if a meat dish stays available every day | `catering_package` |

The constant datasets are dated by their source year and carried forward with `extend()` in the
formulas, because multiplication joins on years too.

The city supplies its meal counts (its municipal balance data, through the same
`dataset_replacements` entry for `gpc/municipal_balance_activity` that it gives the balance module)
and its target path: the
share of vegetarian hot meals by year, starting from its current share, through
`dataset_replacements` of `vegetarian_catering/target_share`. For Mainz that is
`mainz/vegetarian_catering_target_share`, 51.1 % in 2024 to 80 % in 2035
(paths-data `models/mainz/scripts/create_vegetarian_catering_target.py`).

H1 has two categories: the library default `ifeu_2020`, and `city_balance`, the factors of the
city's own balance read from `municipal_items_ef`, so every city compares its own factors with
the library's without supplying anything. Mainz's 2022 workbook factors (3.5 / 5.0 / 2.44 kg)
would be a third, Mainz-only category; not built.

The city puts the action in one of its action groups with `action_group:` on the include entry
(added 15 Nov 2026 for this purpose); a module cannot know the city's groups.

Every document these figures and the arguments rest on is archived in paths-data
`modules/vegetarian-catering/source_material/`, so they stay available if a source disappears.

The ifeu figures are example dishes, not canteen averages. The default factors would be the
means of its meat dishes and of its vegetarian dishes, with the range kept as the uncertainty. A
canteen-average source would be better if one turns up.

## 7. First instance

`mainz-dev`, as a trial of the action library in production that leaves `mainz-bisko` untouched.
The library action **replaces** Mainz's `knsv_vegetarische_verpflegung` there:

* The include puts it in Mainz's group `verwaltung` (`action_group: verwaltung`).
* `mainz-dev` defines its own `knsv_vegetarische_verpflegung`, which wins over the one it includes
  from `mainz-bisko.yaml` (that include now has `allow_override: true`). The override has no
  group, so the action list hides it; it is off in every scenario; and it multiplies the shift by
  zero, so it cannot count the measure twice even when switched on.
* The library action is on wherever the Mainz action was (`mp100_kwp`, `zielpfad_2045`), with
  package A, so the defaults reproduce the Mainz action. The 65 % variant (`selected_number: 1`),
  used only in commented-out scenarios, is gone.
* Mainz's target path starts from its 2024 share in per cent with six decimals. A first version
  rounded the share to four decimals as a fraction, which moved about eight meals in 2030.

If Lorig likes the library version, `mainz-bisko.yaml` takes the same change at the source: the
Mainz action goes, the module is included there with `action_group: verwaltung`, and the override
and `allow_override` in `mainz-dev` go too. That needs a sync and a publish of `mainz-bisko`.

## 8. Order of work

1. ~~Review of this design.~~
2. ~~Producer scripts and CSVs in paths-data; upload; `mainz-dev` pinned to `218720f`.~~
3. ~~The module YAML, with German and English texts.~~
4. ~~`mainz-dev` include, first beside the Mainz action, then (15 Nov) in place of it.~~
5. ~~A pointer in `docs/argumentation-framework.md` to this module as the worked example.~~
6. Pages: the verdict node and the arguments are not on any page yet.

## 9. Verified on `mainz-dev` (14 Nov 2026)

First against a scratch clone of the dataset repository holding the uploader's own output, then
again after the upload, against the real pin `218720f`, with the same results:

* All 273 nodes compute in all four scenarios.
* The three existing scenarios are identical to the last digit, before and after (`net_emissions`,
  `municipal_balance_emissions`, `municipal_items_emissions`, `municipal_items_purchased`; 2024,
  2030, 2035, 2045).
* Package A with the city's own factors saves 207.7 t CO₂e/a in 2035, which is what Mainz's own
  action saves at the same target: the formula reproduces it.
* The shift the chain computes for package A equals Mainz's own KNSV figures to the meal
  (−14,045 meat or fish, −393,114 mixed, +407,159 vegetarian in 2035).
* Rebuilt the same day as a chain of one-step nodes with two registered global parameters, and
  re-verified: same results, existing scenarios unchanged, both parameters visible and
  customizable.

* In place of the Mainz action (15 Nov, against the re-produced target path in a scratch copy of
  the dataset repository): all 278 nodes compute in all three scenarios; `net_emissions`,
  `municipal_balance_emissions`, `municipal_items_emissions` and `municipal_items_purchased`
  equal the original to six decimals in 2024, 2030, 2035 and 2045, and
  `municipal_items_emissions` differs by at most 0.0000008 t/a in any year 2024–2050.

Reduction in 2035, t CO₂e/a:

| Package | H2 | city's factors | ifeu 2020 |
|---|---|---|---|
| A: vegetarian-only days | either | 207.7 | 376.6 |
| B: vegetarian by default | field trials | 119.7 | 217.2 |
| B: vegetarian by default | at scale | 19.3 | 35.0 |

As a share of the city's total emissions without measures: 0.002–0.034 %. Verdicts: A is
accepted by one value framework (negotiable), B by both (robust).

## Decided

* **Health effects are arguments only** (14 Nov 2026).
* **Packages A and B carry the autonomy argument**, rather than Mainz's existing 80 % / 65 %
  scenarios, neither of which removes choice (14 Nov 2026).
* **Mixed meals** are read as a line serving meat on some days and fish on others, and take the
  meat-or-fish factor (14 Nov 2026).
* **The hypotheses stay a dimension** in the comparison nodes, so that the verdict under each can
  be shown, rather than being aggregated to an expectation (14 Nov 2026). The balance itself
  receives package A or B at the effectiveness of one H2 category, both chosen by global
  parameters; the defaults are B and `at_scale` (14 Nov 2026).
