# Heat transition (library actions)

*Produced by Claude Opus 5.5 on 2026-11-17.*
*Responsible: Jouni Tuomisto.*

**Status (17 November 2026):** action 1, district heating decarbonisation, is built and verified
against a scratch copy of the dataset repository. Its datasets are not yet uploaded, and
`schwerin-bisko`, the first city to take it, has the include commented out until they are
(`#HT#`). The other actions are proposals. The analysis they come from is paths-data
`models/schwerin/docs/heat-plan-actions.md`.

Library actions for the heat transition in German BISKO cities. A city's heat plan (*kommunale
Wärmeplanung*) states rates, potentials and milestones, and these actions turn them into effects
on the balance. A city without a heat plan still gets defaults it can defend. Each action acts on
the node of the BISKO module (`modules/bisko/model.yaml`) where its effect happens, through a hook
([`docs/architecture/action-hooks.md`](../../../docs/architecture/action-hooks.md)). So including
the module changes no BISKO calculation.

| # | Action | Acts on | Status |
|---|---|---|---|
| 1 | District heating decarbonisation | `district_heating_emissions` | built (`district_heating_decarbonisation.yaml`) |
| 2 | Replacing fossil heating outside the network | `final_energy_use_corrected` (fossil carriers → environmental heat, biomass) | proposed |
| 3 | District heating expansion | `final_energy_use_corrected` (fossil carriers → district heating) | proposed |
| 4 | Building renovation | `final_energy_use_corrected` (relative cut of heat carriers) | proposed |
| 5 | Biomethane in the gas grid | the natural gas emission factor | proposed |

Binding each action where its effect happens resolves most interactions through the graph. Action
1 scales whatever district heating action 3 adds, for example. Actions 2 and 3 will partition the
fossil heat between them rather than each drawing on all of it.

## Action 1: district heating decarbonisation

The Heat Planning Act (*Wärmeplanungsgesetz*, WPG) requires every heat network to take at least
30 % of its annual net heat generation from renewable energy or unavoidable waste heat from
2030 and 80 % from 2040 (§ 29(1)), and 100 % by the end of 2044 (§ 31(1)). On application, the
deadlines can be extended to the end of 2034 and the end of 2044 (§ 29(2)), but § 31 does not
move. The action follows one of these two paths, from the network's share today, and scales the
district heating emissions by the fossil part:

```
index(t) = (1 − share(t)) / (1 − share today)
effect(t) = district_heating_emissions(t) × (index(t) − 1)
```

"Today" is the latest year of the city's own figure. Between that year and the first milestone,
and between milestones, the share rises in a straight line. A network already above the path
keeps its current share.

| Node | What |
|---|---|
| `district_heating_legal_renewable_share` | the milestones of both paths (`district_heating_path`) |
| `district_heating_minimum_renewable_share` | the path chosen by `district_heating_extended_deadlines` |
| `district_heating_renewable_share` | from the city's share today to the minimum, in straight lines |
| `district_heating_fossil_share` | 1 − share |
| `district_heating_fossil_index` | relative to today (`ratio_to_last_historical_value`) |
| `district_heating_decarbonisation` (action) | `district_heating_emissions * (index - 1)`, hooked on that node |

### What a city provides

* The BISKO module.
* Its network's current renewable share, through `dataset_replacements` of
  `heat_transition/district_heating_renewable_share`. The default is 0 %, dated 2024. That means
  "no information", and it is also what BISKO's standard factor implies (coal-fired combined heat
  and power). For a network that already uses renewable heat it overstates the effect, so a city
  should supply its figure.
* The global parameter `district_heating_extended_deadlines` (bool, `params/global_params.py`) in
  its `params:`. False is the statutory path, true the extended one.

### Choices and limits

* **Every year of the hooked node, whatever BISKO route computed it.** The action reads
  `district_heating_emissions` as computed, so it works on all three of BISKO's district heating
  routes. It also scales heat that other actions bring onto the network.
* **Renewable heat and waste heat count as free of emissions.** BISKO's factors for them include
  upstream emissions, so in reality a small residual remains in 2045. The law counts waste heat
  with renewables, and so does the action.
* **The legal minimum is a floor, not a forecast.** An operator's transformation plan may be
  faster. A city with such a plan should get a third path, its own, which is not built yet.
* **The 2026 amendment of the WPG.** The cabinet adopted it on 27 May 2026; it is not in force.
  It is reported to keep the shares in §§ 29 and 31 largely as they are. Revisit when it passes.
* **The fossil combined heat and power exception (§ 29(5))** changes the 2030 requirement for
  networks where at least 70 % of heat comes from subsidised fossil combined heat and power.
  Treat the extended path as its approximation until a city needs it exactly.

### Verified on 17 November 2026

On `schwerin-bisko` (current share 14.4 %), with the datasets pushed to a scratch bare clone of
the dataset repository and the uploader's own frame and metadata functions:

| | 2024 | 2030 | 2035 | 2040 | 2045 |
|---|---|---|---|---|---|
| Share, statutory path | 14.4 % | 30 % | 55 % | 80 % | 100 % |
| District heating emissions, statutory (kt) | 84.6 | 69.2 | 44.5 | 19.8 | 0 |
| Share, extended deadlines | 14.4 % | 22.9 % | 30 % | 65 % | 100 % |
| District heating emissions, extended (kt) | 84.6 | 76.2 | 69.2 | 34.6 | 0 |

The impact on `net_emissions` by `compute_impact` is −84.6 kt in 2045. With the action off, every
year of `net_emissions`, `stationary_emissions`, `transport_emissions`,
`district_heating_emissions` and `final_energy_use_corrected` agrees with the configuration
without the module to within 1e-9, in both scenarios.

Data: paths-data `modules/heat-transition/` (generic) and `models/<city>/scripts/
create_heat_transition_data.py` (the city's share).
