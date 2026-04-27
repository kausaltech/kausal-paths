# Potsdam CO₂-Zukunftssimulator: EH 55 / EH 40 und Energiewende

---

## Teil A — Fachliche Beschreibung für Expertinnen und Entscheidungsträger in Potsdam

### Ausgangslage

Die Stadt Potsdam steht vor einer konkreten Frage: Soll die bisherige Anforderung für Neubauten – Effizienzhaus 40 (EH 40) – beibehalten werden, oder reicht Effizienzhaus 55 (EH 55) aus? Pro Potsdam Wohnen, das kommunale Wohnungsunternehmen mit rund 18.000 Wohneinheiten (ca. 20 % des Potsdamer Wohnbestands), argumentiert, dass EH 40 angesichts seiner Mehrkosten keinen ausreichenden Klimaschutznutzen bringt – insbesondere dann, wenn die Energieversorgung selbst konsequent dekarbonisiert wird.

Das Herzstück dieser Debatte ist ein Zielkonflikt:

> **Wenn die Energie sauber wird, schrumpft der Klimavorteil von mehr Dämmung gegen null.**

Der CO₂-Zukunftssimulator (Kausal Paths) soll diesen Zusammenhang sichtbar und quantifizierbar machen.

---

### Was das erweiterte Modell zeigen wird

#### 1. Der zeitabhängige Wert der Gebäudedämmung

Der Klimanutzen von EH 40 gegenüber EH 55 ergibt sich aus der Formel:

> **CO₂-Einsparung = eingesparte Energie × Emissionsfaktor der Wärmeversorgung**

Da EWP bis 2030 mindestens 30 % erneuerbare Wärme liefert und bis 2045 100 % anstrebt (Wärmeplanungsgesetz, Geothermie bereits in Bohrung), sinkt der Emissionsfaktor der Fernwärme über die Zeit gegen null. Der Klimavorteil von EH 40 gegenüber EH 55 ist daher heute am größten und wird bis 2045 nahezu verschwunden sein – unabhängig davon, wie viel Energie das Gebäude einspart.

Das Modell wird diesen zeitlichen Verlauf grafisch darstellen.

#### 2. Kosten und Nutzen aus vier Perspektiven

Das Modell rechnet nicht nur CO₂-Emissionen, sondern auch wirtschaftliche Effekte – jeweils aus der Sicht:

| Perspektive | Relevante Fragen |
|-------------|------------------|
| **Mieterinnen und Mieter** | Wie hoch ist die Kaltmiete pro m² bei EH 55 vs. EH 40? Was kostet das pro Wohnung und Monat? |
| **Pro Potsdam** | Welche Investitionskosten entstehen durch EH 40 gegenüber EH 55? Welche Rendite ergibt sich? |
| **Stadt Potsdam** | Welcher Anteil der Mehrkosten muss über Mietsubventionen oder Sozialwohnungsförderung getragen werden? |
| **Klima / Gesellschaft** | Wie groß ist die CO₂-Einsparung tatsächlich – und wie entwickelt sie sich bis 2045? |

#### 3. Vergleich der zwei Handlungsstränge

Das Modell unterscheidet explizit zwischen:

- **Gebäudemaßnahmen** (Dämmung, Effizienzstandard): reduzieren den Energieverbrauch
- **Energiemaßnahmen** (EWP-Wärmewende, Umstieg auf Wärmepumpen außerhalb des Fernwärmenetzes): reduzieren den CO₂-Gehalt je verbrauchter Einheit Energie

Beide Stränge können einzeln oder gemeinsam eingeschaltet werden. Das Modell zeigt, wie sich die Klimawirkung beider Maßnahmen gegenseitig beeinflusst: wer zuerst liefert, macht den anderen weniger dringend.

#### 4. Pro Potsdam als expliziter Akteur

Erstmals wird Pro Potsdam als eigenständiger Teilbestand im Modell abgebildet – mit:
- Anzahl Wohneinheiten und Wohnfläche
- Wärmeverbrauch pro Wohnung und insgesamt
- Renovierungsrate und Effizienzstandard (EH 55 oder EH 40)
- Kaltmiete, Energiekosten und Investitionskosten je Wohnung

Damit ist es möglich, konkrete Beispielrechnungen zu erstellen: „Was kostet EH 40 einen Pro-Potsdam-Mieter monatlich mehr – und wie viel CO₂ spart es wirklich?"

#### 5. Mietsubvention als Schieberegler

Ein steuerbarer Parameter im Modell ermöglicht es, den Anteil der Mehrkosten zwischen Mieterin, Pro Potsdam und Stadt Potsdam zu verschieben. Damit lassen sich Förderpolitiken direkt durchrechnen.

---

### Erwartetes Ergebnis für die politische Diskussion

Das Modell wird voraussichtlich bestätigen, was die Vorbesprechung am 9. April bereits als Konsens formuliert hat: Der entscheidende Hebel für den Klimaschutz im Gebäudebereich ist die Dekarbonisierung der Energieversorgung – nicht die Dämmstärke. EH 40 liefert in den nächsten zehn Jahren noch einen messbaren CO₂-Beitrag, dieser wird jedoch mit jeder weiteren Prozent-Erneuerbare-Energie bei EWP kleiner. Das Modell macht diesen Zusammenhang für Stadtratsmitglieder, Wohnungsunternehmen und Klimakoordinatoren direkt erlebbar.

---

### Nächste Schritte für Potsdam

| Aktion | Zuständig | Zeitpunkt |
|--------|-----------|-----------|
| Gebäude- und Energiedaten Pro Potsdam (Excel) übermitteln | S.M. / Pro Potsdam | baldmöglichst |
| Aktuellen Brennstoffmix EWP Fernwärme bestätigen | C.L. / EWP | baldmöglichst |
| Energiepreise für Fernwärme und Erdgas (EUR/MWh) | EWP / Stadtwerke | baldmöglichst |
| Modellentwicklung und Integration | Kausal (S-M.I. / J.T.) | laufend nach Datenverfügbarkeit |

---

---

## Part B — Technical Implementation Log and Remaining Work

### Context

This document describes the `potsdam-dev` model instance: a development fork of `potsdam-gpc` extended to analyse the EH 55 vs. EH 40 building standard question and the interaction between building renovation and energy supply decarbonisation. The instance is explicitly designed to model Pro Potsdam Wohnen as a standalone sub-system.

Branch: `feature/lucia`. Primary config: `configs/potsdam-dev.yaml`.

---

### Status overview

| Step | Description | Status |
|------|-------------|--------|
| 0 | Create `potsdam-dev.yaml`, fix migrations | ✅ Done |
| 1 | Fix dataset priority (city-specific data over generic framework data) | ✅ Done |
| 2a | Add `stakeholder` dimension (local override) | ✅ Done |
| 2b | Add `rent` category to `cost_type` in `standard_dims.yaml` | ✅ Done |
| 3 | Fix `heat_generation_individual` categories | ✅ Done |
| 4 | Pro Potsdam building-stock sub-model | ✅ Done (placeholder values) |
| 5 | Cost nodes: cold rent, energy price, total costs | ✅ Done (placeholder values) |
| 6 | New actions: eh55, eh40 (multi-metric DVC), rent_subsidy, individual_heating | ✅ Done (placeholder data) |
| 7 | Scenarios: `pro_potsdam_eh55_only`, `pro_potsdam_eh40_only` | ✅ Done |
| — | Replace placeholder data with Pro Potsdam Excel | ⏳ Waiting on data |
| — | Replace 150 EUR/MWh placeholder with EWP actual energy prices | ⏳ Waiting on data |
| — | EWP fuel mix (district heating emission factor) | ⏳ Waiting on data |
| — | `pro_potsdam_renovation_capex` node | ⏳ Waiting on data |
| — | `pro_potsdam_rent_subsidy` action (stakeholder cost split slider) | ⏳ Deferred |
| — | `renovation_without_transition` and `transition_without_renovation` scenarios | ⏳ Deferred |

---

### What was built

#### 0. Instance setup

- Copied `configs/potsdam-gpc.yaml` → `configs/potsdam-dev.yaml`, changed `id` to `potsdam-dev`.
- Fixed missing migration files (0042–0044) that had been applied to the DB from `feat/trailhead` but were absent from `main`.

#### 1. Dataset priority fix

City-specific datasets (e.g. `potsdam/...`) must take priority over generic GPC framework datasets when both supply data to the same node. The fix uses a coalesce / anti-join pattern in `frameworks/datasets.py`: city-specific rows are kept; generic rows are only used for years not present in the city dataset.

#### 2. Dimensions

**`stakeholder`** defined locally in `potsdam-dev.yaml` (not in `standard_dims.yaml` because it is Potsdam-specific):
- `tenant`, `pro_potsdam`, `city`, `society`

**`rent` category** added to `cost_type` in `configs/frameworks/standard_dims.yaml`.

#### 3. Node fix

`heat_generation_individual` previously excluded `[district_heating, solar_thermal, biogas, biomass]`, which also blocked heat pumps (`environmental_heat`). Changed to exclude only `[district_heating]` so that natural gas → heat pump transitions can flow through.

#### 4. Pro Potsdam sub-model

Standalone parallel calculation tree (does NOT feed into `net_emissions` to avoid double-counting with city-wide aggregates):

| Node id | Unit | Value |
|---------|------|-------|
| `pro_potsdam_apartments` | – | 18,000 (placeholder) |
| `pro_potsdam_floor_area_per_apt` | m² | 70 (placeholder) |
| `pro_potsdam_total_floor_area` | m² | apartments × floor_area_per_apt |
| `pro_potsdam_heat_demand_per_m2` | kWh/m²/a | 95 (placeholder) |
| `pro_potsdam_heat_demand` | MWh/a | total_floor_area × heat_demand_per_m2 |
| `pro_potsdam_emissions` | kt/a | heat_demand × corrected_emission_factor[district_heating, heating] |

`pro_potsdam_emissions` shares the `corrected_emission_factor` from the main model, so it automatically reflects the EWP decarbonisation trajectory from the existing district heating actions.

#### 5. Cost nodes

| Node id | Unit | Notes |
|---------|------|-------|
| `pro_potsdam_cold_rent_per_m2` | EUR/m²/month | 18 placeholder; modified by eh55/eh40 actions |
| `pro_potsdam_cold_rent` | MEUR/a | cold_rent_per_m2 × total_floor_area; `cost_type=rent`, `stakeholder=tenant` |
| `pro_potsdam_average_energy_price` | EUR/MWh | 150 placeholder (replace with EWP data) |
| `pro_potsdam_energy_costs` | MEUR/a | heat_demand × energy_price; `cost_type=energy_costs`, `stakeholder=tenant` |
| `total_costs` | MEUR/a | cold_rent + energy_costs; `output_dimensions: [cost_type, stakeholder]`; `is_outcome: true` |

**Note on dimensions:** `total_costs` requires `input_dimensions` and `output_dimensions: [cost_type, stakeholder]` to be explicitly listed. The upstream nodes (`pro_potsdam_cold_rent`, `pro_potsdam_energy_costs`) assign `cost_type` and `stakeholder` via edge `to_dimensions`. The `pro_potsdam_emissions` node carries extra dimensions (`energy_carrier`, `sector`) which are flattened out using `from_dimensions: flatten: true` before reaching cost nodes.

#### 6. New actions

**`pro_potsdam_renovation_eh55`** and **`pro_potsdam_renovation_eh40`** are two-metric `simple.AdditiveAction` nodes backed by a DVC dataset (`potsdam/pro_potsdam_renovation`, commit `508c60da55f4ae36597a79ddee7f9c739756e10a`):

```yaml
output_metrics:
- id: energy_reduction
  unit: kWh/m**2/a
  quantity: energy
- id: additional_cost
  unit: EUR/m**2/month
  quantity: unit_price
input_datasets:
- id: potsdam/pro_potsdam_renovation
  forecast_from: 2025
  interpolate: true
  filters:
  - column: action
    value: eh55
output_nodes:
- id: pro_potsdam_heat_demand_per_m2
  metrics: [energy_reduction]
- id: pro_potsdam_cold_rent_per_m2
  metrics: [additional_cost]
```

The dataset was created by `data/potsdam/create_renovation_csv.py` (placeholder values) and uploaded with `notebooks/upload_new_dataset.py`.

**`individual_heating_transition`**: `simple.LinearCumulativeAdditiveAction`, targets `heat_generation_individual`. Currently uses a placeholder. Needs the individual heating fuel split data from Kommunale Wärmeplanung.

**`pro_potsdam_rent_subsidy`**: Deferred. Will be a parameter action (slider 0–100 %) redistributing rent cost between `tenant` and `city` stakeholder categories.

#### 7. Scenarios

- `pro_potsdam_eh55_only`: EH 55 + EWP district heating actions ON, EH 40 OFF
- `pro_potsdam_eh40_only`: EH 55 + EH 40 + EWP district heating actions ON

Remaining scenarios (`renovation_without_transition`, `transition_without_renovation`) deferred until data is available.

---

### Bug fixes discovered during development

#### `interpolate: true` in `input_datasets` was silently ignored

**File:** `nodes/instance_loader.py`, line 518.

```python
# Before (always overwrote the YAML-defined flag):
ds_obj.interpolate = ds_interpolate

# After (preserves the YAML flag):
ds_obj.interpolate = ds_interpolate or ds_def.interpolate
```

Without this fix, sparse datasets with only a few key years (e.g. 2024, 2030, 2040, 2050) would show zero effect in all intermediate years even when `interpolate: true` was set in the YAML config.

#### Python 2 `except` syntax in `notebooks/upload_new_dataset.py`

```python
# Before:
except ValueError, TypeError:

# After:
except (ValueError, TypeError):
```

This caused a `SyntaxError` at import time.

#### Edge `to_dimensions` does not support `assign_category`

When routing values to a target node that needs a new categorical dimension, `to_dimensions` in the edge definition requires `categories: [cat_id]` syntax — not `assign_category: cat_id`. The `assign_category` field only works in dataset filter context (`DimensionDatasetFilterDef`). Using it in an edge silently produced no dimension assignment and caused "Dimensions do not match" errors.

---

### Remaining work (data-dependent)

#### After Pro Potsdam Excel data arrives

Replace placeholder `historical_values`/`forecast_values` on:
- `pro_potsdam_apartments` (18,000)
- `pro_potsdam_floor_area_per_apt` (70 m²)
- `pro_potsdam_heat_demand_per_m2` (95 kWh/m²/a)

Replace placeholder renovation dataset (`data/potsdam/create_renovation_csv.py`) with actual values for:
- `energy_reduction` per m² (kWh/m²/a) for EH 55 and EH 40 over time
- `additional_cost` per m² (EUR/m²/month) for EH 55 and EH 40 over time
- Annual renovation ramp-up rate (total floor area renovated per year)

After updating the dataset, re-upload with `notebooks/upload_new_dataset.py` and update `dataset_repo.commit` in `potsdam-dev.yaml`.

#### After EWP energy price and fuel mix data arrives

Replace `pro_potsdam_average_energy_price` (currently 150 EUR/MWh constant) with a dataset-backed node. The district heating emission factor (fed to `pro_potsdam_emissions`) may also need to be updated with the real EWP fuel mix.

#### Remaining nodes to add

| Node id | Blocked on |
|---------|-----------|
| `pro_potsdam_renovation_capex` | Investment cost data (EH 40 over EH 55 per m²) |
| `pro_potsdam_energy_cost_per_apt` | Pro Potsdam apartment count confirmed |
| `pro_potsdam_co2_cost` | Agreement on CO₂ shadow price |

#### Remaining actions

| Action id | What is needed |
|-----------|----------------|
| `pro_potsdam_rent_subsidy` | Design: how to implement stakeholder cost redistribution in YAML |
| `individual_heating_transition` | Fuel split for non-DH heating (Kommunale Wärmeplanung) |

#### Remaining scenarios

| Scenario id | Description |
|-------------|-------------|
| `renovation_without_transition` | EH 40 but no EWP decarbonisation |
| `transition_without_renovation` | EWP full transition, no renovation |

---

### 9. Key reference figures (from meeting notes, 9 April 2026)

| Figure | Value | Source |
|--------|-------|--------|
| Pro Potsdam portfolio | 18,000 units, ~20% of Potsdam housing stock | Meeting notes |
| Share on district heating | 90% | Meeting notes |
| Baseline energy demand | ~120,000 MWh/year | Meeting notes |
| Implied heat demand per m² (assuming ~70 m²/apt) | ~95 kWh/m²a | Derived |
| Cold rent: EH 55 | ~18 EUR/m²/month | Meeting notes |
| Cold rent: EH 40 | ~21 EUR/m²/month | Meeting notes |
| Rent difference on 70 m² apartment | ~€3/month | Meeting notes |
| Demand reduction: efficiency alone | 14–15% | Meeting notes |
| EWP target: 30% RE | 2030 | Heat Planning Act |
| EWP target: 100% RE | 2045 | Heat Planning Act |
| Pro Potsdam CO₂ emissions (approx.) | ~25 kt/year | Meeting notes |
