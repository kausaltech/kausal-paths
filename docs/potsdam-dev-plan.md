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
| Gebäude- und Energiedaten Pro Potsdam (Excel) übermitteln | Sebastian Möller / Pro Potsdam | baldmöglichst |
| Aktuellen Brennstoffmix EWP Fernwärme bestätigen | Cordine Lippert / EWP | baldmöglichst |
| Energiepreise für Fernwärme und Erdgas (EUR/MWh) | EWP / Stadtwerke | baldmöglichst |
| Modellentwicklung und Integration | Kausal (Sonja-Maria / Jouni) | laufend nach Datenverfügbarkeit |

---

---

## Part B — Technical Implementation Plan

### Context

This document describes the `potsdam-dev` model instance: a development fork of `potsdam-gpc` extended to analyse the EH 55 vs. EH 40 building standard question and the interaction between building renovation and energy supply decarbonisation. The instance is explicitly designed to model Pro Potsdam Wohnen as a standalone sub-system.

---

### 0. Prerequisites

**Migration issue (must fix before any development)**

Migrations `0042–0044` were applied to the local DB from `feat/trailhead` but the source files do not exist on `main`. Run these three commands in the terminal:

```bash
git show feat/trailhead:nodes/migrations/0042_dataset_port_spec.py \
  > nodes/migrations/0042_dataset_port_spec.py

git show feat/trailhead:nodes/migrations/0043_alter_datasetport_options_datasetport_dataset_index.py \
  > nodes/migrations/0043_alter_datasetport_options_datasetport_dataset_index.py

git show feat/trailhead:nodes/migrations/0044_instanceconfig_is_locked.py \
  > nodes/migrations/0044_instanceconfig_is_locked.py
```

Then add the `is_locked` model field to `InstanceConfig` in `nodes/models.py` (copy from `feat/trailhead` around line 392). After this, `python manage.py migrate` should confirm all migrations as already applied.

---

### 1. Instance file

- Source: `configs/potsdam-gpc.yaml` → copy to `configs/potsdam-dev.yaml`
- Change `id: potsdam-gpc` to `id: potsdam-dev`
- Change `site_url` to a dev URL (e.g. `https://potsdam-dev.paths.staging.kausal.tech`)
- Keep all existing nodes, actions, and scenarios intact — add new content only

---

### 2. New dimensions

#### 2a. `stakeholder` (define locally in potsdam-dev.yaml)

Not in `frameworks/standard_dims.yaml`. Define in the config:

```yaml
- id: stakeholder
  label_en: Stakeholder
  label_de: Akteur
  categories:
  - id: tenant
    label_en: Tenant
    label_de: Mieterin / Mieter
  - id: pro_potsdam
    label_en: Pro Potsdam
    label_de: Pro Potsdam
  - id: city
    label_en: City of Potsdam
    label_de: Stadt Potsdam
  - id: society
    label_en: Society / Climate
    label_de: Gesellschaft / Klima
```

#### 2b. `cost_type` — already in `frameworks/standard_dims.yaml`

Existing categories cover `capex`, `opex`, `energy_costs`. One addition needed: `rent` (cold rent pass-through component). Either add it to `standard_dims.yaml` or define locally.

---

### 3. Fix to existing node (one line)

`heat_generation_individual` at line 543 of `potsdam-gpc.yaml` excludes heat pumps from the individual supply transition. This must be fixed for the `individual_heating_transition` action to work:

**Current:**
```yaml
      categories: [district_heating, solar_thermal, biogas, biomass]
      exclude: true
```

**Change to:**
```yaml
      categories: [district_heating]
      exclude: true
```

This allows `natural_gas → environmental_heat` (heat pump) transitions to flow through `heat_generation_individual` → `emission_change_due_to_ef`.

---

### 4. Pro Potsdam sub-model (standalone parallel track)

**Design decision:** Pro Potsdam nodes do NOT feed into `corrected_emissions` or `net_emissions`. They form a parallel calculation tree that shares the district heating emission factor from the main model but uses Pro Potsdam's own energy data. This avoids double-counting with the city-wide aggregate data and allows development independent of the main model restructuring.

Merge into the main model (making `pro_potsdam` a sub-sector of `heating`) is deferred until the Pro Potsdam data is validated and the city-wide data split is confirmed.

#### Building stock nodes

| Node id | Type | Quantity | Unit | Value / Source |
|---------|------|----------|------|---------------|
| `pro_potsdam_apartments` | `simple.FixedMultiplierNode` or constant dataset | number | dimensionless | 18,000 (meeting notes) |
| `pro_potsdam_floor_area_per_apt` | constant | area | m²/apartment | ~70 m² (to be confirmed by Pro Potsdam data) |
| `pro_potsdam_total_floor_area` | `simple.MultiplicativeNode` | area | m² | apartments × area_per_apt ≈ 1,260,000 m² |
| `pro_potsdam_heat_demand_per_m2` | constant / dataset | energy | kWh/m²a | ~95 kWh/m²a (= 120,000 MWh / 1,260,000 m²) |
| `pro_potsdam_heat_demand` | `simple.MultiplicativeNode` | energy | MWh/a | total_floor_area × heat_demand_per_m2 |
| `pro_potsdam_emissions` | `simple.MultiplicativeNode` | emissions | kt/a | heat_demand × district_heating_emission_factor (shared from main model) |

**Key interaction:** `pro_potsdam_emissions` reads the `corrected_emission_factor` node (district_heating, heating sector) from the main model. This means Pro Potsdam's emissions automatically reflect EWP's decarbonisation trajectory as modelled by the existing `district_heating_until_2030` / `..._2030_2040` / `..._2040_2045` actions.

---

### 5. Energy price and cost nodes

The stub at lines 476–509 of `potsdam-gpc.yaml` can be uncommented and extended.

| Node id | Quantity | Unit | Dims | Notes |
|---------|----------|------|------|-------|
| `energy_prices` | unit_price | EUR/MWh | [energy_carrier] | Data needed: Fernwärme and Erdgas prices, with forecast |
| `pro_potsdam_energy_cost` | currency | EUR/a | — | pro_potsdam_heat_demand × energy_prices[district_heating] |
| `pro_potsdam_energy_cost_per_apt` | currency | EUR/apartment/a | — | energy_cost / apartments |
| `renovation_cost_per_m2` | unit_price | EUR/m²/month | — | EH 55 = 18, EH 40 = 21 (cold rent, from meeting notes) |
| `pro_potsdam_cold_rent` | currency | EUR/a | [stakeholder] | rent_per_m2 × total_floor_area, split by stakeholder via subsidy action |
| `pro_potsdam_renovation_capex` | currency | EUR | — | Additional investment cost of EH 40 over EH 55 (data gap) |
| `pro_potsdam_costs_by_stakeholder` | currency | EUR/a | [stakeholder, cost_type] | Aggregates energy_cost + rent + annualised capex |
| `pro_potsdam_co2_cost` | currency | EUR/a | — | pro_potsdam_emissions × CO₂ shadow price; stakeholder = society |

---

### 6. New actions

#### a. `pro_potsdam_renovation_eh55`

- **Group:** `heat_consumption`
- **Mechanism:** Reduces `pro_potsdam_heat_demand_per_m2` for the renovated/newly-built fraction of the portfolio. Uses a ramp-up dataset (renovation rate × floor area per year).
- **Target node:** `pro_potsdam_heat_demand`
- **Data needed:** Annual renovation rate (units/year or m²/year) from Pro Potsdam Excel; assumed ~14–15 % demand reduction vs. unimproved baseline.

#### b. `pro_potsdam_renovation_eh40`

- **Group:** `heat_consumption`
- **Mechanism:** Additional demand reduction on top of EH 55, same renovation rate assumption, ~14–15 % further reduction (i.e. ~28–30 % total vs. baseline).
- **Target node:** `pro_potsdam_heat_demand`
- **Note:** In the UI, this action should only be visible/meaningful when EH 55 is also enabled. Document this dependency clearly.

#### c. `pro_potsdam_rent_subsidy`

- **Type:** Parameter action (slider 0–100 %)
- **Mechanism:** Redistributes the rent cost component between `tenant` and `city` stakeholder categories in `pro_potsdam_cold_rent`. At 0 % the full cost passes to the tenant; at 100 % the city absorbs the EH-40-driven rent increase.
- **Target node:** `pro_potsdam_cold_rent` (stakeholder dimension)

#### d. `individual_heating_transition`

- **Group:** `heat_generation_individual`
- **Mechanism:** Models the replacement of natural gas boilers in non-district-heating buildings with heat pumps (`natural_gas → environmental_heat`). Feeds into the existing `heat_generation_individual` summary node → `emission_change_due_to_ef`.
- **Data needed:** Share of Potsdam heating outside district heating network that is currently natural gas; annual replacement rate assumption.
- **Prerequisite:** Fix to `heat_generation_individual` node (Section 3 above) must be in place.

#### e. District heating transition (existing)

The actions `district_heating_until_2030`, `district_heating_2030_2040`, `district_heating_2040_2045` already model EWP's trajectory. No new action needed. These feed `pro_potsdam_emissions` automatically via the shared emission factor.

---

### 7. Data gaps — what is needed before the model can be parameterised

| Data item | Source | Impact if missing |
|-----------|--------|-------------------|
| Pro Potsdam: floor area per apartment, units by building type/age, actual energy consumption breakdown | Sebastian Möller, Pro Potsdam (Excel promised) | Cannot set `pro_potsdam_heat_demand_per_m2` or renovation ramp-up accurately |
| EWP current fuel mix in district heating (% natural gas, % other) | EWP / Cordine Lippert | Cannot set baseline emission factor for district heating; currently modelled as single carrier |
| Energy prices: Fernwärme and Erdgas EUR/MWh (historic + forecast) | EWP / Stadtwerke Potsdam | Cannot populate `energy_prices` node |
| Individual heating outside DH network: fuel split (natural gas fraction) | Kommunale Wärmeplanung data | Cannot parameterise `individual_heating_transition` action |
| Additional investment cost: EH 40 over EH 55 per m² (construction, not just rent) | Engineering study (Gregor Heilmann to commission) | Cannot populate `pro_potsdam_renovation_capex` |

---

### 8. Scenarios to add

| Scenario id | Description | Actions enabled |
|-------------|-------------|-----------------|
| `pro_potsdam_eh55_only` | EH 55 standard + full EWP transition | EH 55 renovation ON, EWP actions ON, EH 40 OFF |
| `pro_potsdam_eh40_only` | EH 40 standard + full EWP transition | EH 55 + EH 40 renovation ON, EWP actions ON |
| `renovation_without_transition` | EH 40 but no EWP decarbonisation | Both renovation actions ON, EWP actions OFF |
| `transition_without_renovation` | EWP full transition, no renovation | EWP actions ON, both renovation actions OFF |

Comparing `pro_potsdam_eh55_only` vs. `pro_potsdam_eh40_only` directly answers the EH 55/40 climate question. Comparing the last two scenarios shows which lever dominates.

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
| Pro Potsdam CO₂ emissions (approx.) | ~25,000 t/year | Meeting notes (likely kt, not t) |
