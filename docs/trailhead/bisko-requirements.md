# BISKO dataset ownership and editability

This document records which inputs to a BISKO inventory are methodology-owned,
which are centrally supplied defaults, and which belong to the municipality.
It is based on the BISKO material received from the Agentur fuer kommunalen
Klimaschutz and ifeu under [`Attic/bisko`](../../Attic/bisko/).

The certification protocol is the authority for certification behavior. The
data-acquisition guide contains broader recommendations for producing a useful
inventory; these are not all certification preconditions. In particular,
municipal-facility data is strongly recommended, but the first review protocol
says that its absence must not by itself make an inventory non-conformant.

The latest implementation account considered here is the second-review
kick-off deck dated 17 August 2026. It describes intended corrections, not a
completed certification decision.

## Ownership model

There are three ownership classes:

1. **BISKO reference data** is immutable to municipality users.
2. **Provider defaults** are immutable source snapshots. A municipality may
   select them or supersede them with a separate municipal value, but must not
   edit the provider record in place.
3. **Municipal data** consists of local observations, explicit declarations,
   and overrides. It is editable by authorized municipality users.

Computed values are not a fourth editable class. They inherit the provenance
of their inputs and the version of the calculation method.

This distinction matters for certification. A provider value, a municipality
override, and a municipality-confirmed zero can have the same numeric value but
do not carry the same evidence.

## BISKO reference data: read-only

The following data and behavior must not be editable by municipality users:

| Dataset or behavior | Contents | Requirement |
| --- | --- | --- |
| Stationary final-energy emission factors | Factors for electricity, natural gas, heating oil, biomass, coal, and other carriers, including equivalents and upstream chains | Use the BISKO factors, or another centrally approved reliable factor set that includes equivalents and upstream chains. Users must not modify the factors. |
| Energy-generation emission factors | Fuel factors used for power plants, heating plants, and CHP calculations | These are method inputs, not municipal observations. |
| Federal electricity mix | Annual `Bundesstrommix` emission factor | Use it for all electricity consumption in the BISKO baseline, including electric transport. |
| Standard district-heating factors | BISKO method paper Table 5 factors for the permitted predominant generation variants | A municipality may select an applicable generation variant, but not edit its factor. |
| Transport factors | Road mileage split by fuel, specific energy use per vehicle-km, fuel well-to-wheel factors, rail and ship biofuel shares, and occupancy/load factors | These are national ifeu/TREMOD method data. Municipalities supply or select activity data, not conversion or emission factors. |
| Calculation methods | Exergetic allocation, activity-to-energy conversion, and fuel-to-emission conversion | The algorithm and its constants are versioned methodology. |

The supplied reference workbooks are:

- [`BISKO-Daten_ab_1990.xlsx`](<../../Attic/bisko/BISKO-Daten_ab_1990.xlsx>)
- [`Emissionsfaktoren_2023_Kausal_korrigiert.xlsx`](<../../Attic/bisko/Emissionsfaktoren_2023_Kausal_korrigiert.xlsx>)
- [`Emissionsfaktoren_stationaer_ifeu_2023_Kausal_EN.xlsx`](<../../Attic/bisko/Emissionsfaktoren_station%C3%A4r_ifeu_2023_Kausal_EN.xlsx>)
- [`Uebergabe Kausal Verkehr Faktoren Update 2025 final_EN.xlsx`](<../../Attic/bisko/%C3%9Cbergabe Kausal Verkehr Faktoren Update 2025 final_EN.xlsx>)

The first certification review explicitly requires that stationary and
transport emission factors cannot be changed by tool users. The
[second-review kick-off deck](<../../Attic/bisko/BISKO Certification/Kausal BISKO Certification 2nd Review Kick-Off.pptx>)
says this was corrected in the model editor.

### District heating

District-heating factors require a narrower distinction between input and
result:

- Local plant fuel inputs, heat output, electricity output, and required
  temperature data are municipal or local-provider observations and may be
  edited.
- The resulting factor is computed using the locked exergetic-allocation
  method and locked generation factors. It is not directly editable.
- If detailed plant data is unavailable, the municipality may select an
  applicable predominant generation variant from BISKO Table 5. The selected
  reference factor remains read-only.
- The review protocol also permits the conservative BISKO coal-CHP default
  described in the method paper.

## Provider defaults: immutable but replaceable

Provider defaults are source data rather than methodology constants. They may
be municipality-specific, but that does not make them municipality-owned.

| Provider dataset | Examples | Municipality interaction |
| --- | --- | --- |
| ifeu municipal transport activity | Road vehicle-km by vehicle type and road type; final energy for rail, inland waterways, and commercial aviation | The municipality may explicitly use the BISKO default or supersede it with local activity data. The imported ifeu value remains unchanged. |
| Census and statistical defaults | Buildings and heating systems, household sizes, floor area, population, employment, and industrial-energy estimates | Use as fallbacks and allocation inputs. Store better local evidence as an override. |
| Other national or regional defaults | BAFA-derived heat-pump, biomass, and solar estimates; DWD degree-day data | Retain the provider snapshot and its provenance. Local replacements are separate municipal values. Weather-corrected results are not the BISKO baseline. |

The supplied municipal transport defaults are:

- [`Uebergabe Kausal Verkehrsdaten 2023 20250605 final_EN.xlsx`](<../../Attic/bisko/%C3%9Cbergabe Kausal Verkehrsdaten 2023 20250605 final_EN.xlsx>)
- [`Verkehrsdaten_2010-2023_20260312ifeu.xlsx`](<../../Attic/bisko/Verkehrsdaten_2010-2023_20260312ifeu.xlsx>)

The KSP collection workbook states that its centrally supplied statistical
values, factors, and indicators may be refined by users. In Paths, refinement
should mean a provenance-preserving override, not mutation of the provider
row. See
[`Uebersicht_KSP-Datenerfassung_241009_EN.xlsx`](<../../Attic/bisko/Uebersicht_KSP-Datenerfassung_241009_EN.xlsx>).

## Municipal data: editable

### Required for BISKO conformity

#### Grid-bound stationary energy

The municipality must supply local primary final-energy consumption for:

- electricity;
- natural gas; and
- district heating.

At minimum, the data must cover the whole municipality and come from local
primary sources such as network operators. An explicit zero is permitted.
The data must have the highest data-quality category, A or 1.

Values split between private households, commerce/trade/services, industry,
and municipal facilities produce a better inventory. The first review
protocol, however, states the certification floor as local primary consumption
at least at whole-municipality level. The local
[`BISKO quality criteria.xlsx`](<../../Attic/bisko/BISKO quality criteria.xlsx>)
currently expresses the stricter sector-level interpretation; that difference
must be resolved before treating the workbook as the normative validator.

#### Non-grid stationary energy

There must be an entry for every applicable required non-grid carrier,
including:

- heating oil;
- biomass;
- solar thermal;
- heat pumps or environmental heat; and
- other locally applicable non-grid carriers.

The value may be observed consumption, a locally derived estimate, or an
explicit municipality-entered zero. The certification protocol does not make
the data-quality grade decisive for these carriers.

If credible data is pre-populated, the default may satisfy the input. If no
data is pre-populated, the municipality must enter a value, including zero
where appropriate. A silent placeholder zero must not be mistaken for a
municipality-confirmed zero. The first review found this distinction missing.

#### Transport

The inventory must contain the required activity or consumption data for:

- road traffic, differentiated by the required vehicle categories and, when
  mileage is used, consumption-relevant road type;
- scheduled buses and trams, light rail, or metro where applicable;
- local and long-distance passenger rail;
- rail freight;
- freight inland navigation; and
- commercial aviation where applicable.

Official ifeu activity defaults may be used where supplied. Better local road
traffic data is stored as a municipal override. Public road transport commonly
requires local operator data such as fuel or electricity consumption,
vehicle-km, passenger-km, or seat-km.

The municipality's choice to use provider defaults must be explicit and
visible. Removing a local value must not leave an apparently conformant
inventory unless a valid default remains selected.

#### Provenance and data quality

Every municipal energy input must retain:

- source and source type;
- applicable year;
- data-quality grade where required;
- whether the value is observed, estimated, or an explicit zero; and
- whether it supersedes a provider default.

The tool must calculate and expose data quality for the total inventory as
well as by relevant sector and energy carrier.

### Recommended or conditional municipal data

The following values are municipality-owned when supplied, but are not all
certification prerequisites:

- Municipal buildings, infrastructure, and street-lighting consumption.
  These should be separated from commerce/trade/services and subtracted from
  that sector to avoid double counting. The certification protocol says that
  their absence alone must not make the inventory non-conformant.
- Municipal fleet consumption by vehicle and fuel. The review protocol calls
  this recommended rather than mandatory.
- Detailed local traffic-model results replacing official activity defaults.
- Local plant inputs and outputs used to calculate district-heating factors.
- Local electricity generation data used for a supplementary territorial-mix
  presentation.

## Editable inputs that make a result non-BISKO

Paths may support the following for analysis, but a result using them must not
be labelled BISKO-conformant:

- a custom or local electricity emission factor in place of the federal mix;
- a direct override of any BISKO emission factor;
- weather correction of the baseline inventory; or
- a district-heating factor that is neither an approved BISKO default nor
  computed through the approved exergy method.

These options should live in an explicitly non-BISKO scenario or
supplementary result. The second-review deck describes this behavior for the
local electricity-factor and weather-correction switches.

## Required representation in Paths

The intended data flow is:

```text
versioned BISKO reference release
    + versioned provider defaults
    + municipality observations and overrides
    -> computed BISKO inventory
```

For every result, Paths must retain enough information to identify:

- the selected reference release and calculation-method version;
- the provider dataset release and original value, if used;
- the municipality override, if any;
- whether zero was provided by a source or explicitly confirmed by the
  municipality;
- the data-quality assessment; and
- any option that makes the result non-BISKO.

Provider updates must create a new version or materialization rather than
silently rewriting the evidence behind an already published inventory.

## Primary sources in the attic

- [BISKO method paper, July 2024](<../../Attic/bisko/Agentur_Methodenpapier_BISKO_Juli-24_EN.pdf>)
- [Municipal data-acquisition guide, 2024](<../../Attic/bisko/Agentur_Leitfaden_Datenbeschaffung_2024_EN.pdf>)
- [KSP handbook](<../../Attic/bisko/ksp-handbuch_EN.pdf>)
- [First Kausal certification review protocol, 16 June 2026](<../../Attic/bisko/260616_Zertifizierung_Kausal_erster Pr%C3%BCflauf_Pr%C3%BCfprotokoll.pdf>)
- [Second-review kick-off deck, dated 17 August 2026](<../../Attic/bisko/BISKO Certification/Kausal BISKO Certification 2nd Review Kick-Off.pptx>)

