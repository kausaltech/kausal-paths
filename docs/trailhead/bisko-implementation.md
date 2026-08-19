# BISKO certification architecture — implementation plan

## Purpose

This plan turns the BISKO ownership and certification requirements in
[`bisko-requirements.md`](bisko-requirements.md) into a generic Trailhead
architecture. The immediate goal is to answer the certification review
correctly and explainably. The longer-term goal is to make the same machinery
usable by other frameworks and certification methods.

The current BISKO calculation graph contains useful diagnostics, but its
arithmetic 0/1 availability nodes are not a sufficient certification authority.
They can test whether a dataframe cell exists after loading, but cannot by
themselves establish:

- whether a zero was explicitly confirmed or silently pre-filled;
- whether a value came from an approved provider release or local primary data;
- whether use of a provider default was explicitly selected;
- which requirement version was applied;
- whether an absent category is missing, optional, or not applicable; or
- which exact evidence and model revision supported a published decision.

The intended end state is:

```text
published framework model release
    + versioned certification profile
    + versioned reference and provider data
    + city-owned data and evidence
    + city-owned presentation overlay
    -> effective city graph
    -> revision-scoped certification assessment
```

## Status

Planning document, 18 August 2026. No implementation described here should be
treated as landed merely because adjacent validation, revision, or model-editor
infrastructure already exists.

The current relevant building blocks are:

- `Framework.template_instance`, which currently acts as a source to clone when
  creating a framework instance;
- immutable `InstanceGraph` value objects and revision-backed instance
  snapshots;
- dataset and instance publication with pinned dataset revisions;
- UUID-identified nodes, ports, dimensions, categories, schemas, metrics, and
  data points;
- generic `DimensionScope` and `DatasetSchemaScope`, currently restricted to
  `InstanceConfig` in Paths;
- typed dataset metric validation rules with edit-time and publication-time
  enforcement; and
- `DataSource` and `DatasetSourceReference`, which retain bibliographic sources
  but not a complete evidence assessment.

## Load-bearing decisions

### Certification is an assessment, not a calculation node

The authoritative answer is a structured `CertificationAssessment` evaluated
against an immutable instance revision, model release, certification profile,
and dataset revisions. A node such as `is_bisko_compliant` may remain as a
read-only projection or transitional diagnostic, but it is not the persisted
certification record.

Overall certification has at least three states:

- `conformant`: every applicable blocking requirement is satisfied;
- `non_conformant`: at least one applicable blocking requirement is violated;
- `incomplete`: the evaluator cannot decide because evidence, applicability,
  computation, or required metadata is missing.

Unknown or unevaluated state must never collapse to conformant.

### Frameworks own reusable calculation and collection semantics

The BISKO framework owns:

- its common calculation graph;
- calculation dimensions and categories;
- dataset schemas and their valid category domains;
- reference and provider dataset roles;
- intrinsic validation rules; and
- compatible certification profiles.

City instances own observations, declarations, overrides, explicit source
selections, local extensions, and presentation. A framework-owned object is
effectively read-only when viewed through a city instance even if the city user
can edit the city-owned data connected to it.

The existing `FrameworkDimension` remains separate. It classifies
`FrameworkConfig` objects for such purposes as selecting defaults. Calculation
dimensions continue to use `kausal_common.datasets.Dimension` and gain a
`Framework` scope.

### Shared nodes are consumed through a versioned base graph

The current `bisko` instance becomes the initial authoring surface for the
shared BISKO model, referenced by `Framework.template_instance`. City instances
must not execute the mutable draft of that instance directly.

Publishing the template produces an immutable framework model release. The
effective graph for a city is composed from a pinned release and a city-owned
overlay:

```text
Framework.template_instance draft
        |
        | publish
        v
FrameworkModelRelease ------------------------------+
  - template instance revision                      |
  - framework catalogue snapshot                    |
  - compatible reference-data declarations          |
                                                     v
                                      effective city InstanceGraph
                                                     ^
City overlay ----------------------------------------+
  - municipal datasets and evidence
  - explicit provider/default selections
  - bindings at declared extension points
  - permitted city parameters
  - node layouts
  - optional local nodes
```

Changing the template creates a new release. Framework administrators may
advance many city drafts to it in one controlled operation, but published city
revisions stay pinned to their original release.

### City overlays cannot mutate the certified base

The base release owns node specs, ports, internal edges, and locked method
parameters. A city overlay may only change explicitly exposed extension
points. In particular:

- framework node calculation metadata is protected;
- framework internal ports and edges are protected;
- municipal datasets remain city-editable;
- selection of an accepted provider or municipal source is city-owned;
- node layout is city-owned presentation state; and
- local analysis nodes may be allowed without becoming part of the certified
  result.

Effective GraphQL permissions derive from origin (`framework` or `instance`)
and the requested operation. They do not rely solely on a copied
`is_editable=false` flag.

### Schema validity and certification requiredness are different

`DatasetSchema` declares which category combinations are meaningful. A
certification profile declares which meaningful combinations are required,
recommended, conditional, or acceptable alternatives.

For example:

- the schema says `industry × natural_gas` is a valid final-energy row;
- a certification requirement decides whether that row is mandatory;
- a dataset validation reports an impossible row; and
- a certification finding reports missing acceptable evidence for a required
  row or group of alternative rows.

Requiredness must not be stored on the category combination itself because the
same schema may be used by different certification profile versions.

### Human identifiers are the YAML authoring syntax; UUIDs are the durable contract

Framework YAML refers to shared objects by human-readable identifiers. Profile
compilation resolves them against the framework catalogue and stores UUID
references in the published release. Compilation fails on missing or ambiguous
identifiers.

Published identifiers are effectively immutable. A rename requires an
explicit alias or migration. Published assessments retain both resolved UUIDs
and authored identifiers for durable identity and understandable diagnostics.

### Data quality belongs to evidence, not to a bibliographic source

A `DataSource` identifies a publication, authority, edition, or declaration.
Its quality is contextual: the same source may be quality A for one value and a
lower grade for a derived allocation. Quality therefore belongs to the evidence
assertion connecting sources to a dataset or data point.

The existing parallel `quality` metric in BISKO final-energy datasets becomes
a derived calculation projection of effective evidence. It is not the
authoritative editable quality record.

## Domain model

### FrameworkModelRelease

Introduce an immutable published model release:

```text
FrameworkModelRelease
  uuid
  framework -> Framework
  version
  template_revision -> wagtailcore.Revision
  state: draft | published | retired
  created_at / created_by
  published_at / published_by
  supersedes -> FrameworkModelRelease | null
  catalogue_snapshot
```

`catalogue_snapshot` records the exact dimensions, categories, schemas,
category domains, metrics, and intrinsic validation rules used by the release.
It may initially be a typed `SchemaField`; normalised retention tables can be
added if later cleanup requirements demand them.

`Framework` gains an active/default model release. `FrameworkConfig` records
the release used by its current draft. Every published `InstanceConfig`
revision records the effective model release UUID.

The template instance remains an authoring workspace, not a live dependency of
published city calculations.

### Framework and instance catalogue scopes

Extend the Paths type and service boundaries to support:

```python
DimensionScopeType = Framework | InstanceConfig
DatasetSchemaScopeType = Framework | InstanceConfig
```

Do not scatter `framework OR instance` queries throughout loaders and GraphQL.
Add an effective-catalogue service with explicit methods such as:

```python
effective_dimensions(instance_config)
effective_dataset_schemas(instance_config)
resolve_dimension(instance_config, identifier)
resolve_dataset_schema(instance_config, identifier)
```

Resolution returns the union of framework-scoped resources and instance-local
additions. Duplicate identifiers across the effective catalogue are errors
unless an explicit replacement mechanism is introduced later.

Framework administrators control framework-scoped definitions. Instance
editors may use those definitions and edit their city-owned datasets, but may
not mutate the definitions. This requires separating schema-definition
permissions from dataset/data-point permissions; a framework-owned immutable
schema must not accidentally make all municipal data using it immutable.

### DatasetSchema category domain

Add a dedicated typed field to `DatasetSchema`, rather than using a parallel
template dataset:

```python
from pydantic import Field


class DatasetCategoryCombination(BaseModel):
    id: UUID
    categories: dict[UUID, UUID]  # dimension UUID -> category UUID


class DatasetCategoryDomain(BaseModel):
    mode: Literal['open', 'closed'] = 'open'
    combinations: list[DatasetCategoryCombination] = Field(default_factory=list)
```

Suggested model field:

```python
category_domain = SchemaField(
    schema=DatasetCategoryDomain,
    default=DatasetCategoryDomain,
    blank=True,
)
```

Semantics:

- `open`: combinations are not exhaustively prescribed; undeclared tuples may
  occur unless another rule prohibits them;
- `closed`: every populated tuple must match a declared combination; and
- each combination UUID is stable and may be referenced by findings and UI
  state.

Pydantic validation checks syntactic invariants and duplicate tuples. A
catalogue-aware validator checks that:

- every referenced dimension belongs to the schema;
- every category belongs to its stated dimension;
- a tuple mentions a dimension at most once;
- closed-domain tuples contain the required schema dimensions; and
- all references are available in the effective framework catalogue.

All schema write entry points call the same validator. Dataset edits and
publication validate populated data against closed domains.

The domain is serialized into framework releases, instance snapshots, dataset
schema GraphQL, and published assessments. A later relational representation
is justified only if category-level querying, independent combination edits,
or database deletion protection becomes important enough to outweigh the
simpler source-neutral JSON representation.

### Generic value validation

Generalise the existing dataset metric rule vocabulary into rules over a
tabular value subject. A rule subject may be:

- a dataset metric; or
- a node output port.

Dataset metric rules remain attached to `DatasetMetric`. Node output rules are
attached to `OutputPortDef`, which is the canonical persisted 1:1 description
of a runtime node metric.

Initial reusable rule kinds include:

- `value_range`;
- `dimension_sum`;
- `no_gaps`;
- `allowed_combinations`; and
- `required_combinations`.

Intrinsic validation and certification remain distinct:

- intrinsic rule: this output must never be negative;
- certification requirement: this BISKO profile requires this output to equal
  one for the assessed year.

The evaluator returns structured violations/findings rather than only booleans.
Node computation failure produces an incomplete finding and cannot satisfy a
requirement vacuously.

### CertificationProfile

Add a profile owned by a framework:

```text
CertificationProfile
  uuid
  framework -> Framework
  identifier
  version
  name
  state: draft | published | retired
  compatible_model_release -> FrameworkModelRelease
  supersedes -> CertificationProfile | null
  specification
```

Each published row is an immutable version. A BISKO method update creates a new
profile rather than rewriting an older one.

`specification` is a typed, source-neutral representation containing quality
schemes and certification requirements. If profile reuse across several
frameworks becomes real, replace the direct framework FK with an explicit
through model; do not add that indirection pre-emptively.

### CertificationRequirement

A requirement contains:

```text
identifier
level: required | recommended
subject: dataset metric | node output | instance setting
years
applicability
rule/assertion
accepted evidence constraints
citation
```

The authored YAML uses identifiers. Profile compilation produces a canonical
requirement snapshot whose subject and category references contain UUIDs.

`required` findings affect the overall certification state. `recommended`
findings are displayed and exported but do not prevent conformity. This is how
municipal-fleet data remains visibly recommended without becoming a false
certification precondition.

### DataEvidence

Promote provenance from individual source links to an evidence assertion:

```text
DataEvidence
  uuid
  dataset -> Dataset | null
  data_point -> DataPoint | null
  kind: observed | estimated | explicit_zero | provider_default
  source_classification
  coverage_extent: scope_wide | partial | unknown
  quality: DataQualityAssessment | null
  supersedes -> DataEvidence | null
  assessed_at / assessed_by
  created_at / created_by
  last_modified_at / last_modified_by
```

Exactly one of `dataset` and `data_point` must be populated. Add a database
check constraint for that local invariant.

`source_classification` is a profile-defined identifier such as
`municipal_primary` or `national_provider`. It describes the contextual role
of the evidence, not an immutable property of the cited publication. The
profile compiler validates it against the profile vocabulary.

`coverage_extent=scope_wide` means that the evidence covers the entire scope
being assessed, whether that scope is a municipality, state, country,
organisation, or another modelled entity. It does not require the values to be
stored as one aggregate total: a sector breakdown may still carry scope-wide
evidence. `partial` and `unknown` must not silently satisfy a requirement for
scope-wide coverage.

Quality is typed metadata:

```python
class DataQualityAssessment(BaseModel):
    scheme: str
    scheme_version: str
    level: str
```

The profile defines the available levels, their labels, and any numeric score
used for calculation projections. Evidence retains the scheme version so a
later profile does not reinterpret an old assessment silently.

Source links become:

```text
EvidenceSourceReference
  uuid
  evidence -> DataEvidence
  data_source -> DataSource
  role: primary | supporting
```

`DataSource` remains bibliographic: name, authority, edition, description, and
URL. Whether a source acts as municipal primary data, supporting material, or a
provider default is contextual evidence metadata, not necessarily an intrinsic
property of the publication.

Effective evidence for a data point resolves as:

```text
data-point evidence
    else dataset-level evidence
    else missing evidence
```

This avoids repeating a common source and quality assessment on every data
point while allowing a particular year or cell to override it. If real inputs
later require evidence shared by complex dataset slices, add a typed
metric/year/category selector to dataset-level evidence rather than inventing
implicit matching.

`supersedes` records provenance-preserving replacement of a provider default or
older municipal observation. The superseded evidence and provider value remain
available to published revisions.

Evidence is included in dataset revisions, instance exports, change history,
GraphQL, and certification findings.

### Derived quality calculation data

The materialization layer projects effective categorical quality into numeric
columns when a calculation node needs weighted quality. The profile's quality
scheme owns the categorical-to-numeric mapping.

During migration, the existing BISKO `quality` metric remains readable but is
not treated as a second authority. Existing values are imported into
`DataEvidence.quality`, compared against their paired energy values and source
references, and then regenerated from evidence. Any mismatch becomes a
migration report item rather than being silently resolved.

### CertificationAssessment and findings

Persist assessments against immutable inputs:

```text
CertificationAssessment
  uuid
  instance_config -> InstanceConfig
  instance_revision -> wagtailcore.Revision
  framework_model_release -> FrameworkModelRelease
  certification_profile -> CertificationProfile
  assessed_year
  status: conformant | non_conformant | incomplete
  evaluated_at
  evaluator_version

CertificationFinding
  uuid
  assessment -> CertificationAssessment
  requirement_identifier
  requirement_snapshot
  status: satisfied | missing | needs_confirmation | invalid | not_applicable
  subject UUIDs and authored identifiers
  years
  category-combination UUIDs
  evidence UUIDs
  structured details
```

Store the requirement snapshot used for each finding, not merely an FK to a
profile that might later be retired. Findings must be sufficient to explain a
historical assessment without evaluating current mutable state.

An assessment is recomputed for drafts when relevant graph, dataset, evidence,
selection, or profile state changes. Published assessments are immutable.

## YAML authoring

### Framework-scoped schema domain

The exact surrounding framework YAML format may evolve, but the intended
authoring form is concise and identifier-based:

```yaml
dataset_schemas:
- id: final_energy
  dimensions: [sector, energy_carrier]
  metrics:
  - id: energy
    unit: MWh/a

  category_domain:
    mode: closed
    combinations:
    - id: households_electricity
      categories:
        sector: private_households
        energy_carrier: electricity
    - id: households_natural_gas
      categories:
        sector: private_households
        energy_carrier: natural_gas
    - id: industry_electricity
      categories:
        sector: industry
        energy_carrier: electricity
    - id: industry_natural_gas
      categories:
        sector: industry
        energy_carrier: natural_gas
    # The real domain contains all meaningful combinations.
```

The compiler assigns or resolves stable UUIDs for the schema combinations.
Exported YAML retains identifiers; published snapshots retain UUIDs and
identifiers.

### BISKO grid-bound energy requirement

The certification floor currently documented in
[`bisko-requirements.md`](bisko-requirements.md#grid-bound-stationary-energy)
requires each grid-bound carrier at municipality level, rather than every
sector/carrier cell:

```yaml
certification_profiles:
- id: bisko-2024
  name: BISKO 2024
  version: 2024-07

  source_classifications:
  - municipal_primary
  - national_provider

  quality_schemes:
  - id: bisko
    version: 2024
    levels:
    - {id: A, score: 1.0}
    - {id: B, score: 0.75}
    - {id: C, score: 0.5}
    - {id: D, score: 0.25}

  requirements:
  - id: grid-bound-local-primary-data
    level: required

    dataset: final_energy
    metric: energy
    years: assessed

    cells:
      for_each:
        energy_carrier:
        - electricity
        - natural_gas
        - district_heating

      any_of:
        sector:
        - private_households
        - commerce_trade_services
        - industry

    accept:
      evidence:
      - observed
      - explicit_zero
      source_types:
      - municipal_primary
      coverage_extent:
      - scope_wide
      quality:
      - A

    citation:
      document: certification_protocol
      section: "1.1"
```

Semantics: for the assessed year, every listed energy carrier must have at
least one acceptable value among the listed sectors. A municipality-confirmed
zero is acceptable; a placeholder zero without explicit-zero evidence is not.

If the certification authority resolves the current ambiguity in favour of
mandatory sector-level cells, the profile changes `sector` from `any_of` to
`for_each` without an evaluator code change:

```yaml
    cells:
      for_each:
        energy_carrier:
        - electricity
        - natural_gas
        - district_heating
        sector:
        - private_households
        - commerce_trade_services
        - industry
```

`for_each` creates independently required groups. `any_of` lists alternative
valid cells that may satisfy one group. Both expand only over combinations in
the schema domain; the evaluator never invents a Cartesian product.

### Node-output requirement

Node validation uses the same authoring vocabulary:

```yaml
  - id: no-weather-correction
    level: required

    node: has_no_weather_correction
    output: value
    years: assessed

    assert:
      equals: 1

    citation:
      document: bisko_method
      criterion: 5
```

The compiler resolves the node's stable framework/template origin and output
port. Requirements must not persist a city-specific `NodeConfig` pk or rely on
a mutable display name.

### Conditional transport evidence

Requirements can select alternative evidence routes explicitly:

```yaml
  - id: road-transport-activity-or-energy
    level: required
    years: assessed

    one_of:
    - when:
        setting: use_mileage
        equals: true
      dataset: vehicle_kilometers
      metric: mileage
      cells:
        for_each:
          combination_set: required_road_mileage
      accept:
        evidence: [observed, provider_default, explicit_zero]

    - when:
        setting: use_mileage
        equals: false
      dataset: vehicle_energy
      metric: energy
      cells:
        for_each:
          combination_set: required_road_energy
      accept:
        evidence: [observed, estimated, explicit_zero]

    citation:
      document: certification_protocol
      section: "1.5"
```

This prevents the current class of false positive in which the own-energy
route is selected but only mileage availability is considered.

### Recommended municipal fleet

```yaml
  - id: municipal-fleet-data
    level: recommended
    dataset: municipal_fleet_energy
    metric: energy
    years: assessed
    cells:
      require_any: true
    citation:
      document: certification_protocol
      section: "1.5"
```

Missing data creates a visible recommendation finding but does not change an
otherwise conformant result.

## Evaluation semantics

### Dataset requirement evaluation

For each assessment:

1. Resolve the requirement's subject and selectors from the compiled profile.
2. Expand selectors against the pinned schema category domain.
3. Resolve the assessment year or year range.
4. Load raw stored values before interpolation, extension, imputation, or
   calculation defaults.
5. Resolve effective evidence for each candidate value.
6. Apply accepted evidence kind, source classification, quality, and
   applicability constraints.
7. Produce one structured finding for each independently required group.
8. Aggregate findings into the overall assessment without discarding reasons.

An explicit numeric zero is only acceptable when effective evidence says
`explicit_zero`, or when an accepted provider source itself supplies the zero.
A null, missing row, unconfirmed placeholder, or invalid category combination
does not satisfy presence.

Provider defaults satisfy a requirement only when:

- the requirement permits `provider_default`;
- the exact provider release is compatible with the profile;
- the city has explicitly selected or confirmed use of that default; and
- no active municipal override supersedes it.

### Node requirement evaluation

1. Build the effective graph from the pinned base release and city overlay.
2. Compute only the node outputs needed by the profile, using revision-pinned
   datasets and settings.
3. Evaluate generic output rules against raw node output.
4. Record computation or validation failures as incomplete findings.

Where conformity depends on behavior rather than coincidental output values,
the behavior should declare its semantic property. A consumer must not infer
the method by switching on concrete node classes. A diagnostic node may expose
that declared property as a normal output for profile evaluation.

### Aggregation

- any `required` finding with status `invalid` or a definite failed assertion
  makes the assessment `non_conformant`;
- any `required` finding with status `missing` or `needs_confirmation` makes
  the assessment `incomplete` unless a definite non-conformity also exists;
- `recommended` findings do not affect the overall state; and
- every applicable required finding must be satisfied for `conformant`.

The distinction between `missing` and `invalid` remains visible even if the UI
renders both as blocking.

## Graph composition and model-editor behavior

### Base graph

The framework model release contains framework-origin nodes, internal edges,
ports, method parameters, and declared input slots. Base node and port UUIDs
remain stable across compatible releases. A new release may add objects while
retaining the UUIDs of unchanged semantic objects.

### City overlay

The city overlay refers to base nodes and ports by UUID and contains only
city-owned differences. It must not use cross-instance ORM foreign keys as an
implicit inheritance system. Introduce explicit overlay snapshots or models
and compose them into `InstanceGraph` before constraint solving and hydration.

Municipal input slots should refer to framework-scoped dataset schema roles,
not to an editable empty dataset owned by the template instance. Reference and
provider datasets may be shared read-only releases; municipal datasets are
instantiated in the city scope.

### Layouts

Move effective layout ownership to `(city instance, node origin UUID)`. A new
base node initially uses the framework release's default layout. Moving it in a
city writes only the city overlay. Layout changes do not alter certification
semantics or the shared base graph.

### Local extensions

The first implementation may disallow local nodes in certified framework
instances. If they are later allowed:

- local UUIDs cannot collide with base UUIDs;
- they may connect only through declared extension ports;
- they cannot replace or remove base nodes or edges;
- certified outputs are evaluated from the protected base path; and
- any option that changes a certified result through an unapproved path makes
  that result non-conformant or supplementary.

## API and UI surface

### GraphQL

Expose at least:

- effective object origin and effective edit permissions;
- framework model release and certification profile versions;
- schema category domains and stable combination IDs;
- intrinsic dataset and node validation violations;
- evidence kind, quality, sources, and supersession;
- draft certification assessment and findings;
- published assessment history; and
- per-cell required/recommended/applicability state for the selected profile
  and assessment year.

Mutation inputs use UUIDs. Human identifiers remain YAML and display/export
syntax after compilation.

Evidence mutations should make semantically meaningful actions explicit:

- attach or replace sources;
- mark observed or estimated;
- confirm an explicit zero;
- select a provider default;
- supersede provider evidence with municipal evidence; and
- set a quality grade under a named scheme version.

Do not infer explicit-zero confirmation merely from writing numeric `0`.

### Model editor

The editor renders requirement metadata supplied by the backend:

- `Required`, `Recommended`, and `Conditional` markers;
- a filter for missing required evidence;
- distinct states for missing, unconfirmed zero, invalid, provider default,
  municipal override, and not applicable;
- source and quality controls near the value they describe;
- an explicit `Confirm zero` action;
- an explicit `Use provider default` action showing provider and release;
- a certification checklist with exact findings and citations; and
- generic `Protected` treatment for framework-origin calculation objects.

The UI must not reconstruct BISKO-specific requiredness from category names or
node identifiers.

## Implementation phases

Each phase ends at a review gate. Do not combine all migrations into one large
change: the shared graph, evidence, validation, and assessment boundaries can
be reviewed and tested independently.

### Phase 0 — Resolve normative ambiguities and capture regression fixtures

1. Record the exact review instance, assessed year, dataset revisions, settings,
   and operations used for findings 1.1, 1.2, and 1.5.
2. Obtain a certification-authority decision on municipality-level versus
   sector-level grid-bound requirements.
3. Confirm transport applicability and which provider defaults are accepted
   for rail, inland navigation, aviation, buses, and tram/metro.
4. Confirm whether road energy is a fully acceptable alternative route and
   the required vehicle/fuel combinations for that route.
5. Preserve the supplied prefilled-zero templates as migration fixtures.

**Review gate:** an approved decision table maps every review statement to a
blocking, recommended, conditional, or non-applicable requirement with a
source citation.

### Phase 1 — Typed schema category domains

1. Add `DatasetCategoryDomain` and stable combination IDs.
2. Add the `DatasetSchema.category_domain` `SchemaField`.
3. Implement catalogue-aware domain validation.
4. Add domain GraphQL and snapshot/export support.
5. Add `allowed_combinations` dataset validation.
6. Import the current BISKO `required_categories` and
   `required_road_mileage_categories` structures as schema domains or named
   combination sets.
7. Leave the existing template datasets readable during migration, but compare
   their semantics against the new domains.

**Review gate:** the BISKO final-energy and road-transport schemas represent
their ragged valid domains without consulting an editable dataset or inventing
a Cartesian product.

### Phase 2 — Framework-scoped catalogue

1. Permit calculation `Dimension` and `DatasetSchema` scope to `Framework`.
2. Implement the effective-catalogue service.
3. Split schema-definition permissions from dataset/data-point permissions.
4. Serialize the effective catalogue into snapshots.
5. Update loaders, GraphQL, sync, deletion, and copy paths to use the service.
6. Migrate common BISKO dimensions and schemas from city/template ownership to
   framework ownership without changing city dataset UUIDs.

**Review gate:** two BISKO city instances use the same read-only schema and
dimension definitions while independently editing their own datasets.

### Phase 3 — DataEvidence and quality

1. Add `DataEvidence`, typed quality, evidence kind, and supersession.
2. Replace or migrate `DatasetSourceReference` into
   `EvidenceSourceReference`.
3. Add exactly-one-target and source-role constraints.
4. Extend REST/GraphQL, snapshots, revisions, change history, export/import,
   and permissions.
5. Implement dataset-level default evidence and data-point override resolution.
6. Import existing source references.
7. Import the BISKO quality metric into evidence quality, reporting conflicts.
8. Derive calculation quality columns from effective evidence.
9. Add explicit zero and provider-selection mutations.

**Review gate:** an observed zero, an explicit municipality-confirmed zero, a
placeholder zero, and a provider zero are four distinguishable states through
storage, API, revision round trips, and calculation materialization.

### Phase 4 — Generic node-output validation

1. Extract the existing metric rule vocabulary into a subject-neutral module.
2. Add validation rules to `OutputPortDef`.
3. Evaluate only requested output subjects through the effective graph.
4. Generalize violation payloads while retaining dataset-specific API
   compatibility.
5. Expose node output validation through GraphQL and publication problems.
6. Convert appropriate BISKO plausibility and method diagnostics to generic
   output rules.

**Review gate:** the same typed value rule can validate a dataset metric and a
node output port, and node computation failure blocks a positive assessment.

### Phase 5 — Certification profile compiler and evaluator

1. Add typed profile and requirement specifications.
2. Parse the concise identifier-based YAML.
3. Resolve all references to UUIDs against the effective framework catalogue
   and template graph.
4. Fail compilation on ambiguous, missing, incompatible, or stale references.
5. Implement `for_each`, `any_of`, `one_of`, applicability, assessed-year, and
   evidence acceptance semantics.
6. Produce structured draft findings and an overall tri-state result.
7. Implement the requirements from the Phase 0 decision table.
8. Expose profile compilation diagnostics and draft findings.

**Review gate:** every initial certification comment is reproduced by an exact
test and explained by a structured requirement finding, without reading the
legacy `is_bisko_compliant` output.

### Phase 6 — Framework model releases and effective graph overlays

1. Add `FrameworkModelRelease` and publish the `bisko` template instance into
   its first release.
2. Give base nodes and ports stable release identity.
3. Define municipal dataset slots and permitted overlay operations.
4. Introduce city-owned bindings, settings, and layouts keyed to base UUIDs.
5. Compose release plus overlay into `InstanceGraph`.
6. Make GraphQL queries and mutations origin-aware.
7. Pin model releases in city revisions.
8. Add an upgrade operation that validates and advances selected city drafts.
9. Retain old releases while published revisions reference them.

**Review gate:** adding a protected node to a new BISKO release makes it
available in upgraded city drafts without copying or mutating city node rows,
while a previously published city revision computes against the old release.

### Phase 7 — Persisted assessments and publication integration

1. Add `CertificationAssessment` and `CertificationFinding`.
2. Evaluate against revision-pinned model and dataset inputs.
3. Persist immutable published assessments.
4. Add assessment history and export.
5. Decide whether non-conformity blocks ordinary model publication or only the
   separate act of declaring/certifying an inventory. Prefer separate actions
   unless the product requirement says every published draft must be certified.
6. Project current assessment status to any legacy API/node consumers.

**Review gate:** a published assessment can be explained and reproduced after
the framework, profile, provider data, and city draft have all advanced.

### Phase 8 — UI completion and legacy removal

1. Render backend-provided requiredness and findings in the dataset editor.
2. Add source, evidence, quality, zero-confirmation, and provider-selection
   flows.
3. Render framework-origin nodes and datasets as protected while retaining
   city layout editing.
4. Remove BISKO-specific client conditionals.
5. Compare legacy availability nodes with the new evaluator over all active
   BISKO instances.
6. Remove template requirement datasets and the legacy arithmetic conformity
   authority only after discrepancies are resolved.

**Review gate:** a municipality user can reach a conformant assessment from an
empty city overlay using only UI actions, and every remaining blocker is
visible before attempting certification.

## Migration strategy

### Existing BISKO instances

1. Create the BISKO `Framework` if it does not already exist in the target
   environment and point `template_instance` at the current canonical `bisko`
   instance.
2. Publish a baseline `FrameworkModelRelease` matching current production
   behavior before changing semantics.
3. Link city `FrameworkConfig` rows to that release.
4. Match existing city nodes to template origins by verified semantic identity
   and record a migration report. Do not guess when identifiers or structures
   diverge.
5. Preserve city-specific nodes and bindings as overlay candidates.
6. Move common dimensions and schemas to framework scope only after comparing
   UUIDs, identifiers, categories, and metric semantics.
7. Import data evidence and quality without deleting legacy columns or source
   links.
8. Run old and new conformity evaluation side by side for at least one release.

### Requirement template datasets

For each current requirement dataset:

1. read its raw category tuples;
2. compare them with the target schema dimensions and category catalogue;
3. assign stable combination identifiers;
4. store them in `category_domain` or a named combination set;
5. verify that missing and extra combinations produce the same diagnostic
   cells where the old diagnostic was correct; and
6. retain the old dataset until all linked instances have migrated.

### Source references and quality

Migration groups existing source references by target into evidence records.
Where several sources exist, retain all and require review before designating a
primary source if none is objectively known.

For each legacy quality value:

- map the numeric value to an explicit scheme and grade;
- ensure there is a corresponding energy value and target evidence;
- record conflicting or orphaned quality values;
- do not manufacture explicit-zero evidence from a numeric zero; and
- do not infer provider selection solely from the presence of provider data.

## Acceptance and regression tests

### Certification review cases

Tests use the exact assessed year and settings of the recorded review fixture.

1. Remove all local grid-bound electricity while leaving non-grid and
   transport defaults intact: assessment is not conformant and identifies
   electricity evidence.
2. Remove local natural-gas evidence from every sector: the natural-gas group
   fails.
3. Remove industry natural gas or private-household electricity: behavior
   follows the approved Phase 0 sector-level decision and produces either a
   blocking or recommended finding explicitly.
4. Leave a numeric placeholder zero for non-grid energy: assessment reports
   `needs_confirmation`.
5. Confirm the same zero explicitly: the relevant presence requirement passes.
6. Remove rail, inland-navigation, and aviation values entirely: each
   applicable mode remains represented by the schema domain and cannot
   disappear from aggregation.
7. Select an approved provider default explicitly: the corresponding transport
   requirement may pass and cites the provider release.
8. Remove a municipal value while a valid selected provider default remains:
   assessment uses the default and explains that choice.
9. Remove the municipal value and deselect the provider default: assessment is
   incomplete.
10. Select the mileage route and remove a required vehicle/road combination:
    mileage requirement fails.
11. Select the energy route and remove diesel or another required 2019
    vehicle/fuel combination: energy-route requirement fails; the mileage gate
    cannot make it pass.
12. Omit municipal fleet data: a recommendation is shown but overall conformity
    is unchanged.

### Evidence invariants

- exactly one evidence target is required;
- explicit zero requires value zero and a user/system confirmation record;
- non-zero values cannot carry `explicit_zero`;
- supersession cannot form a cycle;
- an active municipal override and provider default resolve deterministically;
- dataset evidence is inherited and data-point evidence overrides it;
- quality scheme and level references are valid for the pinned profile;
- source and evidence round-trip through dataset and instance revisions; and
- published evidence remains unchanged when draft evidence is edited.

### Framework release invariants

- a city cannot mutate a base node, port, or internal edge;
- a city can edit municipal data connected to an exposed base input slot;
- a city can move a base node without changing the framework layout;
- upgrading one city draft does not upgrade another implicitly;
- bulk upgrade is atomic per city and reports incompatible overlays;
- old published revisions retain their original model release; and
- stable base UUIDs survive compatible releases.

### Category-domain invariants

- closed schemas reject invalid tuples;
- open schemas do not invent required tuples;
- ragged domains do not become Cartesian products;
- duplicate tuples are rejected;
- category/dimension mismatches are rejected;
- profile selectors expand only over valid combinations; and
- human identifiers compile to the expected UUIDs.

## Operational tooling

Add read-only reporting before mutation commands:

- `framework_model_release_status <framework>`: template draft versus active
  release and linked city versions;
- `compile_certification_profile <framework> <profile>`: resolved references,
  errors, and canonical output;
- `certification_status <instance> --year <year>`: structured findings without
  persisting an assessment;
- `compare_certification <instance> --year <year>`: legacy node result versus
  the new evaluator;
- `evidence_status <instance> [dataset]`: missing evidence, unconfirmed zeros,
  quality conflicts, and provider selections; and
- `upgrade_framework_instances <framework> <release> --dry-run`: per-city
  compatibility and findings before applying an upgrade.

Every mutating command supports `--dry-run`, reports exact target UUIDs and
revisions, and operates transactionally per instance. Bulk operations continue
past independently failing cities only when explicitly requested and produce a
machine-readable failure report.

## Observability and audit

Log and measure:

- profile compilation failures by requirement and reference;
- assessment duration and node computations requested;
- finding counts by profile version, requirement, and status;
- evidence coverage and quality by schema, metric, carrier, and sector;
- framework release adoption across city drafts and published revisions;
- overlay incompatibilities during upgrades; and
- legacy/new evaluator disagreements during migration.

Audit entries must identify framework release, certification profile, assessed
instance revision, dataset revisions, evidence changes, user, and operation.

## Open decisions

These decisions are deliberately not hidden inside implementation defaults:

1. **Grid-bound sector granularity:** whether every sector/carrier cell is a
   certification precondition or municipality-level carrier coverage is the
   floor.
2. **Certification versus publication:** whether a non-conformant model may be
   published as a draft/public analysis while remaining uncertified.
3. **Transport applicability:** how a city declares that aviation,
   inland-navigation, tram/metro, or another conditional mode is not
   applicable, and who may approve that declaration.
4. **Quality schemes:** whether BISKO quality is entered directly as a grade or
   derived from more objective provenance attributes. The initial model stores
   the assessed grade and scheme version without preventing later derivation.
5. **Local nodes:** whether certified framework instances initially permit
   local calculation extensions at all.
6. **Evidence slices:** whether dataset-level default plus data-point override
   is sufficient before introducing evidence selectors for a dataset slice.
7. **Cross-framework profiles:** whether a certification profile ever needs to
   be shared by multiple `Framework` rows. Start with framework ownership until
   a real case requires a through model.

## Definition of done

The architecture is complete when:

- shared BISKO calculation objects and schemas have one framework-owned source
  of truth and versioned releases;
- cities can edit their observations, evidence, selections, and layouts without
  mutating certified calculation semantics;
- schema domains represent meaningful category tuples without requirement
  template datasets;
- dataset and node validation share one typed rule vocabulary;
- certification profiles are concise to author in YAML and compile to durable
  UUID references;
- evidence distinguishes observed, estimated, explicit-zero, provider-default,
  and superseded values with versioned quality;
- every assessment is structured, tri-state, explainable, and revision-pinned;
- all initial BISKO review cases have exact regression tests;
- framework upgrades can advance city drafts in bulk without rewriting
  published inventories; and
- the UI marks required and recommended inputs from backend-owned requirement
  metadata rather than BISKO-specific client logic.
