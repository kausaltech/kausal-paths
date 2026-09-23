# BISKO certification architecture — status and implementation plan

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
published template instance revision
    + versioned certification profile
    + versioned reference and provider data
    + instance-owned data and evidence
    + instance-owned bindings, settings, and presentation
    -> effective instance graph
    -> revision-scoped certification assessment
```

## Status

Updated 22 September 2026. This document started as the 18 August plan;
the statuses below distinguish implemented backend foundations from planned
certification and UI work. Implemented means present in the codebase, not
deployed or enabled for every existing instance.

| Area | Current status |
| --- | --- |
| Framework and quality catalogue | Implemented: semantic model modules, `DataQualityScheme`, `DataQualityLevel`, and idempotent BISKO provisioning. |
| Template inheritance | Implemented: published template revision selection, effective graph composition, local binding overrides, and atomic template publication. |
| Shared schemas and dimensions | Implemented: Framework or InstanceConfig scopes, separate schema/data permissions, and one dataset per schema per scope. |
| Editor permissions | Implemented: backend-computed node/port editability, separate binding editability, and editable local nodes in inheriting instances. |
| Dataset validation | Existing backend support includes typed metric rules, category domains, structured violations, and edit/publication enforcement. |
| Data-point quality and evidence | Minimal slice implemented: `DataPointEvidence` (kind + grade per data point), GraphQL read/write, snapshot round trip, derived quality columns, and legacy import. Dataset-level defaults, supersession, and source roles remain planned. |
| Certification criteria and assessment | Planned: no profile/requirement models, compiler, evaluator, or persisted assessment yet. |
| Data-editing UI | Can consume existing dataset validation and permission fields; quality/evidence and certification flows still need backend APIs and UI integration. |

Implementation details live in
[framework quality](../architecture/framework-quality.md) and
[template inheritance](../architecture/template-inheritance.md).
Framework models now live under `src/frameworks/models/`:
`framework.py`, `config.py`, `measures.py`, and `quality.py`, with public
imports preserved by `__init__.py`.

The latest full backend run passed 3,106 tests, with four skipped and three
expected failures. Ruff, targeted mypy, and migration consistency checks passed.
Local conversion experiments compared 504 node outputs across two existing
BISKO models and their default/baseline scenarios against stored reference
outputs. Those comparisons established numerical migration parity, not
certification. Different historical national reference-data editions must
remain pinned to reproduce those baselines.

The next backend slice for the data-editing UI is quality assignment and its
evidence/validation contract. Provisioning grades alone does not make quality
editable on `DataPoint`, and existing validation violations are not certification
findings. The detailed evidence design below remains a proposal for that slice.

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

Dependent instances own observations, declarations, overrides, explicit source
selections, local extensions, and presentation. Framework definitions are
read-only through the dependent instance even when its editor can modify the
local data connected to them.

The existing `FrameworkDimension` remains separate. It classifies
`FrameworkConfig` objects for such purposes as selecting defaults. Calculation
dimensions use `kausal_common.datasets.Dimension` with `Framework` or
`InstanceConfig` scope.

### Shared nodes are consumed through a published template revision

The `bisko` instance is the authoring surface referenced by
`Framework.template_instance`. A dependent instance selects its base through
`InstanceConfig.template_revision -> wagtailcore.Revision`. There is no separate
`FrameworkModelRelease` or `InstanceGraphImport` model, and no release FK on
individual bindings. One selected revision governs the dependent draft.

Throughout this document, a model release means a published template instance
revision. The generic terms are **framework instance**, **dependent instance**,
and **local**; framework membership does not imply a municipality.

```text
Framework.template_instance live draft
        |
        | publish: freeze reference data and validate all dependent drafts
        v
published Wagtail Revision <--- dependent InstanceConfig.template_revision
        |                                      |
        +------------------+-------------------+
                           v
                 effective InstanceGraph
                 + local nodes and datasets
                 + input binding replacements
                 + permitted settings and layouts
                           |
                           | publish dependent instance
                           v
                 self-contained instance snapshot
```

Individual template edits affect only the template draft. Publishing locks the
template and its dependent instances, validates their composed drafts, and
advances the template publication and all dependent draft pointers atomically.
A newly introduced structural conflict, missing target, or cycle aborts the
whole operation. Existing draft conflicts do not by themselves prevent an
otherwise compatible template update.

Published dependent instances retain their complete effective snapshots and
dataset pins; neither template edits nor template publication rewrites them.
Public views can therefore use published snapshots while model editors work on
drafts. Selectively advancing individual dependents is not the current
publication workflow.

### Ownership and editability are relative to the active instance

Inherited node definitions, port definitions, and template-owned bindings are
read-only in the dependent instance. The same nodes and ports are editable in
the template's own draft, subject to permissions. Local nodes remain editable
in an inheriting instance; framework membership is not a blanket edit lock.

`NodeMeta.can_edit()` centralizes the decision using a `NodeEditContext`.
GraphQL flags and mutation guards use the same policy. The effective
`Node.isEditable`, `InputPortType.isEditable`, and
`OutputPortType.isEditable` values account for permissions, draft/published
source, and ownership. The legacy node `is_editable` flag is retained for
ordinary protected local nodes; it is not the inheritance mechanism.

Port definition edits and binding edits are separate operations:
`InputPortType.bindingsEditable` reports whether sources may be replaced.
An inherited input with `binding_owner == 'instance'` accepts local sources;
other inherited inputs retain the template bindings.

Local inputs may be supplied either by a dataset or by a local node output.
District-heating emission factors are the motivating BISKO case: an instance
may compute its local factor in Paths. The same rule applies to other declared
local-input ports. Inherited outputs may also feed local nodes. These
connections must satisfy port contracts and preserve an acyclic graph.

Local layouts, goals, and permitted parameter values are instance settings.
Allowing a local calculation is implemented; deciding whether its evidence and
method satisfy a certification profile remains future assessment work.

### A shared schema still permits only one dataset per scope

`DatasetSchema` and calculation `Dimension` may be scoped to a `Framework`
or an `InstanceConfig`. A shared schema can serve datasets in many instances,
but the database uniqueness constraint on
`(schema, scope_content_type, scope_id)` is retained. Sharing across instances
does not require multiple datasets under the same schema in one instance.

The proposed constraint-removal migration was removed before pushing.
Alternative datasets within one scope need separate schemas under this model;
historical versions use revisions. Schema adoption during conversion retains
an existing local schema if another dataset in the same scope already uses the
target shared schema. The editor rejects a second dataset for that schema and
scope with a validation error.

Shared-definition visibility never grants access to another instance's values.
Schema-definition permissions are separate from dataset/data-point permissions,
so a protected framework schema can still receive locally editable data.
Legacy protected local datasets retain their previous edit restrictions.

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

### Template revision and binding storage — implemented

`Framework.template_instance` identifies the authoring instance.
`InstanceConfig.template_revision` selects the published base for a dependent
draft. Framework members must select a revision of their own framework's
template; nested template inheritance is rejected.

The generic `InputPortBindingSet` identifies
`(instance, node_uuid, port_uuid)` and stores a complete ordered replacement:

- no row: inherit the default binding;
- an empty binding list: explicitly disconnect;
- a populated list: replace the default with those dataset-metric or node-output
  sources.

Targets and sources use stable snapshot UUIDs, not a foreign key to a mutable
template `NodeConfig`. `InputPortBindingReference` supplies protective foreign
keys for referenced local datasets, metrics, and nodes.
`NodeInputPortBinding` remains ordinary local binding storage.
`InstanceConfig.node_settings` stores permitted local settings without copying
shared definitions.

This is generic graph composition, not a family of `FrameworkBinding*` models.
Binding validity is checked against the selected revision. Template publication
validates all affected composed drafts before advancing them.

### Framework and instance catalogue scopes — implemented

Paths supports:

```python
DimensionScopeType = Framework | InstanceConfig
DatasetSchemaScopeType = Framework | InstanceConfig
```

`frameworks.catalogue.dimension_scopes()` and `schema_scopes()` provide the
effective catalogue queries. `DatasetSchema.objects.for_scope_type(Framework)`
encapsulates content-type filtering, using Django's ContentType cache.

List queries annotate dataset schema editability with `Exists`, and graph
dataset/schema details are bulk-loaded and prefetched. Query-count tests cover
dataset lists and inherited ports with default or local dataset bindings.
The UI can consume `schemaIsEditable` separately from value-edit permissions.

### Quality schemes and levels — catalogue implemented

`DataQualityScheme` belongs to a framework and is unique by
`(framework, identifier, version)`. `DataQualityLevel` supplies a stable UUID,
identifier, label, description, order, and score in the inclusive range 0–1.

Provisioning seeds `bisko` scheme version `1` with A=1, B=0.5, C=0.25, D=0.
The version identifies the catalogue definition, not a certification protocol
edition. Model saves protect scheme identity and grade identity/score; changes
to scoring require a new version. Ungraded is absence of an assessment, not D.

These models store vocabulary only. The proposed evidence and certification
models below must reference the versioned catalogue rather than own duplicate
grade definitions. Quality assignment, its editing API, and the projection back
into numeric calculation columns are still to be implemented.

### DatasetSchema category domain — backend foundation implemented

The existing typed field on `DatasetSchema` represents valid combinations
without a parallel template dataset:

```python
from pydantic import Field


class DatasetCategoryCombination(BaseModel):
    id: UUID
    identifier: str
    categories: dict[UUID, UUID]  # dimension UUID -> category UUID


class DatasetCategoryDomain(BaseModel):
    mode: Literal['open', 'closed'] = 'open'
    combinations: list[DatasetCategoryCombination] = Field(default_factory=list)
```

The model field is:

```python
category_domain = SchemaField(
    schema=DatasetCategoryDomain,
    default=empty_category_domain,
    blank=True,
)
```

Semantics:

- `open`: combinations are not exhaustively prescribed; undeclared tuples may
  occur unless another rule prohibits them;
- `closed`: every populated tuple must match a declared combination; and
- each combination UUID is stable and may be referenced by findings and UI
  state.

Pydantic validation checks duplicate IDs, identifiers, and tuples. The intended
catalogue-aware write contract also requires that:

- every referenced dimension belongs to the schema;
- every category belongs to its stated dimension;
- a tuple mentions a dimension at most once;
- closed-domain tuples contain the required schema dimensions; and
- all references are available in the effective framework catalogue.

Completing and auditing this contract across schema write paths and shared
framework scopes remains part of Phase 1. Dataset rule evaluation already
supports allowed/required combination checks against category domains.

The domain is serialized into dataset/instance snapshots and exposed through
GraphQL. Published certification assessments must retain it when implemented.
Importing all BISKO requirement-template semantics into domains remains migration
work; valid combinations must not be confused with certification requiredness.
A later relational representation is justified only if category-level querying, independent combination edits,
or database deletion protection becomes important enough to outweigh the
simpler source-neutral JSON representation.

### Generic value validation — dataset rules implemented; node rules planned

Dataset metric rules and structured validation violations already exist, with
`block_edit` and `block_publish` enforcement. `Dataset.validationViolations`
reads current violations from materialization; data-point batch mutations also
return violations. The data-editing UI can consume this contract now.
Violations identify the rule, metric, years, and category coordinates, with
combination IDs and requirement-group identifiers where applicable. They
describe dataset validity, not quality grades or certification decisions.

The planned extension makes the existing rule vocabulary apply to a generic
tabular value subject. A rule subject may be:

- a dataset metric; or
- a node output port.

Dataset metric rules remain attached to `DatasetMetric`. Planned node output
rules will attach to `OutputPortDef`, which is the canonical persisted 1:1 description
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

### CertificationProfile — planned

Add a profile owned by a framework:

```text
CertificationProfile
  uuid
  framework -> Framework
  identifier
  version
  name
  state: draft | published | retired
  compatible_template_revision -> wagtailcore.Revision
  supersedes -> CertificationProfile | null
  specification
```

Each published row is an immutable version. A BISKO method update creates a new
profile rather than rewriting an older one.

`specification` is a planned typed, source-neutral representation containing
references to versioned quality schemes and certification requirements. If profile reuse across several
frameworks becomes real, replace the direct framework FK with an explicit
through model; do not add that indirection pre-emptively.

### CertificationRequirement — planned

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

### DataEvidence — planned

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

The framework quality catalogue defines levels, labels, and numeric scores.
The profile selects a compatible scheme version and accepted grades. Evidence retains the scheme version so a
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

### Derived quality calculation data — planned

The materialization layer projects effective categorical quality into numeric
columns when a calculation node needs weighted quality. The selected framework quality
scheme owns the categorical-to-numeric mapping.

During migration, the existing BISKO `quality` metric remains readable but is
not treated as a second authority. Existing values are imported into
`DataEvidence.quality`, compared against their paired energy values and source
references, and then regenerated from evidence. Any mismatch becomes a
migration report item rather than being silently resolved.

### CertificationAssessment and findings — planned

Persist assessments against immutable inputs:

```text
CertificationAssessment
  uuid
  instance_config -> InstanceConfig
  instance_revision -> wagtailcore.Revision
  template_revision -> wagtailcore.Revision
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

Certification YAML and its compiler below are proposed syntax, not an
implemented API. Quality schemes are provisioned independently; profiles select
their versions.

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
    version: "1"

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

## Evaluation semantics — planned

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

### Base graph and local bindings — implemented

The selected template revision supplies shared node specs, ports, internal
bindings, and pinned reference data. Composition with local nodes, data, binding
sets, and settings happens before constraint solving and hydration.

For inheriting instances,
`instanceEditor.setInputPortBindings(nodeId, portId, bindings)` writes a full
replacement. `bindings: null` restores the default; `bindings: []` disconnects.
Each source identifies either a node/output port or a dataset/metric, with
optional transformations. New structural conflicts return
`ConstraintViolations` without saving.

The ordinary `bindDataset` and per-edge mutation APIs continue to work for local
nodes, including local targets fed by inherited outputs. They update effective
local overrides where needed rather than writing template rows.

A default dataset connection is a convenience, not a restriction that excludes
a local calculation edge. Nor does connecting a default constitute an explicit
provider-evidence declaration; that future evidence action remains separate.

### Layouts and local extensions — implemented foundation

`InstanceConfig.node_settings` retains local layouts, goals, and allowed
parameter selections keyed by node UUID. Inherited node definitions remain
unchanged. Local nodes may consume inherited outputs and supply declared local
inputs; UUID collisions, invalid bindings, and cycles are rejected.

Certification rules for local calculations and their evidence remain planned.
The ability to connect a local district-heating emission factor does not itself
assert that the resulting inventory meets BISKO certification requirements.

### Local actions acting on framework nodes — engine implemented

A municipality's own measures (Mainz's heat plan, target paths, Masterplan
measures) attach to framework nodes as hooks rather than as new inputs: the
framework node's calculation is unchanged, the hook belongs to the local
action, and contributions only enter years after the last historical year, so
the certified balance cannot move. See
[action hooks](../architecture/action-hooks.md). Two kinds of local
customisation are therefore distinct:

- **replacing a framework input's bindings** (`binding_owner: instance`): data
  slots (`kommune/*`) and the method routes BISKO permits. These can move the
  balance and need citing against the Methodenpapier;
- **acting on a framework node** (hook): forecast-only, no ownership needed.

Until ownership is declared in the YAML, `declare_local_data_slots()` marks
every template input bound only to `kommune/*` datasets as instance-owned when
the template is published.

## API and UI surface

### GraphQL

Already available are effective node/port edit permissions and binding
editability, shared schema discovery and reuse, schema category domains,
typed dataset metric rules, structured dataset violations, and graph
`ConstraintViolations`.

The remaining certification/evidence surface should expose:

- selected template revision and certification profile versions;
- generic node-output validation violations;
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

### Model editor — remaining certification UI

The editor should render requirement metadata supplied by the backend:

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

The numbering below preserves the original plan; implementation order changed.
Framework provisioning, quality vocabulary, and template inheritance were taken
first. The status notes override the original future-tense task lists.
Evidence, generic node validation, and certification remain separate reviewable
slices. The next priority is data-point quality and the data-editing UI contract.

### Phase 0 — Resolve normative ambiguities and capture regression fixtures

**Status:** Open normative decisions. Numerical graph-conversion fixtures have been checked,
but they do not establish the certification decision table below.

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

**Status:** Backend foundation implemented. Catalogue-aware write validation,
shared-scope coverage, BISKO domain population, and comparison with legacy
requirement datasets remain to be completed.

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

**Status:** Implemented for template inheritance, the editor, and explicit BISKO conversion.
One dataset per schema per scope remains enforced; rollout to all instances is
not implied.

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

**Status:** Minimal data-point slice implemented (23 September 2026). It deviates from the
design above in shape, not intent:

- `frameworks.DataPointEvidence` is one row per data point with `kind`
  (`observed | estimated | explicit_zero | provider_default`, null = unknown) and
  `quality_level -> DataQualityLevel` (null = ungraded); at least one is set. It lives
  on the Paths side because `DataPoint` is shared with Watch and the grade vocabulary
  is framework-owned. There is no dataset-level evidence, supersession, or
  `EvidenceSourceReference` yet.
- Applicable grades come from the dataset's scope: framework, member instance, or
  template instance (`frameworks.evidence.quality_schemes_for_dataset`).
- GraphQL: `Dataset.qualitySchemes`, `DataPoint.evidence`, and `evidenceKind` /
  `qualityLevelId` on `createDataPoints` / `updateDataPoints`. A confirmed zero requires
  a zero value and must be cleared or changed in the same write that changes the value.
- A metric whose `spec.quality_of` names another metric is a projection: its column is
  derived from evidence scores in `DBDataset.deserialize_df` (ungraded → null), its
  stored points are ignored, and direct writes are rejected. `DatasetMetric.qualityOf`
  exposes the marker.
- Evidence rides in `DatasetSnapshot` (revisions and instance export/import); grades
  resolve by UUID, else by scheme/version/level identifiers in the target framework.
- `manage.py import_quality_evidence` converts legacy quality columns. A dataset whose
  values fall between grades is not projected unless `--snap-down` is given; Mainz has
  17 such averaged cells out of 1,879.

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

**Status:** Planned. Dataset metric validation is available; generic node-output rule
attachment and evaluation are not yet implemented.

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

**Status:** Planned; no certification criteria models or evaluator have been added.

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

### Phase 6 — Published template inheritance and effective graph overlays

**Status:** Implemented with existing Wagtail revisions instead of a separate
release model.

1. Publish the `bisko` template through `publish_template_instance()`.
2. Select the base with `InstanceConfig.template_revision`.
3. Keep stable node/port UUIDs and compose generic local binding sets and settings.
4. Permit local nodes and both dataset and edge sources at local-input ports.
5. Compute editability centrally in `NodeMeta` for the active graph and user.
6. Validate and advance all dependent drafts atomically on template publication.
7. Store self-contained published instance snapshots and pinned reference data.

**Verified behavior:** template draft edits do not affect dependent drafts;
compatible publication advances dependents; incompatible publication rolls back
the entire update; existing dependent publications remain unchanged.

### Phase 7 — Persisted assessments and publication integration

**Status:** Planned. Existing instance publication is not a certification assessment.

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

**Status:** Planned for evidence and certification. Backend editability and dataset
validation fields are already available for UI integration.

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

Provisioning and graph conversion are implemented; evidence import and
side-by-side certification evaluation in the final steps remain planned.

1. Create the BISKO `Framework` if it does not already exist in the target
   environment and point `template_instance` at the current canonical `bisko`
   instance.
2. Publish a baseline template revision matching the intended reference-data
   edition before changing semantics.
3. Attach `FrameworkConfig` membership and select that revision through the
   dependent `InstanceConfig.template_revision`.
4. Match existing city nodes to template origins by verified semantic identity
   and record a migration report. Do not guess when identifiers or structures
   diverge.
5. Preserve city-specific nodes and bindings as overlay candidates.
6. Move common dimensions and schemas to framework scope only after comparing
   UUIDs, identifiers, categories, and metric semantics. Retain separate local
   schemas where adoption would violate dataset uniqueness within a scope.
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

Certification review and evidence cases below are acceptance targets, not
passing tests for an implemented evaluator. Template inheritance, ownership,
binding, publication rollback, and query-count regressions are covered today.

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
- an instance can edit local data or connect a local calculation to an exposed input;
- the template author can edit its own draft nodes and ports;
- one schema can serve different scopes, but only one dataset in each scope;
- a city can move a base node without changing the framework layout;
- template draft edits do not alter dependent drafts;
- template publication advances all dependent drafts in one transaction;
- an incompatible dependent draft rolls back the whole publication;
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

### Implemented provisioning and conversion

`python -m tools.setup_bisko` supports:

- `--template IDENTIFIER`: select the existing template (default `bisko`);
- `--instance IDENTIFIER`: attach a database-backed instance without conversion;
- `--prepare-from IDENTIFIER`: reconcile template input declarations and category
  vocabulary against explicit migration examples;
- `--publish`: publish the template and atomically advance existing dependents;
- `--convert IDENTIFIER`: replace copied shared nodes with inheritance;
- `--reference-instance IDENTIFIER`: use a historical national reference-data
  edition for publication; requires `--publish`;
- `--dry-run`: run the operation and roll back database changes.

Instance flags can be repeated; `--instance` and `--convert` are alternative
attachment modes. The command is transactional as a whole. Provisioning is
idempotent, preserves existing UUIDs/settings, and rejects conflicting grades or
template identity. It does not create pages, organizations, users, or grants.
YAML-backed membership is rejected until the instance has been migrated to a
verified database-backed model.

Reference-edition selection is not an independent release channel for each
instance: publication still advances existing dependent drafts. Historical
output comparisons must use the intended pinned edition.

### Planned certification tooling

Read-only profile compilation, certification status/comparison, evidence
coverage reports, and assessment history remain to be implemented. Commands
must identify their assessed revision and inputs explicitly. A separate
per-instance or bulk release-upgrade command is not part of the implemented
lifecycle; publication is currently the atomic rollout boundary.

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
   derived from more objective provenance attributes. The proposed evidence model
   stores an assessed grade and scheme version; only the catalogue exists today.
5. **Local calculation acceptance:** local nodes and local input edges are
   supported. What evidence or method constraints a certification profile places
   on those calculations remains to be specified.
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
