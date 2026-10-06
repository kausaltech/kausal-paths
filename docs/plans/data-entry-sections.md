# Data-entry sections

Status: implemented in the backend and Data Studio on 5 October 2026. First design
recorded on 1 October; revised on 3 October after the graph-placement prototype and
on 5 October after design review. Ordered table entries and direct dataset choices
were added on 6 October. Existing database instances require their layout to
be synced and, for framework followers, published in the pinned template revision
before switching the UI. No live instances were migrated as part of implementation.

Implementation entry points:

- `src/nodes/defs/data_entry.py`: authored specs, UUID amendments and copy remapping;
- `src/nodes/data_entry.py`: pure graph resolution and metric-coordinate partition;
- `src/nodes/data_entry_yaml.py`: YAML reference resolution;
- `src/datasets/data_entry.py`: permission-checked, revision-aware data and findings;
- `src/nodes/graphql/types/data_entry.py`: embedded GraphQL section views;
- `configs/modules/bisko/model.yaml`: the existing BISKO page placement;
- Data Studio's `src/loaders/data-entry.ts`: section, table and count queries.

Implementation boundaries: node classes can declare category-preserving dependencies
through `Node.data_entry_dependencies`; unknown mappings widen and report their
approximation. Conditional input requirements that depend on another port remain
model-validation checks in the editor; the section summary layer does not compute
the dependency or count an evaluation reminder as a data finding. Direct required-value and qualifier checks share their
assessment logic with delivered-value validation. Published observations are read-only
and have no mutable data-point UUID in this view, because dataset revision payloads
preserve natural coordinates rather than data-point UUIDs. Neither the layout nor
permission results are stored on observations.

## Problem and scope

Data Studio finds the editable datasets from the effective instance-owned input-port
bindings, then assigns them to pages using node and sector identifiers hard-coded in
`src/config/data-entry.ts`. The loaders fetch the whole ownership graph and every
dataset, then rebuild the section view in the client. The UI is therefore tied to one
model layout, and it cannot ask for a section summary or a single table without
loading everything.

Move section membership into the backend. A section's contents are resolved from the
effective `InstanceGraph`, and GraphQL exposes the resolved sections with their
tables, data and findings. The UI renders what it receives and knows no BISKO node
names.

The implementation places dataset-backed instance inputs and explicitly selected instance datasets. Submodel input forms,
alternative views by data source, and splitting a dataset along delivery lines are
out of scope, but the contracts below are meant to carry them later.

## Two placement mechanisms

Sections are workflow organisation, not accounting categories. The page layout that
serves a municipality best follows the data it receives or produces: grid operators
deliver grid-bound energy, the city's building management delivers municipal
facilities, and so on. That usually coincides with how the model uses the data, but
not always. The municipal collection workflows have not yet been established. For
the initial migration, preserve the current Data Studio placement and page order,
including municipal facilities on their own page and district-heating consumption
on the sector pages. BISKO currently uses explicit placements only: production node
classes do not yet declare the category mappings needed for reliable discovery.
New inputs fall back to the unplaced section until assigned explicitly.

So placement has two mechanisms:

- **Graph-derived placement.** An `autodiscover` table entry names *anchors*: output ports, optionally sliced,
  that stand for the concept the section collects. The resolver walks upstream from
  each anchor through the effective bindings and collects the instance-owned inputs
  it reaches, with a translated category slice and an explicit indication of any
  approximation. This follows the effective graph, so a city's local submodel or a
  port added in a later template revision can receive a default placement without
  anyone editing the layout.
- **Explicit placement.** A table entry names an input port or dataset directly, optionally sliced. This
  corrects the walk where its result is awkward, places what the walk cannot see
  (below), and is the door for instance-specific, manual customisation.

Explicit placement takes precedence, cell by cell. The walk only places cells that no
explicit placement has claimed.

### Partition

Every data cell has at most one section. A *cell* is one coordinate of an
dataset eligible for data entry: a metric and one tuple of categories. Placement assigns
cells, not datasets, because one dataset can feed several sections in disjoint slices.
The initial BISKO layout splits final energy across the four sector pages, including
each sector's district-heating rows. The prototype's derived-only placement split it
across five sections; explicit placements preserve the existing four-way split.

The order that decides a cell's home:

1. Instance explicit placements, in declaration order.
2. Template explicit placements, in declaration order.
3. Exact graph-derived placement: paths whose slice translation required no widening.
4. Approximate graph-derived placement: paths containing a widened selection.
5. Cells nothing reaches go to a built-in *unplaced* group, with the reason. They are
   never dropped silently.

Within each derived tier, a cell reached by several sections goes to the first in
display order. Record competing claims and their precision as diagnostics. An
approximate claim never takes a cell from an exact claim merely because its section
appears earlier. Approximation is sticky along a path; another exact path to the same
cell can still establish an exact claim.

The partition gives each cell one editing home. Findings over several cells can be
relevant to several sections: their lists and counts follow the separate rules below,
so section finding counts are not necessarily additive.

A cell's identity is the dataset's own coordinate, not the section's. Tables do not
own observations, and the existing data-point and dataset UUIDs stay the mutation
identities.

### The walk

The walk is a pure function of `InstanceGraph`. It runs no computation and reads no
observations.

From an anchor `(node, output_port, slice)`, translate the output selection to the
node's contributing input ports, then traverse their bindings:

- Translate the slice backward through the binding's transformations, in reverse
  order:
  - `filter_dimension` / `filter_column` intersect the slice with the selection;
    `exclude` keeps the complement.
  - `flatten` re-introduces the dimension, restricted to the selection.
  - `assign_dimension` prunes the binding when the slice excludes the assigned
    category, and drops the dimension otherwise.
- A `groups` selection, a parameter-referenced filter, or an opaque transformation
  may prevent exact translation. Widen the affected dimensions and record the reason.
  If the affected dimensions are unknown, widen the entire selection. Such paths
  produce approximate claims; they must not silently retain downstream restrictions
  that could exclude contributing source cells. Raw-column filters are exact only
  when their values can be resolved to declared category identities.
- A dataset binding on an instance-owned port contributes the slice, restricted to the
  dataset's declared dimensions, for the bound metric.
- An edge continues into the referenced source output port. When that output is
  another section's anchor, stop only the portion overlapping that anchor's declared
  slices, and continue with the remainder. An anchor for households must not block
  commerce reaching the same output. A whole-output anchor stops the whole selection.
- Inside a node, pass restrictions unchanged only where its semantic contract proves
  preservation and identifies the contributing inputs. Reuse compiled shape rules
  where they establish those facts; shape equality alone does not prove category
  preservation. For unknown behavior, visit all potentially contributing inputs and
  widen any restrictions whose backward translation is unproven. Never assume that
  unchanged propagation through an arbitrary node is conservative.

The step vocabulary is the constraint solver's (`FilterStep`, `AssignStep`,
`OpaqueStep`). The walk should share that translation, not re-implement it, but it is
a separate pass: a requirement and a selection are different questions.

Selections are unions of rectangles: per dimension, a set of categories, with an
absent dimension meaning all. Commerce, for example, can be reached as
`sector=commerce_trade_services` through one path and as
`sector=municipal_facilities` through the double-count correction. A rectangle that
another one covers is dropped within the same precision/provenance treatment.
Intersection and subtraction must preserve the union, including holes left by prior
claims or sliced anchors. For example, `(households, electricity) OR (commerce, gas)`
is two rectangles, not all four combinations of their category sets. An empty set
of categories on a dimension selects nothing; an absent dimension selects all.
Authored `slices: []` means unrestricted, while an empty resolved rectangle union
means no cells. Keep those representations distinct.

### What the walk cannot see

Contributions switched inside a node are not structural. The municipal-facilities
correction reads its input only when the parameter
`municipal_facilities_included_in_commerce` is true, so the walk places those rows in
commerce even where the parameter is false. An explicit placement of
`sector=municipal_facilities` in the municipal facilities section resolves this:
explicit claims come first, so the walk never sees those cells. If parameter-gated
ports turn out to be common, a port could declare the parameter that gates it, and the
walk could evaluate it. Explicit placement is enough until then.

### Prototype findings

The walk ran against the BISKO review fixture, with the four sector emission nodes,
`transport_emissions`, `district_heating_emissions`, the three emission-factor nodes,
`weather_correction` and `net_emissions` as anchors. Compared with the hand-written
mapping in Data Studio:

- The sector, transport, emission-factor and settings sections matched.
- District heating partitions final energy differently. The sector edges exclude
  `energy_carrier=district_heating`, because district heating has its own factor
  route, so the district-heating rows of every sector land in the district heating
  section. Today the sector pages show them. This is a prototype observation, not
  the migration target: explicit sector placements will retain those rows on their
  current pages.
- Commerce picked up the municipal-facilities rows through the parameter-gated
  correction (above).
- Two instance-owned inputs reached no section. One feeds a local node that a template
  revision had disconnected, so it contributes nothing; the other feeds a reporting
  comparison, not the balance. The current UI files them under transport and "other".
  Preserve transport with an explicit placement and "other" through the unplaced
  fallback for the initial migration. Placement does
  not assert that an input contributes to the balance; connectivity diagnostics
  remain available separately. Previously unmapped inputs retain the UI's "other"
  fallback through the built-in unplaced section.

## Stored Pydantic specifications

```python
class DataEntrySliceSpec(BaseModel):
    categories: dict[UUID, list[UUID]]  # one rectangle; omitted dimensions mean all


class DataEntryAnchorSpec(BaseModel):
    node_id: UUID
    output_port_id: UUID
    slices: list[DataEntrySliceSpec] = Field(default_factory=list)


class DataEntryPlacementSpec(BaseModel):
    id: UUID
    kind: Literal['input_port'] = 'input_port'
    node_id: UUID
    port_id: UUID
    slices: list[DataEntrySliceSpec] = Field(default_factory=list)


class DataEntryDatasetSpec(BaseModel):
    id: UUID
    kind: Literal['dataset'] = 'dataset'
    dataset_id: UUID
    metric_ids: list[UUID] | None = None  # omitted: all value metrics
    slices: list[DataEntrySliceSpec] = Field(default_factory=list)


class DataEntryAutodiscoverSpec(BaseModel):
    id: UUID
    kind: Literal['autodiscover'] = 'autodiscover'
    anchors: list[DataEntryAnchorSpec]


type DataEntryTableSpec = Annotated[
    DataEntryPlacementSpec | DataEntryDatasetSpec | DataEntryAutodiscoverSpec,
    Field(discriminator='kind'),
]


class DataEntrySectionSpec(I18nBaseModel):
    id: UUID
    identifier: str | None = None
    name: I18nString
    description: I18nString | None = None
    tables: list[DataEntryTableSpec] = Field(default_factory=list)


class DataEntrySectionAmendmentSpec(I18nBaseModel):
    section_id: UUID
    name: I18nString | None = None  # omitted: inherit; explicit null: invalid
    description: I18nString | None = None  # omitted: inherit; null: clear
    tables: list[DataEntryTableSpec] = Field(default_factory=list)


class DataEntrySpec(BaseModel):
    sections: list[DataEntrySectionSpec] = Field(default_factory=list)
    amendments: list[DataEntrySectionAmendmentSpec] = Field(default_factory=list)
```

`InstanceModelSpec.data_entry` accepts an authored `DataEntrySpec`, a
`ComposedDataEntrySpec` retaining separate template and local definitions, or `None`.
The discriminator is `kind: authored | composed`; composed definitions are used only
for effective editions, never saved back as a follower's authored layout. An absent
layout is omitted from serialization to preserve existing pinned-template hashes.
GUI creation
allocates a section UUID once; the persisted spec requires it. Preserve field presence
for sparse amendment metadata through serialization, as for other local overrides.

**Ordered tables.** Each section contains one ordered `tables` list. Manual entries
select a dataset directly or follow an instance-owned input port. An `autodiscover`
entry expands into zero or more dataset tables at its position, sorted by dataset UUID.
Different datasets remain separate regardless of unit, quantity or dimension compatibility.
Different authored entries remain separate even when they select the same dataset.
The UI may render each value metric as its own editing grid.

Presentation order does not alter explicit-versus-derived ownership: manual cells are
reserved before discovery runs. Overlapping explicit entries at the same precedence
produce an `overlapping_tables` diagnostic; only the first claim renders the overlap.
A local explicit override of a template claim remains intentional and is not such a conflict.

**Manual sources.** An `input_port` entry follows the dataset bound to `(node, port)`;
use it in reusable templates when each follower supplies its own dataset. A `dataset`
entry fixes the dataset UUID, including an unbound dataset belonging to the instance.
It does not create a computational binding. Its slices use the dataset's own coordinates;
port selections use delivered coordinates and are translated back through the binding.
Direct dataset references participate in permission filtering, graph catalogs, revision
pins, export and copying, just like bound data. Missing sources remain visible diagnostics.
Schemas are not source identities: several datasets may share one schema.

**Identity.** A section's identity is its UUID, unique within the composed layout.
Routes and local references use that UUID. The optional `identifier` is a readable
authoring alias; GUI users need not supply one, and renaming a section does not change
its identity. Authored table entries also require UUIDs. Node, port, dataset, metric,
dimension and category references are UUIDs in parsed and persisted models.

For YAML authoring, accept an explicit `uuid`; otherwise require the readable `id`
and derive `uuid3(authoring_namespace_uuid, f'data-entry-section:{id}')`. The namespace
belongs to the declaring template/module or standalone instance and must be stable,
not the follower instance, file path or translated label. Shared module declarations
use the module's stable namespace. Resolve YAML aliases before persistence; amendments
persist the target section UUID. Export explicit UUIDs so aliases can be renamed
without changing identity. Table entries accept `uuid` or derive
`uuid3(section_uuid, f'data-entry-table:{id}')` from a required YAML alias.
YAML dataset choices use `dataset: <identifier>` and optional `metrics: [<identifier>]`;
resolve both against the dataset catalog, using persisted identities during DB sync.
Unknown or ambiguous aliases fail loading. Module dataset replacements also remap
these choices. Repeated imports of the same declaration yield the same UUID.

Exports and imports preserving model identity preserve section UUIDs. Copying a model
with fresh identities remaps locally owned section UUIDs and their references together;
references to sections in an unchanged pinned template retain the template UUIDs.

**Inheritance.** A template follower starts with the pinned template's sections in
template order. Compose its authored `DataEntrySpec` as follows:

- `sections` declares new local sections with their own UUIDs and required names;
  append them in declaration order. A local section cannot shadow an inherited UUID.
- `amendments` targets inherited sections by UUID. Supplied metadata overrides the
  inherited metadata; omitted metadata inherits. Table entries matching an inherited
  table UUID replace that entry in place; new UUIDs append in declaration order.
  Omitted table entries inherit. Local manual selections take precedence over template
  selections, including those in other sections.
- Local placement precedence is deterministic: amendments in declaration order, then
  new local sections in declaration order; within each, placement declaration order.
- At most one amendment may target a given inherited section. A missing target after
  a template upgrade is a resolution problem retaining the amendment and target UUID;
  it must not silently become a new nameless section.
- Removing or reordering inherited sections, or replacing their table
  lists wholesale, is not supported initially. Individual table entries can be overridden by UUID.

Omitted/null `data_entry` and an empty authored spec contribute no local declarations:
followers inherit the template layout, while standalone instances have no authored
sections. An empty spec does not disable inheritance. Inputs without a placement still
appear in the built-in unplaced section. A standalone/template spec cannot contain
amendments without a base layout.

Composition produces a separate immutable resolved layout; never save it back as the
instance's authored spec. Preserve amendment field presence and placement provenance
across snapshots. This keeps instances receiving later template sections and avoids
turning inherited declarations into local overrides.

**Validation and resolution.** Validate new authoring changes against the effective
graph before saving:

- section and table-entry UUIDs are unique, and supplied YAML aliases resolve unambiguously;
- amendment targets exist in the inherited layout;
- anchors name existing node output ports;
- input-port entries name existing **instance-owned** input ports;
- dataset entries name available datasets and value metrics;
- slice dimensions and categories exist and are valid in the referenced port's
  coordinate space (output space for anchors, delivered input space for ports, source space for datasets).

Translate explicit placements backward through their effective bindings to dataset
coordinates too. If a translation needs widening, report it; do not silently broaden
an explicit authoring claim. Require an exactly resolvable placement for that input.

Deserialization and composition of existing snapshots must instead retain stale
references as resolution problems, so a template upgrade cannot make the editor
unloadable. An invalid declaration makes no claim; valid declarations still resolve.
Keep the offending reference and its section or amendment context available for repair.

Inspect instance-owned ports independently of whether they have bindings. A required
unbound port, an explicit empty binding set, an external placeholder, and a binding
whose source cannot be resolved have distinct states; an intentional disconnect is
not automatically a missing-value error. Return unresolved inputs with node, port,
section association where known, and reason, with dataset identity optional. Do not
invent dataset cells for them. If no authored section can be associated, surface them
in the unplaced section. Submodel-backed inputs remain outside the initial form
implementation and must have an explicit unsupported-source state where applicable.

Store only layout and selection. Years, permissions, resolved datasets, cells, counts
and observations are resolved outputs. Include the spec in snapshots, template
composition, export and import. A published snapshot keeps its own layout.

**YAML.**

```yaml
data_entry:
  sections:
  - id: private_households
    name_de: Private Haushalte
    tables:
    - id: energy
      kind: input_port
      input: final_energy_use
      slices:
      - {sector: [private_households]}  # includes district heating, as today
    - id: discovered
      kind: autodiscover
      anchors:
      - node: private_household_emissions  # output_port required if ambiguous
    - id: local_notes_data
      kind: dataset
      dataset: city/additional_energy  # dataset identifier, resolved to UUID
      metrics: [energy]                # optional metric identifiers
  - id: municipal_facilities
    name_de: Kommunale Einrichtungen
    tables:
    - id: energy
      kind: input_port
      input: final_energy_use
      slices:
      - {sector: [municipal_facilities]}
    - id: discovered
      kind: autodiscover
      anchors:
      - node: municipal_facilities_emissions
```

## Resolved layout in InstanceGraph

`InstanceGraph.data_entry` returns the resolved, immutable layout:

- sections in display order, each with tables, including the built-in unplaced section
  last when it has content; its UUID is deterministic within the instance and its
  `kind` distinguishes it from authored sections (BISKO labels it "other");
- a table per `(authored entry UUID, dataset UUID)`, with **metric-specific selections**;
  unplaced tables use their section UUID instead. The resolved table UUID is
  `uuid3(entry_uuid, f'dataset:{dataset_uuid}')`; `entryId` links to the authored entry.
  Tables follow authored entry order, then dataset UUID within each expansion;
  each selection retains its union of rectangles and, where available, declared
  category-domain combinations. Never take a Cartesian product of the table's
  metrics with a shared selection;
- per metric-selection fragment, provenance: explicit (template or instance) or
  derived (anchor, path, precision and widening reasons), plus the ports reading it.
  A table may mix origins; it has no single placement enum;
- unplaced metric-selection fragments in the same table structure, with reasons,
  including partial leftovers of datasets already placed elsewhere;
- unresolved inputs with optional dataset identity, including ports with no binding;
- resolution problems: stale references, competing claims and widened slices.

The resolver is pure. It reads no observations, runs no calculation and caches no
user permissions, so its result can be cached per graph version.

A request-scoped query layer turns the selections into data: it loads the authorised
datasets, data points, evidence and findings in batches. It follows the selected
draft or published source and the dataset revision pins, and never reads newer data
silently.

## GraphQL contract

Expose the view under `instance.editor.dataEntry`. Illustrative fields:

```graphql
dataEntry {
  years
  defaultYear
  canManageYears
  problemCounts(years: $years) { ... }  # distinct instance total
  sections {  # optional sectionId argument for one section
    id  # UUID
    identifier  # optional authoring alias
    kind  # authored | unplaced
    name
    description
    tables {
      id
      entryId
      dataset { id }
      metricSelections {
        metricId
        fragments {
          rectangles { dimensions { dimensionId categoryIds } }
          placement  # template_explicit | instance_explicit | derived | unplaced
          precision  # exact | approximate
          path
          reasons
          nodeId
          portId
        }
      }
      dataPoints(years: $years) { ... }
    }
    unresolvedInputs { nodeId portId datasetId reason }
    problems(years: $years) { id shared affectedSectionIds ... }
    problemCounts(years: $years) { ... }
  }
  resolutionProblems { ... }
}
```

`dataPoints` returns only the union of the table's metric-specific cells and selected
years, without duplicates. Missing values retain those same metric-specific
coordinates. A UI can ask for section summaries without table data, and must never
need every dataset and every year to open one section. Dataset contents are shown only where the user may read
them; resolution problems name an input without revealing data the user cannot see.

## Year selection

Use `years: [Int!]` with the same meaning everywhere:

| Argument | Meaning |
| --- | --- |
| Omitted or `null` | All declared inventory years. |
| `[]` | No years; no data points or annual findings. Yearless problems remain separate. |
| Non-empty list | Those inventory years. |

"All years" means the declared inventory calendar (`YearsSpec.historical`), with its
skipped years. Explicitly requested years outside it are a validation error. Broader
history stays available through the ordinary dataset APIs. With no declared calendar,
expose that and invent no missing-cell grid; framework conversion now infers the
calendar from the inventory inputs, so this is the exception. `defaultYear` is the
latest inventory year, or null. The year-management capability comes from current
permissions, never from the layout.

## Section problems and counts

Distinguish at least:

- missing required values;
- missing required grades or evidence;
- plausibility and other dataset validation findings;
- resolution and configuration problems, exposed separately in `InstanceEditor.problems`.

Missing required cells need stable coordinates (dataset, metric, categories, year)
even when no data point exists. Requiredness comes from the value contracts and
dataset validation rules, not from every combination of schema categories.

A cell finding appears in the section owning its cell. A finding over several cells,
such as a plausibility sum across sectors, appears in every section owning any of its
affected cells, once per section. It carries a stable finding ID, `shared: true` when
applicable, and the affected section UUIDs. Dataset-wide findings without narrower
coordinates appear in every section containing that dataset. Unplaced cells follow
the same rules through the built-in unplaced section.

Each section exposes its own `problems(years:)` and `problemCounts(years:)`, including
shared findings, so the UI can render a badge directly and it agrees with the opened
section. The instance-level `problemCounts` deduplicates findings by their canonical
identity across sections; clients must not sum section badges to obtain that total.
Permissions apply before exposing findings, counts or affected-section references.

Definition problems appear as `DataEntryDefinitionProblem` entries in
`InstanceEditor.problems`, with `enforcement: BLOCK_PUBLISH` and section/node/port/
dataset UUIDs where available. They are excluded from section `problems` and all
`dataEntry.problemCounts`. Draft edits remain possible; ordinary instance and template
publication reject invalid layouts and GraphQL returns a typed
`DataEntryDefinitionProblems` result. `resolutionProblems` remains a diagnostic view.

Unbound, disconnected or external inputs encountered incidentally during discovery
are input states, not automatically definition errors. A manual table that requires
an unavailable source is a definition error. Required-input contracts and graph
constraints are checked independently in the model editor.

Return annual finding totals and a per-year breakdown, with yearless data findings
separate. A finding spanning several selected years counts once in the total and
once in each affected year's breakdown; per-year counts need not add up to the
total. Lists and counts share selection and deduplication rules.

`tools/setup_bisko.py` checks the same categories of instance problems before explicit
or default-quality-triggered template publication and after applying dependent draft
pins. A failed post-upgrade check rolls back the pin and associated changes; the
script's outer transaction rolls back the whole invocation on failure.
`--ignore-problems` reports the problems but allows publication and pin updates.
It does not bypass authorization, malformed snapshot errors or runtime failures.
Use `--dry-run` to exercise the workflow while rolling back all database writes.

Summary queries must not hydrate every data-point ORM object or initialise the
calculation runtime. Plausibility is evaluated per dataset over its whole frame, so
the request layer must load each dataset at most once per request, however many
tables share it. A shared finding is evaluated once and referenced from its sections.

## UI caching and migration

Scope query results to the instance, selected draft/published source and revision,
section UUID and requested years. Changes to values, grades, bindings, the layout or
the inventory calendar invalidate the affected data
and summaries. Observation identity is shared across tables showing the same data;
computed views stay scoped to their parent fields and arguments. Successful data
entry saves refresh the section findings and navigation badges.

Declare the BISKO layout in the module YAML, retaining the current Data Studio page
order and cell placement. Port the existing `src/config/data-entry.ts` mapping into
explicit placements: all four final-energy sector slices (including district heating),
the transport inputs, district-heating-specific inputs, emission factors and settings.
The built-in unplaced section keeps the current "other" fallback, including the review
fixture's reporting-comparison input. Discovery entries are disabled until real-node
category mappings are supported and tested against the BISKO graph. The retired
`passenger_kilometers_own` input has no layout placement.

Instances without a layout stay valid. Deploy the backend contract before switching
Data Studio, use section UUIDs in new routes, and resolve existing readable route
aliases through the backend during migration. Then remove the hard-coded mapping and
the client-side ownership joins. Workflow-driven regrouping is a later product
decision, informed by municipal workflows.

## Implementation sequence and verification

1. Specs, YAML resolution, validation, template composition with local additions, and
   snapshot round trips.
2. Pure resolution in `InstanceGraph`: the walk (sharing the solver's step
   translation), explicit claims, the partition, the unplaced group and diagnostics.
3. The request-scoped data and problem layer and the GraphQL fields, with the
   declared-year contract and permission checks.
4. The BISKO layout; Data Studio loaders and navigation; removal of the hard-coded
   mapping.

Verify:

- each transformation kind's backward translation, including the widening cases;
- sliced-anchor overlap stops, continued traversal of the remainder, and anchors on
  distinct output ports of the same node;
- category-changing and unknown node behavior, with no unproven restriction retained;
- precedence: instance explicit over template explicit over exact derived over
  approximate derived, cell by cell, including competing-claim diagnostics;
- rectangle subtraction and metric-specific selections, without extra Cartesian cells;
- the partition: every instance-owned cell in exactly one section or in the unplaced
  group;
- ordered discovery/manual interleaving, separate slices of one dataset, direct unbound datasets,
  dataset-identifier resolution, table amendments, overlapping manual selections, and UUID remapping;
- input-port placements surviving a dataset replacement and a template upgrade, and failing
  visibly when their port disappears;
- unbound ports, intentional disconnects, external placeholders and invalid sources;
- UUIDv3 stability across YAML imports and template followers, GUI sections without
  aliases, and copy/export/import reference remapping;
- sparse amendment round trips, missing amendment targets after upgrades, and
  omitted/null/empty layout inheritance;
- current Data Studio placement and page order as migration regression cases,
  including district-heating rows retained on sector pages;
- shared findings counted once per affected section and once in the instance total,
  with matching problem lists and distinct yearless counts;
- all/null/empty/subset year arguments, skipped and undeclared years;
- permission isolation, and query counts for summary-only and table queries.
