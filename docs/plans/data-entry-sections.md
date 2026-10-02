# Backend-defined data-entry sections

Status: proposed. Agreed design recorded on 1 October 2026; implementation has
not started.

## Problem and scope

Data Studio currently discovers editable datasets from effective instance-owned
input-port bindings, then assigns them to pages using hard-coded node and sector
identifiers in `src/config/data-entry.ts`. The loaders fetch the ownership graph
and datasets and reconstruct the section view client-side. This couples the UI
to one model layout and prevents efficient queries for section summaries and
selected tables.

Move the layout and selection rules into the model specification. Resolve the
layout against the effective `InstanceGraph`, and expose the selected data and
findings through GraphQL. The UI should render backend-defined sections without
knowing BISKO node names or implementing its own dataset selection rules.

The first implementation supports schema-backed dataset tables. It does not
implement submodel input forms, arbitrary upstream graph traversal, multiple
collection views, or the proposed split of municipal consumption into a separate
collection dataset. Those can build on the same contracts later.

## Presentation and source identity

Sections are framework-authored workflow organization, not accounting sectors
prescribed by BISKO. A section may select part of a dataset, or contain several
tables backed by different datasets. A dataset may appear in several sections
with different slices. The initial BISKO layout can preserve the existing UI's
sector, transport and supporting-input pages.

The Klimaschutz-Planer handbook distinguishes subject-oriented entry from entry
by data source. That supports future alternative views over the same collection
inputs; it does not require us to implement those views initially.

Connect the initial declarations to **dataset schemas**, rather than a particular
municipality's dataset rows. A schema describes the reusable collection shape;
the resolved dataset supplies observations for the current instance. Effective
bindings remain authoritative about which sources actually supply the model.
An unused dataset must not appear as the active source merely because its schema
matches.

The existing uniqueness constraint is one dataset per schema **per scope**.
Several visible scopes can still produce multiple candidates. Resolve against
the effective graph and its instance/default selections; do not choose the first
matching ORM row. Missing, ambiguous and unsupported sources must be explicit
resolution problems. Do not silently omit the affected table.

Future model-input-port-backed declarations can identify a logical collection
input whose effective source may be a dataset or a local submodel. Keep table
declarations separate from sections so that their source can evolve into a
discriminated reference. Do not automatically expose every upstream dataset of
a submodel: its author should declare its collection inputs.

## Stored Pydantic specifications

Add pure specification models, using the existing i18n types:

```python
class DataEntrySliceSpec(BaseModel):
    dimension_id: UUID
    category_ids: list[UUID]


class DataEntryTableSpec(I18nBaseModel):
    id: UUID
    identifier: str
    dataset_schema_id: UUID
    name: I18nString | None = None
    metric_ids: list[UUID] | None = None
    slice: list[DataEntrySliceSpec] = Field(default_factory=list)


class DataEntrySectionSpec(I18nBaseModel):
    id: UUID
    identifier: str
    name: I18nString
    description: I18nString | None = None
    tables: list[DataEntryTableSpec] = Field(default_factory=list)


class DataEntrySpec(BaseModel):
    sections: list[DataEntrySectionSpec] = Field(default_factory=list)
```

Add `data_entry: DataEntrySpec | None = None` to `InstanceModelSpec`.
`None` means inherit the template layout, if any; an explicitly empty
`DataEntrySpec` overrides it with no sections. For the initial implementation,
a local declaration replaces the whole inherited layout. Do not introduce
implicit per-section merging.

Section and table list order defines display order. Persist UUID identity;
identifiers provide readable authoring references and section routes. UUIDs must
be assigned stably at the configuration boundary, not regenerated on every YAML
load. Readable YAML schema, metric, dimension and category names resolve to UUID
references before persistence.

Validate unique section/table identities and identifiers, metrics belonging to
the selected schema, and slice dimensions/categories belonging to that schema.
An omitted metric selection means all applicable entry metrics, excluding legacy
quality projection metrics. An explicit selection must be nonempty. Slice
category lists must be nonempty, with no duplicate dimension selectors; an empty
slice list means no additional category filter.

Store only layout and selection. Years, permissions, actual datasets, problem
counts and observations are resolved outputs, not duplicated specification data.
Include the specification in snapshots, template composition, export/import and
UUID remapping. Historical published snapshots must retain their own layout.

## Resolved layout in InstanceGraph

`InstanceGraph.data_entry` exposes immutable resolved sections and tables using
its existing dataset, metric, dimension and binding catalogs. Pure graph
resolution returns canonical references, selected metrics/categories, and
source-resolution diagnostics. It must not query observations, run the
calculation model, or cache user-specific permissions.

A request-scoped query layer uses those selections to load authorized dataset
objects, data points, evidence and validation findings in batches. This is the
mutable-data portion of the view service; it consumes the graph rather than
maintaining another source catalog. Follow the selected draft/published source
and dataset revision pins instead of silently reading newer data.

## GraphQL contract

Expose the view under `instance.editor.dataEntry`. Illustrative fields:

```graphql
dataEntry {
  years
  defaultYear
  canManageYears
  sections {
    id
    identifier
    name
    description
    tables {
      id
      identifier
      name
      resolutionStatus
      dataset { id }
      metrics { id }
      dimensions { id categories { id } }
      dataPoints(years: $years) { ... }
    }
    problems(years: $years) { ... }
    problemCounts(years: $years) { ... }
  }
}
```

The final schema must represent unresolved tables explicitly, through a typed
resolution result or status plus resolution problems. A resolved dataset-backed
table has a concrete dataset, selected metrics and permitted slice categories;
an unresolved table must not manufacture a dataset or turn into an unexplained
null. Expose source problems without revealing unauthorized dataset contents.

`dataPoints` returns only the table's selected metrics, categories and years.
Underlying dataset and data-point UUIDs remain the mutation identities; tables
and sections do not own duplicate observations. A UI can request small section
summaries separately from table data and deduplicate shared datasets. The API
must not require all datasets and all years to be loaded to open one section.

## Year selection

Use `years: list[int] | None` consistently for data points, problems and counts:

| Argument | Meaning |
| --- | --- |
| Omitted or `null` | All declared inventory years. |
| `[]` | No years; an empty result. |
| Nonempty list | The selected declared inventory years. |

For this view, “all years” means the instance's declared inventory calendar,
respecting skipped years. It does not mean every year present in a dataset or an
unbounded calendar. Reject explicitly requested undeclared years with a clear
validation error. Broader historical observations remain accessible through
ordinary dataset APIs.

Resolve the calendar from the existing model year declaration; do not infer it
from whichever data points happen to exist. If no inventory calendar is declared,
expose that state and produce no invented missing-cell grid. `defaultYear` is the
latest declared inventory year, or null for an empty calendar. Year-management
capability is resolved from current permissions, never stored in the layout.

## Section problems and counts

Distinguish at least:

- missing required values;
- missing required grades/evidence;
- plausibility and other dataset validation findings;
- source-resolution/configuration problems.

Missing required cells need stable dataset/metric/category/year coordinates even
when no `DataPoint` exists. Requiredness comes from applicable collection and
validation contracts, not every theoretical combination of schema categories.
Unassessed computed energy coverage is not a replacement for input completeness.

Intersect point findings and required-cell grids with the table's metric and
category selection and the requested declared years. Section problems are the
union of their tables' relevant problems, deduplicated by canonical finding/cell
identity when table selections overlap. Yearless configuration problems remain
distinguishable from annual cell findings; do not multiply them by the number of
inventory years. An empty year selection returns no problems or counts.

Return total counts and a per-year breakdown, with yearless configuration counts
separate. Counts and problem lists must use the same selection, requiredness and
deduplication rules so a tab badge agrees with the opened section.

Batch dataset, evidence and finding reads. Summary-only queries must not hydrate
every data point or initialize the runtime calculation graph merely to compute
badges. Use existing materializations where suitable; do not introduce a second
plausibility evaluator in section resolvers.

## UI caching and migration

Scope all query results to the active instance and requested years. Changes to
values, grades, bindings or the inventory calendar invalidate affected data and
summaries; session caching must allow refresh. Share observation identity across
tables displaying the same data, while keeping computed view results scoped to
their parent fields and filters.

Declare the initial BISKO layout in model configuration and provision it through
the normal template workflow. Existing configurations without a layout remain
valid. Deploy the backend contract before switching Data Studio, then remove the
hard-coded node-to-section mapping and client-side ownership joins. Regenerate
GraphQL types and retain loader-to-domain conversion in the UI.

## Implementation sequence and verification

1. Add typed specs, readable YAML resolution, stable identities and snapshot
   round trips; implement template inheritance and explicit local replacement.
2. Add pure graph resolution and diagnostics for schema-backed active inputs.
3. Add request-scoped batch data/problem selection and GraphQL fields with the
   declared-year contract and permission checks.
4. Declare and publish the initial BISKO layout; adapt Data Studio loaders and
   section navigation, then remove the hard-coded mapping.

Verify schema/slice validation, export/import identity remapping, inherited and
local layouts, dataset resolution across scopes, and disconnected or
submodel-backed inputs. Test multi-table sections, shared sliced datasets,
missing cells without rows, optional cells, evidence gaps, overlapping findings,
all/null/empty/subset year selections, skipped and undeclared years, and
permission isolation. Add scale-sensitive query-count tests for summary-only and
table-data queries, plus UI cache checks for instance switching and edits.
