# Submission indicators and aggregation

A view over many instances at once (a district's members, a state's comparison) needs each
instance's key figures. Computing every member's model for one page is not possible at a few
thousand members. So the figures are stored on the submission, and such views read them without
building a runtime.

## What a model declares

The instance spec has an optional `submission_indicators` block
(`nodes.defs.indicator_defs.SubmissionIndicatorsSpec`). In a framework it belongs in the template,
and followers inherit it:

```yaml
submission_indicators:
  settings:                       # parameter values every figure is computed under
  - parameter: use_national_values
    value: true
  declarations:                   # parameters whose effective values are recorded
  - heat_pump_electricity_included_in_electricity
  indicators:
  - id: final_energy
    node: total_final_energy_use
    dimensions: [sector, energy_carrier]   # kept; the node's other dimensions are summed away
  - id: emissions_local_mix
    node: net_emissions
    settings:                              # for this indicator alone, on top of the shared ones
    - parameter: use_national_values
      value: false
  - id: data_quality
    node: total_weighted_data_quality
    composition: weighted_mean
    weight: final_energy
```

No code names a node or a parameter. A framework with other figures declares other figures.

- **`settings`** make the figures independent of the visitor. The runtime starts from blank
  settings, takes the declared parameter values, and turns normalization off: a stored figure is
  absolute. The parameters must be customizable ones.
- **`composition`** says how a figure composes over several submissions:
  - `sum` (the default): added cell by cell;
  - `weighted_mean`, with `weight`: the summed products of value and weight over the summed
    weights. It is never averaged;
  - `none`: says nothing about a territory, e.g. an intensity.

## What is stored

`Submission.indicators` holds a `SubmissionIndicators` (`frameworks.models.indicators`):

| Part | Content |
|---|---|
| `status`, `reason` | `computed`, or `not_computable` with the model's error |
| `values` | per indicator: unit, kept dimensions, total, cells, and its composition rule |
| `completeness` | requirements met and applicable in the year, and open plausibility findings |
| `declarations` | effective values of the declared parameters |
| `provenance` | when, the draft head or the pinned revision, the template revision, the settings |

Each value carries its own composition rule. A sum over submissions that were finalised under
different template revisions therefore reads each one's rule from the submission, not from
today's template.

**Completeness** counts requirements, not values. One required value in one year is one
requirement: a contract's year-level requirement, one required group of its shape, or one
combination required by another input's positive value (`nodes.value_validation.RequirementCounts`).
The unmet ones are the `missing_required_value` and `required_qualifier` findings the data-entry
views already show. An instance whose inventory calendar is undeclared has no countable
requirements and reads 0 of 0.

## When the figures are written

`frameworks.indicators` computes; three places call it.

- **Every change of the draft.** `change_operation` ends by sending `instance_draft_changed`
  (`nodes.change_ops`), inside its transaction. The receiver in `frameworks.signals` rewrites the
  figures of *all* open submissions of the instance, because a parameter or a template change moves
  every year. Instances without an open submission, or whose model declares no figures, pay one
  `EXISTS` query.
- **A writer outside a change operation** that announces its change with `announce_draft_change`.
  `upgrade_template_instance` does, and opening a submission does the equivalent directly.
- **Finalising.** The figures are computed once more from the revision the submission pins
  (`InstanceConfig.create_instance_from_revision`) and then never again.

Two failure rules:

- **A draft that cannot be computed is recorded, not refused.** Halfway through data entry a model
  can legitimately fail. The submission then holds `status: not_computable` with the reason, and
  the edit stands.
- **A final submission must have its figures.** If the model cannot give them, finalising fails and
  the publication it made is rolled back with it.

**Staleness.** An import, a sync or the Wagtail admin changes the draft without a change operation
and without announcing it. The figures then predate the draft, and `indicators_are_stale` says so
by comparing the recorded draft head and template revision with the instance's. A reader shows
such figures as stale, never as current. A final submission's figures cannot be stale.

**Backfill.** `python manage.py materialise_submission_indicators [instances] [--all] --apply`
writes the figures of submissions that lack them: open ones from the draft, final ones from the
revision each pins, which need not be the live one.

## Years, drafts and the lock

- **A year being entered has an open submission.** In an instance that belongs to a framework,
  `add_inventory_year` opens a draft for the year. `remove_inventory_year` takes an untouched
  draft with it, so adding a year can be undone. A draft whose year holds entered data keeps the
  year until the draft is discarded.
- **A finalised year is read-only until reopened.** A year is locked when it has a final inventory
  submission and no open one (`frameworks.submissions.locked_years`). The dataset editor's
  GraphQL mutations refuse to create, update or delete its data points, and with them their
  evidence and quality grades. Comments stay possible. Opening a correction unlocks the year;
  finalising or discarding it locks the year again. `dataEntry { lockedYears }` exposes the set.
- **Changes that apply to every year** (a parameter, a template upgrade) are not locked. They
  change what the draft model gives for a finalised year. That is harmless as long as a finalised
  year is shown from its frozen figures and not from the draft model.

## Aggregation

`FrameworkOrganization.aggregation(year:, kind:)` (`frameworks.aggregation`) is computed on read
from the organization tree, the instances and the submissions. Nothing is stored for it.

- **The framework declares its reporting units** (`Framework.reporting`): the organization classes
  that are expected to hold an instance and submit. Coverage has to count the units that are
  absent, which a list of instances cannot do.
- **The basis is the published sum.** A unit counts when it has a final submission for the year,
  and a territory's figures are composed from what each unit froze (`frameworks.composition`).
  Figures whose contributors disagree on unit, dimensions or rule are left out and named; nothing
  is converted. The result also says how many template revisions it adds up.
- **A territory's own balance is beside the sum, never in it.** It covers the same ground as its
  units.
- **A territory whose only child is a reporting unit is presented as that unit.** This is how a
  city that is its own district appears in the tree.
- **A unit counts only in the years it existed**, by its founding and dissolution dates. A merger
  recorded as dissolution plus founding leaves past years as they were.

**Access.** A viewer whose subtree grant or own instance covers the organization gets every member.
A viewer who only holds something inside it gets its totals and the members on the way to their
own. Totals of a territory with fewer than `Framework.reporting.min_units_for_ancestor_total`
units, and sums of fewer finalised units than that, are withheld from such a viewer: one peer's
figures would follow from them.

`Framework.finalSubmissions(year:, organizationId:)` lists the final submissions of the reporting
units the viewer may read, with their figures, for comparison and export.

## Limits

- The year lock guards the GraphQL dataset editor. The REST data point API and the Wagtail admin
  are not guarded.
- Coverage is stated by count and by reference population. Another denominator needs the reference
  table generalised first.
- Computing the figures costs a model run on every edit of an instance that has an open
  submission. Measured on two municipal BISKO instances in November 2026: 0.3 to 1.1 s for
  building the runtime and reading the figures, depending on how much of the node cache the edit
  invalidated, and another 0.6 to 0.9 s for counting plausibility findings. It runs synchronously
  inside the mutation.
