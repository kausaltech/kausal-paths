# Trailhead: First Action-Authoring Flow

## Summary

This document defines the first UI-target use case for Trailhead action authoring.

The flow is intentionally narrow:

- a model builder copies an existing Aarhus-style action
- gives it a new identity
- connects it to new precomputed impact data from Excel
- confirms the output wiring
- saves it as a new action

This is a UI handoff first. It is not a full backend authoring spec and it is not a general-purpose node editor spec.


## Why This Is The First Flow

In `aarhus-c4c`, many actions already follow a very regular pattern:

- they are `simple.AdditiveAction` nodes
- they read data from a shared dataset table
- they select the correct rows through a dataset filter, usually `action = <action_id>`
- they expose one or more output metrics
- they send those metrics to existing target nodes through explicit edge mappings

This means the first valuable authoring slice is not “create any node from scratch”.

It is:

> copy a structurally similar action and retarget its data

That matches what model builders already do mentally today, just without requiring YAML editing.

### Aarhus examples

Energy-style action:

- [`carbon_capture_and_storage`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L1997)
- reads from shared dataset `aarhus/energy_actions`
- outputs `emissions`, `energy`, and `currency`
- routes those metrics to existing emissions, energy-demand, and cost nodes

Transportation-style action:

- [`sustainable_air_and_maritime_transport`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L2359)
- reads from shared dataset `aarhus/transportation_actions`
- outputs `energy` and `currency`
- routes energy into transport energy-consumption nodes and cost into `transportation_costs`

Green mobility variant:

- [`green_mobility_2a2_zero_emission_zone`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L2447)
- reads from shared dataset `aarhus/green_mobility_plan_actions`
- outputs `fraction` and `currency`
- routes the fraction through a tagged edge into an existing target node

The green mobility pattern is slightly more specialized, but it is still close enough to keep visible as a nearby variant when designing the first flow.


## User Story

As a model builder, I want to copy an existing action that is structurally similar to the one I need, connect it to new precomputed impacts from Excel, and review the resulting wiring before saving it, so that I can add a new action without touching YAML.

### Success criteria

- no YAML editing is required
- the user can attach precomputed cost and/or energy impacts from Excel
- the new action becomes a valid scenario action in the model
- target-node connections are explicit and reviewable
- the user can preview that the action has matched data and produces output before leaving the flow


## V1 Defaults

These defaults should be treated as product decisions for the first implementation.

- Entry point is `Copy existing action`, not blank action creation.
- The copied action should inherit:
  - metadata structure
  - dimensions
  - metric definitions
  - output edge suggestions
- The copied action should not silently inherit the old dataset binding as final truth.
- Copied edges are draft suggestions and must remain editable.
- Structured edge editing is the authoritative configuration surface.
- Drag-from-port interactions may exist, but they are a convenience layer on top of structured edge configuration.
- Dataset preparation is part of the same workflow, not a separate admin task.


## Proposed UX Flow

## 1. Copy Action

The user starts by selecting an existing action to copy.

The UI should prefill from the source action:

- node type
- dimensions
- output metrics
- output edge suggestions
- group, names, and descriptions where useful as editable starting values

The copied action should be clearly labeled as a new draft, not as an in-place edit of the source action.

### UI terms

- `Source action`
- `Copied action`


## 2. Basic Info

Require:

- new action id
- display name
- action group

Optional:

- localized labels and descriptions

The user should not be able to continue with an id collision.


## 3. Data Source

The user chooses where the copied action will get its data.

Two options:

- `Existing dataset/table`
- `New dedicated dataset`

This decision should be explicit. The system must not assume that copied actions continue to use the original selector value.

### UI terms

- `Data source`
- `Selector column`
- `Selector value`


## 4. Prepare Dataset

This is the core of the first flow.

### If the user chooses an existing dataset/table

The UI should guide the user through:

- selecting the dataset
- selecting the selector column, usually `action`
- entering the new selector value for the copied action
- checking whether required output metrics already exist in the dataset
- checking whether required dimension categories already exist
- pasting or importing rows from Excel into the dataset editor

The UI should detect and surface:

- missing metric columns
- missing dimension categories
- no rows matching the selector
- rows whose categories do not match the model dimensions

For Aarhus-style actions, the common pattern is:

- dataset: `aarhus/energy_actions`, `aarhus/transportation_actions`, or `aarhus/green_mobility_plan_actions`
- selector column: `action`
- selector value: the new action id or another chosen action key

### If the user chooses a new dedicated dataset

The UI should guide the user through:

- creating the dataset shell
- mapping Excel columns to:
  - `Year`
  - dimensions
  - metrics
  - optional `Forecast`
- confirming units for each metric

V1 does not need to optimize for every future dataset-authoring feature, but it must make the action flow feel continuous.


## 5. Outputs

The copied action should expose output ports by user-facing metric name.

Preferred user-facing output metric labels:

- `Cost`
- `Energy`
- `Emissions`
- `Mileage`
- `Fraction`

The UI should show:

- which output metrics were copied from the source action
- which output metrics are actually present in the selected dataset rows
- which output metrics are currently unwired

Copied edge suggestions should appear here, but as editable draft mappings.

### UI terms

- `Output metric`
- `Target node`
- `Edge mapping`


## 6. Edge Configuration

After selecting or creating a connection, the user configures how that output metric is mapped to the target node.

Each edge configuration must support:

- selected output metric
- target node
- selected target metric when relevant
- dimension flattening
- fixed-category mapping
- tags

The UI should surface compatibility warnings before save, for example:

- metric exists but no compatible target metric is available
- dimensions require flattening or mapping to fit the target
- copied edge config refers to dimensions that are no longer present

### Authoritative configuration surface

The structured edge editor is the source of truth.

Drag-from-port can be used for quick creation, but the user must still land in structured edge configuration before the edge is considered complete.


## 7. Review & Save

Before saving, the UI should show a compact review of:

- source action used as template
- data source selection
- dataset selector
- number of matched rows
- output metrics found
- target nodes selected
- unresolved warnings

The user should not need to mentally inspect the graph to know whether the action is valid.


## 8. Impact Preview

After the action is structurally valid, the UI should offer a lightweight preview.

V1 preview is enough if it shows:

- a small year table of action output
- and/or a small year table of impact on selected target nodes

The goal is not full scenario analysis inside the editor. The goal is to confirm that the copied action has real data and produces the expected kind of effect.


## Aarhus Grounding

The editor should preserve the capabilities Aarhus relies on today without exposing YAML concepts as the primary user language.

### Current source patterns

Aarhus actions commonly use shared filtered datasets:

- `aarhus/energy_actions`
- `aarhus/transportation_actions`
- `aarhus/green_mobility_plan_actions`
- `aarhus/waste_actions`
- `aarhus/ippu_actions`
- `aarhus/afolu_actions`

The action typically receives the correct data through a filter on the `action` column.

Examples:

- [`carbon_capture_and_storage`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L2006)
- [`optimized_and_emission_free_municipal_fleet`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L2330)
- [`green_mobility_2a2_zero_emission_zone`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L2452)

### Current sink patterns

Cost often flows into sector cost aggregators:

- [`energy_costs`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L1932)
- [`transportation_costs`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L1943)
- [`waste_costs`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L1954)
- [`ippu_costs`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L1965)
- [`afolu_costs`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L1976)

Energy, emissions, mileage, and fraction outputs flow into existing domain nodes such as:

- [`electricity_demand`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L569)
- [`waterborne_transport_energy_consumption`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L1551)
- [`chp_emissions`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L535)
- [`onroad_transport_passenger_car_emissions`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L1120)
- [`land_areas`](/home/jey/sync/devel/kausal-paths/configs/aarhus-c4c.yaml#L1723)

The first editor flow should preserve this pattern while hiding implementation detail like `from_dimensions`, `to_dimensions`, and raw YAML edge blocks behind structured UI controls.


## Standardized Terms

Use these terms consistently in the UI spec and UI copy:

- `Source action`
- `Copied action`
- `Data source`
- `Selector column`
- `Selector value`
- `Output metric`
- `Target node`
- `Edge mapping`

When grounding examples in existing config, it is fine to mention current model terms. But the primary UX language should stay at the level above YAML.


## Validation For The Doc

This document is successful if a UI engineer can design the first action-authoring flow without needing to ask:

- how the user starts
- when data source selection happens
- how dataset preparation fits into the flow
- whether copied edges are final or draft
- how output metrics are presented
- whether drag-to-connect is sufficient on its own

This document should remain narrow. It should not drift into a full generic graph-editor or dataset-editor specification.


## Out Of Scope For This First Flow

- blank-from-scratch arbitrary action creation
- authoring custom compute classes
- multi-step action families or parent-child action authoring
- advanced formula authoring
- full generic dataset modeling for every future data shape
- replacing the structured edge editor with canvas interactions alone
