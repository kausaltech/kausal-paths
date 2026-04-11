# Lucia Model Simplification

## Purpose

The lucia model config (`configs/lucia.yaml`) was developed before `FormulaNode` became
available and before `GenericNode` gained its current tag-based arithmetic capabilities.
As a result, many intermediate passthrough nodes were created whose sole purpose is to
reshape, filter, or combine another node's output before it reaches its consumer. These
intermediate nodes add config length and make the computation graph harder to follow
without adding conceptual value.

The goal of this exercise was to identify and remove such nodes — merging their
transformations directly into the consuming node's `input_nodes` port specification —
while preserving all modelled outputs exactly. The `nzc` instance was used as a reference:
if lucia and nzc give identical results after a change, the refactoring is correct.

## Rules Applied

A node can be removed if all of the following hold:

1. **Single consumer.** The node feeds into exactly one other node (checked by scanning
   both `input_nodes` declarations in consumer nodes *and* `output_nodes` declarations in
   the node itself — edges can be defined from either end).

2. **Not protected.** The node does not appear in `result_excels`, which lists nodes whose
   outputs are consumed by Excel sheets. The node does not appear in `visualizations` attribute of any node. Removing such a node would break
   those outputs.

3. **Transformations are expressible at the port.** Everything the intermediate node does
   (dimension flattening via `from_dimensions`, category filtering, tags like
   `inventory_only`, `geometric_inverse`, `extend_values`) can be expressed on the
   consuming node's `input_nodes` entry using `from_dimensions`, `to_dimensions`, and tags.

4. **No side inputs via `output_nodes`.** The node must not receive hidden inputs from
   other nodes that point to it via their `output_nodes`. These edges are not visible in
   the node's own definition. If such edges exist, removing the node would silently drop
   those inputs.

## What Was Done

Two intermediate nodes were successfully removed:

### `building_heat_fossil_ratio` → absorbed into `heating_emission_factor`

This node computed the fraction of building heat energy that is fossil-fuelled
(`fossil_energy / total_non_electric_energy`), then `heating_emission_factor` inverted
it. Both computations were merged: `heating_emission_factor` now takes
`building_heat_fossil_energy` and `building_heat_energy_use` directly as inputs with
`inventory_only`, `geometric_inverse`, and `extend_values` tags. The math is equivalent
because extending the ratio is the same as individually extending numerator and
denominator when both follow the same last-historical-value rule.

### `building_air_pollutant_emissions` → absorbed into `building_air_pollution_cost`

This node filtered `building_heating_emissions` to air-quality pollutants (nox,
pm25_combustion, pm10_combustion) and flattened the heating_type, scope, and fuel_type
dimensions. The identical `from_dimensions` and category filter were moved to the
`input_nodes` port of `building_air_pollution_cost`.

## Limitations Encountered

### 1. Same node cannot appear twice in `input_nodes`

The framework raises an exception (`Node already added to input nodes`) when the same
node is referenced more than once in a single node's `input_nodes`. This prevented
merging `vehicle_kilometres_per_bus` into `number_of_buses`:

```
number_of_buses = vehicle_km[ec, tm=buses] * num_buses_hist[tm] / vehicle_km[ec_flat, tm=buses, hist]
```

This formula requires two different projections of `vehicle_kilometres`, which the
framework does not allow. `FormulaNode` could help here with its sum_dim() function for dimension-flattening operations.

### 2. Dataset format mismatch when inlining goal nodes

`DatasetDifferenceAction` and `DatasetReduceAction` can accept their `goal` input either
from a node (via `input_nodes` with tag `goal`) or from a dataset (via `input_datasets`
with tag `goal` — the `_operate_tags` mechanism will skip the `goal` tag since it is not
a known data operation). However, the raw dataset is in spreadsheet format (label-style
column names, `Sector`/`Quantity`/`Unit` metadata columns). A `GenericNode` intermediary
converts this to clean dimension IDs via `convert_names_to_ids()` and
`get_filtered_dataset_df()`. Without that processing, the action receives malformed data:
dimension columns either have label names that don't match the model's category IDs, or
the data values are misinterpreted. We should consider givin up all GPC-style datasets and use the canonical datasets only. However, this requires that all dataset definitions are updated accordingly in the yaml file.

This prevented:
- Removing `waste_recycling_shares_goal` (a plain `GenericNode` over
  `nzc/waste_recycling_goal`)
- Removing `renovation_intensity_shares_goal` (a `DatasetPlusOneNode` over
  `nzc/defaults`, which additionally applies sector filtering via `params.sector`)

### 3. Hidden inputs via `output_nodes`

`consumer_electricity_intensity` appeared to be a simple passthrough of
`consumer_electricity_intensity_baseline`, but `electricity_need_reduction` also writes
to it via its own `output_nodes` declaration. Removing the node would have silently
dropped the efficiency improvement adjustment. Always scan the full config for
`output_nodes` references before removing a node.

### 4. `DatasetPlusOneNode` applies non-trivial processing

`renovation_intensity_shares_goal` uses `gpc.DatasetPlusOneNode`, which applies sector
filtering (`Sector == "Assumed share of type of renovation in lever"`), name-to-ID
conversion, unit handling, and optional baseline-year-only filtering. None of this can be
replicated by a raw `input_datasets` entry.

## Next Steps

### Near-term (low risk)

- **Systematic audit with edge-direction awareness.** Write a script that builds the full
  bidirectional graph and identifies all single-consumer nodes not in `result_excels`.
  Many more candidates may exist that weren't reviewed in this session.

- **`vehicle_kilometres_per_bus` / `number_of_buses` consolidation.** If the framework is
  extended to allow referencing the same node twice with different port configurations
  (e.g., via named port slots), these two nodes could become one. Alternatively, a
  dedicated node type for "scale historical count by current utilisation" would cleanly
  express this pattern.

### Longer-term (requires framework changes)

- **Dataset preprocessing pipeline.** If the dataset loading layer applied
  `convert_names_to_ids` and basic dimension normalisation before returning data, goal
  datasets could be inlined directly into actions without an intermediate `GenericNode`.
  This would enable removing `waste_recycling_shares_goal` and similar nodes.

- **FormulaNode dimension operations.** Impelementing `sum_dim(node, dim)` with  `FormulaNode` allows expressing computations like the
  bus count formula in a single node.
