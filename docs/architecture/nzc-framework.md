# NZC Framework Instances

This note describes framework-backed NetZeroCities (NZC) instances in Kausal
Paths. It is written for future maintainers who need to reason about the
current implementation and the direction it should move in.

The important distinction: an NZC framework instance is not a separate model
configuration per city. The calculation model is fixed. City-specific input is
stored as framework measure data and injected into the shared model at runtime.

Hand-written YAML instances that reuse NZC modules, such as
`lappeenranta-nzc`, `cork-nzc`, and `dut-transport-nzc`, are related but not
covered here. They are manually maintained exceptions where the model itself
can diverge.

## Conceptual Model

Framework-backed NZC instances have three layers:

1. **Fixed calculation model**

   The runtime model comes from [configs/nzc.yaml](../../configs/nzc.yaml).
   It includes sector modules under [configs/nzc/](../../configs/nzc):
   transport, transport cost, transport health cost, freight transport,
   buildings, electricity, waste, and other.

2. **Framework metadata**

   The `Framework`, `Section`, and `MeasureTemplate` records describe the data
   collection UI: which questions exist, how they are grouped, what units they
   use, which values are high priority, validation bounds, help text, and
   whether a value belongs to data collection or future assumptions.

3. **Per-city measure values**

   Each city has a `FrameworkConfig`. For each `MeasureTemplate`, the city may
   have a concrete `Measure` and one or more `MeasureDataPoint` rows. These
   are the city-specific inputs. They are keyed by measure-template UUID when
   joined back to model datasets.

The external Data Studio UI lives outside this repository. From the Paths side
it should be treated as a specialized client for the framework API. It
hard-codes the NZC framework, creates plans through `createNzcFrameworkConfig`,
reads the `data_collection` and `future_assumptions` sections, and writes
measure values by measure-template UUID.

## Creation Flow

The NZC creation mutation is defined in
[frameworks/schema.py](../../frameworks/schema.py).

`createNzcFrameworkConfig` takes two inputs:

- `FrameworkConfigInput`: framework id, instance identifier, organization
  name, baseline year, target year, and optional UUID.
- `NZCCityEssentialData`: population, low/high yearly temperature, and
  low/high renewable energy mix.

The mutation does the following:

1. Creates a `FrameworkConfig` and a matching `InstanceConfig`.
2. Loads the model instance for that framework config.
3. Reads the instance dataset repository from `instance.context.dataset_repo`.
4. Calls `get_nzc_default_values()` in [frameworks/nzc.py](../../frameworks/nzc.py).
5. Stores the returned defaults as baseline-year `MeasureDataPoint.default_value`
   values through `FrameworkConfig.create_measure_defaults()`.

`get_nzc_default_values()` reads the DVC dataset `nzc/placeholders`. The
placeholder dataset chooses one of four scenario columns from:

- renewable mix: `low` or `high`
- temperature: `low` or `high`

Rows marked `PerCapita` are multiplied by the city population. The output is a
mapping from measure-template UUID to numeric default value.

## Runtime Model Loading

The bridge from framework config to model instance is
`FrameworkConfig.create_model_instance()` in
[frameworks/models.py](../../frameworks/models.py).

It builds the path:

```python
configs/{framework.identifier}.yaml
```

For NZC, this is `configs/nzc.yaml`. The `InstanceLoader` receives the current
framework config as `fw_config`, and the resulting context carries
`framework_config_data`.

This is why framework-backed NZC instances all use the same model definition.
Changing [configs/nzc.yaml](../../configs/nzc.yaml) changes the model for
framework-backed NZC instances. Per-city differences are expected to enter
through measure data, not through copied YAML.

## Measure Data Injection

Measure data enters the calculation graph through
`FrameworkMeasureDVCDataset` in [frameworks/datasets.py](../../frameworks/datasets.py).

When an `InstanceLoader` has `fw_config`, it uses `FrameworkMeasureDVCDataset`
for `gpc.DatasetNode` inputs. It also uses it for any dataset input tagged
`framework_measure_data`.

The dataset class expects a GPC-style dataframe with a `UUID` column. The
`UUID` values are measure-template UUIDs. For matching UUIDs, it queries
`MeasureDataPoint` rows for the current framework config and joins them onto
the dataset.

Value precedence is:

1. `MeasureDataPoint.value`, entered or imported by the city.
2. `MeasureDataPoint.default_value`, computed during NZC plan creation.
3. The original dataset value from `nzc/defaults`.

The joined dataframe receives two marker columns:

- `FromMeasureDataPoint`: true when either `value` or `default_value` supplied
  the value.
- `ObservedDataPoint`: true only when the user-provided `value` supplied the
  value.

Those markers are used by downstream dataset and formula operations to decide
whether to use city-entered observations, placeholder defaults, or modelled
values.

## Data Studio Contract

The Data Studio UI is intentionally simpler than the full Paths model editor.
Its unit of interaction is a framework measure, not a graph node.

The relevant API shape is:

- `framework(identifier: "nzc")`
- `section(identifier: "data_collection")`
- `section(identifier: "future_assumptions")`
- `MeasureTemplate.measure(frameworkConfigId: ...)`
- `updateMeasureDataPoint(...)`
- `updateMeasureDataPoints(...)`

The UI presents two main sheets:

- **Data collection** for baseline-year city data.
- **Future assumptions** for target-year and year-bound assumption data.

The UI filters hidden templates, applies client-side min/max validation, and
filters `yearBound` templates to years between the selected baseline and target
year. That year-bound filtering is a current client-side workaround and should
move backend-side when the framework API grows a stronger representation for
time-varying measures.

Plan export/import is also measure-based. Exports contain measure-template UUIDs
and measure datapoints. Import maps those UUIDs back into `MeasureInput` and
calls `updateMeasureDataPoints`.

## Current DVC Dataset Contract

The current NZC model still depends heavily on DVC-backed wide datasets.

`nzc/defaults` is the main dataset consumed by the model. Many nodes filter it
by columns such as:

- `Sector`
- `Quantity`
- `Unit`
- dimensions represented as human-readable column labels
- `UUID`

`gpc.DatasetNode` then performs GPC-specific normalization:

- filters by `params.sector` or `params.gpc_sector`
- filters to the node quantity
- optionally filters by a specific UUID
- converts label-style dimension names and category labels to model ids
- moves the `Unit` column into Paths dataframe unit metadata

`nzc/placeholders` is used only during NZC framework-config creation. It is not
the full model input dataset. It maps city essential data to default values for
measure-template UUIDs.

The intended direction is to replace these broad DVC-backed inputs with smaller
database-backed datasets that contain only the data each part of the model
actually needs. When doing that, preserve the important contract explicitly:
model data must remain linkable to `MeasureTemplate` identity, and the source of
each value must remain distinguishable as user observation, computed default, or
model fallback.

## Observed-Value Overrides

Some important outputs have both modelled and observed forms. For example:

- passenger kilometres observed in
  [configs/nzc/transport.yaml](../../configs/nzc/transport.yaml)
- total electricity consumption in
  [configs/nzc/electricity.yaml](../../configs/nzc/electricity.yaml)
- building heat energy use observed in
  [configs/nzc/buildings.yaml](../../configs/nzc/buildings.yaml)
- emissions from other sectors in
  [configs/nzc.yaml](../../configs/nzc.yaml)

These nodes typically use an input dataset tagged:

```yaml
tags: [observed, prepare_gpc_dataset, framework_measure_data]
```

and a formula that selects observed data when `measure_data_override` is true,
otherwise modelled data.

`observed_only_extend_all(observed)` uses the marker columns from
`FrameworkMeasureDVCDataset`: if user observations exist for a dimension
combination, it keeps only those observations; otherwise, if framework defaults
exist, it may keep those; otherwise it falls back to the model default
selection behavior. The result is then extended over the model time span.

## Synchronization And Import

Framework metadata can be synchronized with
`python manage.py sync_framework_data`.

The relevant modes are:

- `import`: import framework metadata, including sections and measure templates.
- `export`: export framework metadata.
- `import_configs`: import framework configs and measure values.
- `export_configs`: export framework configs and measure values.

The sync layer is implemented in:

- [frameworks/sync_frameworks.py](../../frameworks/sync_frameworks.py)
- [frameworks/sync_configs.py](../../frameworks/sync_configs.py)

There is also
[`import_framework_config`](../../frameworks/management/commands/import_framework_config.py),
which imports framework configs through GraphQL from one API to another.

`load_nzc_measures` is an older importer that builds NZC framework metadata from
a JSON shape with `dataCollection` and `futureAssumptions`. It is useful for
understanding the original metadata structure, but the sync JSON path is the
more explicit current mechanism.

## New Observation System (lucia model)

The legacy measure data injection described above works but has several
drawbacks (see "Current Legacy Pieces" below). A cleaner alternative has been
implemented and is currently live in `configs/lucia.yaml`. It will eventually
replace the legacy approach in `nzc.yaml` as well.

### ObservationDataset

`ObservationDataset` in [frameworks/datasets.py](../../frameworks/datasets.py)
is a `DVCDataset` subclass that overlays DB-sourced values onto a DVC parquet
dataset. Instead of the wide GPC-style `nzc/defaults`, each observable metric
has its own narrow DVC dataset with `uuid` kept as a real dimension
(`drop_col: false`).

The overlay adds boolean flag columns to the dataset:

- `observed` — a city user entered a value (`MeasureDataPoint.value`)
- `placeholder` — only a comparable-city default exists (`MeasureDataPoint.default_value`)

Both value and flag columns are added in a single pass. Unit conversion is
handled automatically if the measure template unit differs from the dataset
unit. If no `FrameworkConfig` exists for the instance, the dataset is returned
unchanged (the DVC values pass through as-is).

The dataset is tagged `observation_dataset` in the YAML input_datasets list,
which signals `ObservableNode` to treat it specially.

### ObservableNode

`ObservableNode` in [nodes/generic.py](../../nodes/generic.py) extends
`GenericNode` with an `apply_observations` operation. It consumes an
`ObservationDataset` and an optional `modeled` input node.

The `apply_observations` operation blends observations with modelled values
depending on the global boolean parameter `use_observations`:

- **`use_observations=false`** (default / decarbonisation scenario): the
  modelled output is used for all years, except the reference year, which is
  overridden by the observation value if one is present.
- **`use_observations=true`** (progress_tracking scenario): all observed values
  are extended across all historical and future years. If no observed values
  exist, falls back to the same ref-year-only behaviour as the default
  scenario.

The modelled input node is connected with `tags: [modeled]`. This tag causes
`ObservableNode._get_add_multiply_nodes` to exclude it from the normal
add/multiply pipeline; it is consumed by `apply_observations` instead.

### YAML pattern

```yaml
- id: passenger_kilometres_observed
  type: generic.ObservableNode
  quantity: mileage
  unit: Mpkm/a
  input_datasets:
  - id: nzc/passenger_transport_need_fleet
    column: passenger_kilometres
    filters:
    - column: uuid
      drop_col: false      # keep uuid as a dimension for DB lookup
    - column: energy_carrier
    tags: [observation_dataset]
  input_nodes:
  - id: passenger_kilometres
    tags: [modeled]        # excluded from add/multiply; consumed by apply_observations
    from_dimensions:
    - id: energy_carrier
      flatten: true
```

### Scenarios

The `use_observations` parameter replaces the old `measure_data_override`
ConstantNode:

```yaml
scenarios:
- id: default
  params:
  - id: use_observations
    value: false
- id: progress_tracking
  params:
  - id: use_observations
    value: true
```

### Observable nodes in lucia.yaml

Five key outputs are currently implemented as ObservableNodes:

- `emissions_from_other_sectors_observed` — dataset `nzc/other_sectors`
- `passenger_kilometres_observed` — dataset `nzc/passenger_transport_need_fleet`
- `freight_transport_need_observed` — dataset `nzc/freight_transport`
- `building_heat_energy_use_observed` — dataset `nzc/buildings_heating_fuel_tech`
- `total_electricity_consumption` — dataset `nzc/electricity`

---

## Current Legacy Pieces

The following pieces are part of the current implementation but should not be
treated as the target architecture:

- **GPC-style wide datasets.** `nzc/defaults` carries many sectors, quantities,
  units, dimensions, and UUIDs in one broad table. This makes node configs
  depend on string filters and dataframe cleanup.

- **`gpc.DatasetNode` special handling.** It knows about `Sector`, `Quantity`,
  `Unit`, `UUID`, label-to-id conversion, historical extension, observed-value
  replacement, and `measure_data_baseline_year_only`.

- **`measure_data_override`.** This global parameter is used to switch selected
  formula nodes between modelled values and observed measure data.

- **`measure_data_baseline_year_only`.** This global parameter controls whether
  dataset nodes keep only the baseline year and forecast rows from measure data.
  It exists because the historical rows in the wide datasets and framework data
  do not yet have a cleaner shared representation.

- **`DatasetPlusOneNode` and `SCurveAction` variants in `gpc`.** These exist to
  support older NZC YAML patterns. Newer model config should prefer the generic
  node/action forms when possible.

- **Client-side year-bound filtering and validation.** Data Studio currently
  filters `yearBound` templates and validates min/max bounds. The backend should
  eventually expose a cleaner contract so clients do less interpretation.

The migration principle is: isolate these mechanisms behind named compatibility
paths while moving the common path toward narrower datasets, explicit measure
identity, and backend-enforced contracts.

## Future Direction

The target shape should preserve what works about NZC while removing accidental
coupling:

- Keep the fixed-model property. Framework-backed NZC cities should share one
  model definition unless an instance is intentionally forked into a manual YAML
  exception.

- Make the measure-to-model binding explicit. Today it is discovered by scanning
  UUIDs in DVC dataset rows. Future DB-backed datasets should represent this
  link directly.

- Replace broad `nzc/defaults` tables with narrower datasets. Nodes should
  receive data in the shape they need, instead of filtering a large spreadsheet
  format.

- Move UI interpretation backend-side. Year-bound applicability, validation,
  default/fallback semantics, and source classification should be represented in
  the API rather than reconstructed by Data Studio.

- Preserve value provenance. Calculations need to know whether a value came from
  city-entered data, generated defaults, or model fallback data. That distinction
  is currently carried by `ObservedDataPoint` and `FromMeasureDataPoint`.

- Keep framework config import/export measure-based. UUID-addressed measure data
  is a good portability boundary as long as framework metadata UUIDs are stable.
