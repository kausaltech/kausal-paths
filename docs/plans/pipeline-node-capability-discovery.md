# PipelineNode capability discovery

Status: discovery pass completed 2026-08-27; no node replacement is proposed by
this document.

Related plans:

- [Node input-port runtime migration](node-input-port-runtime-migration.md)
- [Loader/spec inversion](loader-spec-inversion.md)

## Purpose

The input-port migration requires us to inspect custom node classes one by one.
Use that inspection to answer a second question: which classes could eventually
be authored as `PipelineNode` definitions, and what reusable operations or
runtime facilities are missing before that is possible?

The two decisions remain independent:

1. Every node can migrate from source-specific accessors to runtime input ports.
2. Only some nodes should cease to be Python classes.

Port migration must not wait for PipelineNode. Conversely, a class is not a
good pipeline candidate merely because it consumes its inputs through ports.

## Method and limits

This pass inspected compute-like methods under `nodes/simple.py`,
`nodes/generic.py`, `nodes/actions/`, `nodes/costs.py`, `nodes/buildings.py`,
`nodes/gpc.py`, `nodes/ch/`, `nodes/finland/`, `nodes/health.py`,
`nodes/emissions/`, and `nodes/kpr.py`. It also counted explicit `type:` values
inside the `nodes` and `actions` sections of repository YAML files.

The count found 5,306 declarations using 112 class paths. These are raw
declarations in top-level files and reusable fragments, not a count of distinct
runtime nodes: includes can cause a declaration to be counted both in a module
and in a composed instance. Database-only specs are not represented. The count
is useful as a migration-risk and reuse signal, not as production telemetry.

Source inspection found 117 compute-like method definitions. Some are base
implementations, mixins, overrides, or currently unused classes, so this is not
the number of independently migratable semantics.

## Existing pipeline surfaces

There are three partially overlapping representations:

1. `nodes.pipeline.PipelineSpec` is typed and executable. It currently supports
   `identity`, `add`, `subtract`, `multiply`, `divide`, and `clip`, with port,
   intermediate, parameter, dataset, and scalar input references plus simple
   conditions.
2. `PipelineNodeIR` adds runtime-only bindings and is already used to compare
   lowered AdditiveNode and MultiplicativeNode execution with their legacy
   implementations.
3. `PipelineConfig.operations` is still a placeholder, while
   `NodeSpec.pipeline` is a separate loosely typed field.

Before authored pipelines are persisted, (1) and (3) need one canonical stored
schema. Discovery and parity lowering can use `PipelineNodeIR` before that
decision.

An authored computation should normally refer only to input ports. The current
`DatasetInputRef` and runtime `InputDatasetBinding` are useful compatibility
devices, but using them as the normal authored API would reintroduce the source
distinction that runtime ports are removing. Operations that genuinely require
raw dataset metadata should declare that exceptional capability explicitly.

### GenericNode is the legacy prototype

`GenericNode` already executes an authored comma-separated sequence from a
registry. Its base and subclasses contain 37 `_operation_*` methods, including:

- input collection: `get_single_dataset`, `add_datasets`, `concat_datasets`,
  `add`, `multiply`, and `coalesce`;
- dimension semantics: split by shares, split evenly, use as totals or shares,
  add dimensions, and fill a complement category;
- temporal and correction semantics: `impute`, `select_variant`, correction,
  year iteration, and baseline-plus-one;
- specialized computations: weighted sum, logit, sector processing, generation
  capacity, CHP allocation, scenario impact, action history, and observables.

This is the best empirical source for PipelineNode requirements, but it should
not be copied literally. Several operations discover inputs through tags, mix
input resolution with algebra, or depend on node/context state. They need to be
split into typed port references, pure operations, and explicit runtime
capabilities.

## Classification

The tables use these dispositions:

- **Near**: representable by the current IR or a small general extension.
- **Ops**: a plausible authored pipeline after reusable dataframe operations
  are added.
- **Runtime**: algebra is pipeline-shaped, but it also needs a scenario, action,
  goal, observation, or graph-runtime facility.
- **Adapter**: primarily imports or interprets a particular external dataset.
  Prefer moving work into dataset/binding transformations over making a node
  pipeline operation.
- **Algorithm**: iterative or domain-heavy computation. Keep it custom unless a
  coherent higher-level operation or reusable sub-pipeline emerges.
- **Alias**: compatibility name or thin specialization; remove or lower to the
  class it aliases rather than expanding PipelineNode for it.
- **Separate**: authored computation of a different kind that should coexist
  with PipelineNode.

The disposition is a hypothesis for the next implementation pass, not a
replacement decision.

### Core and generic nodes

| Configured class | YAML refs | Disposition | Main missing capability or observation |
| --- | ---: | --- | --- |
| `simple.AdditiveNode` | 1,302 | Near | Complete mixed-source, dataset, gap-fill, scaling, share, and tolerant-add lowering. |
| `simple.AdditiveNode2` | 80 | Alias | Merge with AdditiveNode after compatibility users migrate. |
| `simple.MultiplicativeNode` | 697 | Near | Complete additive side-input, dataset, and ordered-imputation lowering. |
| `simple.MultiplicativeNode2` | 35 | Alias | Merge with MultiplicativeNode after compatibility users migrate. |
| `simple.SubtractiveNode` | 1 | Near | Variadic subtraction and model-year filtering. |
| `simple.FixedMultiplierNode` | 94 | Near | Parameter multiplication; dataset replacement belongs at a binding or explicit overlay operation. |
| `simple.ImprovementNode` | 5 | Ops | Dimensionless `1 - x`. |
| `simple.ImprovementNode2` | 1 | Ops | Dimensionless `1 + x`. |
| `simple.RelativeNode` | 1 | Ops | Add, join, fill, and `(x + 1) * reference`. |
| `simple.FillNewCategoryNode` | 9 | Ops | Aggregate over a dimension, complement, assign category, and concatenate. |
| `simple.ChooseInputNode` | 1 | Ops | Select one of several ports from a string parameter. |
| `simple.RelativeYearScaledNode` | 1 | Ops | Scale by a parameter or instance reference year. |
| `simple.AnnuityNode` | 2 | Algorithm | Partition, explode year ranges, compound interest, and aggregate payments. Candidate for a financial macro later. |
| `simple.DiscountNode` | 1 | Ops | Year-relative exponentiation with a rate series. |
| `simple.MixNode` | 21 | Ops | Calculate shares, clamp, normalize across dimensions, and extend history. |
| `simple.EmissionFactorActivity` | 30 | Ops | Detect/select activity and factor metrics, multiply, convert GHG to CO2e, aggregate, and add direct emissions. |
| `simple.SectorEmissions` | 59 | Adapter | Dataset selection and schema normalization followed by ordinary additions. |
| `simple.DataAvailabilityNode` | 9 | Adapter | Deliberately inspects raw datasets and interpolation metadata before transformations. |
| `simple.Activity` | 19 | Alias | Semantic name for AdditiveNode only. |
| `simple.PerCapitaActivity` | 2 | Alias | Existing FIXME already points to GenericNode. |
| `generic.GenericNode` | 866 | Ops | Replace its string operation registry with typed port-based operations incrementally. |
| `generic.ConstantNode` | 19 | Near | Constant dataframe generation plus addition. |
| `generic.CoalesceNode` | 12 | Ops | Ordered null-aware coalescing of port values. |
| `generic.DatasetPlusOneNode` | 1 | Ops | Historical/forecast branching and baseline-plus-one. |
| `generic.DatasetReduceNode` | 8 | Ops | Metric mapping, dimensional filtering, interpolation, forecast synthesis, goal joins, and reduction. |
| `generic.WeightedSumNode` | 7 | Ops | Weight lookup/alignment and weighted reduction over repeated inputs. |
| `generic.LogitNode` | 1 | Ops | Weighted sum followed by a logit transform. |
| `generic.DimensionalSectorEmissions` | 6 | Adapter / Ops | Interpret sector data, then multiply and add through ordinary ports. |
| `generic.DimensionalSectorEnergy` | 9 | Adapter / Ops | Interpret sector data, then multiply and add through ordinary ports. |
| `generic.DimensionalSectorEmissionFactor` | 9 | Adapter / Ops | Interpret sector data and calculate emission factors. |
| `generic.GenerationCapacityNode` | 1 | Algorithm | Stock/capacity evolution over years; possible temporal macro. |
| `generic.CohortNode` | 1 | Algorithm | Annual cohort simulation and age-group aggregation. |
| `generic.BiskoChpNode` | 3 | Algorithm | Domain allocation with several coupled outputs. |
| `generic.BiskoExergeticAllocationNode` | 1 | Algorithm | BISKO-specific conformity/allocation calculation. |
| `generic.ScenarioImpactNode` | 1 | Runtime | Evaluate one source under active and reference scenarios and calculate impact. |
| `generic.LeverNode` | 1 | Runtime / Ops | Inspect action enabled state and conditionally replace forecast rows. |
| `generic.ObservableNode` | 5 | Runtime | Observation lookup and imputation. |
| `formula.FormulaNode` | 340 | Separate | Authored expression language; share ports and primitives where useful, but do not presume replacement by an ordered pipeline. |
| `emissions.GlobalWarmingPotential` | 1 | Adapter / Ops | Generate a GHG lookup table; a general lookup/join operation may absorb it. |
| `health.AttributableFractionRR` | 1 | Ops | Join exposure and risk, exponentiation, and conditional attributable-fraction formula. |

`costs.AdditiveNode` has one YAML reference but resolves to an imported
AdditiveNode compatibility name rather than an independent computation.

### Action nodes

| Configured class | YAML refs | Disposition | Main missing capability or observation |
| --- | ---: | --- | --- |
| `simple.AdditiveAction` | 357 | Runtime / Near | Repeated per-output metric pipelines plus the common action enabled/no-effect and impact envelope. |
| `simple.GenericAction` | 207 | Runtime / Ops | GenericNode-style operations inside the common action envelope. |
| `simple.ValueAction` | 9 | Runtime / Ops | Parameterized target trajectory and action state. |
| `simple.ParameterAction` | 4 | Runtime | Produce a time series from an action parameter. |
| `simple.CumulativeAdditiveAction` | 17 | Runtime / Ops | Dataset effect plus cumulative temporal operation. |
| `simple.LinearCumulativeAdditiveAction` | 22 | Runtime / Ops | Cumulative effect with scenario-sensitive linear interpolation. |
| `simple.TrajectoryAction` | 1 | Runtime / Ops | Dataset trajectory gated by action state. |
| `simple.SCurveAction` | 2 | Runtime / Ops | S-curve trajectory with action state. |
| `values.BudgetingAction` | 6 | Runtime / Ops | Multi-metric additive action plus budget-specific output shaping. |
| `energy_saving.BuildingEnergySavingAction` | 70 | Runtime / Algorithm | Coupled building-stock/energy effects and investment lifetime across several outputs. |
| `energy_saving.CfFloorAreaAction` | 6 | Runtime / Algorithm | Building action plus dataset-triggered floor-area changes. |
| `energy_saving.EnergyAction` | 2 | Runtime / Ops | Dataset-derived energy effect gated by action state. |
| `linear.ReduceAction` | 34 | Runtime / Ops | Parameterized reductions and flow-specific dimensional behavior. |
| `linear.DatasetReduceAction` | 61 | Runtime / Ops | DatasetReduce-style interpolation, goals, and action effect. |
| `linear.DatasetReduceAction2` | 2 | Alias | Consolidate with DatasetReduceAction. |
| `linear.DatasetDifferenceAction` | 3 | Runtime / Ops | Dataset/goal difference, interpolation, and action gating. |
| `shift.ShiftAction` | 15 | Runtime / Ops | Shift values between dimension categories according to structured parameters. |
| `parent.ParentActionNode` | 26 | Runtime | Coordinate child actions and graph relationships; not just dataframe algebra. |
| `gpc.DatasetAction` | 17 | Runtime / Adapter | GPC dataset interpretation within the action envelope. |
| `gpc.DatasetAction2` | 11 | Alias | Thin DatasetAction compatibility subclass. |
| `gpc.DatasetActionMFM` | 3 | Runtime / Adapter | Framework dataset interpolation and action trajectory. |
| `gpc.SCurveAction` | 5 | Runtime / Adapter | Legacy GPC dataset adapter plus S-curve effect. |
| `gpc.DatasetRelationAction` | 2 | Runtime / Ops | Typed relationship operation plus action envelope; strong early pipeline candidate. |

The action envelope itself should be a runtime composition facility rather than
copied into each operation. It covers enabled state, no-effect values, scenario
overrides, impact calculation, and multi-output metric handling.

### Costs and buildings

| Configured class | YAML refs | Disposition | Main missing capability or observation |
| --- | ---: | --- | --- |
| `costs.SelectiveNode` | 4 | Ops | Select/group inputs using a global parameter, then add. |
| `costs.ExponentialNode` | 19 | Ops | Generate model years and apply parameterized exponential growth. |
| `costs.EnergyCostNode` | 2 | Ops | Add energy components, fill dataset gaps, and apply taxes/fees. |
| `costs.DilutionNode` | 5 | Algorithm | Year-by-year recurrence over tagged rate/change inputs. |
| `buildings.FloorAreaNode` | 2 | Algorithm | Coupled old/new stock projection, action triggers, and custom dimensions. |
| `buildings.CfNode` | 4 | Algorithm | Building-stock counterfactual calculation with heterogeneous inputs. |
| `buildings.EnergyNode` | 4 | Ops | Cumulative action effects with per-action exceptions. |
| `buildings.HistoricalNode` | 3 | Ops | Filter inherited additive result to historical rows. |
| `buildings.CCSNode` | 11 | Ops | Join emissions/share, conditional capture, and split into three output metrics/scopes. |

Source-only `costs.InternalGrowthNode` and `costs.IterativeNode` are recurrence
algorithms. They should not drive low-level pipeline vocabulary unless active
uses return. `simple.FixedScenarioNode` needs scenario evaluation and is not
currently referenced directly by repository YAML.

### GPC and national dataset adapters

| Configured class | YAML refs | Disposition | Main missing capability or observation |
| --- | ---: | --- | --- |
| `gpc.DatasetNode` | 291 | Adapter | GPC schema interpretation, variant selection, unit column handling, framework overlays, and temporal completion. |
| `gpc.DetailedDatasetNode` | 7 | Adapter | GPC dataset selection and schema normalization. |
| `gpc.CorrectionNode2` | 2 | Ops | Simple parameterized correction after DatasetNode processing. |
| `finland.Population` | 6 | Adapter | Legacy population source normalization. |
| `finland.population.Population` | 2 | Adapter | Combine historical and forecast population datasets. |
| `finland.aluesarjat.BuildingStock` | 1 | Adapter | Pandas-era Aluesarjat grouping plus additive changes. |
| `finland.aluesarjat.FutureBuildingStock` | 3 | Algorithm | Population-driven stock projection using cumulative differences. |
| `finland.hsy.HsyNode` | 4 | Adapter | HSY municipality/sector schema parsing and interpolation. |
| `finland.syke.AlasNode` | 4 | Adapter | ALas schema parsing, grouping, and unit conversion. |
| `finland.syke.AlasEmissions` | 254 | Adapter / Ops | Select and project ALas emissions from an adapter output; high-volume reason to separate loading from algebra. |

These classes expose an important boundary: parsing an external schema should
be a dataset adapter or binding transformation, after which an authored
pipeline can perform general algebra. PipelineNode should not accumulate
operations named after HSY, ALas, GPC, or particular source columns.

### Zürich nodes

| Configured class | YAML refs | Disposition | Main missing capability or observation |
| --- | ---: | --- | --- |
| `ch.zuerich.BuildingEnergy` | 3 | Ops | Named additive components and an efficiency factor. |
| `ch.zuerich.BuildingFloorAreaHistorical` | 4 | Adapter / Ops | Rename/recode source columns and aggregate dimensions. |
| `ch.zuerich.BuildingHeatHistorical` | 4 | Ops | Wide/narrow reshape and combine carrier columns. |
| `ch.zuerich.BuildingUsefulHeat` | 4 | Ops | Conditional COP adjustment and multiplication. |
| `ch.zuerich.BuildingHeatPerArea` | 4 | Ops | Aggregate, join, divide, overlay efficiency, and extend years. |
| `ch.zuerich.BuildingGeneralElectricityEfficiency` | 3 | Ops | Join energy/area/intensity inputs and calculate weighted efficiency. |
| `ch.zuerich.BuildingHeatUseMix` | 4 | Ops | Aggregate activity and normalize mix. |
| `ch.zuerich.BuildingHeatByCarrier` | 3 | Ops | Join and reshape efficiency and system-share inputs. |
| `ch.zuerich.ElectricityProductionMix` | 3 | Ops | Multi-input weighted mix with explicit internal, subsidized, and external branches. |
| `ch.zuerich.ElectricityProductionMixLegacy` | 5 | Alias / Ops | Migrate configurations to the non-legacy recipe rather than preserve a second operation family. |
| `ch.zuerich.DistrictHeatProductionMix` | 3 | Ops | Required base, optional additions, normalization, and conditional gas-grid adjustment. |
| `ch.zuerich.GasGridNode` | 4 | Ops | Apply gas-grid category substitutions over dimension combinations. |
| `ch.zuerich.EnergyProductionEmissionFactor` | 7 | Ops | Weighted factors, aggregation, optional CCS adjustment, and fill semantics. |
| `ch.zuerich.EmissionFactor` | 4 | Adapter / Ops | Select factor metric, combine datasets, deduplicate/aggregate, and extend. |
| `ch.zuerich.EmissionFactorActivity` | 6 | Ops | Join activity/factor, multiply, aggregate output dimensions. |
| `ch.zuerich.PassengerKilometers` | 4 | Ops | Aggregate mileage and multiply by occupancy. |
| `ch.zuerich.VehicleKilometersPerInhabitant` | 4 | Ops | Aggregate, divide by population, and overlay additive projections. |
| `ch.zuerich.VehicleEngineTypeSplit` | 8 | Ops | Calculate shares, join grouping metadata, and normalize projections. |
| `ch.zuerich.VehicleMileage` | 4 | Ops | Join mileage and engine-type shares and multiply. |
| `ch.zuerich.VehicleMileageHistorical` | 1 | Adapter | Select and normalize the source mileage metric. |
| `ch.zuerich.TransportFuelFactor` | 4 | Adapter / Ops | Select per-output dataset metrics, fill, filter, and normalize units. |
| `ch.zuerich.TransportEmissionFactor` | 4 | Ops | Join energy carriers with factors and add an electricity branch. |
| `ch.zuerich.TransportEmissionsForFuel` | 4 | Ops | Join factors, divide, aggregate, and add transport residuals. |
| `ch.zuerich.TransportEmissions` | 16 | Near | Multiplicative composition followed by a null invariant check. |
| `ch.zuerich.TransportEmissions2kW` | 15 | Ops | Join scope components, clamp/fill, and append scope-3 rows. |
| `ch.zuerich.NonroadMachineryEmissions` | 4 | Ops | Join fuel/factor, multiply, aggregate, and add projections. |
| `ch.zuerich.WasteIncinerationEmissions` | 1 | Ops | Join activity/factors, calculate fossil/biogenic split, and concatenate scopes. |
| `ch.zuerich.SewageSludgeProcessingEmissions` | 1 | Ops | Join CCS share and conditionally remap gas/scope outputs. |
| `ch.zuerich.WastewaterTreatmentEmissions` | 1 | Ops | Population normalization, factor multiplication, and scope assignment. |

The Zürich bundle is strong evidence that a modest set of general operations
can replace many bespoke classes. It is not evidence for Zürich-specific
operations: repeated joins, reshapes, aggregates, category assignments,
conditional expressions, and output mappings should be extracted instead.

## Capability backlog

### 1. Canonical authoring model

- Replace placeholder `PipelineConfig.operations` and the parallel loose
  `NodeSpec.pipeline` field with one typed, versionable specification.
- Use `PortInputRef` as the ordinary source reference.
- Define stable intermediate and output-port references, including several
  output metrics from one pipeline.
- Compile and validate pipeline shape rules before runtime execution.

This is required before replacing classes in stored specs, but not before
lowering them to IR for parity comparison.

### 2. General dataframe operations

The first reusable tranche should cover:

- select, rename, drop, assign, and cast metrics or dimensions;
- filter rows/categories and conditional expressions;
- join with explicit join and index-source policies;
- concatenate, coalesce, fill null/NaN, and assert non-null;
- aggregate over dimensions and group-by reduction;
- wide/narrow reshaping where a more semantic operation cannot avoid it;
- normalize shares, calculate complements, and assign a category;
- exponent, logarithm/logit, lookup/replace, and unit-aware scalar arithmetic;
- split/map one intermediate into several output ports.

Operations should carry PathsDataFrame metadata and unit rules themselves.
They must not expose raw Polars expressions as the authored contract.

### 3. Temporal operations

- select historical or forecast rows;
- extend, interpolate, backfill, and synthesize model years;
- scale or index to a reference year;
- difference, shift, cumulative sum/product, and recurrence/window operations;
- parameterized trajectories including linear and S-curve interpolation.

Some of these already exist as binding transformations. Reuse their semantics
or invoke them at a clearly different stage; do not create two subtly
different implementations with the same name.

### 4. Collection and control flow

- reduce an ordered repeated port by add, multiply, coalesce, or weighted sum;
- optional inputs without placeholder frames;
- choose a port or branch from a parameter;
- conditions over parameters and scalar intermediate values;
- reusable sub-pipelines or macros for coherent computations such as annuity,
  rather than hundreds of low-level dataframe steps.

### 5. Runtime capabilities, not dataframe operations

- evaluate a port under a named/reference scenario;
- action enabled/no-effect behavior and the common impact envelope;
- obtain goals or observations as typed inputs;
- use instance years and reference year as scalar inputs;
- map a computation over output metrics for multi-metric actions;
- inspect raw dataset/schema metadata only through an explicitly restricted
  source capability;
- tolerate or attribute an input failure according to a declared reduction
  policy.

Topology inspection and `target_node` behavior should not become pipeline
capabilities. Port bindings must deliver the intended transformed value
without the computation rediscovering graph edges or source node objects.

### 6. Operational completeness

- Pipeline errors identify the node, operation, intermediate, and binding.
- Cache identity includes the typed pipeline, referenced parameters, and
  binding identities.
- Explanations can describe the authored sequence.
- GraphQL can edit and validate the typed union without camelCase backend
  field names.
- Legacy and pipeline execution can be compared on recorded instance
  baselines before changing the stored node kind.

## Recommended implementation order

1. During input-port migration, add a short inventory entry and representative
   node-unit test for each touched class.
2. Complete lowering and parity for AdditiveNode, MultiplicativeNode, and
   AdditiveAction, including their currently unsupported branches.
3. Type the persisted PipelineConfig around the canonical PipelineSpec.
4. Replace GenericNode's most frequent pure operations with typed operations,
   starting with collection, joins, dimension aggregation, fill, temporal
   extension, and output mapping.
5. Lower promising classes to IR and compare them without changing configs.
   Zürich provides a good broad second wave after the generic primitives.
6. Design runtime facilities from ScenarioImpactNode, action nodes, and
   ObservableNode separately from dataframe operations.
7. Move source-specific parsing toward dataset adapters/binding transforms.
8. Convert stored node kinds only after class-level and instance-baseline parity
   is demonstrated.

Parallel conversion work should be divided by node module. One owner should
control shared pipeline schemas, operations, executor behavior, and the parity
harness so separate migrations do not invent competing primitives.

## Questions for the first implementation checkpoint

1. Is a pipeline always a single-output recipe, with multi-output nodes owning
   one recipe per output port, or may one recipe publish several named
   intermediates as outputs?
2. Should higher-level macros be stored as versioned operations, or expanded to
   primitive operations when the spec is persisted?
3. Which binding transformations may also appear inside a node computation,
   and how is their execution stage made unambiguous?
4. Are goals, observations, instance years, and scenario-evaluated ports all
   variants of a typed runtime input, or separate host services?
5. Should external-schema adapters become declarative dataset recipes, Python
   adapter classes, or both?
