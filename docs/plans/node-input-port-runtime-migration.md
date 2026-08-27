# Node input port runtime migration

Status: planned; implementation has not started.

## Goal

Make an input port the computation boundary for every node class:

```python
df = self.get_input(self.energy_port)
```

The returned value has already been selected from its source output port or
dataset metric and passed through the binding's ordered transformation recipe.
The consuming node does not know whether the binding came from a node or a
dataset.

This removes the arbitrary runtime distinction between
`get_input_dataset*()` and `other_node.get_output_pl(target_node=self)`, and in
particular allows several datasets, or a mixture of datasets and nodes, to be
connected wherever the port contract permits it.

The persistent half of this boundary already exists:

- `InputPortDef` gives a port durable instance-local identity;
- `NodeInputPortBinding` is the authoritative edge-or-dataset binding table;
- `PortBindingDef` and its edge/dataset variants are the immutable,
  ORM-independent graph representation;
- binding position orders values within a port across both source kinds;
- `transformations` is the binding's complete ordered adaptation recipe.

The missing half is runtime construction. `InstanceLoader` currently projects
bindings into two older, lossy structures: grouped `Edge` objects and
`Node.input_dataset_instances`. Node implementations consequently select
sources by tags, quantities and list positions, then invoke different APIs for
node and dataset values.

This plan introduces one runtime binding seam, migrates node computations to
declared ports, and removes those compatibility structures after their last
consumer is gone.

Related plans and architecture:

- [Loader and spec inversion](loader-spec-inversion.md)
- [Instance graph and dimension constraints](instance-graph-dimension-constraints.md)
- [Fault tolerance](../architecture/fault-tolerance.md)
- [MetricDataFrame](../architecture/metric-dataframe.md)
- [Architecture principles](../architecture/principles.md)

## Scope

This work includes:

- the class-facing input-port declaration and accessor API;
- static and conditional input requiredness;
- runtime binding construction from `InstanceGraph` definitions;
- source output-port/dataset-metric resolution;
- binding transformation execution;
- compatibility views for unmigrated node classes;
- migration of Python node computations away from tag/quantity selection;
- a succinct unit-test harness for node computations;
- removal of the obsolete runtime edge/dataset-input projections.

This work does not include:

- implementing `MetricDataFrame`;
- completing authored `PipelineNode` execution;
- converting every suitable custom class into a pipeline definition;
- removing compatibility for supported old snapshot revisions before the
  normal revision-retention window permits it.

Port boundaries nevertheless adopt the future MDF invariant now: one input
port value contains exactly one metric. The initial concrete return type is
`PathsDataFrame`.

## Settled semantics

### Ports are addressed through class declarations

Python node classes do not handle port UUIDs. They hold class-level semantic
declarations and pass those declarations to the input API. UUID lookup remains
inside the graph/runtime layer.

Use convenience constructors over `InputPortDeclaration` so cardinality is
visible at the declaration site:

```python
energy_port = InputPort.one('energy')
gas_mix_port = InputPort.optional('gas_mix')
additive_port = InputPort.multi('additive', required=False, aggregation='sum')
impute_port = InputPort.multi('impute', required=False)
factors_port = InputPort.repeatable('factors', min_count=2)
```

The exact helper namespace may change during implementation, but direct
`InputPortDeclaration(...)` construction must remain possible. The helpers
must produce the same declaration model rather than create a parallel port
vocabulary.

The three existing identities keep distinct responsibilities:

- UUID is durable graph and binding identity;
- `identifier` is user/formula-facing and renameable;
- `role` is stable class vocabulary shared by computation and shape rules.

Backend port roles and identifiers use `snake_case`. GraphQL may expose them
through its normal casing conventions at the API boundary, but backend Defs,
Python declarations and serialized model specifications do not use camelCase.

Instance-authored FormulaNode and future PipelineNode ports may be addressed by
identifier or UUID inside their authored algebra. That exception does not
leak into ordinary Python node classes.

### Multiplicity and aggregation are separate

`multi` means that one port may receive several homogeneous bindings. It does
not by itself prescribe how the values are combined.

- `get_input(port)` returns one port value. For a multi-port it is valid only
  when the declaration defines an automatic aggregation, initially `sum`.
- `iter_inputs(port)` yields ordered values without automatic aggregation.
  For a multi-port these are its binding values; for a repeatable role these
  are the values of its ordered port instances.
- `iter_input_ports(port)` preserves the identity and metadata of each
  repeatable port instance. Computations that need a per-port relationship,
  rather than only anonymous values, use this API.
- A non-multi port accepts at most one binding.
- A repeatable role creates several distinct ports. Products happen across
  those ports, never by treating a multi-port as a heterogeneous product.

This supports the current and foreseeable computations without assigning
addition universally to multi-ports:

- AdditiveNode reads a summing multi-port through `get_input()`;
- MultiplicativeNode iterates factor ports, reads its summing additive port,
  and iterates ordered imputation bindings;
- FormulaNode and PipelineNode may iterate collection inputs and apply their
  authored operation explicitly.

Binding position remains observable. It controls floating-point association
order for addition and precedence for ordered operations such as imputation.

### Requiredness is a port cardinality, not a call-site boolean

Static requiredness belongs to the declaration and is materialized into the
instance port definition so graph validation, revisions and the editor see the
same contract as computation.

The initial model needs to express:

- required single port: exactly one binding;
- optional single port: zero or one binding;
- required multi-port: at least one binding;
- optional multi-port: zero or more bindings;
- repeatable role: a minimum number of port instances, each with its own
  binding cardinality.

The implementation should prefer an explicit minimum binding count over
deriving requiredness from `multi`. A compatibility-friendly first schema may
store `InputPortDef.required` while the constructors expose the clearer
cardinalities above. Do not overload `InputPortDeclaration.min_count`: it
counts port instances in a role, not bindings on a port.

Static shape constraints and input availability are different facts. Shape
rules continue to describe dimensions, units and quantities. The constraint
program reports missing required bindings separately.

Some requirements depend on non-customizable node configuration. An optional
port may become required when a feature parameter is enabled. Node classes use
`require_input(port)` at computation time and may declare the same conditional
requirement through a small validation hook so publication fails before
calculation:

```python
def required_input_ports(self) -> Iterable[InputPortDeclaration]:
    yield self.base_mix_port
    if self.get_parameter_value('use_gas_network', required=False):
        yield self.grid_share_port
        yield self.gas_mix_port
```

Do not introduce scenario-dependent graph topology as part of this work.

### Transformations execute at the input boundary

From a node class's perspective, binding transformations happen in
`get_input()` or `iter_inputs()`. Internally the responsibilities are layered:

```text
Node.get_input(declaration)
  -> resolve declaration to instance port(s)
  -> RuntimeInputBinding.get_value()
       -> obtain source output-port or raw dataset-metric value
       -> apply the binding transformation recipe exactly once
       -> validate the delivered one-metric frame
  -> enforce cardinality and aggregate when declared
```

`Node.get_input()` must not become a node-versus-dataset switch. Source
resolution and transformation execution belong to the runtime binding.

For datasets, source-owned loading and overlays remain distinct from
binding-owned adaptation. Several bindings to one dataset share the decoded
source payload but apply their own recipes independently. Framework overlays
that require raw join keys run before the first temporal-fill transformation;
the stored transformation list remains the complete recipe and is never
shadowed by parallel implicit flags.

Metric selection, target-specific row selection, unit conversion, dimension
filtering/assignment and other existing edge adaptation must either be an
explicit source-port selection or a stored binding transformation before the
old target-aware `get_output_pl()` path is removed.

### Failure, caching and attribution

The runtime binding UUID is the diagnostic identity for failures while still
recording both source and target nodes/datasets.

- A transformation failure is attributed to the binding, with its source and
  target available in the error event.
- An unavailable required input makes the consumer fail or become incomplete
  according to the existing strict/tolerant context.
- An absent optional input returns `None` or yields no values.
- Skipping an unavailable value is allowed only where the node/port contract
  says missing input is a zero contribution. It is not a generic accessor
  behavior.
- A product cannot silently omit a failed factor.

The consumer cache identity includes, directly or through the instance/source
hashes:

- binding UUID and position;
- source identity and selected output port/dataset metric;
- the ordered transformation recipe and its contextual cache data;
- upstream node output or dataset payload identity;
- port aggregation semantics where automatic aggregation is used.

Resolving a binding must not compute an upstream node twice merely to inspect
availability.

## Runtime ownership

### Temporary runtime seam

Introduce an immutable-definition-backed runtime adapter:

```python
class RuntimeInputBinding:
    definition: EdgeBindingDef | DatasetBindingDef
    target_node: Node
    source: RuntimeNodeSource | RuntimeDatasetSource

    def get_value(self) -> PathsDataFrame: ...
```

Concrete edge and dataset subclasses, or source objects behind a common
protocol, should own source-specific behavior. Avoid scattering
`isinstance(EdgeBindingDef)` switches across nodes and consumers.

Each runtime Node gains:

```python
graph_node: NodeMeta
input_bindings: tuple[RuntimeInputBinding, ...]


def bindings_for(self, declaration: InputPortDeclaration) -> tuple[RuntimeInputBinding, ...]: ...
```

The immutable `PortBindingDef` remains directly inspectable as
`RuntimeInputBinding.definition`. Do not attach request-local mutable Node
objects back onto graph models: `InstanceGraph` values are immutable and may
come from a shared cache, while runtime Nodes are per-context mutable objects.

### Revisit after InstanceGraph constructs Context

The preferred final ownership is closer to the Def models. Once
`InstanceGraph` itself is the Context/runtime factory, revisit whether:

- `PortBindingDef` should expose an `instantiate(runtime)` operation;
- source-specific Def subclasses should construct their runtime binding
  counterparts;
- the separate `RuntimeInputBinding` factory can disappear;
- input aggregation/cardinality logic can live directly on an immutable port
  runtime value built by the graph.

This revisit is an explicit exit criterion, not an invitation to delay the
input migration. `RuntimeInputBinding` is the temporary seam that avoids
making the current snapshot loader grow another lossy projection.

## Node-facing examples

### BuildingEnergy

`BuildingEnergy` currently selects two datasets by tag and carries a configured
`transport_electricity` edge whose computation has been commented out. Its
declared form should be:

```python
class BuildingEnergy(Node):
    energy_port = InputPort.one('energy', label=_('Building energy'))
    other_fuel_use_port = InputPort.one('other_fuel_use', label=_('Other fuel use'))
    input_port_declarations = (energy_port, other_fuel_use_port)

    def compute(self) -> PathsDataFrame:
        df = self.get_input(self.energy_port)
        other = self.get_input(self.other_fuel_use_port)
        ...
```

Both sources are statically required. Either may later be supplied by a node
or dataset without changing the computation. Negating other fuel use is node
algebra, not binding adaptation.

The migration must explicitly decide whether to restore the dormant transport
electricity behavior or delete the stale binding. It must not silently map the
ignored edge into one of the two declared operands.

The class likely no longer needs to inherit AdditiveNode because it neither
uses nor exposes the generic additive computation. Check inherited parameters
and other behavior before changing the superclass.

### DistrictHeatProductionMix

This class separates a required base mix, ordinary additions, and two
conditionally required gas-grid controls:

```python
class DistrictHeatProductionMix(MixNode, GasGridMixin):
    base_mix_port = InputPort.one('base_mix')
    additive_port = InputPort.multi('additive', required=False, aggregation='sum')
    gas_mix_port = InputPort.optional('gas_mix')
    grid_share_port = InputPort.optional('grid_share')
    input_port_declarations = (
        base_mix_port,
        additive_port,
        gas_mix_port,
        grid_share_port,
    )
```

`GasGridMixin.use_gas_grid()` receives resolved dataframes instead of reaching
back through tags:

```python
if self.get_parameter_value('use_gas_network', required=False):
    df = self.use_gas_grid(
        df,
        grid_share=self.require_input(self.grid_share_port),
        gas_mix=self.require_input(self.gas_mix_port),
    )
```

The conditional validation hook declares the same requirement for publication.

### MultiplicativeNode

The migrated computation uses three different collection semantics:

```python
def compute(self) -> PathsDataFrame:
    df = multiply_inputs(
        self.iter_inputs(self.factors_port),
        unit=self.single_metric_unit,
    )

    additive = self.get_input(self.additive_port)
    if additive is not None:
        df = sum_inputs((df, additive), unit=self.single_metric_unit)

    for impute in self.iter_inputs(self.impute_port):
        df = impute_input(df, impute)
    return df
```

Factor ports are repeatable and heterogeneous; each has one binding. The
additive multi-port has an automatic sum. Imputation uses ordered individual
bindings and has no automatic aggregation.

Reconcile the existing declaration's `min_count=1` with the computation's
normal requirement of two factors. Required dataset factors and node factors
must count identically.

### Source-polymorphic goal/history nodes

Classes such as `DatasetReduceNode` currently try a tagged node and fall back
to a tagged dataset for both `historical` and `goal`. These become the simplest
proof that the abstraction works:

```python
historical = self.get_input(self.historical_port)
goal = self.get_input(self.goal_port)
```

There is no source-priority rule because the graph cardinality permits one
binding. Supplying both is a structural error rather than an implicit choice.

## Replacing tags and other implicit selectors

Before migrating a class, classify each use of tags, quantities or positional
selection into one of these categories:

1. **Named semantic operand** -> declared port role.
2. **Source-polymorphic operand** -> one port accepting either source kind.
3. **Homogeneous additive collection** -> summing multi-port.
4. **Ordered overlay/priority collection** -> multi-port consumed through
   `iter_inputs()`.
5. **Heterogeneous operand collection** -> repeatable role.
6. **Conditional control operand** -> optional port plus conditional
   requirement.
7. **Dataframe adaptation** -> explicit binding transformation.
8. **Genuine source metadata/topology use** -> a separate binding/topology
   inspection API, not a distortion of `get_input()`.
9. **Dead configuration** -> remove or restore deliberately.

Migrated computation code must not select inputs through binding tags, source
node tags, quantity heuristics or source-list positions. Tags may remain at
serialization compatibility boundaries until all supported configurations and
revisions carry explicit port roles.

## Compatibility during incremental migration

YAML, database draft and published calculation already converge on the typed
snapshot loader. Their remaining differences are snapshot upgrading and
dataset payload selection, not node computation semantics. Runtime binding
construction must therefore happen after source normalization and be shared by
all three modes.

Unmigrated APIs become projections of the runtime binding registry:

- `get_input_dataset*_pl()` selects dataset-backed bindings and returns their
  delivered values;
- `get_input_node(s)()` returns the source Nodes of edge-backed bindings;
- `input_nodes`, `input_dataset_instances` and edge traversal remain
  compatibility views;
- `get_output_pl(target_node=...)` remains during caller migration because it
  is a source-side API and cannot always be expressed as a literal wrapper
  around target-side `get_input()`.

Do not create separate YAML, database or published compatibility
implementations. Old snapshots upgrade at the existing snapshot/graph
boundary, after which the runtime sees the same port and binding definitions.

Legacy methods may continue to select by tags only for unmigrated callers.
Their implementation must use the new registry so source resolution,
transformations, errors and cache identity have one implementation.

## Succinct node test harness

Add a non-ORM `NodeTestCase`-style helper that constructs a minimal Context,
NodeSpec and runtime binding set while executing the real input resolver and
transformation pipeline.

Target usage:

```python
def test_building_energy(node_case):
    case = node_case(
        BuildingEnergy,
        unit='GWh/a',
        dimensions={'energy_carrier': ['electricity', 'natural_gas']},
    )
    case.bind(BuildingEnergy.energy_port, energy_frame)
    case.bind(BuildingEnergy.other_fuel_use_port, other_fuel_frame)

    assert_frame(case.compute(), expected)
```

The harness provides:

- compact one-metric frame construction with year, forecast, dimensions,
  quantity and unit metadata;
- `bind(port, frame, source_kind='node' | 'dataset', transformations=...)`;
- `bind_many(port, frames, ...)` with stable positions;
- parameter and scenario setup;
- `compute()` and concise value/frame assertions;
- access to binding errors and node status.

Most node-class behavior tests should require neither Django nor database
factories. They must pass through `RuntimeInputBinding`, not monkeypatch
`Node.get_input()` or populate the old `Edge`/dataset lists.

Keep a smaller integration layer that builds real snapshots/InstanceGraphs and
proves that YAML, draft and published construction create equivalent runtime
bindings. The class harness tests computation contracts; integration tests test
construction and persistence.

Every migrated node class gets focused tests for:

- required and optional inputs;
- node-backed and dataset-backed parity where both are legal;
- its multi/repeatable ordering and aggregation semantics;
- relevant transformation behavior;
- missing or failed inputs;
- any conditional input requirements.

## Implementation sequence

### Review checkpoint: runtime foundation (2026-08-27)

Implemented before the first production-node conversion:

- `InputPort.one()`, `.optional()`, `.multi()` and `.repeatable()` declaration
  constructors, with declaration-level requiredness and optional `sum`
  aggregation;
- request-local `RuntimeInputBinding` adapters retaining the immutable graph
  binding definition, binding UUID, position and source identity;
- loader attachment of graph-derived runtime bindings while the legacy Edge
  and dataset views remain available;
- declaration-addressed `get_input()`, `iter_inputs()` and `require_input()`;
- binding recipe metadata in cache identity and binding UUIDs in accessor
  errors;
- a compact frame/binding test harness plus one real snapshot-loader topology
  test;
- requiredness and aggregation in the GraphQL declaration catalog.

Deliberately left for review before pilot migrations:

- requiredness is not yet persisted on `InputPortDef` or enforced by the
  whole-graph constraint program;
- the loader still constructs the legacy views independently instead of
  projecting them from the runtime registry;
- dataset runtime sources are temporarily joined through snapshot
  `dataset_index`, until `InstanceGraph` owns Context construction;
- source-mode equivalence, complete transformation coverage and generalized
  tolerant-input behavior still need the later stage gates;
- no Zürich or other production node class has been converted.

### Pilot checkpoint: Zürich sample (2026-08-27)

Converted after review:

- `BuildingEnergy` now consumes required `energy` and `other_fuel_use`
  declarations through `require_input()`; its configured but dormant
  `transport_electricity` edge remains deliberately unclassified and ignored;
- `DistrictHeatProductionMix` now consumes `base_mix`, an optional summing
  `additive` port, and conditionally required `gas_mix` and `grid_share`
  inputs through the runtime-binding API;
- `MixNode.normalize_mix()` separates normalization from the legacy
  node-source traversal, allowing a mixed-source aggregated value to be
  normalized without reconstructing source identity;
- legacy YAML bindings for all four Zürich variants resolve to the same
  semantic roles, while the loader synthesizes a structural dataset catalog
  only when a YAML snapshot has no persisted catalog available.

The live `configs/zuerich.yaml` computations for both pilot nodes were compared
in-process against copies of their legacy algorithms and matched exactly.
`GasGridNode` and the remaining Zürich classes still use the old accessors.

### Core additive/multiplicative checkpoint (2026-08-27)

- `AdditiveNode` now consumes its additive and imputation roles through runtime
  bindings. Its additive multiport accepts mixed node/dataset sources and more
  than one dataset, while retaining the v1 dataset preprocessing and tolerant
  failure behavior.
- `MultiplicativeNode` now consumes factor, additive and imputation node roles
  through runtime bindings. Legacy dataset bindings remain unclassified for
  this class: some are currently ignored operands while others are auxiliary
  replacement/fill data, so treating all of them as factors would silently
  change existing models. `MultiplicativeNode2` remains the explicit path for
  dataset factors until those configurations are disambiguated.
- `AdditiveAction` and its `Hypothesis` and `BudgetingAction` subclasses expose
  one generated, non-editable single-metric input per output metric. Each
  `InputPortDef.paired_output_port_id` records the durable relationship;
  runtime computation iterates the instantiated ports and emits each value
  through its paired output. Adding an output in the editor creates the input
  atomically, while bindings on that input remain editable.
- Existing YAML, database and retained snapshot specs acquire these pairs at
  their explicit graph/parser adapter boundaries. A wide dataset remains one
  dataset source with a separate metric binding to each generated input port;
  computation never reconstructs a wide input from binding column names.
- Runtime bindings retain their request-local source object and can be iterated
  without resolving the value. This preserves per-binding tolerant failure and
  source-aware compatibility behavior without exposing UUID lookup to node
  computations.
- Runtime nodes are bound to their `NodeMeta`, and `RuntimeInputPort` groups
  bindings by the actual target-port UUID before node computation begins.
- Inline `historical_values` / `forecast_values` are still absent from
  `InstanceGraph`; an explicit class-declared compatibility role adapts their
  `FixedDataset` until the graph represents them as ordinary dataset bindings.

The additive inheritance audit found that only generic subclasses should rely
on these inherited roles. Subclasses with tagged, positional or domain-specific
input algebra must redeclare their complete port contract as they migrate; an
inherited declaration alone is not evidence that their overridden computation
uses runtime inputs.

### 0. Characterize and build the harness

- Inventory old input access by semantic pattern rather than performing a
  blind mechanical call-site rewrite.
- Record dead or ignored bindings found during the inventory.
- Extract the compact frame/context helpers already repeated in
  `test_add_multiply_semantics.py` and related tests.
- Add the Node test harness against a minimal temporary runtime-binding
  protocol so later stages can develop test-first.

**Gate:** a test-only node demonstrates required, optional, multi and
repeatable ports with node/dataset parity and no ORM setup.

### 1. Complete the declaration/cardinality contract

- Add declaration convenience constructors.
- Represent requiredness in `InputPortDef` and the GraphQL/editor projection.
- Define optional/required single and multi cardinality validation.
- Define automatic aggregation metadata, initially `sum` or absent.
- Make repeatable-role minimums and per-port binding requirements distinct.
- Add conditional-requiredness validation hook.
- Ensure class declaration defaults materialize identically in YAML parsing,
  node creation, sync, copy and revision snapshots.

**Gate:** editor and whole-graph validation agree on occupancy; old snapshots
upgrade without changing existing calculated outputs.

### 2. Construct RuntimeInputBinding from InstanceGraph

- Make calculation construction build or receive the same `InstanceGraph`
  used by structural consumers.
- Bind each runtime Node to its `NodeMeta` and runtime input adapters.
- Preserve binding UUID, target port, position, source port/metric, tags and
  transformations.
- Resolve runtime source Nodes and dataset payload sources by UUID.
- Retain the immutable Def on every adapter for diagnostics and inspection.
- Keep the old Edge/dataset projections generated from the adapters for now.

**Gate:** one port can interleave edge and dataset bindings in stored position
order without losing identity; all three instance source modes construct the
same runtime binding topology.

### 3. Resolve and transform one binding value

- Select exactly the declared source output port or dataset metric.
- Split reusable dataset source loading from per-binding adaptation.
- Execute the complete transformation recipe once.
- Enforce the one-metric port boundary.
- Attribute errors to the binding and include both endpoints.
- Include the binding/source/recipe in cache identity.
- Preserve framework overlay and temporal-fill ordering.

**Gate:** binding-level tests show node/dataset equivalence for all shared
operations, source-kind-specific operations are rejected consistently, and no
operation executes twice.

### 4. Add the node-facing accessor and compatibility views

- Implement declaration-to-instance-port resolution.
- Implement `get_input()`, `iter_inputs()` and `require_input()`.
- Apply declared automatic aggregation in stable position order.
- Generalize tolerant additive skip behavior at the binding boundary without
  enabling it for other algebra.
- Reimplement legacy accessors and topology properties as registry views.

**Gate:** legacy and new access on the same runtime produce identical values,
errors, hashes and statuses.

### 5. Pilot migrations

Migrate representative semantics before bulk work:

1. AdditiveNode/AdditiveNode2: summing mixed-source multi-port and tolerant
   failure behavior.
2. A source-polymorphic historical/goal class such as DatasetReduceNode.
3. BuildingEnergy: two required named inputs and the dormant transport binding
   decision.
4. DistrictHeatProductionMix: required base, optional additions and
   conditionally required gas-grid controls.
5. MultiplicativeNode2, then MultiplicativeNode: repeatable factors, additive
   aggregate and ordered imputation.

Each pilot receives class-level unit tests before its production configuration
is changed.

**Gate:** recorded outputs remain unchanged except for separately approved bug
fixes; pilot configurations contain no computation-selection tags.

### 6. Migrate remaining classes by pattern

- Generic/simple core classes first.
- Shared action classes next.
- Region-specific classes last.
- Convert suitable classes to authored pipelines only when PipelineNode is
  independently ready; port migration does not wait for it.
- Re-declare ports on subclasses whose overridden computation changes the
  inherited algebra. Do not consider a class migrated merely because it
  inherits declarations.

Track every removed tag heuristic and every retained topology inspection use.

**Gate:** no migrated compute method calls `get_input_dataset*()`,
`get_input_node(s)()`, or `source.get_output_pl(target_node=self)`.

### 7. Remove compatibility runtime state

- Remove target-aware transformation behavior from `Node.get_output_pl()`.
- Remove legacy runtime `Edge` grouping once graph/topology consumers use the
  binding registry.
- Remove `input_dataset_instances` once dataset source consumers use runtime
  bindings/source stores.
- Remove tag/quantity/position selection from compatibility methods and then
  delete the methods.
- Retire transitional dataset grouping state (`dataset_index`,
  `DatasetPortSpec`) when every dataset binding executes directly.
- Revisit RuntimeInputBinding ownership as part of making InstanceGraph the
  Context factory; move construction/behavior closer to the Def models where
  that boundary permits it.

**Gate:** one canonical graph-to-runtime binding path remains, with no
edge-versus-dataset distinction exposed to node computation.

## Verification matrix

Focused tests must cover:

- required, optional and conditionally required ports;
- non-multi occupancy rejection;
- required and optional multi-ports;
- repeatable heterogeneous ports;
- multiple datasets on one port;
- edge and dataset inputs mixed on one port;
- stable position ordering after insert, reorder and delete;
- multi-metric sources narrowed to the selected output port/metric;
- every transformation supported by both source kinds;
- source-kind-specific transformation rejection;
- framework dataset overlays before temporal reshaping;
- strict failure, additive tolerant skip and all-inputs-unavailable behavior;
- binding-level error identity and endpoint reporting;
- cache invalidation after source, metric, transformation, order or
  aggregation changes;
- YAML, database draft and published revision parity;
- old snapshot upgrading within the supported revision window.

Repository-level gates:

- focused node and binding unit tests;
- full pytest suite;
- mypy and ruff on changed code;
- parse-oracle parity;
- recorded `test_instance --compare` outputs for every available baseline;
- explicit before/after comparison for every production instance containing a
  migrated class.

Floating-point comparisons retain current input association order. Any
intentional semantic correction is isolated, documented and approved rather
than hidden inside the migration baseline.

## Completion criteria

The migration is complete when:

- every Python computation obtains bound values by declared port;
- node and dataset sources are interchangeable wherever their port contract is
  the same;
- multiple datasets are not artificially restricted by the runtime API;
- port cardinality and conditional requirements are validated before
  publication;
- binding transformations execute once at the input boundary;
- node computation contains no input-selection tags or quantity heuristics;
- binding UUIDs anchor runtime diagnostics and cache dependencies;
- concise unit tests describe each migrated node class's input contract;
- old input accessors and lossy runtime projections are removed;
- RuntimeInputBinding's final ownership has been revisited after
  InstanceGraph-to-Context construction lands.
