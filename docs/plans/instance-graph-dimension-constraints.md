# InstanceGraph and dimensional constraints

*Status: proposed. This plan depends on the stable-port work described in
[`loader-spec-inversion.md`](loader-spec-inversion.md), but the metadata graph
and solver can be introduced before the runtime loader inversion is complete.*

Coordination with the other tracks, so no session re-derives the ordering:

- The native `from_snapshot()` construction (loader plan step 2) is delivered
  here as step 10: the native loader consumes `InstanceGraph`, not a second
  snapshot-reading path. The two are one work item.
- The `flatten` placeholder retirement is step 2 here — deliberately early,
  per [`dimension-constraints.md`](../architecture/dimension-constraints.md)
  §"`FlattenTransformation` is not a flatten". The loader plan's "assertion
  moves to declared port dimensions" bullet lands with it.
- `NodeInputPortBinding` (dimension-constraints next-steps item 1) is step 9.
- The class-level typed port declaration introduced for `shape_rules()` is
  the same declaration the loader plan's `get_input(port)` accessor consumes
  (see [Identity boundary](#identity-boundary)).

## Goal

Introduce one immutable, serializable metadata graph that can answer structural
questions without constructing a runtime `Context` or loading dataset payloads.
Use that graph to compile and solve port constraints for dimensions, categories,
units, and quantities.

The target construction path is:

```text
YAML / DB draft / published revision
                 |
                 v
         InstanceSnapshot
                 |
                 v
          InstanceGraph  ----> constraint result
                 |
                 v
              Context    ----> dataframes and computation
```

`InstanceGraph` is the resolved structural aggregate. `Context` is the mutable
execution environment constructed from it. Editor queries, graph validation,
dependency inspection, and connection compatibility should stop at
`InstanceGraph`; only calculation and data inspection should create a
`Context`.

## Decisions already made

- Durable graph identity is UUID-based. Node, port, binding, dataset, metric,
  dimension, and category references inside snapshots and `InstanceGraph` use
  UUIDs. Human-readable identifiers are labels and authoring conveniences, not
  graph keys.
- YAML and legacy GraphQL may continue accepting identifiers at explicit
  compatibility boundaries. They resolve identifiers once and emit the UUID
  form. New domain APIs must not accept `UUID | str` unions.
- `NodeMeta` and binding definitions are graph-owned objects. They hold a
  non-serialized back-reference to their `InstanceGraph` and are not valid for
  navigation until the graph binds them.
- `NodeMeta.node_class_path` is serialized. `node_class`, graph indexes,
  NetworkX projections, topological order, and compiled constraints are cached
  derived properties and are not serialized.
- Reuse `PortBindingDef`, `EdgeBindingDef`, and `DatasetBindingDef`. Do not add
  parallel `BindingMeta` or public `BindingValueRef` types. A binding UUID is
  the solver's identity for the value delivered to an input port.
- Constraint results are derived state. They are neither Django models nor
  authored fields in `NodeSpec`.
- `supported_dimensions` has no demonstrated semantic meaning. It is generated
  together with `required_dimensions` for grouped multi-ports and is otherwise
  used only by the current editor connection check. Remove it; do not reinterpret
  it as an authored upper bound.
- Dataset schema describes declared shape. A separate, payload-light profile
  describes categories that actually have datapoints. External placeholders
  have unknown observed coverage until import/sync records it.
- `InstanceGraph` gets a thin `create_context()` entry point, but runtime
  construction remains implemented by an injected builder/loader so the
  metadata package does not depend on runtime internals.

## Identity boundary

“UUIDs everywhere” does not mean that readable semantic names disappear from
node-class code. A class needs to be able to say “the `factors` input” and “the
`output` output”; those are role selectors, not references stored in the graph.
`shape_rules(meta)` resolves selectors against that particular node and returns
rules containing UUIDs.

For example:

```python
@classmethod
def shape_rules(cls, meta: NodeMeta) -> tuple[PortShapeRule, ...]:
    factors = meta.input_port_ids_for_roles('factors')
    output = meta.require_output_port('output')
    return (ProductShapeRule(inputs=factors, output=output.id),)
```

If port identifiers later become optional in persisted data, class-declared
ports still need stable semantic roles. That can be represented by a typed
class-level port declaration rather than by making the instance identifier a
durable reference. The compiled rule remains UUID-only.

That class-level port declaration is shared infrastructure, not private to
rules: it is the same declaration the runtime `get_input(port)` accessor
([`loader-spec-inversion.md`](loader-spec-inversion.md), step 1) resolves at
computation time. Rule compilation and runtime input access must consume one
role namespace — two parallel role systems on the same classes is exactly the
drift this plan exists to prevent.

Portable exports carry UUIDs as their canonical references and may additionally
carry identifiers for diagnostics and legacy import matching. Import either
preserves the exported UUID or records an explicit old-to-new UUID remap; it
must not silently fall back to identifier lookup in the common load path.

## Target metadata model

Place the metadata graph in a small module that does not import `Context`,
runtime `Node`, dataframe types, or dataset payload stores. A likely initial
split is:

```text
nodes/instance_graph.py
nodes/constraints/rules.py
nodes/constraints/values.py
nodes/constraints/solver.py
nodes/dataset_shape.py
```

The module boundary governs imports at module scope, not behavior:
`NodeMeta.node_class` and rule compilation import runtime node classes lazily
through `node_class_path`. That one-time import cost is accepted (editor
cold-start is not critical) and buys freedom from cyclical imports — the
metadata package never imports runtime modules at module scope, so runtime
classes may import the metadata package freely.

### Common graph-bound base

Use a common Pydantic base for graph-owned definitions:

```python
class InstanceGraphBoundModel(BaseModel):
    _graph: InstanceGraph | None = PrivateAttr(default=None)

    @property
    def graph(self) -> InstanceGraph:
        if self._graph is None:
            raise RuntimeError(f'{type(self).__name__} is not bound to an InstanceGraph')
        return self._graph

    def _bind_graph(self, graph: InstanceGraph) -> None:
        if self._graph is not None and self._graph is not graph:
            raise RuntimeError(f'{type(self).__name__} is already bound to another graph')
        self._graph = graph
```

The exact base/mixin shape should be chosen after checking the Pydantic MRO of
`I18nBaseModel`. The invariant matters more than inheritance syntax:

- `_graph` is a `PrivateAttr` and never appears in `model_dump()`;
- rebinding to another graph fails;
- graph navigation fails clearly on an unbound value;
- graph-owned values are frozen after construction;
- `model_copy()` of an already bound child is not an editing API.

Use this base initially for `NodeMeta` and `PortBindingDef`. Bind other metadata
objects only when they need graph navigation; do not make every small value
carry a back-reference pre-emptively.

### NodeMeta

`NodeMeta` is the graph's resolved node view, not a second persisted node
schema:

```python
class NodeMeta(InstanceGraphBoundModel):
    id: UUID
    identifier: str | None = None
    node_class_path: str
    spec: NodeSpec

    @cached_property
    def node_class(self) -> type[Node]: ...

    @property
    def input_bindings(self) -> tuple[AnyPortBindingDef, ...]:
        return self.graph.bindings_for_node(self.id)

    def bindings_for_port(self, port_id: UUID) -> tuple[AnyPortBindingDef, ...]: ...
    def require_input_port(self, role: str) -> InputPortDef: ...
    def require_output_port(self, role: str) -> OutputPortDef: ...
```

Keep display metadata needed by metadata-only GraphQL queries either directly
on `NodeMeta` or in its embedded snapshot-derived metadata. Do not include CMS
bodies or layout unless a concrete metadata consumer needs them; layout writes
must not invalidate a computation graph merely because the graph cache happens
to contain editor coordinates.

### Binding definitions

Evolve the existing definitions rather than wrapping them:

```python
class NodePortRef(BaseModel):
    node_id: UUID
    port_id: UUID


class PortBindingDef(InstanceGraphBoundModel):
    id: UUID
    port_ref: NodePortRef
    position: int
    tags: tuple[str, ...]
    transformations: tuple[PortTransformOp, ...]


class EdgeBindingDef(PortBindingDef):
    kind: Literal['edge'] = 'edge'
    from_ref: NodePortRef


class DatasetBindingDef(PortBindingDef):
    kind: Literal['dataset'] = 'dataset'
    dataset_id: UUID
    metric_id: UUID
```

External repository coordinates remain source metadata on the referenced
dataset, not a substitute graph identity. A placeholder dataset and its metric
therefore also have UUIDs.

Useful navigation (`target_node`, `target_port`, `source_node`, `source_port`,
`dataset`, `metric`) is implemented as properties or `cached_property` values
through `binding.graph`. The serialized form contains only IDs and authored
values.

### InstanceGraph

The public fields should be plain versioned Pydantic data:

```python
class InstanceGraph(BaseModel):
    format_version: int
    instance_id: UUID
    metadata: InstanceMetadata
    spec: InstanceModelSpec
    nodes: tuple[NodeMeta, ...]
    bindings: tuple[AnyPortBindingDef, ...]
    dimensions: tuple[DimensionMeta, ...]
    datasets: tuple[DatasetMeta, ...]
```

After validation, `model_post_init()` binds each graph-owned child. Derived
state is exposed through cached properties, including:

- `node_by_id` and optional identifier indexes for compatibility lookup;
- input/output port indexes keyed by `(node UUID, port UUID)`;
- `binding_by_id`, `bindings_by_input`, and bindings in `position` order;
- dimension/category, dataset, and metric indexes keyed by UUID;
- the directed NetworkX graph and topological order;
- the compiled constraint program.

Deserialization must run the same binding step as fresh construction. Tests
must cover model validation, JSON round-trip, cache round-trip, and failure to
reuse a bound child in a different graph.

## Serialization and caching

Use two cache levels:

- an in-process L1 may retain a hydrated `InstanceGraph`;
- the shared/persistent L2 stores a versioned public representation, such as
  `model_dump(mode='json')` encoded as JSON or MessagePack.

Do not pickle the hydrated Pydantic object. Pickle can capture private
back-references, cached properties, imported classes, and NetworkX objects,
turning disposable implementation details into the cache contract. If pickle
is measured to be materially faster, pickle only the plain dumped structure
and keep the explicit `format_version` validation boundary.

The cache key must distinguish at least:

- graph format/builder version;
- instance UUID;
- source mode: draft, published revision, or YAML;
- source version: draft `cache_invalidated_at`, published revision ID/content
  hash, or YAML dependency hash.

The graph builder must not read datapoints or materialized dataset payloads.
Record build time, serialized size, L1/L2 hit rate, and query count. Add a query
assertion proving that metadata-only construction issues no datapoint or
payload-content query.

## Dataset shape metadata

Keep two concepts separate:

```python
class DatasetMeta(BaseModel):
    id: UUID
    metric_ids: tuple[UUID, ...]
    declared_dimensions: tuple[UUID, ...]
    # Declared category registry and external source metadata as needed.


class DatasetShapeProfile(BaseModel):
    dataset_id: UUID
    metric_id: UUID
    categories_by_dimension: Mapping[UUID, frozenset[UUID] | None]
    has_datapoints: bool | None
    source_version: str
```

`None` means unknown; an empty set means observed and empty. Build profiles in
one grouped query over distinct dataset, metric, dimension, and category IDs.
Load profiles only for dataset bindings participating in the requested
validation. Cache them independently by dataset materialization generation or
published dataset revision.

For published graphs, derive or store the profile with the immutable dataset
revision so validation cannot observe live draft datapoints. For current
external placeholders, use declared schema for dimension-level reasoning and
unknown observed categories. Editor validation must never fetch DVC data.

## Constraint rule contract

Rules are immutable, declarative values containing node-local port UUIDs:

```python
class SameShapeRule(BaseModel):
    kind: Literal['same'] = 'same'
    inputs: tuple[UUID, ...]
    output: UUID


class ProductShapeRule(BaseModel):
    kind: Literal['product'] = 'product'
    inputs: tuple[UUID, ...]
    output: UUID


class DimensionTransformRule(BaseModel):
    kind: Literal['dimension_transform'] = 'dimension_transform'
    input: UUID
    output: UUID
    requires: frozenset[UUID] = frozenset()
    consumes: frozenset[UUID] = frozenset()
    produces: frozenset[UUID] = frozenset()
    transparent: bool = True


type PortShapeRule = SameShapeRule | ProductShapeRule | DimensionTransformRule
```

The base runtime node class exposes:

```python
@classmethod
def shape_rules(cls, meta: NodeMeta) -> tuple[PortShapeRule, ...]: ...
```

`NodeMeta` is sufficient because it owns its graph reference. A separate
`NodeMetaBuilderView` would add a type and argument indirection without adding
an ownership boundary.

The graph validates every rule as it is compiled: referenced ports belong to
the node and have the correct direction, `consumes` is a subset of `requires`,
and produced/consumed dimension identities exist in the instance registry.
Pipeline nodes compile their operation IR to the same rule union.

### AdditiveNode

Migrate the node to an explicit multi input role (`additive`) and one output
role (`output`). Its rule is:

```python
return (
    SameShapeRule(
        inputs=(meta.require_input_port('additive').id,),
        output=meta.require_output_port('output').id,
    ),
)
```

For a multi-port, `SameShapeRule` constrains every delivered binding value,
the port aggregate, and the output to the same dimensions; units must be
convertible and quantities equal. Imputation is a separate explicit `impute`
multi port (declared on `AdditiveNode` as of 2026-08-11), not a tag exception
hidden inside equality — its values overlay the result and therefore share
the output shape, so it joins the same rule.

### MultiplicativeNode

Migrate the node to explicit roles (decision 2026-08-11, revising the earlier
multi-factor-port sketch):

- `factors`: **repeatable** single ports — each factor is its own port
  instance carrying its own unit, quantity and dimension expectations. The
  product happens *across* these ports;
- `additive`: optional multi input equal to the computed result shape;
- `impute`: optional multi input equal to the final output shape;
- `output`: output port.

This gives the two multiplicities exactly one meaning each: a **multi** port
(one port, many bindings) is always a homogeneous ``same``-shaped aggregate,
while a **repeatable** role holds heterogeneous instances. Products only ever
happen across distinct ports; the earlier "multi-port product over delivered
bindings" definition is retired. Declarations carry ``min_count`` /
``default_count`` (factors: min 1, default 2; additive: min 0, default 1) so
node creation and the editor's add-port affordances read the same catalog.

Its initial rules are conceptually:

```python
return (
    ProductShapeRule(
        inputs=meta.input_port_ids_for_roles('factors'),
        output=meta.require_output_port('output').id,
    ),
    SameShapeRule(
        inputs=meta.input_port_ids_for_roles('additive', 'impute'),
        output=meta.require_output_port('output').id,
    ),
)
```

Additive and imputation values match the output shape. Two rules constraining
one output port is legitimate; only intermediates need a unique producing
rule.

During migration only, graph construction may classify legacy bindings from
`impute` / `non_additive` tags and current output-unit compatibility. Keep that
adapter at snapshot-to-graph construction and emit diagnostics. Do not put the
heuristic in `shape_rules()`: changing a unit must not silently change a
binding's computational role. Remove the adapter once persisted ports have
explicit roles.

### Formula and pipeline nodes (decided 2026-08-11)

Class declarations describe *class-fixed* algebra; `FormulaNode` and the
upcoming `PipelineNode` have *instance-authored* algebra, and their ports are
authored artifacts rather than role instances:

- The identifier is load-bearing — it is the variable the formula references —
  so `role` stays `None` and no generic `operand` role is invented. Each
  port's semantics come from usage, recovered by compilation. The legacy
  inference hook correctly does nothing for these classes.
- `shape_rules(meta)` compiles the authored artifact: the pipeline compiler
  already exists, and formula-AST compilation emits the same rule union
  (`convert_gwp()` is the natural first real producer of
  `DimensionTransformRule`).
- `multi` is a per-port authoring decision; several multi ports are fine —
  each is its own homogeneous aggregate, products still happen only across
  ports.
- Editor affordance is a class capability flag (`supports_authored_ports`
  or similar) offered alongside the declaration catalog: "add input port"
  with user-supplied identifier and multi toggle. Add-then-reference is the
  expected flow — an unreferenced port is a benign draft diagnostic, while a
  formula variable with no port is a publication-blocking conflict.
- Renaming: pipeline operation refs are already port UUIDs, so
  `PipelineNode` ports rename freely; formula text references names, so a
  rename must atomically rewrite the formula (inside the step-9 aggregate
  write) or be refused. Storing formulas in a resolved UUID-referencing form
  is the eventual fix, not v1.

## Solver model

Compile rules, authored declarations, bindings, transformations, dataset
schemas, and optional dataset profiles into a constraint program. The program
uses UUID keys only:

```python
@dataclass(frozen=True)
class PortKey:
    node_id: UUID
    port_id: UUID
    direction: Literal['input', 'output']


@dataclass(frozen=True)
class ConstraintOrigin:
    kind: Literal['declaration', 'node_rule', 'binding', 'transformation', 'dataset_schema', 'dataset_profile']
    node_id: UUID | None = None
    port_id: UUID | None = None
    binding_id: UUID | None = None
    transformation_index: int | None = None
```

Internally distinguish:

- an exact known produced dimension/category set;
- an unknown produced set;
- dimensions/categories required by consumers;
- units/quantities that are known, derived, or unconstrained.

This distinction represents exact scalar (`known = empty`) separately from
unknown (`known = None`) without assigning invented semantics to
`supported_dimensions`. Constraint propagation is a bidirectional monotone
fixpoint: forward facts derive output shapes, backward facts derive input
requirements, and binding transformations translate between them.

The result contains effective shapes, unresolved facts, and structured
conflicts with origins. It does not throw on the first contradiction; callers
need the complete set for editor diagnostics. A strict validation facade may
turn conflicts into one domain exception for publication or computation.

Support hypothetical edits as overlays:

```python
result = graph.solve_constraints(
    profiles=profiles,
    overlay=GraphOverlay(add_bindings=(candidate,), remove_binding_ids=()),
)
```

An overlay must not clone/rebind `NodeMeta` values or mutate the cached graph.
This is the connect-time compatibility path.

### Quantity algebra: validation only in v1

Output quantities stay authored. The solver checks them where a rule applies
and stays silent where none does; incompleteness must never block a
computation whose units check out. Pint remains the hard gate for dimensional
arithmetic — quantities are semantic templates layered over it, not a parallel
exponent system (do not rebuild pint over quantity labels).

The v1 rule set, measured against the 1,043 multiplicative nodes with known
inputs in current configs (2026-08-07; these three rules cover 82%):

- **Scalar identities**: `fraction` / `ratio` / `mix` / dimensionless
  `number` factors preserve the other operand's quantity.
- **Factor cancellation**: a factor quantity (`emission_factor`,
  `energy_factor`, …) times an activity yields the factor's numerator
  (`emissions`, `energy`, …), guarded by pint unit cancellation. The
  *concrete* activity is recovered from the port's unit, never enumerated in
  the quantity — this dissolves "emission factor per what?" without a
  quantity-level dimensional algebra.
- **Price**: a `unit_price` quantity times anything yields `currency`.

The quantity classes these rules need already exist in `nodes/constants.py`
(`ACTIVITY_QUANTITIES`, `ACTIVITY_FACTOR_QUANTITIES`, `UNIT_PRICE_QUANTITIES`);
they are not wired to any algebra yet. The 18% of nodes the rules cannot check
are vocabulary debt, each recoverable from the unit — `per_capita` erasing its
numerator, heating values hiding in `ratio`, `number` acting as an activity
versus a scalar — and get cleaned up opportunistically; each cleanup raises
checkable coverage and none blocks anything. The generic `factor` quantity is
this system's `Any` and stays advisory. No auto-derivation of output
quantities in v1.

## GraphQL migration

Add UUID-canonical references without breaking the editor's deployment order.
The exact schema names should follow the existing `portRef`, `fromRef`, and
`bindingEditor` vocabulary:

- output references add canonical `nodeUuid: UUID!`; the existing `portId:
  UUID!` is already canonical and stays unchanged;
- retain identifier-valued `nodeId` as a deprecated alias while clients
  migrate;
- new mutation inputs accept UUID reference objects, for example
  `fromRef: NodePortRefInput!` and `portRef: NodePortRefInput!`, containing
  `nodeUuid` and `portId`;
- retain identifier-based `fromNodeId` / `toNodeId` and legacy port strings as
  deprecated, mutually exclusive compatibility inputs;
- make the formerly required legacy fields optional at the schema level, then
  validate that exactly one complete reference form was supplied (GraphQL
  cannot deprecate a required input field without a default);
- resolve legacy identifiers at the GraphQL boundary, then call the same
  UUID-only application service as canonical inputs;
- reject a request that supplies both forms with different targets;
- add usage metrics/logging for the legacy fields and remove them only after
  the UI and stored queries no longer use them.

Do not silently change the meaning of the existing GraphQL `nodeId` from an
identifier to a UUID. Even though it is typed as `ID`, that would be a semantic
breaking change without an introspectable migration path.

Expose effective constraint data separately from authored port fields, for
example `effectiveShape` and `constraintConflicts`. Never overwrite or accept
computed values through `InputPortInput` / `OutputPortInput`.

Replace `_validate_edge_ports()` and the dataset metric compatibility check
with one application service that:

1. loads the cached graph and relevant dataset profiles;
2. applies a candidate binding overlay;
3. solves constraints;
4. reports conflicts whose origins involve the candidate or facts it makes
   contradictory;
5. writes only after validation succeeds.

Publication runs strict whole-graph validation. Ordinary metadata reads may
return conflicts so incomplete drafts remain inspectable.

The strictness boundary against
[`fault-tolerance.md`](../architecture/fault-tolerance.md): structural
constraint conflicts (dimension/category/unit/quantity contradictions) block
publication, while the metadata-level failures that fault tolerance
deliberately degrades (a broken visualization, bad action metadata) do not —
those surface as node status, not as graph invalidity.

## Persisted binding convergence

The solver does not need to wait for the ORM tables to merge: the first graph
builder can normalize `EdgeSnapshot` and `DatasetPortSnapshot` into the common
binding definitions. Persisted convergence is still required because ordering
and identity belong to the common domain concept.

Introduce `NodeInputPortBinding` as described in
[`dimension-constraints.md`](../architecture/dimension-constraints.md), with:

- one binding UUID;
- target node/port UUIDs;
- a shared `position` across edge and dataset sources;
- exactly one UUID-based source branch;
- the common transformation pipeline and tags.

Backfill existing `NodeEdge` UUIDs directly. Dataset bindings that currently
span several `DatasetPort` metric rows need one explicit binding identity
before backfill; do not choose an arbitrary row UUID as the durable group ID.
Preserve observed iteration order because floating-point addition makes it
observable.

Sequence the `position` backfill deliberately with the loader plan's YAML
edge-declaration normalization (moving `output_nodes` entries onto the target
side): both reassign the order bindings are observed in, and doing them
independently invites a silent reordering that only the full-compare gate
would catch.

Upgrade `InstanceSnapshot` to one discriminated binding list. A pure Pydantic
upgrader can normalize legacy edge arrays, but it cannot invent the dataset and
metric UUIDs absent from an old `DatasetPortSnapshot`. Keep that resolution in
one explicit legacy snapshot adapter with an injected source-specific catalog:
the current DB catalog for drafts and the pinned dataset-revision metadata for
published revisions. Reading an old pinned payload to recover its schema is an
acceptable compatibility cost; new snapshots include the UUID metadata and do
not need it. New snapshots never serialize the legacy form or resolve by
identifier during common graph construction.

## Implementation sequence

### 1. Correct the architecture contract and pin baselines

- Verify `dimension-constraints.md` agrees: as of `c431ea41` it does not
  interpret `supported_dimensions` as an authored upper bound, and it records
  that only the multi-group export path still writes computed
  `required_dimensions` / `supported_dimensions` into authored fields. Add
  the known/unknown produced-shape distinction there if anything still
  contradicts it.
- Record representative snapshot/metadata graph query counts, build time, and
  size. Keep the existing no-datapoint metadata measurement as a regression
  target.
- Add inventory tests for legacy generated `required_dimensions ==
  supported_dimensions` so cleanup does not accidentally delete an independently
  authored requirement.

**Gate:** the document, compatibility assumptions, and measured baseline agree
with current production-shaped data.

### 2. Retire the `flatten` placeholder onto port declarations

Independent of the graph and deliberately early: it removes a non-executable
op from every stored pipeline (3,300 of 11,225 `NodeEdge` rows across 52
instances carry one), and it spares the step-4 graph builder from
interpreting `flatten` ops as declarations. The runtime already excludes them
from execution (`6d798054`).

- Populate authored `InputPortDef.required_dimensions` from bare
  `to_dimensions` entries in both the exporter and the parser (the
  commented-out TODO in `_export_input_ports()`).
- Stop emitting `flatten` in `Edge.to_transforms()` and the parser's
  `_edge_to_transforms()`; have `modernized_transformations()` drop incoming
  `flatten` ops. That tolerance is permanent: pinned published revisions are
  immutable, and republishing is the established migration path.
- Point the output-shape assertion and `_transforms_to_config()` at the port
  declaration instead of the ops. Preserve the all-or-nothing trigger: today
  the assertion fires only when any `to_dimensions` entry exists, and then
  asserts the full set.
- Remove `FlattenType` from the GraphQL edge-transformation union after
  checking editor usage; schema export + diff.
- Resync all instances; `parse_oracle --refresh` (row contents change); full
  `test_instance --compare`.

**Gate:** no pipeline emitted by sync contains `flatten`; old revisions still
load; the output-shape assertion fires exactly where it did before.

### 3. Establish UUID contracts and stable port identity

- Change graph-level `NodePortRef.node_id` to UUID.
- Persist the structural UUID catalog in snapshot v8. Published pins had a
  dataset UUID but did not retain metric, dimension, or category UUIDs, so
  resolving every graph against the current catalog would let published
  semantics drift after a rename. The edge and dataset-port arrays remain as
  the explicit legacy shape until the unified binding migration in step 9;
  the step-4 builder normalizes them into graph bindings.
- Make port UUID preservation explicit in parser/sync: authored UUID, then
  stored UUID matched by structural role, then deterministic UUID only for
  first creation.
- Add stable semantic roles for the generic node ports needed by the first
  rules.
- Add canonical GraphQL UUID fields/inputs and deprecated identifier shims.

**Gate:** renaming a node, dataset, metric, dimension, category, or port label
does not change a graph reference; legacy GraphQL tests and new UUID-only tests
both pass.

### 4. Build and cache InstanceGraph

- Add `InstanceGraphBoundModel`, `NodeMeta`, UUID binding definitions, and
  `InstanceGraph`.
- Build directly from `InstanceSnapshot`; initially adapt its two legacy
  binding arrays.
- Add child binding, indexes, graph validation, NetworkX projection, and
  topological ordering as derived cached properties.
- Add versioned dump/load and L1/L2 caching.
- Expose request-owned lazy snapshot/graph accessors to GraphQL. Do not build
  `InstanceGraph` for fields that can be resolved from `InstanceConfig` or the
  selected published snapshot merely to prove the boundary.

Implementation note (2026-08-10): the request's `PathsObjectCache` is the L1
and Django's configured cache is the shared L2. `InstanceRequestResources`
selects and request-caches snapshots; a graph L2 miss consumes that same
snapshot lazily. The root `instance` field uses live `InstanceConfig` metadata
for draft/YAML and the selected snapshot for published revisions, while
`instance.model` remains the explicit runtime boundary. Published v8 snapshots
are self-contained, while old published snapshots and YAML use the persisted
catalog only inside the compatibility adapter. The first graph-specific
GraphQL field should use `InstanceType.graph()` when it is introduced rather
than making metadata queries hydrate the graph eagerly.

**Gate:** round-trip equality holds, derived values are absent from serialized
bytes, cache invalidation follows the selected draft/revision/YAML source, and
construction performs no dataset payload/datapoint query.

### 5. Add dataset shape profiles

- Implement the grouped observed-category query and immutable profile model.
- Key draft profiles by dataset materialization generation/content hash and
  published profiles by pinned revision.
- Bulk-load only bound dataset/metric pairs involved in validation.
- Represent external placeholder coverage as unknown without accessing DVC.

Implementation note (2026-08-10): exact UUID-based observed facts are computed
from relational datapoints when a `DatasetMaterialization` is refreshed and
stored beside, but separately from, its dataframe payload. Publication copies
those facts to `InstanceRevisionDatasetPin`, so profile reads never load the
payload and published validation cannot observe draft rows. Missing or stale
current materializations are repaired through one shared freshness service used
by profiles and runtime loading; this replaces the runtime's transitional
live-row fallback. Legacy publication pins without recorded facts remain
unknown rather than being reconstructed from pivoted JSON, where an absent cell
and a real null-valued datapoint are indistinguishable.

**Gate:** profile query count is constant with dataset count, published
profiles do not observe live draft data, and unknown differs from known empty.

### 6. Add rule declarations and compilation

- Implement the initial `PortShapeRule` union and validation.
- Add the `shape_rules(meta)` hook to the base node class.
- Declare the port roles through the typed class-level port declaration
  shared with the runtime `get_input(port)` accessor: one role namespace for
  rules and computation.
- Implement explicit rules for `AdditiveNode`, `MultiplicativeNode`, and one
  consumes/produces node such as GWP.
- Compile pipeline operations to the same rules.
- Add a contained diagnostic adapter for legacy multiplicative binding roles.

**Gate:** compiled rules contain UUIDs only, invalid class rules fail with the
node class and port role in the error, and unit changes do not reclassify an
explicit binding role.

Implementation note (2026-08-11): rules, compilation and the pipeline
compiler live in `nodes/constraints/` (`rules.py`, `compile.py`,
`pipeline_compile.py`); `InstanceGraph.shape_rule_compilation` is the cached
entry point. Ports gained a persisted `role` field — the durable link to
class semantics, so identifiers stay freely renameable — emitted by both
`spec_export` and the parser mirror; `InputPortDeclaration` gained
`repeatable` / `min_count` / `default_count` / i18n `label` (see the revised
MultiplicativeNode section: multi = homogeneous aggregate, repeatable =
heterogeneous instances). The legacy classification is class-owned:
`Node.infer_legacy_port_roles(meta, candidates)`, implemented per class from
binding tags and unit compatibility (`Unit.is_compatible_with`), with
`NodeMeta` computing the candidates (authored roles and
declaration-identifier matches filtered out, so the heuristic can never
override them), validating inferred roles against the class declarations,
and formatting uniform `inferred_port_role` / `unclassified_port_role`
diagnostics. Classification is *derived* state recomputed per hydrated graph
— this revises the earlier "keep the adapter at snapshot-to-graph
construction" line, whose intent (heuristic outside `shape_rules()`, unit
changes never reclassify explicit roles) is preserved; graph format bumped
to v3. Deviations from the step text: no
consumes/produces node class ships — the co2e conversions are conditional
inside `compute()` and a rule must not lie about conditionality, so
`DimensionTransformRule` is validated via a test-only class and real
coverage arrives with flatten transformation constraints in step 7; pipeline
compilation targets the canonical operation specs (authored `PipelineConfig`
is still a stub), chaining rules through deterministic intermediate UUIDs.
Missing role ports are `missing_role_port` diagnostics, not errors. Step 8
work queued from this session: expose the declaration catalog to the editor
(per-role add-port affordances from `role`/`multi`/`repeatable`/counts/label
against instantiated ports) and default-port creation in the node-create
mutation (two factors + one additive for a new MultiplicativeNode). Lazy
node-class imports during build/compile run inside the instance's i18n
context — some runtime modules construct i18n values at import time.

### 7. Implement the fixpoint solver

- Add value facts, origins, transformation constraints, merge operations, and
  conflict types.
- Solve dimensions first while retaining the multi-facet value shape; add
  category facts, unit conversion/product, and quantity algebra in small
  independently tested steps.
- Add overlays for candidate add/remove/reorder operations.
- Cache the compiled program on `InstanceGraph`; cache solve results only by
  graph identity plus profile/overlay identity.

**Gate:** tests cover additive pinning/unpinning, multiplicative union,
consumes/produces, assign/filter/flatten transforms, disjoint category filters,
unit convertibility/product, quantity mismatch, multiple conflicts, and origin
provenance.

Implementation note (2026-08-11): `nodes/constraints/values.py` (value keys,
multi-facet facts, monotone merge ops, conflicts) and `solver.py` (program
compilation + fixpoint); the v1 quantity algebra lives in the existing
quantity-kind registry (`nodes/quantities.py` + `configs/quantities/
quantity_kinds.yaml`), which gained `is_scalar_identity` flags and per-factor
`numerator` references (validated at load) rather than a parallel
classification module;
`InstanceGraph.constraint_program` is the cached program and
`solve_constraints(profiles=…, overlay=…)` the entry point, memoizing results
in-process only — solver logic is code, so no L2 for solve results. All four
facets shipped together rather than dimensions-first (the facets share the
propagation skeleton; sequencing them would have meant re-opening every
constraint). Decisions taken while validating against all 66 buildable DB
instances (all solve <100 ms, all converge):

- `ProductShapeRule` gained `inverse_inputs` (divide compiles the divisor
  there); dimension semantics stay the union, unit semantics become
  product-over-quotient. A separate quotient rule was rejected as a fourth
  union member with nothing to say.
- Authored `required_dimensions` seeds a *lower bound* on the port aggregate;
  the exact per-edge assertion mirrors the runtime
  (`_get_output_for_target()`): bare declared entries **plus that binding's
  own assigned dimensions**, asserted on the delivered value. Reading
  `required_dimensions` alone as exact reproduced none of espoo's real
  behavior and produced 7 false `assigned_dimension_missing` conflicts.
- Legacy bare declarations survive only as `FlattenTransformation` rows in
  never-re-synced DB rows (framework-instantiated instances) and pinned
  revisions; `build_instance_graph()` recovers them into
  `EdgeBindingDef.declared_dimensions` *before* modernization drops them
  (graph format v4). Without this, CADS-family instances mis-solve.
- Rules are only trusted where the declaring class's computation is intact:
  a subclass overriding `compute()`/`_compute()`/`perform_operation()`/
  `operate_pairwise()` below the `shape_rules` owner compiles to no rules
  plus an `inherited_shape_rules_skipped` diagnostic, and its multi-port
  aggregates are not shape-equalized either (legacy specs of such classes
  group heterogeneous inputs onto one port). Re-declaring `shape_rules` in
  the subclass is the explicit opt-in. This is the enforcement half of "a
  rule must not lie" — zuerich's mix-weighted `AdditiveNode` subclasses were
  the forcing case.
- `FilterColumnOp` is shape-neutral *except* when its column is one of the
  bound dataset's declared dimensions (then it filters and, with `drop_col`,
  removes that dimension — surrey's pattern); a name-matching raw column
  outside the declared schema stays raw (CADS's pattern).
- Binding tags: a whitelist of provably shape/unit-preserving registered tag
  operations passes through; any other registered operation
  (`geometric_inverse`, `complement`, ratio-family…) makes the binding
  opaque — facts stop, rather than lie. Unregistered tags select behavior
  and stay neutral.
- External placeholders with no declared dimensions are *unknown*, not
  scalar; dataset metric quantities are unknown in v1 (no quantity field on
  `DatasetMetricMeta`).
- Facts keep their **born** origin as they propagate, so a conflict names
  the two authored sources that disagree, not the propagation step that
  collided. Category facts are first-writer-wins per (value, dimension) with
  same-writer recompute allowed — that plus per-rule `rule_index` origins is
  what makes the fixpoint monotone with two rules writing one output port.
- Fleet residue after the fixes: 103 `quantity_mismatch` + 84
  `unit_incompatible` + 1 `dimension_mismatch` across 66 instances — spot
  checks say genuine vocabulary/model debt (health module units,
  `fraction`-vs-`mix`, a FormulaNode port authored dimensionless receiving
  kg/a), i.e. findings, not solver noise. Step 8 decides how they surface.

### 8. Put validation and derived GraphQL fields on the solver

- Replace ad-hoc edge and dataset binding checks with overlay validation.
- Expose effective shapes and conflicts from metadata-only resolvers.
- Validate the complete graph before publication and before constructing a
  strict computation context.
- Remove the editor's use of `supported_dimensions`.

**Gate:** connecting and replacing either binding kind uses the same validator;
rejected replacements leave the old binding intact; draft conflicts are
inspectable; invalid graphs cannot publish.

Implementation note (2026-08-11): the application service is
`nodes/constraints/validation.py`. A candidate edit is a `BindingChange`
(`add_bindings` / `remove_binding_ids` / `add_input_ports` / `add_datasets`)
validated by **baseline diff**: solve the current graph, solve with the change
applied, and only conflicts absent from the baseline reject — pre-existing
model debt never blocks an unrelated edit, which also settles how the step-7
fleet residue surfaces (decision 2026-08-11: publication is strict on *all*
conflicts, viable because no instance has published yet; drafts stay
inspectable via the new read fields). Additions the overlay deliberately
cannot express — a planned input port, a dataset absent from the bound-only
catalog — go through `graph_with_additions()`, an independent hypothetical
graph built by the same serialized round-trip the L2 cache uses, so cached
`NodeMeta` values are never cloned or rebound. Other decisions taken:

- Whole-graph solves are request-memoized on `InstanceRequestResources`
  keyed by `ResolvedInstanceSource` (`require_constraint_solve()`), so
  per-port `effectiveShape` resolvers share one solve and one profile load.
- Rejections are data, not errors: mutations return a `ConstraintViolations`
  union member carrying typed conflicts (strawberry-django flattens a
  declared union before appending `OperationInfo`, so
  `CreateEdgePayload = NodeEdgeType | ConstraintViolations | OperationInfo`).
  `createEdge`, `bindDataset`, `updateDatasetBinding`, `updateEdgeBinding`
  and `publishModelInstance` all use it; binding updates validate as
  remove-plus-re-add under the same binding UUID. Occupancy of a non-multi
  port stays a hard `GraphQLValidationError`: it is structural capacity, not
  shape. `_validate_edge_ports()` and `_check_metric_fits_port()` are gone.
- Read surface: `InstanceEditor.constraintConflicts`,
  `NodeSpecType.constraintConflicts` (filtered by node involvement through
  conflict values and origins), `effectiveShape` + `role` on both port
  types, `NodeSpecType.inputPortDeclarations` (the role catalog with
  instantiated port ids) and `supportsAuthoredPorts` (ClassVar; true on
  `FormulaNode` — deliberately not on `PipelineCompatibleNode`, which is a
  legacy lowering mixin, so the flag waits for the real authored
  `PipelineNode`).
- Connect-time planning replaced the mirror-port auto-create:
  `_plan_target_port()` *plans* without writing (the port is persisted only
  after validation passes, inside the change operation), instantiating a
  declared role — repeatable roles always, missing non-repeatable roles
  unless their `default_count` is 0 (`impute` is authoring-only) — and
  falling back to the source-mirroring port only for declaration-less
  classes. The single-input-port convenience now applies only when that
  port has capacity, so a one-factor `MultiplicativeNode` grows a second
  factor instead of rejecting. `createNode` without `inputPorts`
  instantiates every declaration at its default count (explicit `[]` opts
  out); `InputPortInput` gained `role` and its `supportedDimensions` is
  deprecated and ignored.
- Publication: `InstanceConfig.validate_draft_constraints()` runs inside
  `publish_instance()` after the materialization refresh (profiles read the
  same observed facts the revision pins) and raises
  `InstanceConstraintError`; shape-rule *diagnostics* (missing/inferred
  roles) do not block. The strict-computation-context hook is deferred to
  step 10 with the loader work.
- Solver strictness findings the old checks missed, confirmed as findings
  rather than noise: filters referencing dimensions the instance lacks,
  filters on dimensions the bound dataset cannot carry, and
  `disjoint_category_filter` on datasets whose observed coverage is known
  empty (note: a bind-then-fill workflow on a fresh, empty dataset will hit
  that last one; revisit its severity if it bites editors in practice).

### 9. Converge persisted bindings

- Add the unified binding model and aggregate write service.
- Backfill and dual-read through the existing binding-definition projection.
- Upgrade the snapshot to one binding list.
- Move GraphQL writes, sync, copy, revision, and change-history paths.
- Remove `NodeEdge` / `DatasetPort` only after the native snapshot loader reads
  unified bindings directly.

**Gate:** one multi-port can interleave edge and dataset inputs in stable
position order; binding UUID survives reorder; old revisions upgrade; rejected
aggregate writes are atomic.

Implementation note (2026-08-12, first half — model, backfill, dual-read):
`NodeInputPortBinding` exists as a **derived mirror**; `NodeEdge` /
`DatasetPort` remain authoritative until the write paths move. Decisions:

- `ordered_binding_snapshots()` (`instance_serialization.py`) is the single
  ordering authority: `build_instance_graph()`, the mirror service
  (`nodes/input_bindings.py:sync_input_bindings`) and the backfill migration
  all assign positions through it, so graph and ORM projections cannot
  disagree. Per port, edges come first in snapshot-queryset order (the old
  implicit `NodeEdge.Meta.ordering`, now spelled explicitly in
  `edge_qs_for()` **plus a pk tiebreak** — a determinism hardening for
  parallel edges between the same ports), then dataset ports in the
  established `(node, dataset_index, port, metric)` sort.
- Binding identity is the legacy row UUID, preserved across rebuilds; a
  reorder or unrelated delete renumbers positions densely but never changes
  a surviving binding's UUID or row pk (`sync_input_bindings` diffs against
  desired state; the position uniqueness constraint is deferred to commit).
  The fanned-out column-less dataset binding stays one row per metric —
  each row is its own binding on its own port, so no arbitrary group UUID
  was minted. The *group* identity (today `(node, dataset, dataset_index)`,
  used by `binding_editor`) still needs an explicit answer when writes move
  in the second half.
- Write boundaries: the outermost `change_operation` exit resyncs when the
  operation recorded changes to `NodeEdge` / `DatasetPort` / `NodeConfig`
  (node deletes cascade bindings without per-row records — that is why
  `NodeConfig` is in the trigger set); command-level writers call
  `sync_input_bindings` directly (`spec_sync`, runtime `spec_export` sync,
  `import_instance`, `import_instance_edges_and_ports`, `setup_cads`).
  Tests that create legacy rows directly must call it too before reading
  the projection.
- Dual-read: `NodeConfigQuerySet.annotate_ports()` now serves the
  `PortBindingDef` projection from the mirror — ORM-projected defs carry
  real `position` values for the first time. The instance-level GraphQL
  resolvers (`NodeEdgeType` / `DatasetPortType` lists) still read the
  legacy tables; they move with the write paths.
- Backfill migration (0061) uses historical models for all ORM access and
  imports only the pure ordering/snapshot helpers. Verified on the dev DB:
  0 graph-vs-mirror position mismatches on all buildable instances, and a
  post-migration `sync_input_bindings` pass over all 402 local instances
  changed 0 rows (migration ≡ service).

Remaining for step 9: snapshot upgrade to one discriminated binding list,
moving GraphQL/sync/copy/revision/change-history writes onto the unified
table (authority flip), and the group-identity decision above.

Implementation note (2026-08-12, canonical edge order fix): DB-sourced
GraphQL outputs diverged from the same instance's YAML baseline (permuted
sector `metricDim` categories/values, `upstreamNodes` order). Root cause:
`build_instance_snapshot` read edges in `NodeEdge.Meta.ordering`
(source-node pk) while the authored order is creation order — the parser
mirrors the YAML runtime's edge-creation sequence and sync bulk-creates in
that sequence, so pk order *is* the authored order. `edge_qs_for()` (and the
0061 backfill) now order by pk; graph format bumped to v5; mirror re-synced
fleet-wide (29 instances, 501 rows — binding UUIDs unchanged by design).
This had to land **before** the step-9 authority flip: positions captured
under Meta ordering would have baked the corrupted order in as authored
truth. Query-store baselines recorded from DB-sourced serving under the old
order may need re-recording; baselines recorded from YAML serving now match.

Implementation note (2026-08-13, binding UUIDs survive re-sync): both sync
write halves (`spec_sync._write_edges` / `_write_dataset_ports` and
`spec_export._update_edges` / `_update_dataset_ports`) previously
delete-all + recreated their rows, minting fresh UUIDs on every YAML
re-sync — unacceptable once binding UUIDs become authoritative identity.
Rows are still rebuilt (pk order must remain the authored order), but their
UUIDs are now carried over by ordered structural matching:
`match_preserved_uuids()` in `instance_serialization.py` (beside
`ordered_binding_snapshots()` — identity authority next to ordering
authority) pairs replacement rows to pre-rewrite rows through successive
key passes, most specific first, zipping duplicates in authored order.
Edges match on `(from_node, from_port, to_node, to_port)` then loosely
`(from_node, to_node)` (survives port changes); dataset ports on
`(node, dataset, dataset_index, metric)` then `(node, dataset, metric)`
(survives dataset reordering). A rebind to a *different dataset* is
deliberately a new identity — verified on muenchen-bisko, where the only
non-preserved rows were genuine YAML drift to different datasets. Espoo
double-sync: 186/186 edge and 86/86 dataset-port UUIDs stable, mirror
resync a no-op. Group identity decision (2026-08-13): fanned-out dataset
binding groups get **no durable identity** — each per-metric row is its own
binding; the aggregate write service may carry a non-durable grouping key
for the editor surface.

### 10. Make InstanceGraph the Context factory input

- Add `InstanceGraph.create_context(options, payload_store)` as a thin delegate
  to the native snapshot/runtime builder.
- Change `InstanceLoader.from_snapshot()` to build or receive an
  `InstanceGraph` and construct runtime nodes/ports/bindings from it.
- Move structural helpers out of `Context`; keep scenario state, caches,
  tracing, payload stores, dataframe operations, and runtime node instances in
  `Context`.
- Route YAML through snapshot then graph as the loader-inversion plan lands.

**Gate:** draft, published, and YAML calculation share graph construction;
metadata-only queries never create `Context`; calculation parity and existing
dataset revision isolation tests pass.

### 11. Remove compatibility state

- Stop generating `supported_dimensions`, migrate away any independently
  authored occurrences after auditing them, then remove the field from
  Pydantic and GraphQL.
- Remove identifier-based GraphQL inputs after measured client migration.
- Remove snapshot identifier upgraders only when the supported revision window
  allows it; keep offline export upgrade tooling longer if needed.
- Remove legacy multiplicative role inference and the split-binding projection.
- Retire `DatasetPortSpec.output_dimensions` once schema + ops derive it
  (the non-executing `flatten` placeholder is already gone — step 2).

## Verification

Each phase gets focused unit/query-count tests. Before enabling strict
publication validation or deleting compatibility paths, run:

- snapshot upgrade and revision tests;
- parser/sync round trips for representative YAML and DB instances;
- GraphQL stored queries plus canonical/deprecated binding mutations;
- dataset draft/published isolation tests;
- `test_instance --compare` on representative additive, multiplicative,
  dimension-transforming, external-placeholder, and framework models;
- full pytest, mypy, Ruff, and `git diff --check`.

Record performance separately for cold graph build, L2 decode, L1 lookup,
constraint compilation, solve without profiles, and solve with profiles. A
cache win must not hide an unbounded cold-path query or hydration cost.

## Deliberate non-goals

- `InstanceGraph` does not load dataframes or execute nodes.
- The solver does not infer meaning from arbitrary binding tags in the final
  model.
- Effective constraints are not written back into node specs.
- NetworkX is not part of the serialized cache contract.
- This work does not make every instance metadata field part of the graph; a
  field is included only when structural consumers need it.
