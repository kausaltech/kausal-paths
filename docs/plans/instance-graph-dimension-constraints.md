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
    factors = meta.require_input_port('factors')
    output = meta.require_output_port('output')
    return (ProductShapeRule(inputs=(factors.id,), output=output.id),)
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
convertible and quantities equal. Imputation is a separate explicit port/rule
when supported, not a tag exception hidden inside equality.

### MultiplicativeNode

Migrate the node to explicit roles:

- `factors`: multi input whose aggregate follows product algebra;
- `additive`: optional multi input equal to the computed result shape;
- `impute`: optional multi input equal to the final output shape;
- `output`: output port.

Its initial rules are conceptually:

```python
return (
    ProductShapeRule(
        inputs=(meta.require_input_port('factors').id,),
        output=meta.require_output_port('output').id,
    ),
    SameShapeRule(
        inputs=tuple(meta.optional_input_port_ids('additive', 'impute')),
        output=meta.require_output_port('output').id,
    ),
)
```

The solver defines a multi-port product as the product of each delivered
binding: output dimensions are their union and output unit is their unit
product. Additive and imputation values match the output shape.

During migration only, graph construction may classify legacy bindings from
`impute` / `non_additive` tags and current output-unit compatibility. Keep that
adapter at snapshot-to-graph construction and emit diagnostics. Do not put the
heuristic in `shape_rules()`: changing a unit must not silently change a
binding's computational role. Remove the adapter once persisted ports have
explicit roles.

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

### 8. Put validation and derived GraphQL fields on the solver

- Replace ad-hoc edge and dataset binding checks with overlay validation.
- Expose effective shapes and conflicts from metadata-only resolvers.
- Validate the complete graph before publication and before constructing a
  strict computation context.
- Remove the editor's use of `supported_dimensions`.

**Gate:** connecting and replacing either binding kind uses the same validator;
rejected replacements leave the old binding intact; draft conflicts are
inspectable; invalid graphs cannot publish.

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
