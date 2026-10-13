# Dimension Constraints and Transformations

## Overview

Data in Paths flows downstream through edges. Dimensional shape
requirements flow upstream: an outcome node declares which dimensions
it expects, and those requirements propagate back through the graph,
modified by each node's signature and each edge's transformations.
Some shapes are also derived forward from connected inputs, so the full
picture is a bidirectional fixpoint (see
[Edit-time inference is bidirectional](#edit-time-inference-is-bidirectional)),
and the same constraint walk covers units and quantities, not only
dimensions.

This document captures the vocabulary and the design direction.


## The vocabulary

### Node dimension signature

Every node (or pipeline step) has a shape signature relating its ports. For a
one-input/one-output dimension transformation, the relation has four facets:

| Facet | Meaning | Example |
|---|---|---|
| **requires** | Dimensions the node must receive | GWP node requires `ghg_species` |
| **consumes** | Subset of `requires` that the node removes from its output | GWP consumes `ghg_species` (flattens into CO2e) |
| **produces** | New dimensions the node adds to its output | A disaggregation node might produce `building_class` |
| **transparent** | Whether extra dimensions pass through untouched | Most nodes are transparent; outcome nodes are not |

`consumes` is always a subset of `requires`. `produces` is disjoint
from `requires`.

Additive and multiplicative nodes also need a relation across several ports;
their signatures are equality and union/product rules respectively. The
concrete rule model is described under [Node shape rules](#node-shape-rules).

For datasets, only `produces` applies — a dataset declares the shape
of what it emits.

For outcome nodes, the requested result shape is set explicitly by the user
(e.g. “I want `sector` in the final output”). That is an output-port
declaration, not the node signature's `requires` facet; the signature relates
the declared output back to the node's inputs.


### Dimension transformations on edges

Edge transformations are adapters between an upstream `produces` and a
downstream `requires`. They reshape data as it flows across an edge.

The core operations:

| Operation | What it does |
|---|---|
| **FilterDimension** | Keep or exclude specific categories within a dimension. Optionally flatten (sum over) the dimension afterward. |
| **AssignDimension** | Tag every row with a fixed category in a new dimension. Adds a dimension that didn't exist upstream. |

These two operations cover every edge transformation pattern that
actually reshapes data, and they are the same operations used in dataset
input pipelines.

Dataset pipelines add data-prep operations on top (renaming, null
handling, temporal limits, qualifier setting) that mostly don't apply to
edges, because node outputs are already well-formed. "Mostly" is
deliberate: null handling and temporal limits are meaningful on both
sides. See [Unification with dataset transforms](#unification-with-dataset-transforms)
for how that is modelled.

Metric selection is **not** a transformation. A binding names the single
metric it carries (a `DatasetMetric` for dataset bindings, an output port
for edges); the op pipeline only reshapes what that selection produced.


### `FlattenTransformation` was not a flatten

The retired `FlattenTransformation` op was a misnomer and was deliberately not
folded into `FilterDimension(flatten=True)`.

It was only ever produced from a bare `to_dimensions` entry (one with no
`categories`), and the runtime skipped such entries during execution. Its only
effect was that its dimension id joined the set asserted against the edge
output. In other words it was a **shape declaration about the consuming
port**, not an operation. Real flattening on an edge is
`FilterDimension(categories=[], flatten=True)`, produced from a bare
`from_dimensions` entry.

The declaration now lives on the port: `InputPortDef.required_dimensions`
(instance-graph plan step 2, 2026-08). The parser and the exporter populate it
from bare `to_dimensions` entries, no sync emits `flatten` any more, and the
GraphQL `flatten` input and output types are gone (2026-09-15). The op class
survives only as tolerated *stored* input: `modernized_transformations()`
drops it on read, and `build_instance_graph()` first recovers its dimension
into `EdgeBindingDef.declared_dimensions`, because pinned published revisions
are immutable and database rows that were never re-synced still carry it.


### How constraint propagation works

1. Start at outcome nodes. Their output ports declare a concrete
   dimension set.

2. Walk upstream. At each node, the node's signature determines what
   each input port requires:
   - Additive: every input must match the output dims (after edge
     transforms)
   - Multiplicative: output dims = union of input dims, so each input
     covers its own subset
   - GWP-style: output dims = input dims minus `consumes`

3. Edge transformations modify the requirement as it crosses:
   - A `FilterDimension(flatten=True)` means the upstream node must
     *have* that dimension, even though the downstream port doesn't
     require it after flattening.
   - An `AssignDimension` means the upstream node does *not* need to
     have that dimension — it's added in transit.

   Constraints are not only dimension-level. A `FilterDimension` with
   `categories` requires those categories to exist upstream and narrows
   what flows downstream to that subset; propagation can carry category
   sets where they are known (chained filters that select disjoint sets
   are a detectable configuration error). Category-level propagation is
   a refinement of the dimension-level walk, not a separate mechanism.

4. The propagated requirement at an input port is a function of:
   downstream shape + node signature + edge transforms.


### Edit-time inference is bidirectional

The walk above is downstream-to-upstream from outcome nodes. That alone
is not enough for the editor, for two reasons.

**Some output shapes are derived forward.** A multiplicative node's
output dimensions are the union of its connected inputs (and its output
unit is their product — see [Units and quantities](#units-and-quantities-constrain-the-same-way)).
Such an output port has a *computed* shape that changes as connections
are made, and the change re-propagates downstream. The signature facets
express this: `produces` can be a function of the connected inputs, not
only a static declaration.

**Constraints emerge on partial graphs.** The editor needs answers while
the graph is being built, before any outcome-node requirement reaches
the node under construction:

- Connecting the first input to an additive node pins a shape; every
  subsequent connection must match it (after its edge transforms), and
  the editor should say so at connect time, not at first computation.
- Each further connection to a multiplicative node extends the output
  union, which may invalidate or newly satisfy constraints downstream.

Propagation is therefore a fixpoint over both directions rather than a
single upstream pass: forward derivation of computed output shapes,
backward propagation of requirements, iterated until stable. On the
graphs Paths works with this converges trivially: information only tightens
during one evaluation (required lower bounds grow and allowed upper bounds
shrink). The implementation is a fixpoint all the same
(`nodes/constraints/solver.py`), not one walk with special cases.

Two consequences for the editor:

- **Port compatibility is answerable at connect time**: "which output
  ports can legally bind here" is a query against the current fixpoint,
  and a new connection that contradicts a pinned shape fails validation
  on the binding being created — with the conflicting constraint's
  origin (which connection pinned it) in the error.
- **Computed shapes must be recomputed, never stored** into authored
  fields (see [Authored vs computed declarations](#authored-vs-computed-declarations));
  an edit that removes the first input of an additive node legitimately
  *unpins* its shape.


### Units and quantities constrain the same way

A port's shape is really **dimensions × unit × quantity**. Ports already
carry the latter two (`OutputPortDef.unit` is required;
`InputPortDef.unit` and `quantity` are optional), and the same
propagation walk applies to all three facets:

- **Additive**: inputs and output must be unit-*compatible* (convertible,
  not identical — pint conversion at the boundary is fine) and share a
  quantity kind.
- **Multiplicative**: the output unit is the product of the input units;
  the output quantity follows from the input quantities where the
  quantity algebra knows the combination (energy × emission factor →
  emissions).
- **Flatten-sum** preserves unit and quantity; `AssignDimension` and
  category filters touch neither.
- **`ensure_unit`** is the unit analog of an edge dimension adapter: an
  explicit conversion declared on the binding.

Quantity kinds are coarser than units and catch errors units cannot:
two `dimensionless` ports may still be incompatible because one is a
share of buildings and the other a ratio of prices. Where quantities are
declared, they constrain; where not, only units do.

The quantity algebra is data in the quantity-kind registry
(`nodes/quantities.py`, `configs/quantities/quantity_kinds.yaml`): a kind may
be a scalar identity for products (`is_scalar_identity`) and may name the kinds
whose product it is (per-factor `numerator` references, validated at load).
In v1 the algebra only *validates*; it never invents a quantity a port did not
declare.

Validation reports each facet separately — "dimensions match but units
are incompatible" and "units match but quantities differ" are distinct,
actionable errors. The fixpoint carries the triple; there is no separate
unit-propagation machinery.


### Structural dimensions only

Signature facets (`requires` / `consumes` / `produces` / `transparent`)
and collapse policy apply to **structural** dimensions only.

- **Temporal** axes are always present and are not part of
  requires/produces bookkeeping. A node does not "require `Year`".
- **Ensemble** and **decomposition** axes (Monte Carlo iteration,
  `action_id`) are transparent by construction. They are never consumed
  implicitly, and collapsing them requires an explicit reducer, not
  summation.

See [`metric-dataframe.md`](metric-dataframe.md) for the dimension kinds
this rule refers to. A consequence: `FilterDimension(flatten=True)` must
refuse non-structural dimensions rather than silently summing them.


### Where the declarations live

| What | Where | Static or computed? |
|---|---|---|
| Node class shape rules | `Node.shape_rules(meta)` on the class, or the compiled pipeline | Static (per class/pipeline) |
| Outcome node required dims | `OutputPortDef.dimensions` | Static (user-configured) |
| Consuming-port shape declaration (ex-`FlattenTransformation`) | `InputPortDef.required_dimensions` | Static (user-configured) |
| Port role | `InputPortDef.role` / `OutputPortDef.role`, matching a class-level declaration | Static (set at port creation) |
| Binding transformations | `NodeInputPortBinding.transformations`, presented as `PortBindingDef.transformations` on the **consuming** port | Static (user-configured) |
| Dataset produced dims and observed categories | Dataset schema; `DatasetShapeProfile` facts recorded at materialization | Static / observed |
| Effective shapes, conflicts, provenance | `InstanceGraph.solve_constraints()` | Computed per graph, never stored |


### Authored vs computed declarations

Stored dimension fields on ports are **authored** data. Propagation
results are **computed** and must not be written back over them.

One exception remains in the code: the multi-port grouping
(`_apply_input_port_multi_hints()` in the exporter and its parser mirror)
still fills `required_dimensions` on the group's port from observed runtime
dimensions, which stores computed data in an authored field — the
drifting-registry failure that [`metric-dataframe.md`](metric-dataframe.md)
warns against. The other two offenders are gone: `supported_dimensions` was
removed on 2026-08-31 (every stored value equalled the generated
`required_dimensions`, none was authored), and `DatasetPortSpec.output_dimensions`
the same day (inert everywhere; schema plus pipeline derive it).

Rules:

- `InputPortDef.required_dimensions` is authored, and meaningful for explicit
  consuming-port shape declarations. The solver reads it as a *lower bound* on
  the port aggregate; the exact per-binding assertion mirrors the runtime:
  bare declared entries plus that binding's own assigned dimensions. An
  outcome's requested result shape is instead authored on its output port; the
  node's shape rule carries it back to the inputs. Everywhere else the input
  requirement is computed.
- Propagation results are exposed as separate derived GraphQL fields —
  `effectiveShape` on both port types, `constraintConflicts` on the instance
  editor and on nodes — never by overwriting the authored one.
- No mutation accepts computed dimension sets as input.


## Data model

*Implemented 2026-08 (instance-graph plan steps 3–9); this section records the
shape and the reasons for it.*

There are four different kinds of state here. Keeping them separate is more
important than the exact class names:

| Layer | Representation | Persisted? |
|---|---|---|
| Authored port declarations | Pydantic values inside `NodeSpec` | Yes, in `NodeConfig.spec` |
| Node shape algebra | Node-class or pipeline rules | No duplicate copy in the database |
| Input bindings and transformations | One `NodeInputPortBinding` row per binding | Yes |
| Effective shapes, provenance and conflicts | Constraint-engine values | No; derived for the current graph |

In particular, there is no `DimensionConstraint` Django model. A constraint is
invalidated by ordinary graph edits and is cheap to recompute. Persisting it
would create a cache-coherency problem and, worse, a second authored-looking
source of truth.


### Authored port declarations

The authored shape is the existing pair of fields, not a wrapper type (a
`DimensionSetSpec` was considered and found unnecessary once
`supported_dimensions` was gone):

```python
class InputPortDef(I18nBaseModel):
    id: UUID
    identifier: NodePortIdentifier | None = None
    role: MixedCaseIdentifier | None = None
    required_dimensions: UniqueList[DimensionRef] = []   # authored lower bound
    unit: Unit | None = None
    quantity: QuantityKindRef | None = None
    multi: bool = False


class OutputPortDef(I18nBaseModel):
    id: UUID
    identifier: MixedCaseIdentifier | None = None
    role: MixedCaseIdentifier | None = None
    dimensions: UniqueList[DimensionRef] = []             # seeds the output when non-empty
    unit: Unit
    quantity: QuantityKindRef | None = None
```

`OutputPortDef.dimensions == []` is ambiguous on its own: it can mean "scalar
output" or "not known yet". The solver keeps the distinction in its derived
facts instead of in a new authored field: an empty authored list seeds nothing,
and the port's exact set is then whatever the node rule derives from its
inputs; only a rule (or an outcome declaration) pins an exact empty set.
Internally facts carry `known: frozenset[UUID] | None`, where `None` is unknown
and an empty set is an exact scalar, with the consumer requirement as a
separate lower-bound set. Neither fact is written back into the authored port.
External placeholder datasets with no declared dimensions are *unknown*, not
scalar.

Port UUIDs are durable instance-local identity and are what bindings refer to.
Sync preserves them: the parser keeps an authored UUID, otherwise matches the
stored port by structural role, and mints a deterministic UUID only on first
creation. A port's link to its class semantics is the persisted `role` field,
matching a class-level `InputPortDeclaration` / `OutputPortDeclaration`. Port
identifiers remain human/formula names and are freely renameable without
detaching the port from its role. The declarations distinguish two
multiplicities that must not be conflated: a **multi** port (one port, many
bindings) is a *homogeneous* aggregate — every binding shape-equal, summed —
while a **repeatable** role (many port instances, e.g. each factor of a
product) is *heterogeneous*, each instance carrying its own unit, quantity
and dimension expectations. Products therefore only ever happen across
distinct ports; bindings on one port are always shape-equal. The declaration
catalog is what the editor's connect-time planning instantiates: connecting to
a node with a free declared role gets a port of that role (repeatable roles
always, missing non-repeatable roles unless their `default_count` is 0), and
`createNode` without explicit ports instantiates every declaration at its
default count.

An anonymous legacy port can still be executed, but it cannot participate in
a class-level rule until it has a role. During migration the node class
itself classifies such ports — `Node.infer_legacy_port_roles(meta,
candidates)`, implemented per class from binding tags and unit compatibility,
mirroring the runtime's behavior. The framework side (`NodeMeta`) computes the
candidates (authored roles and declaration-identifier matches are filtered
out, so the heuristic can never override them), validates that inferred roles
exist in the class declarations, and formats uniform `inferred_port_role` /
`unclassified_port_role` diagnostics. The classification is derived state,
recomputed per hydrated graph, never serialized. Implementing the hook for a
new node class is always wrong — new classes declare roles at port creation —
and the whole mechanism dies once persisted ports carry explicit roles.


### Node shape rules

*Implemented in `nodes/constraints/` (instance-graph plan step 6, 2026-08-11).*

The four signature facets are sufficient for a one-input/one-output
transformation, but they do not say how several ports relate. Additive and
multiplicative nodes need relations between named ports. Represent the common
algebras explicitly rather than storing a Python callback name in `NodeSpec`.
Rules are compiled per node — `Node.shape_rules(meta)` resolves role
selectors against one node's ports — so the compiled rules contain UUIDs
only:

```python
class SameShapeRule(BaseModel):
    kind: Literal['same'] = 'same'
    inputs: tuple[UUID, ...]
    output: UUID
    # Dimensions are equal; units are convertible; quantities are equal.


class ProductShapeRule(BaseModel):
    kind: Literal['product'] = 'product'
    inputs: tuple[UUID, ...]
    inverse_inputs: tuple[UUID, ...] = ()   # divisors: same union, unit in the denominator
    output: UUID
    # Output dimensions are the union, units the product over the quotient,
    # and quantity is obtained from the quantity algebra when one is registered.


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

The node class exposes a list of these rules. A pipeline compiles its
operations to the same rule list, chaining through deterministic intermediate
value UUIDs (`nodes/constraints/pipeline_compile.py`); several rules may
constrain one output port (that is how "the additive aggregate conforms to
the product result" is expressed), while an intermediate is defined by
exactly one rule. `NodeSpec` stores the selected node type or authored
pipeline, not a copied result of that compilation. This prevents class
semantics and stored signature JSON from drifting apart.

The three rules cover the initial implementation:

- additive nodes use `same`;
- multiplicative nodes use `product`;
- GWP, reducers and disaggregation use `dimension_transform`. No production
  class declares one yet — the existing co2e conversions are conditional
  helpers inside `compute()`, and a rule must not lie about conditionality —
  so the union member is validated and tested, waiting for the first class
  with a declared signature. Real-world consumes/produces coverage comes from
  flatten binding transformations in the solver.

Nodes with genuinely different algebra may implement the same constraint-rule
protocol in Python. That escape hatch should return constraints and derived
facts; it must not mutate port declarations. If custom rules become common,
add another declarative union member based on the repeated algebra rather than
persisting arbitrary callback names.

`consumes ⊆ requires` and `requires ∩ produces = ∅` are construction-time
invariants of `DimensionTransformRule`. A non-transparent rule also places an
upper bound on its output: dimensions not required or produced do not pass.

Compilation keeps two failure modes apart: a structurally invalid rule (wrong
port direction, unknown value or dimension, intermediate cycle) raises
`ShapeRuleError` naming the node class — it is a class bug — while a node
whose legacy spec lacks ports for a required role compiles to no rules and a
`missing_role_port` diagnostic. Incompleteness never blocks anything.

Rules are only trusted where the declaring class's computation is intact. A
subclass that overrides `compute()` / `_compute()` / `perform_operation()` /
`operate_pairwise()` below the class that declared `shape_rules` compiles to
no rules plus an `inherited_shape_rules_skipped` diagnostic, and its
multi-port aggregates are not shape-equalized either (legacy specs of such
classes group heterogeneous inputs onto one port). Re-declaring `shape_rules`
in the subclass is the explicit opt-in. This is the enforcement half of "a
rule must not lie".


### One input-binding table

`NodeEdge` and `DatasetPort` were two storage forms for one domain concept: a
source bound to a consuming input port. They converged on one model so that
ordering, transformations, tags and constraint provenance have one identity:

```python
class NodeInputPortBinding(EditableInstanceChild):
    instance = models.ForeignKey(
        InstanceConfig,
        on_delete=models.CASCADE,
        related_name='input_bindings',
    )
    node = models.ForeignKey(
        NodeConfig,
        on_delete=models.CASCADE,
        related_name='input_bindings',
    )
    port_id = models.UUIDField()
    position = models.PositiveIntegerField(default=0)

    # Exactly one source branch is populated.
    source_node = models.ForeignKey(
        NodeConfig,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='output_bindings',
    )
    source_port_id = models.UUIDField(null=True, blank=True)
    dataset = models.ForeignKey(
        Dataset,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name='node_input_bindings',
    )
    metric = models.ForeignKey(
        DatasetMetric,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name='node_input_bindings',
    )

    transformations = SchemaField(
        schema=list[PortTransformOp],
        default=list,
        blank=True,
    )
    tags = ArrayField(models.CharField(max_length=200), default=list, blank=True)

    class Meta:
        constraints = (
            models.CheckConstraint(
                condition=(
                    Q(
                        source_node__isnull=False,
                        source_port_id__isnull=False,
                        dataset__isnull=True,
                        metric__isnull=True,
                    )
                    | Q(
                        source_node__isnull=True,
                        source_port_id__isnull=True,
                        dataset__isnull=False,
                        metric__isnull=False,
                    )
                ),
                name='node_input_binding_has_one_source',
            ),
            models.UniqueConstraint(
                fields=('node', 'port_id', 'position'),
                name='node_input_binding_position_is_unique',
            ),
        )
```

The source kind is derived from which branch is populated rather than stored as
a redundant discriminator. Domain validation additionally enforces facts the
database cannot see through `NodeSpec` JSON:

- `node` and `source_node` belong to `instance`;
- `port_id` names an input port and `source_port_id` an output port;
- a non-`multi` port has at most one binding, at position zero;
- positions on a `multi` port are contiguous after a write;
- the dataset metric belongs to the selected dataset's schema;
- every transformation applies to the selected source kind.

`position` replaces `DatasetPort.dataset_index` and also orders edge bindings.
That matters because one multi-port may contain both kinds and because
floating-point addition makes iteration order observable. The binding UUID,
not `(node, port, position)`, is its durable identity; reordering does not make
a new binding.

The model deliberately keeps port references as UUID fields instead of foreign
keys because ports remain embedded in `NodeSpec`. Normalizing ports into ORM
rows solely to obtain an FK would split one authored node specification across
two revision mechanisms. Referential checks belong in the aggregate write
service that updates a node spec or its bindings atomically.

The nullable ORM branches do not leak into snapshots or the runtime. Those
use a discriminated source value:

```python
class NodePortSource(BaseModel):
    kind: Literal['node'] = 'node'
    node_id: UUID
    port_id: UUID


class DatasetMetricSource(BaseModel):
    kind: Literal['dataset'] = 'dataset'
    # Natural references keep portable exports restore-stable; the UUIDs make a
    # published snapshot self-contained, and the pinned revision is what it computed from.
    dataset: str
    metric: str
    dataset_uuid: UUID | None = None
    metric_uuid: UUID | None = None
    dataset_revision: int | None = None


type InputBindingSource = NodePortSource | DatasetMetricSource


class InputBindingSnapshot(ModelSnapshot):
    uuid: UUID | None = None
    node_id: UUID
    port_id: UUID
    position: int = 0
    source: InputBindingSource = Field(discriminator='kind')
    transformations: list[PortTransformOp] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
```

`EdgeBindingDef` and `DatasetBindingDef` remain as narrowed graph views, both
constructed from this one shape. The ORM resolves the dataset and metric
strings to its FKs; unlike graph-internal node and port references, these are
intentionally natural keys at the portable snapshot boundary.

The convergence landed in stages (2026-08-12 → 2026-09-01): a backfilled
mirror first, binding UUIDs made stable across re-sync by structural matching
(`match_preserved_uuids()`; a rebind to a *different* dataset is a new
identity by design), one discriminated snapshot list, the authority flip
(2026-08-18), the table drop (`NodeEdge` / `DatasetPort`, 2026-08-30), and
finally snapshot v11 with `bindings: list[InputBindingSnapshot]` and the
transitional `dataset_spec` / `dataset_index` fields retired. Two authorities
came out of it and both must stay single:

- `ordered_binding_snapshots()` assigns positions — per port, edges first in
  authored (creation) order, then dataset rows — for the parser, the sync
  writers and `build_instance_graph()` alike, so the graph and the ORM cannot
  disagree about order. Positions are stored; nothing recomputes them from
  primary keys.
- `group_unified_dataset_bindings()` recovers the runtime's dataset groups
  from native fields only: a row whose pipeline selects a metric is a
  singleton; the column-less rows of one (node, dataset) are the per-metric
  fan-out of one whole-frame binding, closed when a metric repeats. The
  loader, the sync resolution, the explanation system, the placeholders, the
  export ordering and the GraphQL fan-out all derive groups from it. Fanned-out
  groups have **no durable identity** — each per-metric row is its own binding.

The dataset-only compatibility data that used to ride on `DatasetPortSpec`
was not copied onto the common model: `column` is the metric FK plus a
`select_metric` marker in the pipeline, `interpolate` / `backfill` / `extend`
are ordered ops, `input_dataset` left the persisted path (YAML-only until the
YAML era ends), and `output_dimensions` was removed.


### Computed constraint values

*`nodes/constraints/values.py` and `solver.py` (plan step 7, 2026-08-11).*

The solver needs bounds, not only a concrete set. Every value it reasons
about gets a key — `PortValue` (the aggregate at one port), `BindingValue`
(one delivered binding), `DatasetSourceValue` (a metric before its binding
pipeline), `IntermediateValue` (a compiled pipeline stage) — and a mutable,
multi-facet `ValueFacts` record inside a `FactStore`: dimensions,
categories, unit and quantity together. Facts are monotone: unknown may
become known, requirement sets only grow, known upper bounds only shrink, and
a contradicting fact never overwrites an established one — it records a
`ConstraintConflict` carrying both origins instead. That is what makes the
fixpoint terminate and makes its result independent of evaluation order.

`allowed=None` means unknown/unbounded, not an empty set. Combining independent
requirements unions their lower bounds and intersects known upper bounds. A
conflict exists when the resulting required set is not a subset of the allowed
set. Category bounds use the same operation within a dimension, first-writer-
wins per (value, dimension) with same-writer recompute allowed. Chained filters
that select disjoint categories are the same kind of contradiction as
incompatible dimension sets (`disjoint_category_filter`, evaluated against the
dataset's *observed* categories when a shape profile is known). Separate
downstream branches may require disjoint categories without a conflict: their
requirements union at the shared producer.

Units merge differently from set bounds. Two known units satisfy an equality
rule when they are convertible; the solver retains the consuming port's
preferred unit for boundary conversion rather than requiring identical unit
strings. A product rule derives a new unit (product over quotient for
`inverse_inputs`), and `ensure_unit` replaces the representative unit after
checking convertibility. Quantities use exact kind equality except where the
registered quantity algebra derives a product.

Binding tags are handled by whitelist: the registered tag operations that
provably preserve shape and unit pass facts through; any other registered
operation (`geometric_inverse`, `complement`, the ratio family) makes the
binding opaque — facts stop rather than lie. Unregistered tags select behavior
and stay neutral.

A `ConstraintOrigin` attaches to every fact and is kept as the fact's *born*
origin while it propagates, so a conflict names the two authored sources that
disagree — a declaration, a node rule, a binding, a transformation (binding
UUID plus list index; transformations are whole-list values without identity
of their own) or a dataset schema — not the propagation step that collided.
Conflict codes include `dimension_mismatch`, `unit_incompatible`,
`quantity_mismatch`, `assigned_dimension_missing`,
`produced_dimension_missing`, `disjoint_category_filter`,
`unknown_dimension_reference`, `unknown_category_reference` and
`multiple_bindings_on_single_port`.

GraphQL projects the result into read-only derived types: `EffectiveShape`
(exact `dimensionUuids` or null when unknown, `requiredDimensionUuids`,
`forbiddenDimensionUuids`, per-dimension categories, unit, quantity) as
`effectiveShape` on both port types, and `ConstraintConflict` (code, message,
the value it is about, its origins) as `constraintConflicts` on the instance
editor and, filtered by involvement, on each node. Whole-graph solves are
request-memoized on `InstanceRequestResources` so every port resolver shares
one solve and one profile load. The compiled program is a cached property of
the `InstanceGraph`; solve results are memoized in-process only, by profile
versions and overlay content, because solver logic is code and must never
outlive the hydrated graph. Nothing here is included in `NodeSnapshot`,
accepted by a mutation, or restored as authored state.

Validation uses the solver by **baseline diff** (`nodes/constraints/validation.py`):
a candidate edit is a `BindingChange` (bindings to add, binding ids to remove,
input ports and datasets to add), the current graph is solved, the graph with
the change applied is solved, and only conflicts absent from the baseline
reject the edit. Pre-existing model debt therefore never blocks an unrelated
edit, while publication (`InstanceConfig.validate_draft_constraints()`) is
strict on every conflict. Mutations return rejections as data — a
`ConstraintViolations` union member carrying typed conflicts — because a
rejected connection is an answer, not an error; occupancy of a non-multi
port stays a hard validation error, since it is structural capacity, not
shape.


### Reference identity

Constraint provenance uses node, port and binding UUIDs, and the `InstanceGraph`
carries a UUID catalog of dimensions, categories, datasets and metrics
(snapshot v8), so that a published graph does not drift after a rename.
`InstanceGraph.describe_uuid()` turns those back into identifiers for
diagnostics only. The `DimensionRef` and `DimensionCategoryRef` identifier
vocabulary remains at the authored YAML/`NodeSpec` boundary; the solver
resolves it once against the graph's dimension catalog. This proposal does
not make identifiers into a new durable graph identity. If dimensions later
become renameable editor objects, their existing ORM UUIDs should become the
stored references through an explicit snapshot-version migration rather than
by silently changing the meaning of `DimensionRef`.


### Transformations attach to the consuming port

Propagation walks upstream *through input ports*, so it needs
transformations attached to the binding at the consuming port. The runtime
agrees with that now: the loader constructs one `RuntimeInputBinding` per
graph binding (`nodes/runtime_input.py`), whose loader reads the source
value — the bound output column of the source node, or the bound metric of
the dataset — applies the binding's own pipeline through the shared executor,
and delivers it to the consuming node, which resolves inputs by role through
`get_input(port)` / `iter_inputs(port)` / `require_input(port)` against its
class-level port declarations. The pipeline that used to be derived from
producer-side `from_dimensions` / `to_dimensions` dicts is gone with the
config-dict loader (plan step 10, 2026-08-16); a binding executes exactly the
ops it stores.

Not every node class has moved onto the role accessors yet. The remaining
classes still read inputs through the legacy `get_input_dataset*()` /
`get_output_pl(target=…)` accessors, which the loader feeds from the same
bindings, so the two paths cannot disagree about *what* is delivered — only
about how the class asks for it. Migrating a class means declaring its ports
and replacing those calls; nothing else changes.

Port identity is authored and stable (see [Authored port declarations](#authored-port-declarations)),
so a stored binding's `port_id` survives a re-sync.


## Unification with dataset transforms

The dimension-aware subset of dataset transforms and edge transforms are
the same operations. The legacy edge encodings mapped like this:

```
Edge: select_categories  ->  filter_dimension
Edge: assign_category    ->  assign_dimension
Edge: flatten            ->  (nothing — it was a port shape declaration; see above)
```

The legacy kinds survive only as tolerated *stored* input on old rows and
pinned revisions: `modernized_transformations()` rewrites them on every read,
no sync emits them, and the GraphQL inputs and output types for them were
removed on 2026-09-15.

### Decision: one op type, not two layers

There is **one** op union, `PortTransformOp`, covering dimension ops and
data-prep ops together. Applicability is a property of each op, validated
against the binding kind that carries it — not a second type hierarchy.

Rationale: two unions means two GraphQL input types and two executors,
which is the duplication this unification exists to remove. And the
dimension/data-prep line does not fall cleanly between edges and datasets
anyway — null handling and temporal limits are meaningful on both sides.

The ops:

| Op | Applies to | Notes |
|---|---|---|
| `filter_dimension` | edge, dataset | Categories or groups; optional exclude and flatten. Refuses non-structural dimensions when flattening. |
| `assign_dimension` | edge, dataset | Was `assign_category` on the dataset dimension filter. |
| `drop_nulls` | edge, dataset | |
| `filter_temporal` | edge, dataset | Currently the yearly specialization (`min_year` / `max_year`). |
| `ensure_unit` | edge, dataset | Explicit conversion declared on the binding; the unit analog of a dimension adapter. |
| `tag_operation` | edge, dataset | A registered tag behavior applied as an ordered stage. |
| `filter_column` | dataset | Legacy, pre-dimension column filtering. Shape-neutral to the solver unless its column is one of the dataset's declared dimensions, in which case it filters and, with `drop_col`, removes that dimension. |
| `rename_column` | dataset | Legacy wide-DVC column labels. |
| `rename_item` | dataset | Category value remapping. |
| `set_forecast_from` | dataset | Sets the forecast **qualifier**; see `metric-dataframe.md`. |
| `interpolate` | dataset | Fills missing interior years linearly; the interpolation qualifier follows the `MetricDataFrame` migration. |
| `backfill` | dataset | Copies the first known value backwards over existing leading null rows. |
| `extend` | dataset | Carries the last historical value to the instance model end year. |
| `select_metric`, `index_temporal`, `remap_legacy_years` | dataset | **System-managed** stage markers the compiler inserts; clients render them read-only and pass them back unchanged. |

Metric *selection* is not a transformation: a binding names the single
metric it carries (a `DatasetMetric` for dataset bindings, an output port for
edges). The `select_metric` marker only records where in the pipeline the
bound column is aliased to `Value`, so that a pipeline can be executed
literally; it does not choose the metric.


## Current state and next steps

**Done (2026-08 → 2026-09):**

- `PortTransformOp` is the one vocabulary; both binding kinds execute it as a
  pipeline at the consuming port, and every reader sees the current kinds.
- `flatten` retired onto `InputPortDef.required_dimensions` (step 2).
- Stable port UUIDs, the persisted `role`, and the snapshot UUID catalog
  (step 3).
- `InstanceGraph`: the immutable, cached metadata graph built from a snapshot
  without a `Context` (step 4); dataset shape profiles recorded at
  materialization and pinned at publication (step 5).
- Shape rules per node class with pipeline compilation to the same rules
  (step 6); the multi-facet fixpoint solver with provenance (step 7);
  baseline-diff validation, strict publication, and the derived GraphQL
  fields (step 8).
- One `NodeInputPortBinding` table, one snapshot binding list, legacy tables
  dropped, transitional fields retired (step 9 and step 11).
- The native loader: YAML, draft and published all build the runtime from the
  snapshot through the graph (step 10); the explanation system reads typed
  inputs built from the same graph (2026-09-15).
- `supported_dimensions`, `DatasetPortSpec.output_dimensions`, and the
  deprecated identifier-era GraphQL surface removed.

**Open, in no particular dependency order:**

1. The multi-group `required_dimensions` fill in the exporter and parser is
   the last place computed dimensions land in an authored field.
2. Legacy port-role inference (`infer_legacy_port_roles`) dies when every
   persisted port carries a role. Count the `inferred_port_role` diagnostics
   across the fleet after a resync before removing it.
3. Node classes still on the legacy input accessors should declare their
   ports and move to `get_input(port)`.
4. `dimension_transform` rules have no production declarer yet; the first
   class with a genuine consumes/produces signature (GWP, disaggregation)
   makes the union member real.
5. Publication is strict on all conflicts and nothing has published yet;
   revisit `disjoint_category_filter` severity if a bind-then-fill workflow
   on a fresh empty dataset hits it.
6. The quantity algebra is validation-only; deriving a product quantity is a
   later step once the registry has enough coverage.
7. Snapshot identifier upgraders go only when the supported revision window
   allows.
