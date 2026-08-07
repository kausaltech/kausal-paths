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


### `FlattenTransformation` is not a flatten

The current `FlattenTransformation` type is a misnomer and must not be
folded into `FilterDimension(flatten=True)`.

It is only ever produced from a bare `to_dimensions` entry (one with no
`categories`), and `_get_output_for_target()` skips to-dimension entries
that have no categories. Its only runtime effect is that its dimension id
joins the set asserted against the edge output. In other words it is a
**shape declaration about the consuming port**, not an operation.

Real flattening on an edge is `SelectCategoriesTransformation(categories=[],
flatten=True)`, produced from a bare `from_dimensions` entry.

The target state is therefore:

- the shape declaration moves to the port (`InputPortDef`), where
  constraint propagation can compute or validate it
- `FlattenTransformation` disappears from the op vocabulary rather than
  merging into another op


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
shrink). The implementation should still be written as a fixpoint, not as one
walk with special cases.

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
| Node class dimension rules | Node class or pipeline definition | Static (per class/pipeline) |
| Outcome node required dims | `OutputPortDef.dimensions` | Static (user-configured) |
| Port shape declarations (ex-`FlattenTransformation`) | `InputPortDef` | Static (user-configured) |
| Binding transformations | `PortBindingDef.transformations` on the **consuming** port | Static (user-configured) |
| Input port effective requirement | Computed from downstream | Computed at validation/editor time |
| Dataset produced dims | Dataset schema | Static |


### Authored vs computed declarations

Stored dimension fields on ports are **authored** data. Propagation
results are **computed** and must not be written back over them.

This is not yet fully true in the code: the multi-port grouping in
`_apply_input_port_multi_hints()` still fills `required_dimensions` and
`supported_dimensions` on the group's port from observed runtime
dimensions, which stores computed data in authored fields — the
drifting-registry failure that [`metric-dataframe.md`](metric-dataframe.md)
warns against. (The blanket per-port fill this section used to describe is
gone; only the multi-group path remains.) The same applies to
`DatasetPortSpec.output_dimensions`, a manual override of what the dataset
schema plus the op pipeline should derive.

Rules going forward:

- The current `InputPortDef.required_dimensions` (eventually
  `dimensions.required`) is authored, and meaningful for explicit
  consuming-port shape declarations. An outcome's requested result shape is
  instead authored on its output port; the node signature carries it back to
  the inputs. Everywhere else the input requirement is computed.
- Propagation results are exposed as a separate derived field (for
  example `effectiveRequiredDimensions` in GraphQL), never by overwriting
  the authored one.
- No mutation may accept computed dimension sets as input.
  `DatasetPortSpec.output_dimensions` stays read-only and is retired once
  schema + ops can derive it.


## Proposed data model

There are four different kinds of state here. Keeping them separate is more
important than the exact class names:

| Layer | Representation | Persisted? |
|---|---|---|
| Authored port declarations | Pydantic values inside `NodeSpec` | Yes, in `NodeConfig.spec` |
| Node shape algebra | Node-class or pipeline rules | No duplicate copy in the database |
| Input bindings and transformations | One ORM row per binding | Yes |
| Effective shapes, provenance and conflicts | Constraint-engine values | No; derived for the current graph |

In particular, there should be no `DimensionConstraint` Django model. A
constraint is invalidated by ordinary graph edits and is cheap to recompute.
Persisting it would create a cache-coherency problem and, worse, a second
authored-looking source of truth.


### Authored port declarations

The current fields have two ambiguous empty values:

- `OutputPortDef.dimensions=[]` can mean either “scalar output” or “not known
  yet”.
- `InputPortDef.supported_dimensions=[]` can mean either “no dimensions are
  supported” or “there is no upper bound”.

The constraint engine needs those states to be distinct. Model dimension sets
as lower and upper bounds: `required` dimensions must occur; `allowed=None`
means there is no upper bound; an exact set has the same `required` and
`allowed` members. Thus an exact scalar is `required=[]`, `allowed=[]`, while a
wholly unconstrained port is `required=[]`, `allowed=None`.

One possible Pydantic shape is:

```python
class DimensionSetSpec(BaseModel):
    required: UniqueList[DimensionRef] = Field(default_factory=list)
    allowed: UniqueList[DimensionRef] | None = None

    @classmethod
    def exact(cls, dimensions: list[DimensionRef]) -> Self:
        return cls(required=dimensions, allowed=dimensions)


class InputPortDef(I18nBaseModel):
    id: UUID
    identifier: NodePortIdentifier | None = None
    dimensions: DimensionSetSpec = Field(default_factory=DimensionSetSpec)
    unit: Unit | None = None
    quantity: QuantityKindRef | None = None
    multi: bool = False


class OutputPortDef(I18nBaseModel):
    id: UUID
    identifier: NodePortIdentifier | None = None
    # None has no authored declaration; exact([]) is an authored scalar.
    dimensions: DimensionSetSpec | None = None
    unit: Unit | None = None
    quantity: QuantityKindRef | None = None
```

`OutputPortDef.unit` becomes optional for the same reason as dimensions: a
multiplicative output can derive its unit from its inputs. It remains required
by validation for node rules that do not derive it.

This does not require an immediate JSON migration. The first implementation
can translate the current fields at the boundary:

```text
required_dimensions != []      -> dimensions.required
supported_dimensions != []     -> dimensions.allowed
OutputPortDef.dimensions != []  -> DimensionSetSpec.exact(...)
```

The empty `supported_dimensions` and output `dimensions` cases need deliberate
migration rules; their meanings cannot be recovered from the values alone. The
safe defaults are unbounded input dimensions and no authored output
declaration, with node classes that are genuinely scalar-only declaring an
exact empty set.

Port UUIDs are durable instance-local identity and are what bindings refer to.
Port identifiers are the structural names used by node-class rules and
formulas. A class-declared port therefore needs an identifier even though it
remains optional during the migration of anonymous YAML-derived ports. An
anonymous port can still be executed, but it cannot participate in a
class-level signature until it is given a structural name.


### Node shape rules

The four signature facets are sufficient for a one-input/one-output
transformation, but they do not say how several ports relate. Additive and
multiplicative nodes need relations between named ports. Represent the common
algebras explicitly rather than storing a Python callback name in `NodeSpec`:

```python
class EqualShapeRule(BaseModel):
    kind: Literal['equal'] = 'equal'
    inputs: list[NodePortIdentifier]
    output: NodePortIdentifier
    # Dimensions are equal; units are convertible; quantities are equal.


class ProductShapeRule(BaseModel):
    kind: Literal['product'] = 'product'
    inputs: list[NodePortIdentifier]
    output: NodePortIdentifier
    # Output dimensions are the union, units the product, and quantity is
    # obtained from the quantity algebra when one is registered.


class DimensionSignatureRule(BaseModel):
    kind: Literal['dimension_signature'] = 'dimension_signature'
    input: NodePortIdentifier
    output: NodePortIdentifier
    requires: UniqueList[DimensionRef] = Field(default_factory=list)
    consumes: UniqueList[DimensionRef] = Field(default_factory=list)
    produces: UniqueList[DimensionRef] = Field(default_factory=list)
    transparent: bool = True


type PortShapeRule = EqualShapeRule | ProductShapeRule | DimensionSignatureRule
```

The node class exposes a list of these rules. A pipeline compiles its steps to
the same rule list. `NodeSpec` stores the selected node type or authored
pipeline, not a copied result of that compilation. This prevents class
semantics and stored signature JSON from drifting apart.

The three rules cover the initial implementation:

- additive nodes use `equal`;
- multiplicative nodes use `product`;
- GWP, reducers and disaggregation use `dimension_signature`.

Nodes with genuinely different algebra may implement the same constraint-rule
protocol in Python. That escape hatch should return constraints and derived
facts; it must not mutate port declarations. If custom rules become common,
add another declarative union member based on the repeated algebra rather than
persisting arbitrary callback names.

`consumes ⊆ requires` and `requires ∩ produces = ∅` are construction-time
invariants of `DimensionSignatureRule`. A non-transparent rule also places an
upper bound on its output: dimensions not required or produced do not pass.


### One input-binding table

`NodeEdge` and `DatasetPort` are two storage forms for one domain concept: a
source bound to a consuming input port. They should converge on one model so
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

The nullable ORM branches should not leak into snapshots or the runtime. Those
use a discriminated source value:

```python
class NodePortSource(BaseModel):
    kind: Literal['node'] = 'node'
    node_id: UUID
    port_id: UUID


class DatasetMetricSource(BaseModel):
    kind: Literal['dataset'] = 'dataset'
    # Natural references keep portable exports restore-stable, matching the
    # current dataset snapshot boundary.
    dataset: str
    metric: str


type InputBindingSource = NodePortSource | DatasetMetricSource


class InputBindingSnapshot(ModelSnapshot):
    node_id: UUID
    port_id: UUID
    position: int
    source: InputBindingSource = Field(discriminator='kind')
    transformations: list[PortTransformOp] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
```

`EdgeBindingDef` and `DatasetBindingDef` may remain as convenient narrowed
runtime views, but both are constructed from this one persisted shape. The ORM
resolves the dataset and metric strings to its FKs; unlike graph-internal node
and port references, these are intentionally natural keys at the portable
snapshot boundary.

A staged migration is safer than replacing both tables at once:

1. Create `NodeInputPortBinding` and dual-read it behind the existing
   `PortBindingDef` projection.
2. Backfill `NodeEdge` rows, preserving their UUIDs and deterministic current
   iteration order; backfill `DatasetPort` rows using `dataset_index` as the
   initial `position`.
3. Move the snapshot to `input_bindings: list[InputBindingSnapshot]` in a new
   schema version, with an upgrader for `edges` plus `dataset_ports`.
4. Make GraphQL writes target the new table and validate the complete node
   aggregate in one transaction.
5. Remove the old tables only after the native snapshot loader consumes the
   unified binding directly.

During steps 1–4, dataset-only compatibility data still present in
`DatasetPortSpec` needs an explicit adapter. `column` is replaced by the metric
FK, `interpolate` becomes an ordered op, `input_dataset` is resolved into the
dataset source reference, and `output_dimensions` is removed after derivation
works. Do not copy those fields onto the common model: that would make the
transitional encoding permanent.


### Computed constraint values

The solver needs bounds, not only a concrete set. A small immutable value model
is enough:

```python
@dataclass(frozen=True)
class SetBounds[T]:
    required: frozenset[T] = frozenset()
    allowed: frozenset[T] | None = None


@dataclass(frozen=True)
class ValueShape:
    dimensions: SetBounds[DimensionRef]
    categories: Mapping[DimensionRef, SetBounds[DimensionCategoryRef]]
    unit: Unit | None
    quantity: QuantityKindRef | None


@dataclass(frozen=True)
class PortKey:
    node_id: UUID
    port_id: UUID
    direction: Literal['input', 'output']


@dataclass(frozen=True)
class ConstraintOrigin:
    kind: Literal['declaration', 'node_rule', 'binding', 'transformation', 'dataset_schema']
    node_id: UUID | None = None
    port_id: UUID | None = None
    binding_id: UUID | None = None
    transformation_index: int | None = None


@dataclass(frozen=True)
class ConstraintConflict:
    code: str
    facet: Literal['dimensions', 'categories', 'unit', 'quantity']
    origins: tuple[ConstraintOrigin, ...]
    message: str
```

`allowed=None` means unknown/unbounded, not an empty set. Combining independent
requirements unions their lower bounds and intersects known upper bounds. A
conflict exists when the resulting required set is not a subset of the allowed
set. Category bounds use the same operation within a dimension. This gives the
fixpoint a monotone representation and makes chained filters that select
disjoint categories the same kind of contradiction as incompatible dimension
sets. Separate downstream branches may require disjoint categories without a
conflict: their requirements union at the shared producer.

Units merge differently from set bounds. Two known units satisfy an equality
rule when they are convertible; the solver retains the consuming port's
preferred unit for boundary conversion rather than requiring identical unit
strings. A product rule derives a new unit, and `ensure_unit` replaces the
representative unit after checking convertibility. Quantities use exact kind
equality except where the registered quantity algebra derives a product.

Origins attach to individual facts inside the solver, even though the compact
example above shows them only on conflicts. A transformation is addressed by
stable binding UUID plus list index: transformations are intentionally
whole-list values and do not need their own persistent identity.

The in-memory graph is keyed by `PortKey`. GraphQL projects it into derived
types such as:

```graphql
type EffectivePortShape {
  requiredDimensions: [Dimension!]!
  allowedDimensions: [Dimension!]
  unit: String
  quantity: QuantityKind
  conflicts: [PortConstraintConflict!]!
}
```

This result may be memoized by an instance revision or a deterministic graph
hash, but such a memo is a disposable cache. It is never included in
`NodeSnapshot`, accepted by a mutation, or restored as authored state.


### Reference identity

Constraint provenance should use node, port and binding UUIDs. The current
`DimensionRef` and `DimensionCategoryRef` identifier vocabulary can remain at
the authored YAML/`NodeSpec` boundary for the first implementation; the solver
resolves it once against the instance dimension registry. This proposal does
not make identifiers into a new durable graph identity. If dimensions later
become renameable editor objects, their existing ORM UUIDs should become the
stored references through an explicit snapshot-version migration rather than
by silently changing the meaning of `DimensionRef`.


### Transformations attach to the consuming port

Propagation walks upstream *through input ports*, so it needs
transformations attached to the binding at the consuming port. The code
does not do this yet: edge transformations execute on the **producing**
node, in `_get_output_for_target()`, keyed by consumer identity. Dataset
filters execute inside dataset loading. Neither runs at the port.

Propagation also needs port identity that is authored and stable. Ports
are currently derived at export time and their ids are hashed from
`(node, direction, key)`, so a structural edit can regenerate them on sync and
stored bindings cannot safely treat them as durable authored identity.


## Unification with dataset transforms

The dimension-aware subset of dataset transforms and edge transforms are
the same operations. Today's encodings map like this:

```
Edge: SelectCategoriesTransformation  ≡  Dataset: FilterDimensionDatasetTransformOp
Edge: AssignCategoryTransformation    ≡  Dataset: FilterDimensionDatasetTransformOp(assign_category=...)
Edge: FlattenTransformation           ≡  (nothing — it is a port shape declaration)
```

The second line describes a **legacy encoding** that is being retired:
assignment is its own operation (`AssignDimension`), not a field on a
dimension filter.

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
| `filter_column` | dataset | Legacy, pre-dimension column filtering. |
| `rename_column` | dataset | Legacy wide-DVC column labels. |
| `rename_item` | dataset | Category value remapping. |
| `set_forecast_from` | dataset | Sets the forecast **qualifier**; see `metric-dataframe.md`. |
| `interpolate` | dataset | Fills gaps and sets the interpolation qualifier. |

There is deliberately no `select_column` op: metric selection is the
binding's source reference, not a transformation.


## Current state and next steps

**Done:**
- `PortTransformOp` is the one vocabulary; dataset bindings execute it as a
  pipeline, and `PortBindingDef` carries `transformations` for both kinds,
  always presented in the current vocabulary.
- `NodeEdge.transformations` stores the unified kinds: `select_categories` and
  `assign_category` were migrated to `filter_dimension` / `assign_dimension`
  (`0054`), sync emits the new kinds, and the legacy kinds survive only as
  tolerated input. `flatten` remains stored as the placeholder for a port
  shape declaration.
- Editing over GraphQL: `bindingEditor` resolves both kinds;
  `updateDatasetBinding` / `updateEdgeBinding` / `deleteBinding`, plus
  `bindDataset` and `createEdge`. Each kind has its own `oneOf` input type,
  so applicability is the input type's field list, introspectable by the
  editor UI; `createEdge` still accepts the deprecated legacy fields.
- Ports may carry an optional human-readable `identifier`.
- The runtime executes the typed op pipeline at the edge boundary
  (`6d798054`): `_get_output_for_target()` derives ops via
  `Edge.to_transforms()` and runs the shared executor, with `flatten`
  excluded from execution (it only feeds the output-shape assertion).
  Execution still happens on the producing node, fed from the legacy
  dicts — the consuming-port half of step 3 below remains.

**Next, in dependency order.** The first three are prerequisites for
propagation, not independent work:

1. Give ports authored, stable identity that survives a sync, and make
   `instance_from_db` emit port wiring that the loader consumes. Until
   then port ids are cosmetic. Related: unify `NodeEdge` and `DatasetPort`
   into one `NodeInputPortBinding` table. That is where a binding's ordering
   within a `multi` port can live — ordering has to be shared, because a port
   may hold both an edge and a dataset — and it retires the
   `(node, dataset_index)` grouping that stands in for binding identity today.
2. Move the ex-`FlattenTransformation` shape declarations onto
   `InputPortDef` and retire the `flatten` kind.
3. Consume the op pipeline **at the consuming port** from the stored
   binding. `_get_output_for_target()` already executes the typed ops
   through the shared executor, but derives them from the producer-side
   `from_dimensions` / `to_dimensions` dicts, and
   `_get_output_for_node()`'s metric selection and node-column filter are
   still outside the pipeline. Until the stored binding feeds the runtime
   directly, `_transforms_to_config()` in `instance_from_db.py` is the seam
   that translates stored transformations back into those dicts, and it
   bounds what an edge can execute — which is why `EdgeTransformationInput`
   accepts only the dimension-reshaping kinds for now. This step widens it
   with `drop_nulls` / `filter_temporal` / `ensure_unit` (additive,
   non-breaking).
4. Add node dimension signatures (requires/consumes/produces/transparent)
   to node classes or pipeline definitions.
5. Implement upstream constraint propagation for the editor's validation
   and port compatibility checks, exposing results as derived fields.

Removed from this list: "remove the `side` field from edge
transformations" — there is no such field in `edge_def.py`.
