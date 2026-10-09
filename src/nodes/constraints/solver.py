"""
Compile an ``InstanceGraph`` into a constraint program and solve it.

The program relates value facts (``nodes/constraints/values.py``) through
four constraint kinds:

* ``TransformConstraint`` — one binding's delivery: the source value (an
  upstream output port or a dataset metric) pushed forward through the
  binding's resolved transformation steps into the delivered
  ``BindingValue``, and consumer requirements translated backward the same
  way;
* ``AggregateConstraint`` — the delivered values of one input port and the
  port aggregate share a shape (a multi port is a homogeneous aggregate;
  its observed categories are the union over deliveries);
* ``SameConstraint`` / ``ProductConstraint`` / ``TransformRuleConstraint`` —
  the compiled ``shape_rules()`` union relating port aggregates, pipeline
  intermediates, and outputs.

Propagation is a bidirectional monotone fixpoint: forward facts derive
output shapes, backward facts derive input requirements, and binding
transformations translate between them. Contradictions are collected as
``ConstraintConflict`` values — the solver never throws on the first one,
because editor diagnostics need the complete set. Facts born at a
declaration or dataset keep that origin as they travel, so a conflict names
the two authored sources that disagree rather than the propagation step
that happened to collide.

Anything the program cannot model honestly (an unresolvable dimension
reference, a tag operation that reshapes data) makes the affected value
*unknown* rather than guessed at: incompleteness never blocks, only
contradiction does.
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from nodes.constraints.rules import (
    AnyShapeRule,
    DimensionTransformRule,
    ProductShapeRule,
    SameShapeRule,
)
from nodes.constraints.values import (
    BindingValue,
    ConstraintConflict,
    ConstraintOrigin,
    DatasetSourceValue,
    EffectiveValueShape,
    FactStore,
    IntermediateValue,
    PortValue,
    ValueKey,
)
from nodes.defs.binding_def import DatasetBindingDef, EdgeBindingDef
from nodes.defs.transform_def import (
    AssignDimensionOp,
    EnsureUnitOp,
    FilterColumnOp,
    FilterDimensionOp,
    TagOperationOp,
)
from nodes.quantities import QuantityOperand, derive_product_quantity
from nodes.units import Unit, unit_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from nodes.dataset_shape import DatasetMetricPair, DatasetShapeProfile
    from nodes.defs.binding_def import AnyPortBindingDef
    from nodes.instance_graph import InstanceGraph

MAX_SOLVER_SWEEPS = 100

NEUTRAL_TAG_OPERATIONS = frozenset({
    'abs',
    'absolute',
    'add_missing_years',
    'arithmetic_inverse',
    'cumulative',
    'drop_infs',
    'drop_nans',
    'drop_zeros',
    'empty_to_zero',
    'extend_all',
    'extend_both_ways',
    'extend_forecast_values',
    'extend_to_history',
    'extend_values',
    'extrapolate',
    'fill_metrics_nan_null_zero',
    'forecast_only',
    'ignore_content',
    'inventory_only',
    'linear_interpolate',
    'make_nonnegative',
    'make_nonpositive',
    'minus',
    'observed_only_extend_all',
    'round_to_five',
    'truncate_before_start',
    'truncate_beyond_end',
    'use_observations',
})
"""
Registered tag operations that provably preserve dimensions, categories,
unit, and quantity. Any *other* registered operation (``complement``,
``geometric_inverse``, ``ratio_to_last_historical_value``, …) reshapes or
re-units the value, so the binding carrying it goes opaque. A tag that is
not a registered operation at all selects behavior instead of transforming
the frame and stays neutral.
"""


def _tag_is_opaque(tag: str) -> bool:
    from common.polars_ext import PathsExt

    return tag in PathsExt._OPERATION_METHODS and tag not in NEUTRAL_TAG_OPERATIONS


# --- Resolved transformation steps -------------------------------------------


@dataclass(frozen=True, slots=True)
class FilterStep:
    dimension_id: UUID
    selection: frozenset[UUID] | None
    """Selected category UUIDs; ``None`` when the selection is unresolvable (e.g. groups)."""
    exclude: bool
    flatten: bool
    index: int


@dataclass(frozen=True, slots=True)
class AssignStep:
    dimension_id: UUID
    category_id: UUID | None
    index: int


@dataclass(frozen=True, slots=True)
class UnitStep:
    unit: Unit
    index: int


@dataclass(frozen=True, slots=True)
class OpaqueStep:
    reason: str
    index: int


type TransformStep = FilterStep | AssignStep | UnitStep | OpaqueStep


# --- Program ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactSeed:
    key: ValueKey
    dims: frozenset[UUID] | None
    unit: Unit | None
    quantity: str | None
    origin: ConstraintOrigin
    required_dims: frozenset[UUID] = frozenset()
    """Lower-bound dimension requirements, distinct from the exact ``dims`` fact."""


@dataclass(frozen=True, slots=True)
class DatasetSourceInfo:
    binding_id: UUID
    dataset_id: UUID
    metric_id: UUID


@dataclass(frozen=True, slots=True)
class TransformConstraint:
    source: ValueKey
    target: BindingValue
    steps: tuple[TransformStep, ...]
    binding_id: UUID


@dataclass(frozen=True, slots=True)
class AggregateConstraint:
    port: PortValue
    members: tuple[BindingValue, ...]
    multi: bool


@dataclass(frozen=True, slots=True)
class SameConstraint:
    node_id: UUID
    rule_index: int
    inputs: tuple[ValueKey, ...]
    output: ValueKey


@dataclass(frozen=True, slots=True)
class ProductConstraint:
    node_id: UUID
    rule_index: int
    inputs: tuple[ValueKey, ...]
    inverse_inputs: tuple[ValueKey, ...]
    output: ValueKey


@dataclass(frozen=True, slots=True)
class TransformRuleConstraint:
    node_id: UUID
    rule_index: int
    input: ValueKey
    output: ValueKey
    rule: DimensionTransformRule


type AnyConstraint = TransformConstraint | AggregateConstraint | SameConstraint | ProductConstraint | TransformRuleConstraint


@dataclass(frozen=True, slots=True)
class ConstraintProgram:
    seeds: tuple[FactSeed, ...]
    constraints: tuple[AnyConstraint, ...]
    dataset_sources: tuple[DatasetSourceInfo, ...]
    static_conflicts: tuple[ConstraintConflict, ...]


@dataclass(frozen=True, slots=True)
class ConstraintSolveResult:
    shapes: dict[ValueKey, EffectiveValueShape]
    conflicts: tuple[ConstraintConflict, ...]
    converged: bool


class GraphOverlay(BaseModel):
    """A hypothetical binding edit applied at solve time without touching the graph."""

    model_config = ConfigDict(frozen=True)

    add_bindings: tuple[EdgeBindingDef | DatasetBindingDef, ...] = ()
    remove_binding_ids: frozenset[UUID] = frozenset()

    def apply(self, bindings: Sequence[AnyPortBindingDef]) -> tuple[AnyPortBindingDef, ...]:
        kept = tuple(binding for binding in bindings if binding.id not in self.remove_binding_ids)
        return (*kept, *self.add_bindings)

    def cache_key(self) -> str:
        return self.model_dump_json()


# --- Program compilation --------------------------------------------------------


def _resolve_dimension_refs(
    graph: InstanceGraph,
    refs: Sequence[str],
    origin: ConstraintOrigin,
    conflicts: list[ConstraintConflict],
    value: ValueKey | None,
) -> frozenset[UUID] | None:
    """Resolve identifier refs to dimension UUIDs; any unresolvable ref makes the set unknown."""
    ids: set[UUID] = set()
    ok = True
    for ref in refs:
        dimension = graph.dimension_by_identifier.get(ref)
        if dimension is None:
            conflicts.append(
                ConstraintConflict(
                    code='unknown_dimension_reference',
                    message=f'Unknown dimension {ref!r}',
                    value=value,
                    origins=(origin,),
                )
            )
            ok = False
        else:
            ids.add(dimension.id)
    return frozenset(ids) if ok else None


def _resolve_binding_steps(  # noqa: C901, PLR0912
    graph: InstanceGraph,
    binding: AnyPortBindingDef,
    conflicts: list[ConstraintConflict],
    dataset_dims: frozenset[UUID] | None = None,
) -> tuple[TransformStep, ...]:
    steps: list[TransformStep] = []
    target = BindingValue(binding.id)
    for index, op in enumerate(binding.transformations):
        origin = ConstraintOrigin('transformation', binding_id=binding.id, transformation_index=index)
        match op:
            case FilterDimensionOp():
                dimension = graph.dimension_by_identifier.get(op.dimension)
                if dimension is None:
                    conflicts.append(
                        ConstraintConflict(
                            code='unknown_dimension_reference',
                            message=f'Unknown dimension {op.dimension!r} in filter',
                            value=target,
                            origins=(origin,),
                        )
                    )
                    steps.append(OpaqueStep(reason=f'unresolvable filter dimension {op.dimension!r}', index=index))
                    continue
                selection: frozenset[UUID] | None = None
                if op.categories and not op.groups:
                    categories_by_identifier = {
                        category.identifier: category.id for category in dimension.categories if category.identifier
                    }
                    resolved: set[UUID] = set()
                    complete = True
                    for ref in op.categories:
                        category_id = categories_by_identifier.get(ref)
                        if category_id is None:
                            conflicts.append(
                                ConstraintConflict(
                                    code='unknown_category_reference',
                                    message=f'Unknown category {ref!r} in dimension {op.dimension!r}',
                                    value=target,
                                    origins=(origin,),
                                )
                            )
                            complete = False
                        else:
                            resolved.add(category_id)
                    if complete:
                        selection = frozenset(resolved)
                steps.append(
                    FilterStep(
                        dimension_id=dimension.id,
                        selection=selection,
                        exclude=op.exclude,
                        flatten=op.flatten,
                        index=index,
                    )
                )
            case AssignDimensionOp():
                dimension = graph.dimension_by_identifier.get(op.dimension)
                if dimension is None:
                    conflicts.append(
                        ConstraintConflict(
                            code='unknown_dimension_reference',
                            message=f'Unknown dimension {op.dimension!r} in assignment',
                            value=target,
                            origins=(origin,),
                        )
                    )
                    steps.append(OpaqueStep(reason=f'unresolvable assigned dimension {op.dimension!r}', index=index))
                    continue
                categories_by_identifier = {
                    category.identifier: category.id for category in dimension.categories if category.identifier
                }
                category_id = categories_by_identifier.get(op.category)
                if category_id is None:
                    conflicts.append(
                        ConstraintConflict(
                            code='unknown_category_reference',
                            message=f'Unknown category {op.category!r} in dimension {op.dimension!r}',
                            value=target,
                            origins=(origin,),
                        )
                    )
                steps.append(AssignStep(dimension_id=dimension.id, category_id=category_id, index=index))
            case EnsureUnitOp():
                steps.append(UnitStep(unit=op.unit, index=index))
            case FilterColumnOp():
                # A raw-column filter is shape-neutral — unless the column is
                # one of the dataset's *declared* dimensions, in which case
                # selecting on it (and dropping the column, the default)
                # removes that dimension. A name-matching raw column outside
                # the declared schema stays a raw column.
                dimension = graph.dimension_by_identifier.get(op.column)
                if dimension is None or dataset_dims is None or dimension.id not in dataset_dims:
                    continue
                steps.append(
                    FilterStep(
                        dimension_id=dimension.id,
                        selection=None,  # raw values are labels, not category identifiers
                        exclude=op.exclude,
                        flatten=op.drop_col or op.flatten,
                        index=index,
                    )
                )
            case TagOperationOp():
                if _tag_is_opaque(op.tag):
                    steps.append(OpaqueStep(reason=f'tag operation {op.tag!r}', index=index))
            case _:
                # Temporal, forecast, raw-column and marker ops do not touch
                # dimensions, categories, unit, or quantity.
                continue
    for tag in binding.tags:
        if _tag_is_opaque(tag):
            steps.append(OpaqueStep(reason=f'tag operation {tag!r}', index=len(binding.transformations)))
            break
    return tuple(steps)


def _parse_metric_unit(unit: str) -> Unit | None:
    if not unit:
        return None
    try:
        return unit_registry.parse_units(unit)
    except Exception:
        return None


def _rule_value_key(node_id: UUID, value_id: UUID, *, input_ids: set[UUID], output_ids: set[UUID]) -> ValueKey:
    if value_id in input_ids:
        return PortValue(node_id, value_id, 'input')
    if value_id in output_ids:
        return PortValue(node_id, value_id, 'output')
    return IntermediateValue(node_id, value_id)


def compile_constraint_program(  # noqa: C901, PLR0912, PLR0915
    graph: InstanceGraph,
    bindings: Sequence[AnyPortBindingDef] | None = None,
) -> ConstraintProgram:
    effective_bindings = tuple(graph.bindings) if bindings is None else tuple(bindings)
    seeds: list[FactSeed] = []
    constraints: list[AnyConstraint] = []
    dataset_sources: list[DatasetSourceInfo] = []
    static_conflicts: list[ConstraintConflict] = []

    # Authored input-port dimensions come from bare YAML ``to_dimensions``
    # entries. The runtime asserts, per edge, that the delivered value's
    # dimensions equal the *full* declared set — bare entries plus the
    # dimensions the binding's own assignments add (nodes/node.py,
    # ``_get_output_for_target``). Mirror that split exactly: the port
    # aggregate gets a lower-bound requirement here, and each edge binding
    # value gets the exact per-binding set below.
    declared_port_dims: dict[tuple[UUID, UUID], frozenset[UUID] | None] = {}
    for node in graph.nodes:
        for port in node.spec.input_ports:
            origin = ConstraintOrigin('declaration', node_id=node.id, port_id=port.id)
            key = PortValue(node.id, port.id, 'input')
            required: frozenset[UUID] | None = None
            if port.required_dimensions:
                required = _resolve_dimension_refs(graph, port.required_dimensions, origin, static_conflicts, key)
                declared_port_dims[(node.id, port.id)] = required
            seeds.append(
                FactSeed(
                    key=key,
                    dims=None,
                    unit=port.unit,
                    quantity=port.quantity,
                    origin=origin,
                    required_dims=required or frozenset(),
                )
            )
        for output_port in node.spec.output_ports:
            origin = ConstraintOrigin('declaration', node_id=node.id, port_id=output_port.id)
            key = PortValue(node.id, output_port.id, 'output')
            dims: frozenset[UUID] | None = None
            if output_port.dimensions:
                dims = _resolve_dimension_refs(graph, output_port.dimensions, origin, static_conflicts, key)
            seeds.append(FactSeed(key=key, dims=dims, unit=output_port.unit, quantity=output_port.quantity, origin=origin))

    compilation = graph.shape_rule_compilation
    for node_id, rules in compilation.rules_by_node.items():
        node_meta = graph.node_by_id[node_id]
        input_ids = {port.id for port in node_meta.spec.input_ports}
        output_ids = {port.id for port in node_meta.spec.output_ports}
        key_for = partial(_rule_value_key, node_id, input_ids=input_ids, output_ids=output_ids)

        rule: AnyShapeRule
        for rule_index, rule in enumerate(rules):
            match rule:
                case SameShapeRule():
                    constraints.append(
                        SameConstraint(
                            node_id=node_id,
                            rule_index=rule_index,
                            inputs=tuple(key_for(value) for value in rule.inputs),
                            output=key_for(rule.output),
                        )
                    )
                case ProductShapeRule():
                    constraints.append(
                        ProductConstraint(
                            node_id=node_id,
                            rule_index=rule_index,
                            inputs=tuple(key_for(value) for value in rule.inputs),
                            inverse_inputs=tuple(key_for(value) for value in rule.inverse_inputs),
                            output=key_for(rule.output),
                        )
                    )
                case DimensionTransformRule():
                    constraints.append(
                        TransformRuleConstraint(
                            node_id=node_id,
                            rule_index=rule_index,
                            input=key_for(rule.input),
                            output=key_for(rule.output),
                            rule=rule,
                        )
                    )

    members_by_port: defaultdict[tuple[UUID, UUID], list[AnyPortBindingDef]] = defaultdict(list)
    for binding in effective_bindings:
        node_uuid = binding.port_ref.node_uuid
        if node_uuid is None or (node_uuid, binding.port_ref.port_id) not in graph.input_port_by_id:
            continue  # structural breakage is already an InstanceGraph diagnostic
        source: ValueKey
        dataset_dims: frozenset[UUID] | None = None
        if isinstance(binding, EdgeBindingDef):
            from_uuid = binding.from_ref.node_uuid
            if from_uuid is None or (from_uuid, binding.from_ref.port_id) not in graph.output_port_by_id:
                continue
            source = PortValue(from_uuid, binding.from_ref.port_id, 'output')
        else:
            assert isinstance(binding, DatasetBindingDef)
            if binding.dataset_uuid is None or binding.metric_uuid is None:
                continue
            dataset = graph.dataset_by_id.get(binding.dataset_uuid)
            if dataset is None:
                continue
            metric = dataset.metric_by_id.get(binding.metric_uuid)
            if metric is None:
                continue
            source = DatasetSourceValue(binding.id)
            schema_origin = ConstraintOrigin('dataset_schema', binding_id=binding.id)
            # An external placeholder with no declared dimensions has *unknown*
            # shape, not scalar shape: its schema was never imported.
            schema_dims: frozenset[UUID] | None = frozenset(dataset.declared_dimension_ids)
            if dataset.is_external_placeholder and not dataset.declared_dimension_ids:
                schema_dims = None
            dataset_dims = schema_dims
            seeds.append(
                FactSeed(
                    key=source,
                    dims=schema_dims,
                    unit=_parse_metric_unit(metric.unit),
                    quantity=None,
                    origin=schema_origin,
                )
            )
            dataset_sources.append(DatasetSourceInfo(binding_id=binding.id, dataset_id=dataset.id, metric_id=metric.id))
        steps = _resolve_binding_steps(graph, binding, static_conflicts, dataset_dims=dataset_dims)
        if isinstance(binding, EdgeBindingDef):
            # The per-edge output-dimension assertion: bare declared entries
            # (on the port for re-synced snapshots, on the binding for legacy
            # rows) plus this binding's own assigned dimensions, asserted
            # against the delivered (post-transformation) value when any part
            # of the declaration exists.
            port = graph.input_port_by_id[(node_uuid, binding.port_ref.port_id)]
            declared = declared_port_dims.get((node_uuid, binding.port_ref.port_id))
            binding_origin = ConstraintOrigin('declaration', node_id=node_uuid, port_id=port.id, binding_id=binding.id)
            legacy_declared: frozenset[UUID] | None = None
            legacy_unresolvable = False
            if binding.declared_dimensions:
                legacy_declared = _resolve_dimension_refs(
                    graph, binding.declared_dimensions, binding_origin, static_conflicts, BindingValue(binding.id)
                )
                legacy_unresolvable = legacy_declared is None
            assigned = frozenset(step.dimension_id for step in steps if isinstance(step, AssignStep))
            if not legacy_unresolvable and (declared is not None or legacy_declared is not None or assigned):
                base = (declared or frozenset()) | (legacy_declared or frozenset())
                seeds.append(
                    FactSeed(
                        key=BindingValue(binding.id),
                        dims=base | assigned,
                        unit=None,
                        quantity=None,
                        origin=binding_origin,
                    )
                )
        constraints.append(
            TransformConstraint(source=source, target=BindingValue(binding.id), steps=steps, binding_id=binding.id)
        )
        members_by_port[(node_uuid, binding.port_ref.port_id)].append(binding)

    untrusted_node_ids = compilation.untrusted_node_ids
    for (node_uuid, port_id), members in members_by_port.items():
        if node_uuid in untrusted_node_ids:
            # The class computation is overridden below its rule declaration;
            # the multi-port homogeneity contract is part of the same untrusted
            # class vocabulary, so no aggregate constraint applies either.
            continue
        port = graph.input_port_by_id[(node_uuid, port_id)]
        ordered = sorted(members, key=lambda member: (member.position, str(member.id)))
        if not port.multi and len(ordered) > 1:
            static_conflicts.append(
                ConstraintConflict(
                    code='multiple_bindings_on_single_port',
                    message=f'{len(ordered)} values are bound to a port that accepts one',
                    value=PortValue(node_uuid, port_id, 'input'),
                    origins=tuple(ConstraintOrigin('binding', binding_id=member.id) for member in ordered),
                )
            )
        constraints.append(
            AggregateConstraint(
                port=PortValue(node_uuid, port_id, 'input'),
                members=tuple(BindingValue(member.id) for member in ordered),
                multi=port.multi,
            )
        )

    return ConstraintProgram(
        seeds=tuple(seeds),
        constraints=tuple(constraints),
        dataset_sources=tuple(dataset_sources),
        static_conflicts=tuple(static_conflicts),
    )


# --- Constraint application ------------------------------------------------------


def _apply_transform_forward(constraint: TransformConstraint, store: FactStore) -> bool:  # noqa: C901, PLR0912, PLR0915
    source = store.get(constraint.source)
    dims = source.dims_exact
    dims_origin = source.dims_exact_origin
    categories: dict[UUID, tuple[frozenset[UUID], ConstraintOrigin]] = {
        dim: (cats, source.categories_origin[dim]) for dim, cats in source.categories.items()
    }
    unit = source.unit
    unit_origin = source.unit_origin
    quantity = source.quantity
    quantity_origin = source.quantity_origin

    for step in constraint.steps:
        origin = ConstraintOrigin('transformation', binding_id=constraint.binding_id, transformation_index=step.index)
        match step:
            case OpaqueStep():
                dims = None
                dims_origin = None
                categories = {}
                unit = None
                unit_origin = None
                quantity = None
                quantity_origin = None
            case UnitStep():
                if unit is not None and not unit.is_compatible_with(step.unit):
                    assert unit_origin is not None
                    store.add_conflict(
                        'unit_incompatible',
                        f'Unit {unit} is not convertible to {step.unit}',
                        constraint.target,
                        (unit_origin, origin),
                    )
                unit = step.unit
                unit_origin = origin
            case FilterStep():
                if dims is not None and step.dimension_id not in dims:
                    assert dims_origin is not None
                    store.add_conflict(
                        'filter_missing_dimension',
                        f'Filtered dimension {store.describe(step.dimension_id)} is not present',
                        constraint.target,
                        (origin, dims_origin),
                    )
                    return False  # facts beyond a misapplied filter would be guesses
                known = categories.get(step.dimension_id)
                if step.selection is not None and known is not None:
                    known_categories, known_origin = known
                    kept = known_categories - step.selection if step.exclude else known_categories & step.selection
                    if not kept:
                        store.add_conflict(
                            'disjoint_category_filter',
                            f'Filter on dimension {store.describe(step.dimension_id)} keeps no observed category',
                            constraint.target,
                            (origin, known_origin),
                        )
                    categories[step.dimension_id] = (kept, origin)
                elif step.selection is None:
                    categories.pop(step.dimension_id, None)
                if step.flatten:
                    if dims is not None:
                        dims = dims - {step.dimension_id}
                        dims_origin = origin
                    categories.pop(step.dimension_id, None)
            case AssignStep():
                if dims is not None and step.dimension_id in dims:
                    assert dims_origin is not None
                    store.add_conflict(
                        'assign_existing_dimension',
                        f'Assigned dimension {store.describe(step.dimension_id)} is already present',
                        constraint.target,
                        (origin, dims_origin),
                    )
                    return False
                if dims is not None:
                    dims = dims | {step.dimension_id}
                    dims_origin = origin
                if step.category_id is not None:
                    categories[step.dimension_id] = (frozenset({step.category_id}), origin)

    changed = False
    if dims is not None:
        assert dims_origin is not None
        changed |= store.set_dims_exact(constraint.target, dims, dims_origin)
    for dim, (cats, cats_origin) in categories.items():
        changed |= store.set_categories(constraint.target, dim, cats, cats_origin)
    if unit is not None:
        assert unit_origin is not None
        changed |= store.set_unit(constraint.target, unit, unit_origin)
    if quantity is not None:
        assert quantity_origin is not None
        changed |= store.set_quantity(constraint.target, quantity, quantity_origin)
    return changed


def _apply_transform_backward(constraint: TransformConstraint, store: FactStore) -> bool:  # noqa: C901, PLR0912
    target = store.get(constraint.target)
    required = dict(target.dims_required)
    forbidden = dict(target.dims_forbidden)
    exact = target.dims_exact
    exact_origin = target.dims_exact_origin

    for step in reversed(constraint.steps):
        origin = ConstraintOrigin('transformation', binding_id=constraint.binding_id, transformation_index=step.index)
        match step:
            case OpaqueStep():
                return False
            case UnitStep():
                continue
            case FilterStep():
                if step.flatten:
                    if exact is not None:
                        if step.dimension_id in exact:
                            assert exact_origin is not None
                            store.add_conflict(
                                'flattened_dimension_present',
                                f'Dimension {store.describe(step.dimension_id)} is flattened away but present downstream',
                                constraint.target,
                                (origin, exact_origin),
                            )
                            exact = None
                            exact_origin = None
                        else:
                            exact = exact | {step.dimension_id}
                            exact_origin = origin
                    required.pop(step.dimension_id, None)
                    forbidden.pop(step.dimension_id, None)
                required[step.dimension_id] = origin
            case AssignStep():
                if exact is not None:
                    if step.dimension_id not in exact:
                        assert exact_origin is not None
                        store.add_conflict(
                            'assigned_dimension_missing',
                            f'Assigned dimension {store.describe(step.dimension_id)} is absent downstream',
                            constraint.target,
                            (origin, exact_origin),
                        )
                        exact = None
                        exact_origin = None
                    else:
                        exact = exact - {step.dimension_id}
                        exact_origin = origin
                required.pop(step.dimension_id, None)
                forbidden.pop(step.dimension_id, None)
                forbidden[step.dimension_id] = origin

    changed = False
    for dim, dim_origin in required.items():
        changed |= store.require_dimension(constraint.source, dim, dim_origin)
    for dim, dim_origin in forbidden.items():
        changed |= store.forbid_dimension(constraint.source, dim, dim_origin)
    if exact is not None:
        assert exact_origin is not None
        changed |= store.set_dims_exact(constraint.source, exact, exact_origin)
    return changed


def _equalize_values(store: FactStore, keys: tuple[ValueKey, ...]) -> bool:  # noqa: C901
    """Propagate dims/unit/quantity facts and dimension requirements among shape-equal values."""
    changed = False
    facts = [store.get(key) for key in keys]

    dims_ref = next(((f.dims_exact, f.dims_exact_origin) for f in facts if f.dims_exact is not None), None)
    if dims_ref is not None:
        dims, origin = dims_ref
        assert origin is not None
        for key in keys:
            changed |= store.set_dims_exact(key, dims, origin)

    unit_ref = next(((f.unit, f.unit_origin) for f in facts if f.unit is not None), None)
    if unit_ref is not None:
        unit, unit_origin = unit_ref
        assert unit_origin is not None
        for key in keys:
            changed |= store.set_unit(key, unit, unit_origin)

    quantity_ref = next(((f.quantity, f.quantity_origin) for f in facts if f.quantity is not None), None)
    if quantity_ref is not None:
        quantity, quantity_origin = quantity_ref
        assert quantity_origin is not None
        for key in keys:
            changed |= store.set_quantity(key, quantity, quantity_origin)

    all_required = {dim: origin for f in facts for dim, origin in f.dims_required.items()}
    for dim, origin in all_required.items():
        for key in keys:
            changed |= store.require_dimension(key, dim, origin)
    all_forbidden = {dim: origin for f in facts for dim, origin in f.dims_forbidden.items()}
    for dim, origin in all_forbidden.items():
        for key in keys:
            changed |= store.forbid_dimension(key, dim, origin)
    return changed


def _union_categories(
    store: FactStore,
    sources: tuple[ValueKey, ...],
    target: ValueKey,
    origin: ConstraintOrigin,
) -> bool:
    """Write the per-dimension union of source categories onto the target, where every source is known."""
    changed = False
    source_facts = [store.get(key) for key in sources]
    candidate_dims = {dim for facts in source_facts for dim in facts.categories}
    for dim in candidate_dims:
        if any(dim not in facts.categories for facts in source_facts):
            continue
        union = frozenset().union(*(facts.categories[dim] for facts in source_facts))
        changed |= store.set_categories(target, dim, union, origin)
    return changed


def _apply_aggregate(constraint: AggregateConstraint, store: FactStore) -> bool:
    if not constraint.members:
        return False
    keys: tuple[ValueKey, ...] = (constraint.port, *constraint.members)
    changed = _equalize_values(store, keys)
    origin = ConstraintOrigin('binding', node_id=constraint.port.node_id, port_id=constraint.port.port_id)
    changed |= _union_categories(store, constraint.members, constraint.port, origin)
    return changed


def _apply_same(constraint: SameConstraint, store: FactStore) -> bool:
    changed = _equalize_values(store, (*constraint.inputs, constraint.output))
    origin = ConstraintOrigin('node_rule', node_id=constraint.node_id, rule_index=constraint.rule_index)
    changed |= _union_categories(store, constraint.inputs, constraint.output, origin)
    return changed


def _product_unit(units: Sequence[Unit], inverse_units: Sequence[Unit]) -> Unit:
    quantity = unit_registry.Quantity(1.0)
    for unit in units:
        quantity = quantity * unit
    for unit in inverse_units:
        quantity = quantity / unit
    return quantity.units


def _apply_product(constraint: ProductConstraint, store: FactStore) -> bool:
    operand_keys = (*constraint.inputs, *constraint.inverse_inputs)
    operands = [store.get(key) for key in operand_keys]
    origin = ConstraintOrigin('node_rule', node_id=constraint.node_id, rule_index=constraint.rule_index)
    changed = False

    operand_dims = [facts.dims_exact for facts in operands]
    if all(dims is not None for dims in operand_dims):
        union: frozenset[UUID] = frozenset().union(*(dims for dims in operand_dims if dims is not None))
        changed |= store.set_dims_exact(constraint.output, union, origin)

        candidate_dims = {dim for facts in operands for dim in facts.categories}
        for dim in candidate_dims:
            contributors = [facts for facts in operands if facts.dims_exact is not None and dim in facts.dims_exact]
            if not contributors or any(dim not in facts.categories for facts in contributors):
                continue
            intersection = frozenset.intersection(*(facts.categories[dim] for facts in contributors))
            changed |= store.set_categories(constraint.output, dim, intersection, origin)

    input_units = [store.get(key).unit for key in constraint.inputs]
    inverse_units = [store.get(key).unit for key in constraint.inverse_inputs]
    if all(unit is not None for unit in (*input_units, *inverse_units)):
        product = _product_unit(
            [unit for unit in input_units if unit is not None],
            [unit for unit in inverse_units if unit is not None],
        )
        changed |= store.set_unit(constraint.output, product, origin)

    quantity = derive_product_quantity(
        tuple(QuantityOperand(quantity=store.get(key).quantity, unit=store.get(key).unit) for key in constraint.inputs),
        tuple(QuantityOperand(quantity=store.get(key).quantity, unit=store.get(key).unit) for key in constraint.inverse_inputs),
    )
    if quantity is not None:
        changed |= store.set_quantity(constraint.output, quantity, origin)
    return changed


def _apply_transform_rule(constraint: TransformRuleConstraint, store: FactStore) -> bool:  # noqa: C901, PLR0912
    rule = constraint.rule
    origin = ConstraintOrigin('node_rule', node_id=constraint.node_id, rule_index=constraint.rule_index)
    changed = False

    for dim in rule.requires:
        changed |= store.require_dimension(constraint.input, dim, origin)

    input_facts = store.get(constraint.input)
    if input_facts.dims_exact is not None:
        if rule.transparent:
            output_dims = (input_facts.dims_exact - rule.consumes) | rule.produces
        else:
            output_dims = ((input_facts.dims_exact & rule.requires) - rule.consumes) | rule.produces
        changed |= store.set_dims_exact(constraint.output, output_dims, origin)
        for dim, cats in input_facts.categories.items():
            if dim in output_dims and dim not in rule.produces:
                changed |= store.set_categories(constraint.output, dim, cats, origin)

    output_facts = store.get(constraint.output)
    if rule.transparent and output_facts.dims_exact is not None:
        if not rule.produces.issubset(output_facts.dims_exact):
            assert output_facts.dims_exact_origin is not None
            store.add_conflict(
                'produced_dimension_missing',
                'A dimension this node produces is absent from its declared output',
                constraint.output,
                (origin, output_facts.dims_exact_origin),
            )
        else:
            input_dims = (output_facts.dims_exact - rule.produces) | rule.consumes
            changed |= store.set_dims_exact(constraint.input, input_dims, origin)
        for dim, required_origin in output_facts.dims_required.items():
            if dim not in rule.produces:
                changed |= store.require_dimension(constraint.input, dim, required_origin)

    unit_ref = next(
        ((f.unit, f.unit_origin) for f in (input_facts, output_facts) if f.unit is not None),
        None,
    )
    if unit_ref is not None:
        unit, unit_origin = unit_ref
        assert unit_origin is not None
        changed |= store.set_unit(constraint.input, unit, unit_origin)
        changed |= store.set_unit(constraint.output, unit, unit_origin)

    quantity_ref = next(
        ((f.quantity, f.quantity_origin) for f in (input_facts, output_facts) if f.quantity is not None),
        None,
    )
    if quantity_ref is not None:
        quantity, quantity_origin = quantity_ref
        assert quantity_origin is not None
        changed |= store.set_quantity(constraint.input, quantity, quantity_origin)
        changed |= store.set_quantity(constraint.output, quantity, quantity_origin)
    return changed


def _apply_constraint(constraint: AnyConstraint, store: FactStore) -> bool:
    match constraint:
        case TransformConstraint():
            changed = _apply_transform_forward(constraint, store)
            return _apply_transform_backward(constraint, store) or changed
        case AggregateConstraint():
            return _apply_aggregate(constraint, store)
        case SameConstraint():
            return _apply_same(constraint, store)
        case ProductConstraint():
            return _apply_product(constraint, store)
        case TransformRuleConstraint():
            return _apply_transform_rule(constraint, store)


def solve_constraint_program(  # noqa: C901, PLR0912
    program: ConstraintProgram,
    *,
    describe: Callable[[UUID], str] | None = None,
    profiles: Mapping[DatasetMetricPair, DatasetShapeProfile] | None = None,
) -> ConstraintSolveResult:
    store = FactStore(describe=describe)
    for conflict in program.static_conflicts:
        store.add_conflict(conflict.code, conflict.message, conflict.value, conflict.origins)

    for seed in program.seeds:
        if seed.dims is not None:
            store.set_dims_exact(seed.key, seed.dims, seed.origin)
        for dim in seed.required_dims:
            store.require_dimension(seed.key, dim, seed.origin)
        if seed.unit is not None:
            store.set_unit(seed.key, seed.unit, seed.origin)
        if seed.quantity is not None:
            store.set_quantity(seed.key, seed.quantity, seed.origin)

    if profiles:
        for info in program.dataset_sources:
            profile = profiles.get((info.dataset_id, info.metric_id))
            if profile is None:
                continue
            key = DatasetSourceValue(info.binding_id)
            origin = ConstraintOrigin('dataset_profile', binding_id=info.binding_id)
            for dim, cats in profile.categories_by_dimension.items():
                if cats is not None:
                    store.set_categories(key, dim, cats, origin)

    converged = False
    for _sweep in range(MAX_SOLVER_SWEEPS):
        changed = False
        for constraint in program.constraints:
            changed |= _apply_constraint(constraint, store)
        if not changed:
            converged = True
            break

    return ConstraintSolveResult(shapes=store.snapshot(), conflicts=store.conflicts, converged=converged)
