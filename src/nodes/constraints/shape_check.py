"""
The static check of shape references: whether a dataset's entry form can feed the port it is bound to.

It runs while the constraint program is compiled and reads no data. Each binding from a dataset
that refers to a shape, into a port whose contract refers to one, is projected forward through
the binding's resolved steps: a filter restricts the combinations, an assignment adds a fixed
coordinate, and flattening drops a dimension and merges what becomes identical. The projection is
compared with the port's shape: a closed shape must hold every projected combination, and each
required group needs one, or the data-entry route can never satisfy it.

Mismatches are constraint conflicts, so they block publication and reject an edit that introduces
them. A binding whose steps cannot be projected is a notice instead: unchecked, never passed.
See docs/architecture/shapes.md.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nodes.constraints.steps import AssignStep, FilterStep, OpaqueStep, UnitStep
from nodes.constraints.values import BindingValue, ConstraintConflict, ConstraintOrigin, PortValue
from nodes.defs.binding_def import DatasetBindingDef
from nodes.defs.transform_def import FilterDimensionOp

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from uuid import UUID

    from nodes.constraints.steps import TransformStep
    from nodes.defs.binding_def import AnyPortBindingDef
    from nodes.instance_graph import InstanceGraph
    from nodes.shapes import EffectiveShape

type Combination = frozenset[tuple[UUID, UUID]]
"""(dimension, category) pairs."""


@dataclass(frozen=True, slots=True)
class ShapeCheck:
    conflicts: tuple[ConstraintConflict, ...]
    notices: tuple[ConstraintConflict, ...]
    """Bindings the check could not project; they block nothing."""


@dataclass(slots=True)
class _Projection:
    combinations: set[Combination]
    named: frozenset[UUID]
    """The dimensions the combinations name."""
    dimensions: frozenset[UUID] | None
    """The dimensions the delivered value has, when the dataset declares them."""


@dataclass(slots=True)
class _PortState:
    bindings: list[UUID] = field(default_factory=list)
    combinations: set[Combination] = field(default_factory=set)
    complete: bool = True
    """Every binding into the port is a projected data-entry route."""


class _Unprojectable(Exception):  # noqa: N818
    pass


class _Check:
    def __init__(self, graph: InstanceGraph) -> None:
        self.graph = graph
        self.conflicts: list[ConstraintConflict] = []
        self.notices: list[ConstraintConflict] = []
        self.categories = {
            dimension.identifier: (dimension.id, {category.identifier: category.id for category in dimension.categories})
            for dimension in graph.dimensions
            if dimension.identifier
        }
        self.names = {
            **{dimension.id: dimension.identifier or str(dimension.id) for dimension in graph.dimensions},
            **{
                category.id: category.identifier or str(category.id)
                for dimension in graph.dimensions
                for category in dimension.categories
            },
        }

    def combinations(self, shape: EffectiveShape) -> tuple[frozenset[UUID], dict[UUID, Combination]]:
        """Return the shape's dimensions and its combinations by UUID, as dimension and category UUIDs."""
        dimensions = frozenset(self.categories[dimension][0] for dimension in shape.dimensions if dimension in self.categories)
        if len(dimensions) != len(shape.dimensions):
            raise _Unprojectable(f'shape {shape.spec.label} names a dimension the instance does not have')
        resolved: dict[UUID, Combination] = {}
        for combination in shape.combinations:
            pairs = set()
            for dimension, category in combination.categories.items():
                dimension_id, categories = self.categories[dimension]
                if category not in categories:
                    raise _Unprojectable(f'shape {shape.spec.label} names {dimension}:{category}, which the instance lacks')
                pairs.add((dimension_id, categories[category]))
            resolved[combination.uuid] = frozenset(pairs)
        return dimensions, resolved

    def describe(self, combination: Combination) -> str:
        return ', '.join(sorted(f'{self.names[dimension]}={self.names[category]}' for dimension, category in combination))

    def project(self, binding: DatasetBindingDef, shape: EffectiveShape, steps: Sequence[TransformStep]) -> _Projection:
        named, combinations = self.combinations(shape)
        dataset = self.graph.dataset_by_id[binding.dataset_uuid] if binding.dataset_uuid is not None else None
        declared = frozenset(dataset.declared_dimension_ids) if dataset is not None and dataset.declared_dimension_ids else None
        projection = _Projection(set(combinations.values()), named, declared)
        for step in steps:
            match step:
                case FilterStep():
                    self._filter(projection, binding, step)
                case AssignStep():
                    if step.category_id is None:
                        raise _Unprojectable(f'it assigns an unknown category of {self.names.get(step.dimension_id)}')
                    pair = (step.dimension_id, step.category_id)
                    projection.combinations = {
                        frozenset({*(p for p in combination if p[0] != step.dimension_id), pair})
                        for combination in projection.combinations
                    }
                    projection.named |= {step.dimension_id}
                    if projection.dimensions is not None:
                        projection.dimensions |= {step.dimension_id}
                case OpaqueStep():
                    raise _Unprojectable(f'it goes through {step.reason}')
                case UnitStep():
                    pass
        return projection

    def _filter(self, projection: _Projection, binding: DatasetBindingDef, step: FilterStep) -> None:
        selection = step.selection
        if selection is None:
            op = binding.transformations[step.index] if step.index < len(binding.transformations) else None
            whole_dimension = isinstance(op, FilterDimensionOp) and not op.categories and not op.groups
            if not whole_dimension:
                raise _Unprojectable(f'its filter on {self.names.get(step.dimension_id)} cannot be resolved')
        elif step.dimension_id in projection.named:
            projection.combinations = {
                combination
                for combination in projection.combinations
                if (dict(combination)[step.dimension_id] in selection) != step.exclude
            }
        if step.flatten:
            projection.combinations = {
                frozenset(pair for pair in combination if pair[0] != step.dimension_id) for combination in projection.combinations
            }
            projection.named -= {step.dimension_id}
            if projection.dimensions is not None:
                projection.dimensions -= {step.dimension_id}


def check_shape_references(
    graph: InstanceGraph,
    bindings: Iterable[AnyPortBindingDef],
    steps_by_binding: Mapping[UUID, Sequence[TransformStep]],
) -> ShapeCheck:
    if not graph.dimensions:
        # A graph parsed from YAML carries no dimension catalogue, and none of its dimension facts
        # can be checked; it gets one when its instance is synced to the database.
        return ShapeCheck((), ())
    check = _Check(graph)
    ports: dict[tuple[UUID, UUID], _PortState] = {}
    for binding in bindings:
        node_uuid = binding.port_ref.node_uuid
        if node_uuid is None or (node_uuid, binding.port_ref.port_id) not in graph.input_port_by_id:
            continue
        port = graph.input_port_by_id[(node_uuid, binding.port_ref.port_id)]
        if port.validation is None or port.validation.shape is None:
            continue
        state = ports.setdefault((node_uuid, port.id), _PortState())
        state.bindings.append(binding.id)
        projected = _check_binding(check, binding, port.validation.shape, steps_by_binding.get(binding.id, ()))
        if projected is None:
            state.complete = False
        else:
            state.combinations |= projected
    for (node_uuid, port_id), state in ports.items():
        if state.complete:
            _check_requirements(check, node_uuid, port_id, state)
    return ShapeCheck(tuple(check.conflicts), tuple(check.notices))


def _check_binding(
    check: _Check, binding: AnyPortBindingDef, port_shape_id: UUID, steps: Sequence[TransformStep]
) -> set[Combination] | None:
    """Report one binding's mismatches; return what it can deliver, or None when that is not known."""
    graph = check.graph
    if not isinstance(binding, DatasetBindingDef) or binding.dataset_uuid is None:
        # Another route feeds the port; only its delivered values can say what it holds.
        return None
    dataset = graph.dataset_by_id.get(binding.dataset_uuid)
    if dataset is None or dataset.shape_id is None:
        return None
    node_uuid = binding.port_ref.node_uuid
    assert node_uuid is not None
    origins = _origins(node_uuid, binding.port_ref.port_id, binding.id)
    port_shape = graph.shapes.get(port_shape_id)
    dataset_shape = graph.shapes.get(dataset.shape_id)
    if port_shape is None or dataset_shape is None:
        missing = port_shape_id if port_shape is None else dataset.shape_id
        check.conflicts.append(
            ConstraintConflict(
                code='unknown_shape',
                message=f'{_route(graph, binding)} refers to shape {missing}, which the instance does not declare',
                value=BindingValue(binding.id),
                origins=origins,
            )
        )
        return None
    try:
        projection = check.project(binding, dataset_shape, steps)
        port_dimensions, port_combinations = check.combinations(port_shape)
    except _Unprojectable as reason:
        check.notices.append(
            ConstraintConflict(
                code='shape_unchecked',
                message=f'{_route(graph, binding)} cannot be checked against shape {port_shape.spec.label}: {reason}',
                value=BindingValue(binding.id),
                origins=origins,
            )
        )
        return None
    return _compare(check, binding, projection, port_shape, port_dimensions, port_combinations)


def _check_requirements(check: _Check, node_uuid: UUID, port_id: UUID, state: _PortState) -> None:
    graph = check.graph
    port = graph.input_port_by_id[(node_uuid, port_id)]
    assert port.validation is not None
    assert port.validation.shape is not None
    port_shape = graph.shapes[port.validation.shape]
    _, port_combinations = check.combinations(port_shape)
    for group in port_shape.required:
        if any(port_combinations[member] in state.combinations for member in group.combinations):
            continue
        choices = ' or '.join(f'({check.describe(port_combinations[member])})' for member in group.combinations)
        check.conflicts.append(
            ConstraintConflict(
                code='shape_requirement_unreachable',
                message=(
                    f'{_port_label(graph, node_uuid, port_id)} requires {choices} '
                    f'(shape {port_shape.spec.label}), which no data-entry table bound to it can hold'
                ),
                value=PortValue(node_uuid, port_id, 'input'),
                origins=(
                    ConstraintOrigin('declaration', node_id=node_uuid, port_id=port_id),
                    *(ConstraintOrigin('binding', binding_id=binding_id) for binding_id in state.bindings),
                ),
            )
        )


def _compare(
    check: _Check,
    binding: DatasetBindingDef,
    projection: _Projection,
    port_shape: EffectiveShape,
    port_dimensions: frozenset[UUID],
    port_combinations: Mapping[UUID, Combination],
) -> set[Combination] | None:
    """Report one binding's mismatches; return its projection onto the port's dimensions, if it has one."""
    graph = check.graph
    node_uuid = binding.port_ref.node_uuid
    assert node_uuid is not None
    origins = _origins(node_uuid, binding.port_ref.port_id, binding.id)
    unnamed = port_dimensions - projection.named
    if unnamed:
        names = ', '.join(sorted(check.names[dimension] for dimension in unnamed))
        if projection.dimensions is not None and not unnamed <= projection.dimensions:
            check.conflicts.append(
                ConstraintConflict(
                    code='shape_dimension_missing',
                    message=f'{_route(graph, binding)} delivers no {names}, which shape {port_shape.spec.label} constrains',
                    value=BindingValue(binding.id),
                    origins=origins,
                )
            )
        else:
            check.notices.append(
                ConstraintConflict(
                    code='shape_unchecked',
                    message=(
                        f'{_route(graph, binding)} cannot be checked against shape {port_shape.spec.label}: '
                        f"the dataset's shape does not constrain {names}"
                    ),
                    value=BindingValue(binding.id),
                    origins=origins,
                )
            )
        return None
    projected = {frozenset(pair for pair in combination if pair[0] in port_dimensions) for combination in projection.combinations}
    if port_shape.closed:
        allowed = set(port_combinations.values())
        for combination in sorted(projected - allowed, key=check.describe):
            check.conflicts.append(
                ConstraintConflict(
                    code='outside_shape',
                    message=(
                        f'{_route(graph, binding)} can deliver {check.describe(combination)}, '
                        f'which shape {port_shape.spec.label} does not allow'
                    ),
                    value=BindingValue(binding.id),
                    origins=origins,
                )
            )
    return projected


def _origins(node_uuid: UUID, port_id: UUID, binding_id: UUID) -> tuple[ConstraintOrigin, ...]:
    return (
        ConstraintOrigin('declaration', node_id=node_uuid, port_id=port_id),
        ConstraintOrigin('binding', binding_id=binding_id),
    )


def _port_label(graph: InstanceGraph, node_uuid: UUID, port_id: UUID) -> str:
    node = graph.node_by_id[node_uuid]
    port = graph.input_port_by_id[(node_uuid, port_id)]
    name = port.identifier or port.role
    return f'{node.identifier or node_uuid}:{name}' if name else f'{node.identifier or node_uuid}'


def _route(graph: InstanceGraph, binding: DatasetBindingDef) -> str:
    assert binding.port_ref.node_uuid is not None
    assert binding.dataset_uuid is not None
    dataset = graph.dataset_by_id[binding.dataset_uuid]
    return f'{dataset.identifier or dataset.id} -> {_port_label(graph, binding.port_ref.node_uuid, binding.port_ref.port_id)}'
