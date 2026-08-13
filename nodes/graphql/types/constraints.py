"""
Strawberry types exposing constraint-solver results to the model editor.

These are derived, metadata-only views: they resolve from the request-cached
``InstanceGraph`` solve and never hydrate a runtime ``Context``. Effective
shapes and conflicts are read-only — they are never written back into node
specs or accepted through port inputs.
"""

from typing import TYPE_CHECKING
from uuid import UUID

import strawberry as sb

from paths.graphql_types import UnitType

from nodes.constraints.values import (
    BindingValue,
    ConstraintConflict,
    ConstraintOrigin,
    DatasetSourceValue,
    EffectiveValueShape,
    IntermediateValue,
    PortValue,
    ValueDirection,
    ValueKey,
)
from nodes.defs.binding_def import EdgeBindingDef

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nodes.constraints.solver import ConstraintSolveResult
    from nodes.instance_graph import InstanceGraph
    from nodes.units import Unit


@sb.type(name='ConstraintValueRef', description='The solver value a conflict is about.')
class ConstraintValueRefType:
    kind: str = sb.field(description='One of: port, binding, dataset_source, intermediate.')
    node_uuid: UUID | None = None
    port_id: UUID | None = None
    direction: str | None = sb.field(default=None, description='For port values: input or output.')
    binding_id: UUID | None = None

    @classmethod
    def from_value(cls, value: ValueKey) -> ConstraintValueRefType:
        match value:
            case PortValue():
                return cls(kind='port', node_uuid=value.node_id, port_id=value.port_id, direction=value.direction)
            case BindingValue():
                return cls(kind='binding', binding_id=value.binding_id)
            case DatasetSourceValue():
                return cls(kind='dataset_source', binding_id=value.binding_id)
            case IntermediateValue():
                return cls(kind='intermediate', node_uuid=value.node_id)


@sb.type(name='ConstraintOrigin', description='Where a conflicting fact or requirement was authored.')
class ConstraintOriginType:
    kind: str = sb.field(
        description='One of: declaration, node_rule, binding, transformation, dataset_schema, dataset_profile.',
    )
    node_uuid: UUID | None = None
    port_id: UUID | None = None
    binding_id: UUID | None = None
    transformation_index: int | None = None

    @classmethod
    def from_origin(cls, origin: ConstraintOrigin) -> ConstraintOriginType:
        return cls(
            kind=origin.kind,
            node_uuid=origin.node_id,
            port_id=origin.port_id,
            binding_id=origin.binding_id,
            transformation_index=origin.transformation_index,
        )


@sb.type(name='ConstraintConflict', description='One structural contradiction found by the constraint solver.')
class ConstraintConflictType:
    code: str
    message: str
    value: ConstraintValueRefType | None
    origins: list[ConstraintOriginType]

    @classmethod
    def from_conflict(cls, conflict: ConstraintConflict) -> ConstraintConflictType:
        return cls(
            code=conflict.code,
            message=conflict.message,
            value=ConstraintValueRefType.from_value(conflict.value) if conflict.value is not None else None,
            origins=[ConstraintOriginType.from_origin(origin) for origin in conflict.origins],
        )


@sb.type(
    name='ConstraintViolations',
    description=(
        'The mutation was rejected because it would introduce these structural conflicts. '
        'Nothing was written; pre-existing conflicts never appear here.'
    ),
)
class ConstraintViolationsType:
    conflicts: list[ConstraintConflictType]

    @classmethod
    def from_conflicts(cls, conflicts: Iterable[ConstraintConflict]) -> ConstraintViolationsType:
        return cls(conflicts=[ConstraintConflictType.from_conflict(conflict) for conflict in conflicts])


@sb.type(name='EffectiveShapeDimensionCategories')
class EffectiveShapeCategoriesType:
    dimension_uuid: UUID
    category_uuids: list[UUID]


@sb.type(name='EffectiveShape', description='Solver-derived shape of one port value. Read-only.')
class EffectiveShapeType:
    dimension_uuids: list[UUID] | None = sb.field(
        description='Exact dimensions of the value; null when the solver could not determine them.',
    )
    required_dimension_uuids: list[UUID]
    forbidden_dimension_uuids: list[UUID]
    categories: list[EffectiveShapeCategoriesType]
    quantity: str | None

    _unit: sb.Private['Unit | None'] = None

    @sb.field(graphql_type=UnitType | None)
    @staticmethod
    def unit(root: 'EffectiveShapeType') -> 'Unit | None':
        return root._unit

    @classmethod
    def from_shape(cls, shape: EffectiveValueShape) -> EffectiveShapeType:
        obj = cls(
            dimension_uuids=sorted(shape.dimensions, key=str) if shape.dimensions is not None else None,
            required_dimension_uuids=sorted(shape.required_dimensions, key=str),
            forbidden_dimension_uuids=sorted(shape.forbidden_dimensions, key=str),
            categories=[
                EffectiveShapeCategoriesType(
                    dimension_uuid=dimension_id,
                    category_uuids=sorted(category_ids, key=str),
                )
                for dimension_id, category_ids in sorted(shape.categories.items(), key=lambda item: str(item[0]))
            ],
            quantity=shape.quantity,
        )
        obj._unit = shape.unit
        return obj


def effective_port_shape(
    result: ConstraintSolveResult,
    node_uuid: UUID,
    port_id: UUID,
    direction: ValueDirection,
) -> EffectiveShapeType | None:
    shape = result.shapes.get(PortValue(node_uuid, port_id, direction))
    return EffectiveShapeType.from_shape(shape) if shape is not None else None


def _binding_node_ids(graph: InstanceGraph, binding_id: UUID, ids: set[UUID]) -> None:
    binding = graph.binding_by_id.get(binding_id)
    if binding is None:
        return
    if binding.port_ref.node_uuid is not None:
        ids.add(binding.port_ref.node_uuid)
    if isinstance(binding, EdgeBindingDef) and binding.from_ref.node_uuid is not None:
        ids.add(binding.from_ref.node_uuid)


def conflict_node_ids(graph: InstanceGraph, conflict: ConstraintConflict) -> set[UUID]:
    """Every node a conflict touches, through its value and each origin."""
    ids: set[UUID] = set()
    match conflict.value:
        case PortValue() | IntermediateValue():
            ids.add(conflict.value.node_id)
        case BindingValue() | DatasetSourceValue():
            _binding_node_ids(graph, conflict.value.binding_id, ids)
        case None:
            pass
    for origin in conflict.origins:
        if origin.node_id is not None:
            ids.add(origin.node_id)
        if origin.binding_id is not None:
            _binding_node_ids(graph, origin.binding_id, ids)
    return ids


def conflicts_for_node(
    graph: InstanceGraph,
    result: ConstraintSolveResult,
    node_uuid: UUID,
) -> list[ConstraintConflictType]:
    return [
        ConstraintConflictType.from_conflict(conflict)
        for conflict in result.conflicts
        if node_uuid in conflict_node_ids(graph, conflict)
    ]
