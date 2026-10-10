from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, TypeVar
from uuid import UUID

from pydantic import AfterValidator, Field
from pydantic_core import PydanticCustomError

from .identifiers import (
    DimensionCategoryIdentifier,
    DimensionIdentifier,
    QuantityKindIdentifier,
)
from .uuid_kinds import Provenance, Ref

"""
Reference-oriented validation helpers for Paths domain objects.

This module is intended to hold runtime-aware validation concepts:
references to existing nodes, dimensions, scenarios, and other graph objects.
These validators may use a live validation context to check referential
integrity.

Migration note:
- Existing code still imports these helpers from `paths.pydantic`.
- This module is introduced first as scaffolding so "identifier" and "ref"
  can become separate concepts without doing a large migration in one step.
"""

if TYPE_CHECKING:
    from pydantic import ValidationInfo

    from nodes.context import Context
    from nodes.dimensions import Dimension
    from nodes.node import Node


T = TypeVar('T', bound=Hashable)


def _validate_unique_list[T: Hashable](v: list[T]) -> list[T]:
    if len(v) != len(set(v)):
        raise PydanticCustomError('unique_list', 'List must be unique')
    return v


UniqueList = Annotated[list[T], AfterValidator(_validate_unique_list), Field(json_schema_extra={'uniqueItems': True})]


@dataclass(slots=True)
class ValidationContext:
    context: Context
    node: Node | None = None
    dimension: Dimension | None = None


class InvalidContextError(ValueError):
    pass


def get_validation_context(info: ValidationInfo) -> ValidationContext | None:
    if not isinstance(info.context, ValidationContext):
        return None
    return info.context


def require_validation_context(info: ValidationInfo) -> ValidationContext:
    ctx = get_validation_context(info)
    if ctx is None:
        raise InvalidContextError('Context is required')
    return ctx


def require_context(info: ValidationInfo) -> Context:
    return require_validation_context(info).context


def require_node_context(info: ValidationInfo) -> Node:
    ctx = require_validation_context(info)
    if ctx.node is None:
        raise InvalidContextError('Node context is required')
    return ctx.node


def require_dimension_context(info: ValidationInfo) -> Dimension:
    ctx = require_validation_context(info)
    if ctx.dimension is None:
        raise InvalidContextError('Dimension context is required')
    return ctx.dimension


def validate_dimension_id(v: Any, info: ValidationInfo) -> Any:
    ctx = get_validation_context(info)
    if ctx is None:
        return v
    context = ctx.context
    if v not in context.dimensions:
        raise ValueError(f'Dimension with id {v} not found')
    return v


def validate_dimension_category_id(v: str, info: ValidationInfo) -> str:
    ctx = get_validation_context(info)
    if ctx is None:
        return v
    dimension = ctx.dimension
    if dimension is None:
        raise InvalidContextError('Dimension context is required')
    if v not in dimension.cat_map:
        raise ValueError(f'Dimension {dimension.id} does not have a category with id {v}')
    return v


def validate_action_group_ref(v: UUID, info: ValidationInfo) -> UUID:
    ctx = get_validation_context(info)
    if ctx is None:
        return v
    context = ctx.context
    group_uuids = [group.uuid for group in context.instance.action_groups]
    if v not in group_uuids:
        raise ValueError(f'Action group with UUID {v} not found')
    return v


def validate_quantity_kind_ref(v: str, info: ValidationInfo) -> str:
    from nodes.quantities import get_registry

    ctx = get_validation_context(info)
    if ctx is None:
        # Without a validation context, still check the registry
        reg = get_registry()
        if v not in reg:
            raise ValueError(f'Unknown quantity kind: {v!r}')
        return v
    reg = get_registry()
    if v not in reg:
        raise ValueError(f'Unknown quantity kind: {v!r}')
    return v


# -- uuids that point at an entity (see `paths.uuid_kinds`) ---------------------
# A copy rewrites each when its target was copied with it. None is checked against the
# runtime context, which indexes nodes and dimensions by identifier only; `ActionGroupRef`
# is, because action groups are kept by uuid.

NodeRef = Annotated[UUID, Ref('node'), Field(description='The uuid of a node.')]
PortRef = Annotated[UUID, Ref('port')]
ActionGroupRef = Annotated[UUID, Ref('action_group'), AfterValidator(validate_action_group_ref)]
DimensionRef = Annotated[UUID, Ref('dimension')]
DimensionCategoryRef = Annotated[UUID, Ref('category')]
DatasetRef = Annotated[UUID, Ref('dataset')]
DatasetMetricRef = Annotated[UUID, Ref('metric')]
ShapeRef = Annotated[UUID, Ref('shape')]
ShapeCombinationRef = Annotated[UUID, Ref('shape_combination')]
DataSourceRef = Annotated[UUID, Ref('data_source')]
DataEntrySectionRef = Annotated[UUID, Ref('data_entry_section')]
FrameworkRef = Annotated[UUID, Ref('framework')]
FrameworkCategoryRef = Annotated[UUID, Ref('framework_category')]
MeasureTemplateRef = Annotated[UUID, Ref('measure_template')]

InstanceCopyOf = Annotated[UUID, Provenance('instance')]
NodeCopyOf = Annotated[UUID, Provenance('node')]

# -- references by identifier ----------------------------------------------------
# Checked against the runtime context when one is given.

DimensionIdentifierRef = Annotated[DimensionIdentifier, AfterValidator(validate_dimension_id)]
DimensionCategoryIdentifierRef = Annotated[DimensionCategoryIdentifier, AfterValidator(validate_dimension_category_id)]
QuantityKindRef = Annotated[QuantityKindIdentifier, AfterValidator(validate_quantity_kind_ref)]
