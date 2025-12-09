from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, TypeVar

from pydantic import AfterValidator, Field
from pydantic_core import PydanticCustomError

if TYPE_CHECKING:
    from pydantic import ValidationInfo

    from nodes.context import Context
    from nodes.dimensions import Dimension
    from nodes.node import Node


T = TypeVar('T', bound=Hashable)

def _validate_unique_list(v: list[T]) -> list[T]:
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


def require_validation_context(info: ValidationInfo) -> ValidationContext:
    if not isinstance(info.context, ValidationContext):
        raise InvalidContextError('Context is required')
    return info.context


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


def validate_node_id(v: str, info: ValidationInfo) -> str:
    context = require_context(info)
    if v not in context.nodes:
        raise ValueError(f'Node with id {v} not found')
    return v


def validate_scenario_id(v: str, info: ValidationInfo) -> str:
    context = require_context(info)
    if v not in context.scenarios:
        raise ValueError(f'Scenario with id {v} not found')
    return v




def validate_node_output_metric_id(v: str, info: ValidationInfo) -> str:
    node = require_node_context(info)
    if v not in node.output_metrics:
        raise ValueError(f'Metric with id {v} not found')
    return v


def validate_dimension_id(v: Any, info: ValidationInfo) -> Any:
    context = require_context(info)
    if v not in context.dimensions:
        raise ValueError(f'Dimension with id {v} not found')
    return v


def validate_node_output_dimension_id(v: str, info: ValidationInfo) -> str:
    node = require_node_context(info)
    if v not in node.output_dimensions:
        raise ValueError(f'Node {node.id} does not have a dimension with id {v}')
    return v


def validate_dimension_category_id(v: str, info: ValidationInfo) -> str:
    dimension = require_dimension_context(info)
    if v not in dimension.cat_map:
        raise ValueError(f'Dimension {dimension.id} does not have a category with id {v}')
    return v


NodeIdentifier = Annotated[str, AfterValidator(validate_node_id)]
ScenarioIdentifier = Annotated[str, AfterValidator(validate_scenario_id)]
NodeOutputMetricIdentifier = Annotated[str, AfterValidator(validate_node_output_metric_id)]
DimensionIdentifier = Annotated[str, AfterValidator(validate_dimension_id)]
NodeOutputDimensionIdentifier = Annotated[str, AfterValidator(validate_node_output_dimension_id)]
DimensionCategoryIdentifier = Annotated[str, AfterValidator(validate_dimension_category_id)]
