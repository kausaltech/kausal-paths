# ruff: noqa: ANN401
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self

import strawberry as sb
from pydantic import BaseModel, Discriminator, Field, RootModel, ValidationInfo, field_validator, model_validator

import polars as pl

from paths.pydantic import (
    DimensionCategoryIdentifier,
    NodeIdentifier,
    NodeOutputDimensionIdentifier,
    NodeOutputMetricIdentifier,
    ScenarioIdentifier,
    UniqueList,
    ValidationContext,
    require_node_context,
    require_validation_context,
)

from common.i18n import I18nBaseModel, I18nStringInstance
from nodes.constants import VALUE_COLUMN

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.metric import DimensionalMetric
    from nodes.node import Node


class VisualizationNodeDimension(BaseModel):
    """Filters the values of the output of a node by dimension."""

    id: NodeOutputDimensionIdentifier
    """The id of the node's output dimension to filter by."""

    categories: UniqueList[DimensionCategoryIdentifier] | None = None
    """Categories to filter by. If not provided, all categories are included."""

    flatten: bool = True
    """If true, the dimension will be summed up and removed from the output."""

    @field_validator('id', mode='after')
    @classmethod
    def validate_dimension_id(cls, id: str, info: ValidationInfo) -> str:
        context = require_validation_context(info)
        context.dimension = context.context.dimensions[id]
        return id

    @model_validator(mode='after')
    def restore_context(self, info: ValidationInfo) -> Self:
        context = require_validation_context(info)
        context.dimension = None
        return self


@sb.enum(description='Desired (benificial) direction for the values of the output of a node')
class DesiredOutcome(str, Enum):
    increasing = 'increasing'
    decreasing = 'decreasing'


class VisualizationKind(str, Enum):
    node = 'node'
    group = 'group'


class VisualizationEntry(I18nBaseModel):
    kind: VisualizationKind
    label: I18nStringInstance | None = None


class VisualizationNodeOutput(VisualizationEntry):
    """Visualization based on the output of a node."""

    kind: Literal[VisualizationKind.node]
    node_id: NodeIdentifier
    desired_outcome: DesiredOutcome
    dimensions: list[VisualizationNodeDimension]
    scenarios: list[ScenarioIdentifier] | None = None
    """Scenarios to include in the visualization."""

    output_metric_id: NodeOutputMetricIdentifier | None = Field(validate_default=True, default=None)
    """The id of the node output metric to use. If not provided, the node's default output metric is used."""

    @field_validator('node_id', mode='after')
    @classmethod
    def add_node_context(cls, node_id: str, info: ValidationInfo) -> str:
        ctx = require_validation_context(info)
        ctx.node = ctx.context.nodes[node_id]
        return node_id

    @field_validator('output_metric_id', mode='after')
    @classmethod
    def validate_output_metric_id(cls, val: str | None, info: ValidationInfo) -> str | None:
        node = require_node_context(info)
        if val is None and node.single_metric_unit is not None:
            metric = node.get_default_output_metric(required=False)
            if metric is None:
                raise ValueError(f'Must provider output metric id for {node.id}')
        return val

    def get_metric(self, node: Node) -> DimensionalMetric:
        from nodes.metric import DimensionalMetric
        assert node.id == self.node_id
        metric = DimensionalMetric.from_visualization(node, self)
        if metric is None:
            raise ValueError(f'Unable to generate metric output for {node.id}')
        return metric

    def get_output(self, node: Node) -> PathsDataFrame:
        df = node.get_output_pl()
        if self.output_metric_id is not None:
            m = node.output_metrics[self.output_metric_id]
            df = df.select_metrics(m.id, rename=VALUE_COLUMN)
        else:
            m = node.get_default_output_metric()

        for dim in self.dimensions:
            if dim.categories is not None:
                df = df.filter(pl.col(dim.id).is_in(dim.categories))
            if dim.flatten:
                df = df.paths.sum_over_dims(dim.id)

        return df


class VisualizationGroup(VisualizationEntry):
    """A grouped set of visualizations."""

    kind: Literal[VisualizationKind.group]
    label: I18nStringInstance | None = None
    children: list[VisualizationEntryType]


type VisualizationEntryType = Annotated[VisualizationGroup | VisualizationNodeOutput, Discriminator('kind')]


class NodeVisualizations(RootModel):
    ValidationContext: ClassVar = ValidationContext
    root: list[VisualizationEntryType] = Field(default_factory=list)
