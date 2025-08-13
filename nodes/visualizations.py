from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self

import strawberry as sb
from pydantic import BaseModel, Discriminator, Field, RootModel, ValidationInfo, field_validator, model_validator

import polars as pl

from paths.pydantic import (
    DimensionCategoryIdentifier,
    InvalidContextError,
    NodeIdentifier,
    NodeOutputDimensionIdentifier,
    NodeOutputMetricIdentifier,
    ScenarioIdentifier,
    UniqueList,
    ValidationContext,
    require_node_context,
)

from common.i18n import I18nBaseModel, I18nStringInstance
from nodes.constants import VALUE_COLUMN

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.context import Context
    from nodes.metric import DimensionalMetric
    from nodes.node import Node


@dataclass(slots=True)
class VisualizationValidationContext(ValidationContext):
    _: KW_ONLY
    root_node: Node
    id_counter: int = field(default=0)


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


def require_validation_context(info: ValidationInfo) -> VisualizationValidationContext:
    if not isinstance(info.context, VisualizationValidationContext):
        raise InvalidContextError('Context is required')
    return info.context


AUTO_ID = 'auto'


class VisualizationEntry(I18nBaseModel):
    id: str = Field(default=AUTO_ID)
    kind: VisualizationKind
    label: I18nStringInstance | None = None

    def make_id(self, ctx: VisualizationValidationContext) -> str:
        s = f'{ctx.root_node.id}:{ctx.id_counter}'
        ctx.id_counter += 1
        return s

    @model_validator(mode='after')
    def set_id(self, info: ValidationInfo) -> Self:
        if self.id == AUTO_ID:
            ctx = require_validation_context(info)
            self.id = self.make_id(ctx)
        return self


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

    def get_metric_data(self, node: Node) -> DimensionalMetric | None:
        from nodes.metric import DimensionalMetric
        assert node.id == self.node_id
        metric = DimensionalMetric.from_visualization(node, self)
        if metric is None:
            return None
        return metric

    def get_measure_datapoint_years(self, node: Node) -> list[int]:
        if hasattr(node, 'get_measure_datapoint_years'):
            print('test for datapoints in node', node.id)
            if hasattr(node, 'get_measure_datapoint_numbers'):
                print(node.get_measure_datapoint_numbers())
            return node.get_measure_datapoint_years()
        return []

    def get_output(self, node: Node) -> PathsDataFrame:
        df = node.get_output_pl()
        if self.output_metric_id is not None:
            m = node.output_metrics[self.output_metric_id]
            df = df.select_metrics(m.id, rename=VALUE_COLUMN)  # FIXME Shouldn't this be after the if statement?
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
    ValidationContext: ClassVar = VisualizationValidationContext
    root: list[VisualizationEntryType] = Field(default_factory=list)

    def _plot_recursive(self, context: Context, viz: VisualizationEntryType, charts: list) -> None:
        from nodes.metric import DimensionalMetric

        if isinstance(viz, VisualizationNodeOutput):
            metric = DimensionalMetric.from_visualization(context.nodes[viz.node_id], viz)
            assert metric is not None
            charts.append(metric.plot())
            return
        if isinstance(viz, VisualizationGroup):
            for child in viz.children:
                self._plot_recursive(context, child, charts)

    def plot(self, context: Context):
        import altair as alt  # type: ignore # noqa: TC002

        charts: list[alt.Chart] = []
        for viz in self.root:
            self._plot_recursive(context, viz, charts)
        return charts
