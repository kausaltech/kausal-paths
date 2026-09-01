import dataclasses
from typing import TYPE_CHECKING, Annotated, Any, Protocol, cast

import strawberry as sb

from kausal_common.strawberry.pydantic import StrawberryPydanticType
from kausal_common.strawberry.registry import register_strawberry_type

from nodes import visualizations as viz
from nodes.metric import (
    DimensionalMetric,
    Metric,
    MetricCategory,
    MetricCategoryGroup,
    MetricDimension,
    MetricDimensionGoal,
    MetricYearlyGoal,
    NormalizerNode,
    YearlyValue,
)

if TYPE_CHECKING:
    from paths.graphql_types import UnitType
    from paths.types import GQLInstanceContext


class YearlyValueProtocol(Protocol):
    year: int
    value: float


@sb.type
class ForecastMetricType:
    id: sb.ID | None
    name: str | None
    unit: Annotated['UnitType', sb.lazy('paths.graphql_types')] | None
    yearly_cumulative_unit: Annotated['UnitType', sb.lazy('paths.graphql_types')] | None

    @sb.field
    @staticmethod
    def historical_values(root: Metric, latest: int | None = None) -> list[YearlyValue]:
        ret = root.get_historical_values()
        if latest:
            if latest >= len(ret):
                return ret
            return ret[-latest:]
        return ret

    @sb.field
    @staticmethod
    def forecast_values(root: Metric) -> list[YearlyValue]:
        return root.get_forecast_values()

    @sb.field
    @staticmethod
    def cumulative_forecast_value(root: Metric) -> float | None:
        return root.get_cumulative_forecast_value()

    @sb.field
    @staticmethod
    def baseline_forecast_values(root: Metric) -> list[YearlyValue] | None:
        return root.get_baseline_forecast_values()


@register_strawberry_type
@sb.experimental.pydantic.type(model=MetricCategory)
class MetricDimensionCategoryType:
    id: sb.ID
    original_id: sb.ID | None
    label: sb.auto
    color: sb.auto
    order: sb.auto
    group: sb.auto


@register_strawberry_type
@sb.experimental.pydantic.type(model=MetricCategoryGroup)
class MetricDimensionCategoryGroupType:
    id: sb.ID
    original_id: sb.ID
    label: sb.auto
    color: sb.auto
    order: sb.auto


@register_strawberry_type
@sb.experimental.pydantic.type(model=MetricDimension)
class MetricDimensionType:
    id: sb.ID
    original_id: sb.ID | None
    label: sb.auto
    help_text: sb.auto
    categories: sb.auto
    groups: sb.auto
    kind: sb.auto


@register_strawberry_type
@sb.experimental.pydantic.type(model=MetricYearlyGoal, all_fields=True)
class MetricYearlyGoalType:
    pass


@register_strawberry_type
@sb.experimental.pydantic.type(model=MetricDimensionGoal, all_fields=True)
class DimensionalMetricGoalEntry:
    pass


@register_strawberry_type
@sb.experimental.pydantic.type(model=NormalizerNode, all_fields=True)
class NormalizerNodeType:
    pass


@register_strawberry_type
@sb.experimental.pydantic.type(model=DimensionalMetric)
class DimensionalMetricType:
    id: sb.ID
    name: sb.auto
    dimensions: sb.auto
    values: sb.auto
    years: sb.auto
    stackable: sb.auto
    forecast_from: sb.auto
    goals: sb.auto
    normalized_by: sb.auto
    unit: Annotated['UnitType', sb.lazy('paths.graphql_types')]
    measure_datapoint_years: sb.auto

    if TYPE_CHECKING:

        @staticmethod
        def from_pydantic(instance: DimensionalMetric, extra: dict[str, Any] | None = None) -> DimensionalMetricType: ...  # pyright: ignore[reportUnusedParameter]


@sb.type
class FlowNodeType:
    id: str
    label: str
    color: str | None


@sb.type
class FlowLinksType:
    year: int
    is_forecast: bool
    sources: list[str]
    targets: list[str]
    values: list[float | None]
    absolute_source_values: list[float]


@sb.type
class DimensionalFlowType:
    id: str
    nodes: list[FlowNodeType]
    unit: Annotated['UnitType', sb.lazy('paths.graphql_types')]
    sources: list[str]
    links: list[FlowLinksType]


@sb.type
class NodeGoal:
    year: int
    value: float


@register_strawberry_type
@sb.experimental.pydantic.interface(model=viz.VisualizationEntry)
class VisualizationEntry(StrawberryPydanticType[viz.VisualizationEntry]):
    id: sb.ID
    kind: viz.VisualizationKind
    label: str | None


@register_strawberry_type
@sb.experimental.pydantic.type(model=viz.VisualizationNodeDimension)
class VisualizationNodeDimension:
    id: str
    categories: list[str] | None
    flatten: bool | None


@register_strawberry_type
@sb.experimental.pydantic.type(model=viz.VisualizationNodeOutput)
class VisualizationNodeOutput(VisualizationEntry):  # type: ignore[override]
    node_id: str
    desired_outcome: viz.DesiredOutcome
    dimensions: list[VisualizationNodeDimension]
    scenarios: list[str] | None = None

    @sb.field(graphql_type=DimensionalMetricType | None)
    def metric_dim(self, info: sb.Info) -> DimensionalMetric | None:
        e = cast('viz.VisualizationNodeOutput', self)
        req = cast('GQLInstanceContext', info.context)
        dm = e.get_metric_data(req.instance.context.nodes[self.node_id])
        if dm is None:
            return None
        return dm

    def to_pydantic(self, **kwargs: Any) -> viz.VisualizationNodeOutput:
        data = dataclasses.asdict(self)  # type: ignore[call-overload]
        data.update(kwargs)
        return viz.VisualizationNodeOutput.model_validate(data)


@register_strawberry_type
@sb.experimental.pydantic.type(model=viz.VisualizationGroup)
class VisualizationGroup(VisualizationEntry):  # type: ignore[override]
    children: list[VisualizationEntry]

    def to_pydantic(self, **kwargs: Any) -> viz.VisualizationGroup:
        data = dataclasses.asdict(self)  # type: ignore[call-overload]
        data.update(kwargs)
        return viz.VisualizationGroup.model_validate(data)
