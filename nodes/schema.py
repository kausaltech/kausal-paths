import dataclasses
import re
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Protocol, cast
from uuid import UUID

import strawberry as sb
from graphql.error import GraphQLError
from strawberry import auto
from wagtail.blocks.stream_block import StreamValue
from wagtail.rich_text import RichText as WagtailRichText, expand_db_html

import sentry_sdk
from grapple.types.streamfield import StreamFieldInterface
from loguru import logger
from markdown_it import MarkdownIt

from kausal_common.strawberry.grapple import grapple_field
from kausal_common.strawberry.pydantic import StrawberryPydanticType, pydantic_type
from kausal_common.strawberry.registry import register_strawberry_type

from paths import gql
from paths.const import INSTANCE_CHANGE_GROUP, INSTANCE_CHANGE_TYPE
from paths.graphql_helpers import ensure_instance, get_instance_context, pass_context
from paths.graphql_types import UnitType

from nodes.defs import InstanceSpec, SimpleConfig
from nodes.defs.binding_def import DatasetPortBindingDef
from nodes.defs.instance_defs import ActionGroup, InstanceFeatures
from nodes.defs.node_defs import ActionConfig, FormulaConfig, NodeKind, NodeSpec, PipelineConfig
from nodes.goals import GoalActualValue, NodeGoalsEntry
from nodes.graph_layout import GraphLayout, NodeGraphLayoutMeta
from nodes.scenario import Scenario, ScenarioKind
from nodes.schema_spec import (
    InputPortType,
    InstanceSpecType,
    OutputPortType,
    YearsDefType,
)
from pages.models import ActionListPage
from params import Parameter

from . import visualizations as viz
from .constants import FORECAST_COLUMN, IMPACT_COLUMN, IMPACT_GROUP, YEAR_COLUMN, DecisionLevel
from .instance import Instance
from .metric import (
    DimensionalFlow,
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
from .models import InstanceConfig
from .quantities import get_registry as get_quantity_registry
from .units import Unit

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset as DatasetModel, DatasetMetric

    from paths.types import GQLInstanceContext

    from common import polars as ppl
    from nodes.actions.action import ImpactOverview
    from nodes.defs.binding_def import EdgeBindingDef
    from nodes.node import Node  # noqa: TC004
    from nodes.normalization import Normalization
    from nodes.quantities import QuantityKind
    from params.schema import ParameterInterface

    from .actions.action import ActionNode
    from .context import Context
    from .models import NodeConfig, NodeEdge


logger = logger.bind(name='nodes.schema')
markdown = MarkdownIt('commonmark', {'html': True})


@sb.type
class QuantityKindType:
    id: str
    label: str
    icon: str | None
    qudt_iri: str | None
    is_stackable: bool
    is_activity: bool
    is_factor: bool
    is_unit_price: bool

    @classmethod
    def from_kind(cls, kind: QuantityKind) -> QuantityKindType:
        return cls(
            id=kind.id,
            label=str(kind.label),
            icon=kind.icon,
            qudt_iri=kind.qudt_iri,
            is_stackable=kind.is_stackable,
            is_activity=kind.is_activity,
            is_factor=kind.is_factor,
            is_unit_price=kind.is_unit_price,
        )


@sb.type
class InstanceHostname:
    hostname: str
    base_path: str


@sb.type
class NodePortRef:
    node_id: sb.ID
    port_id: UUID


@sb.type
class NodeEdgeType:
    id: sb.ID
    from_ref: NodePortRef
    to_ref: NodePortRef
    transformations: sb.scalars.JSON
    tags: list[str]

    @classmethod
    def from_binding(cls, binding: EdgeBindingDef) -> NodeEdgeType:
        return NodeEdgeType(
            id=sb.ID(str(binding.id)),
            from_ref=NodePortRef(node_id=sb.ID(str(binding.from_ref.node_id)), port_id=binding.from_ref.port_id),
            to_ref=NodePortRef(node_id=sb.ID(str(binding.to_ref.node_id)), port_id=binding.to_ref.port_id),
            transformations=sb.scalars.JSON([]),
            tags=binding.tags,
        )

    @classmethod
    def from_node_edge(cls, edge: NodeEdge) -> NodeEdgeType:
        return NodeEdgeType(
            id=sb.ID(str(edge.uuid)),
            from_ref=NodePortRef(node_id=sb.ID(str(edge.from_node.identifier)), port_id=edge.from_port),
            to_ref=NodePortRef(node_id=sb.ID(str(edge.to_node.identifier)), port_id=edge.to_port),
            transformations=edge.transformations,
            tags=edge.tags or [],
        )


@sb.type
class DatasetExternalRefType:
    """Stable source reference for an externally backed dataset."""

    repo_url: str = sb.field(description='URL of the external dataset repository.')
    commit: str | None = sb.field(description='Repository commit used for this dataset snapshot.')
    dataset_id: str = sb.field(description='Path-like identifier of the dataset inside the external repository.')


@sb.type
class DatasetRefType:
    """Lightweight reference to a dataset object bound in the model."""

    id: sb.ID = sb.field(description='Globally unique identifier of the dataset object.')
    identifier: str | None = sb.field(description='Scoped identifier of the dataset object.')
    is_external_placeholder: bool = sb.field(
        description='Whether the dataset object is only a placeholder without imported datapoints.'
    )
    external_ref: DatasetExternalRefType | None = sb.field(
        description='External source reference for externally backed datasets.'
    )

    @classmethod
    def from_binding(cls, binding: DatasetPortBindingDef) -> DatasetRefType | None:
        if binding.dataset_uuid is None:
            return None
        return DatasetRefType(
            id=sb.ID(str(binding.dataset_uuid)),
            identifier=_external_dataset_id_from_dataset(binding),
            is_external_placeholder=binding.dataset_is_external_placeholder,
            external_ref=_dataset_external_ref_to_gql(binding.dataset_external_ref),
        )

    @classmethod
    def from_model(cls, dataset: DatasetModel) -> DatasetRefType:
        return DatasetRefType(
            id=sb.ID(str(dataset.uuid)),
            identifier=dataset.identifier,
            is_external_placeholder=dataset.is_external_placeholder,
            external_ref=_dataset_external_ref_to_gql(dataset.external_ref),
        )


@sb.type
class DatasetMetricRefType:
    """Lightweight reference to a dataset metric object bound in the model."""

    id: sb.ID = sb.field(description='Globally unique identifier of the dataset metric object.')
    name: str | None = sb.field(description='Stable identifier of the metric within its dataset schema.')
    label: str = sb.field(description='Human-readable label of the metric.')

    @classmethod
    def from_binding(cls, binding: DatasetPortBindingDef) -> DatasetMetricRefType | None:
        if binding.metric_uuid is None:
            return None
        return DatasetMetricRefType(
            id=sb.ID(str(binding.metric_uuid)),
            name=binding.external_metric_id,
            label=binding.external_metric_id or '',
        )

    @classmethod
    def from_model(cls, metric: DatasetMetric) -> DatasetMetricRefType:
        return DatasetMetricRefType(
            id=sb.ID(str(metric.uuid)),
            name=metric.name,
            label=metric.label_i18n,
        )


@pydantic_type(DatasetPortBindingDef)
class DatasetPortType:
    """Binding of an external dataset metric to one node input port."""

    id: sb.ID = sb.field(description='Globally unique identifier of this dataset-port binding.')
    node_ref: NodePortRef = sb.field(description='Reference to the node that owns the bound input port.')
    dataset: DatasetRefType | None = sb.field(description='Dataset object bound to this port.')
    metric: DatasetMetricRefType | None = sb.field(description='Dataset metric object bound to this port.')
    external_dataset_id: str | None = sb.field(
        description='Stable identifier of the external dataset, usually the dataset repo path without extension.',
    )
    external_metric_id: str | None = sb.field(
        description='Stable identifier of the metric within the external dataset.',
    )

    @classmethod
    def from_binding(cls, binding: DatasetPortBindingDef) -> DatasetPortType:
        return DatasetPortType(
            id=sb.ID(str(binding.id)),
            node_ref=NodePortRef(node_id=sb.ID(str(binding.node_ref.node_id)), port_id=binding.node_ref.port_id),
            dataset=DatasetRefType.from_binding(binding),
            metric=DatasetMetricRefType.from_binding(binding),
            external_dataset_id=binding.external_dataset_id,
            external_metric_id=binding.external_metric_id,
        )


InputPortBinding = Annotated[NodeEdgeType | DatasetPortType, sb.union('InputPortBindingUnion')]


@sb.type
class ActionGroupType:
    id: sb.ID
    name: str
    color: str | None

    @sb.field(graphql_type=list[Annotated['ActionNodeType', sb.lazy('nodes.schema')]])
    @pass_context
    @staticmethod
    def actions(root: ActionGroup, context: 'Context') -> list['ActionNode']:
        return [act for act in context.get_actions() if act.group == root]


@sb.experimental.pydantic.type(
    model=InstanceFeatures,
    all_fields=True,
    name='InstanceFeaturesType',
    description=InstanceFeatures.__doc__.strip() if InstanceFeatures.__doc__ else None,
)
class InstanceFeaturesType:
    pass


@sb.experimental.pydantic.type(model=GoalActualValue, name='InstanceYearlyGoalType')
class InstanceYearlyGoalType(StrawberryPydanticType[GoalActualValue]):
    year: auto
    goal: auto
    actual: auto
    is_interpolated: auto
    is_forecast: auto


class InstanceGoalDimensionProtocol(Protocol):
    dimension: str
    categories: list[str]
    groups: list[str]


@sb.type
class InstanceGoalDimension:
    dimension: str
    categories: list[str]
    groups: list[str]

    @sb.field(deprecation_reason='replaced with categories')
    @staticmethod
    def category(root: InstanceGoalDimensionProtocol) -> str:
        return root.categories[0]


@sb.type
class InstanceGoalEntry:
    id: sb.ID
    label: str | None
    disabled: bool
    disable_reason: str | None
    outcome_node: 'Node' = sb.field(graphql_type=Annotated['NodeType', sb.lazy('nodes.schema')])
    dimensions: list[InstanceGoalDimension]
    default: bool

    _goal: sb.Private[NodeGoalsEntry]

    @sb.field
    def values(self) -> list[InstanceYearlyGoalType]:
        actual_values = self._goal.get_actual()
        return [InstanceYearlyGoalType.from_pydantic(x) for x in actual_values]

    @sb.field(graphql_type=UnitType)
    def unit(self) -> Unit:
        df = self._goal._get_values_df()
        return df.get_unit(self.outcome_node.get_default_output_metric().column_id)


@sb.type
class InstanceType:
    id: sb.ID
    uuid: UUID
    name: str
    owner: str | None
    default_language: str
    supported_languages: list[str]
    base_path: str
    years: YearsDefType
    target_year: int | None
    model_end_year: int
    reference_year: int | None
    minimum_historical_year: int
    maximum_historical_year: int | None
    theme_identifier: str | None
    action_groups: list[ActionGroupType]
    features: InstanceFeaturesType

    @sb.field(graphql_type=InstanceHostname | None)
    @staticmethod
    def hostname(root: Instance, hostname: str) -> InstanceHostname | None:
        hn = root.config.hostnames.filter(hostname__iexact=hostname).first()
        if not hn:
            return None
        return InstanceHostname(hostname=hn.hostname, base_path=hn.base_path)

    @sb.field
    @staticmethod
    def lead_title(root: Instance) -> str:
        return root.config.lead_title_i18n or ''

    @sb.field
    @staticmethod
    def lead_paragraph(root: Instance) -> str | None:
        return root.config.lead_paragraph_i18n

    @sb.field
    @staticmethod
    def identifier(root: Instance) -> str:
        return root.id

    @sb.field
    @staticmethod
    def config_source(root: Instance) -> str:
        return root.config.config_source

    @sb.field
    @staticmethod
    def live(root: Instance) -> bool:
        return root.config.live

    @sb.field
    @staticmethod
    def has_unpublished_changes(root: Instance) -> bool:
        return root.config.has_unpublished_changes

    @sb.field
    @staticmethod
    def first_published_at(root: Instance) -> datetime | None:
        return root.config.first_published_at

    @sb.field
    @staticmethod
    def last_published_at(root: Instance) -> datetime | None:
        return root.config.last_published_at

    @sb.field
    @staticmethod
    def goals(root: Instance, id: sb.ID | None = None) -> list[InstanceGoalEntry]:
        ret = []
        for goal in root.get_goals():
            node = goal.get_node()
            goal_id = goal.get_id()
            if id is not None and goal_id != id:
                continue

            dims = []
            for dim_id, path in goal.dimensions.items():
                dims.append(
                    InstanceGoalDimension(
                        dimension=dim_id,
                        categories=path.categories,
                        groups=path.groups,
                    )
                )

            out = InstanceGoalEntry(
                id=sb.ID(goal_id),
                label=str(goal.label) if goal.label else str(node.name),
                outcome_node=node,
                dimensions=dims,
                default=goal.default,
                disabled=goal.disabled,
                disable_reason=str(goal.disable_reason),
                _goal=goal,
            )
            out._goal = goal
            ret.append(out)
        return ret

    @grapple_field
    @staticmethod
    def action_list_page(root: Instance) -> ActionListPage | None:
        return root.config.action_list_page

    @sb.field(graphql_type=list[StreamFieldInterface] | None)
    @staticmethod
    def intro_content(root: Instance) -> StreamValue:
        return root.config.site_content.intro_content

    @sb.field(graphql_type=Annotated[InstanceSpecType | None, sb.lazy('nodes.schema_spec')])
    @staticmethod
    def spec(root: Instance, info: gql.Info) -> InstanceSpec | None:
        ic = root.config
        if not ic.gql_action_allowed(info, 'change'):
            return None
        return ic.spec

    @sb.field(graphql_type=list[NodeEdgeType])
    @staticmethod
    def edges(root: Instance) -> list[NodeEdgeType]:
        edges = root.config.edges.select_related('from_node', 'to_node')
        return [NodeEdgeType.from_node_edge(edge) for edge in edges]

    @sb.field(graphql_type=list[DatasetPortType])
    @staticmethod
    def dataset_ports(root: Instance) -> list[DatasetPortType]:
        dataset_ports = root.config.dataset_ports.select_related('node', 'dataset', 'metric')
        return [
            DatasetPortType(
                id=sb.ID(str(dp.uuid)),
                node_ref=NodePortRef(node_id=sb.ID(str(dp.node.identifier)), port_id=dp.port_id),
                dataset=DatasetRefType.from_model(dp.dataset),
                metric=DatasetMetricRefType.from_model(dp.metric),
                external_dataset_id=_external_dataset_id_from_dataset(dp.dataset),
                external_metric_id=dp.metric.name,
            )
            for dp in dataset_ports
        ]

    @sb.field(graphql_type=list[Annotated['NodeInterface', sb.lazy('nodes.schema')]])
    @staticmethod
    def nodes(root: Instance, id: list[sb.ID] | None = None) -> list['Node']:
        if id is not None:
            nodes: list['Node'] = []
            for obj_id in id:
                node = root.context.nodes.get(obj_id)
                if node is None:
                    continue
                nodes.append(node)
            return nodes
        return sorted(
            root.context.nodes.values(),
            key=lambda node: (node.order is None, node.order or 0, node.id),
        )

    @sb.field
    @staticmethod
    def graph_layout(root: Instance) -> GraphLayout:
        classifier = root.context.node_graph_classifier
        return GraphLayout(
            thresholds=classifier.thresholds,
            core_node_ids=[sb.ID(node_id) for node_id in classifier.core_nodes],
            ghostable_context_source_ids=[sb.ID(node_id) for node_id in classifier.ghostable_context_sources],
            hub_ids=[sb.ID(node_id) for node_id in classifier.hubs],
            action_ids=[sb.ID(node_id) for node_id in classifier.actions],
            outcome_ids=[sb.ID(node_id) for node_id in classifier.outcomes],
            main_graph_node_ids=[sb.ID(node_id) for node_id in classifier.main_graph_node_ids],
        )


class YearlyValueProtocol(Protocol):
    year: int
    value: float


@sb.type
class ForecastMetricType:
    id: sb.ID | None
    name: str | None
    unit: UnitType | None
    yearly_cumulative_unit: UnitType | None

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
    label: auto
    color: auto
    order: auto
    group: auto


@register_strawberry_type
@sb.experimental.pydantic.type(model=MetricCategoryGroup)
class MetricDimensionCategoryGroupType:
    id: sb.ID
    original_id: sb.ID
    label: auto
    color: auto
    order: auto


@register_strawberry_type
@sb.experimental.pydantic.type(model=MetricDimension)
class MetricDimensionType:
    id: sb.ID
    original_id: sb.ID | None
    label: auto
    help_text: auto
    categories: auto
    groups: auto
    kind: auto


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


@sb.type
class ScenarioParameterOverrideType:
    parameter_id: str
    value: sb.scalars.JSON


@register_strawberry_type
@sb.experimental.pydantic.type(
    model=DimensionalMetric,
)
class DimensionalMetricType:
    id: sb.ID
    name: auto
    dimensions: auto
    values: auto
    years: auto
    stackable: auto
    forecast_from: auto
    goals: auto
    normalized_by: auto
    unit: Annotated[UnitType, sb.lazy('paths.graphql_types')]
    measure_datapoint_years: auto

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
    unit: UnitType
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
        req: 'GQLInstanceContext' = info.context
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


@pydantic_type(model=Scenario)
class ScenarioType:
    id: sb.ID
    name: auto
    kind: auto
    all_actions_enabled: bool
    is_selectable: auto
    actual_historical_years: auto

    @sb.field
    @staticmethod
    def identifier(root: Scenario) -> str:
        return root.id

    @sb.field
    @staticmethod
    def description(root: Scenario) -> str | None:
        return str(root.description) if root.description is not None else None

    @sb.field
    @staticmethod
    def parameter_overrides(root: Scenario) -> list[ScenarioParameterOverrideType]:
        return [
            ScenarioParameterOverrideType(parameter_id=param_id, value=cast('sb.scalars.JSON', value))
            for param_id, value in root.param_values.items()
        ]

    @sb.field
    @pass_context
    @staticmethod
    def is_active(root: Scenario, context: 'Context') -> bool:
        return context.active_scenario == root

    @sb.field
    @staticmethod
    def is_default(root: Scenario) -> bool:
        return root.default


@sb.type
class ScenarioValue:
    scenario: ScenarioType
    value: float | None
    year: int


@sb.type
class MetricDimensionCategoryValue:
    dimension: MetricDimensionType
    category: MetricDimensionCategoryType
    value: float | None
    year: int


@sb.type
class ActionImpactType:
    action: 'ActionNodeType'
    value: float
    year: int


@sb.type
class ScenarioActionImpacts:
    scenario: ScenarioType
    impacts: list[ActionImpactType]


def _get_impact_metric(source_node: ActionNode, target_node: Node, goal: NodeGoalsEntry | None = None) -> Metric | None:
    import polars as pl

    df: ppl.PathsDataFrame = source_node.compute_impact(target_node)
    if goal is not None:
        df = goal.filter_df(df)

    df = df.filter(pl.col(IMPACT_COLUMN).eq(IMPACT_GROUP)).drop(IMPACT_COLUMN)
    if df.dim_ids:
        # FIXME: Check if can be summed?
        df = df.paths.sum_over_dims()

    try:
        m = target_node.get_default_output_metric()
    except Exception:
        return None

    df = df.select([*df.primary_keys, FORECAST_COLUMN, m.column_id])
    active_normalization = target_node.context.active_normalization
    if active_normalization and active_normalization.get_normalized_unit(m) is not None:
        _, df = active_normalization.normalize_output(m, df)

    metric = Metric(
        id='%s-%s-impact' % (source_node.id, target_node.id),
        name='Impact',
        df=df,
        unit=df.get_unit(m.column_id),
    )
    return metric


@pydantic_type(model=ActionConfig)
class ActionConfigType:
    node_class: auto
    decision_level: auto
    group: auto
    parent: auto
    no_effect_value: auto


@pydantic_type(model=SimpleConfig)
class SimpleConfigType:
    node_class: auto


@pydantic_type(model=FormulaConfig)
class FormulaConfigType:
    formula: str


@pydantic_type(model=PipelineConfig)
class PipelineConfigType:
    operations: sb.scalars.JSON  # FIXME


def _dataset_external_ref_to_gql(external_ref: object) -> DatasetExternalRefType | None:
    if not isinstance(external_ref, dict):
        return None
    repo_url = external_ref.get('repo_url')
    dataset_id = external_ref.get('dataset_id')
    if not isinstance(repo_url, str) or not isinstance(dataset_id, str):
        return None
    commit = external_ref.get('commit')
    return DatasetExternalRefType(
        repo_url=repo_url,
        commit=commit if isinstance(commit, str) else None,
        dataset_id=dataset_id,
    )


def _external_dataset_id_from_dataset(dataset: DatasetModel | DatasetPortBindingDef) -> str | None:
    if isinstance(dataset, DatasetPortBindingDef):
        external_ref = dataset.dataset_external_ref
        if isinstance(external_ref, dict):
            dataset_id = external_ref.get('dataset_id')
            if isinstance(dataset_id, str):
                return dataset_id
        return dataset.external_dataset_id

    external_ref = dataset.external_ref
    if isinstance(external_ref, dict):
        dataset_id = external_ref.get('dataset_id')
        if isinstance(dataset_id, str):
            return dataset_id
    return dataset.identifier


def _require_nc(spec_type: NodeSpecType) -> NodeConfig:
    if spec_type._node is None:
        raise ValueError('NodeSpecType has no Node instance')
    nc = spec_type._node.db_obj
    if nc is None:
        raise ValueError('NodeSpecType has no NodeConfig instance')
    return nc


@pydantic_type(model=NodeSpec)
class NodeSpecType(StrawberryPydanticType[NodeSpec]):
    type_config: Annotated[
        ActionConfigType | SimpleConfigType | FormulaConfigType | PipelineConfigType, sb.union('NodeConfigUnion')
    ]

    _node: sb.Private['Node | None'] = None

    @sb.field
    @staticmethod
    def input_ports(root: 'NodeSpecType') -> list[InputPortType]:
        nc = _require_nc(root)
        spec = root._original_model
        edge_bindings = nc.port_edge_bindings
        dataset_bindings = nc.port_dataset_bindings
        port_objs = []
        edges_by_id: dict[UUID, list[NodeEdgeType | DatasetPortType]] = {}
        for edge in edge_bindings:
            if edge.to_ref.node_id != spec.identifier:
                continue
            sb_edge = NodeEdgeType.from_binding(edge)
            edges_by_id.setdefault(edge.to_ref.port_id, []).append(sb_edge)
        for dataset in dataset_bindings:
            sb_dataset = DatasetPortType.from_binding(dataset)
            assert sb_dataset.node_ref.node_id == spec.identifier
            edges_by_id.setdefault(dataset.node_ref.port_id, []).append(sb_dataset)
        for port in spec.input_ports:
            edges = edges_by_id.get(port.id, [])

            # edges = [NodeEdgeType.from_binding(binding) for binding in edge_bindings if binding.to_port == port.id]
            # datasets = [DatasetPortType.from_binding(binding) for binding in dataset_bindings if binding.port_id == port.id]
            port_obj = InputPortType.from_def(
                port,
                bindings=edges,
            )
            port_objs.append(port_obj)
        return port_objs

    @sb.field(graphql_type=list[OutputPortType])
    @staticmethod
    def output_ports(root: 'NodeSpecType') -> list[OutputPortType]:
        nc = _require_nc(root)
        edge_bindings = nc.port_edge_bindings
        spec = root._original_model
        return [
            OutputPortType.from_def(
                port,
                edges=[
                    NodeEdgeType.from_binding(binding)
                    for binding in edge_bindings
                    if binding.from_ref.port_id == port.id and binding.from_ref.node_id == spec.identifier
                ],
            )
            for port in spec.output_ports
        ]


@sb.interface
class NodeInterface:
    id: sb.ID
    short_name: str | None
    order: int | None
    unit: UnitType | None
    quantity: str | None

    input_nodes: list['NodeInterface']
    output_nodes: list['NodeInterface']

    @sb.field
    @staticmethod
    def quantity_kind(root: 'Node') -> QuantityKindType | None:
        if root.quantity is None:
            return None
        registry = get_quantity_registry()
        kind = registry.get(root.quantity)
        if kind is None:
            return None
        return QuantityKindType.from_kind(kind)

    @sb.field
    @staticmethod
    def name(root: 'Node') -> str:
        nc = root.db_obj
        if nc is not None and nc.name_i18n:
            return nc.name_i18n
        return str(root.name)

    @sb.field
    @staticmethod
    def kind(root: 'Node') -> NodeKind | None:
        if root._spec is None:
            return None
        return root.spec.kind

    @sb.field
    @staticmethod
    def identifier(root: 'Node') -> str:
        return root.id

    @sb.field
    @staticmethod
    def spec(root: 'Node', info: gql.Info) -> Annotated[NodeSpecType | None, sb.lazy('nodes.schema_spec')]:
        nc = root.db_obj
        if nc is None:
            return None
        if not nc.gql_action_allowed(info, 'change'):
            return None
        if root.has_spec:
            spec_type = NodeSpecType.from_pydantic(root.spec)
        else:
            spec_type = NodeSpecType.from_pydantic(nc.spec)
            root._spec = nc.spec
        spec_type._node = root
        return spec_type

    @sb.field
    @staticmethod
    def color(root: 'Node') -> str | None:
        nc = root.db_obj
        if nc and nc.color:
            return nc.color
        if root.color:
            return root.color
        if root.quantity == 'emissions':
            for parent in root.output_nodes:
                if parent.color:
                    root.color = parent.color
                    return root.color
        return None

    @sb.field
    @staticmethod
    def is_visible(root: 'Node') -> bool:
        nc = root.db_obj
        if nc and nc.is_visible:
            return nc.is_visible
        return root.is_visible

    @sb.field
    @staticmethod
    def uuid(root: 'Node') -> str | None:
        nc = root.db_obj
        if nc is None:
            return None
        return str(nc.uuid)

    @sb.field
    @staticmethod
    def node_group(root: 'Node') -> str | None:
        nc = root.db_obj
        if nc is not None and nc.spec.node_group is not None:
            return nc.spec.node_group
        return root.node_group

    @sb.field
    @staticmethod
    def layout_meta(root: 'Node') -> NodeGraphLayoutMeta:
        return root.context.node_graph_classifier.for_node(root.id)

    @sb.field(deprecation_reason='Replaced by "goals".')
    @staticmethod
    def target_year_goal(root: 'Node') -> float | None:
        if root.goals is None:
            return None
        goal = root.goals.get_dimensionless()
        if not goal:
            return None
        target_year = root.context.target_year
        vals = goal.get_values()
        for val in vals:
            if val.year == target_year:
                break
        else:
            return None
        return val.value

    @sb.field
    @pass_context
    @staticmethod
    def goals(root: 'Node', context: 'Context', active_goal: sb.ID | None = None) -> list[NodeGoal]:
        if root.goals is None:
            return []
        instance = context.instance
        goal = None
        if active_goal:
            agoal = instance.get_goals(active_goal)
            if agoal.dimensions:
                # FIXME
                dim_id, cats = next(iter(agoal.dimensions.items()))
                goal = root.goals.get_exact_match(
                    dim_id,
                    groups=cats.groups,
                    categories=cats.categories,
                )
        if not goal:
            goal = root.goals.get_dimensionless()
        if not goal:
            return []
        return [NodeGoal(year=val.year, value=val.value) for val in goal.get_values()]

    @sb.field(deprecation_reason='Use __typeName instead')
    @staticmethod
    def is_action(root: 'Node') -> bool:
        from nodes.actions.action import ActionNode

        return isinstance(root, ActionNode)

    @sb.field
    @pass_context
    @staticmethod
    def explanation(root: 'Node', context: 'Context') -> str | None:
        if context.instance.features.show_explanations:
            return None
        return root.get_explanation()

    @sb.field
    @staticmethod
    def node_type(root: 'Node') -> str:
        typ = str(type(root))
        typstr = re.search(r"'([^']*)'", typ)
        if typstr is not None:
            return typstr.group(1)
        return ''

    @sb.field
    @staticmethod
    def tags(root: 'Node') -> list[str] | None:
        return list(root.tags)

    @sb.field
    @staticmethod
    def input_dimensions(root: 'Node') -> list[str] | None:
        return list(root.input_dimensions.keys())

    @sb.field
    @staticmethod
    def output_dimensions(root: 'Node') -> list[str] | None:
        return list(root.output_dimensions.keys())

    @sb.field(graphql_type=list[VisualizationEntry] | None)
    @staticmethod
    def visualizations(root: 'Node') -> list[viz.VisualizationEntryType] | None:
        node_viz = root.visualizations
        if not node_viz:
            return None
        return node_viz.root

    @sb.field(graphql_type=list['NodeInterface'])
    @staticmethod
    def downstream_nodes(
        root: 'Node',
        info: gql.Info,
        max_depth: int | None = None,
        only_outcome: bool = False,
        until_node: sb.ID | None = None,
    ) -> list['Node']:
        info.context._upstream_node = root  # type: ignore
        if until_node is not None:
            try:
                to_node = root.context.get_node(until_node)
            except KeyError:
                raise GraphQLError('Node %s not found' % until_node, info.field_nodes) from None
        else:
            to_node = None
        return root.get_downstream_nodes(max_depth=max_depth, only_outcome=only_outcome, until_node=to_node)

    @sb.field(graphql_type=list['NodeInterface'])
    @staticmethod
    def upstream_nodes(
        root: 'Node',
        same_unit: bool = False,
        same_quantity: bool = False,
        include_actions: bool = True,
    ) -> list['Node']:
        from nodes.actions.action import ActionNode

        def filter_nodes(node: Node) -> bool:
            if same_unit and root.unit != node.unit:
                return False
            if same_quantity and root.quantity != node.quantity:
                return False
            if not include_actions and isinstance(node, ActionNode):
                return False
            return True

        return root.get_upstream_nodes(filter_func=filter_nodes)

    # TODO: Many nodes will output multiple time series. Remove metric
    # and handle a single-metric node as a special case in the UI??
    @sb.field(graphql_type=ForecastMetricType | None)
    @staticmethod
    def metric(root: 'Node', goal_id: sb.ID | None = None) -> Metric | None:
        return Metric.from_node(root, goal_id=goal_id)

    @sb.field(graphql_type=DimensionalMetricType | None)
    @staticmethod
    def outcome(root: 'Node') -> DimensionalMetric | None:
        return getattr(root, 'outcome', None)

    # If resolving through `descendant_nodes`, `impact_metric` will be
    # by default be calculated from the ancestor node.
    @sb.field(graphql_type=ForecastMetricType | None)
    @pass_context
    @staticmethod
    def impact_metric(
        root: 'Node',
        info: gql.Info,
        context: 'Context',
        target_node_id: sb.ID | None = None,
        goal_id: sb.ID | None = None,
    ) -> Metric | None:
        from nodes.actions.action import ActionNode

        instance = context.instance
        upstream_node: Node | None = getattr(info.context, '_upstream_node', None)

        if goal_id is not None:
            try:
                goal = instance.get_goals(goal_id=goal_id)
            except Exception:
                raise GraphQLError('Goal not found', info.field_nodes) from None
        else:
            goal = None

        target_node: 'Node'
        if target_node_id is not None:
            if target_node_id not in context.nodes:
                raise GraphQLError('Node %s not found' % target_node_id, info.field_nodes)
            source_node = root
            target_node = context.get_node(target_node_id)
        elif upstream_node is not None:
            source_node = upstream_node
            target_node = root
        elif goal is not None:
            source_node = root
            target_node = goal.get_node()
        else:
            # FIXME: Determine a "default" target node from instance
            outcome_nodes = context.get_outcome_nodes()
            if not len(outcome_nodes):
                raise GraphQLError('No default target node available', info.field_nodes)
            source_node = root
            target_node = outcome_nodes[0]

        if not isinstance(source_node, ActionNode):
            return None

        return _get_impact_metric(source_node, target_node, goal)

    @sb.field(graphql_type=list[ForecastMetricType])
    @staticmethod
    def impact_metrics(root: 'Node') -> list[Metric]:
        from nodes.actions.action import ActionNode

        if not isinstance(root, ActionNode):
            return []
        metrics = []
        for outcome_node in root.get_downstream_nodes(only_outcome=True):
            metric_val = _get_impact_metric(root, outcome_node)
            if metric_val is not None:
                metrics.append(metric_val)
        return metrics

    @sb.field(graphql_type=list[ForecastMetricType] | None)
    @staticmethod
    def metrics(root: 'Node') -> list[Metric] | None:
        return getattr(root, 'metrics', None)

    @sb.field(graphql_type=DimensionalFlowType | None)
    @staticmethod
    def dimensional_flow(root: 'Node') -> DimensionalFlow | None:
        from nodes.actions.action import ActionNode

        if not isinstance(root, ActionNode):
            return None
        return DimensionalFlow.from_action_node(root)

    @sb.field(graphql_type=DimensionalMetricType | None)
    @staticmethod
    def metric_dim(
        root: 'Node',
        info: gql.Info,
        with_scenarios: list[str] | None = None,
        include_scenario_kinds: list[ScenarioKind] | None = None,
    ) -> DimensionalMetric | None:
        context = get_instance_context(info)
        extra_scenarios: list[Scenario] = []
        for scenario_id in with_scenarios or []:
            if scenario_id not in context.scenarios:
                # FIXME: workaround; remove later
                sentry_sdk.capture_message('Scenario %s not found' % scenario_id, level='error')
                continue
                raise GraphQLError('Scenario %s not found' % scenario_id, info.field_nodes)
            extra_scenarios.append(context.get_scenario(scenario_id))

        for kind in include_scenario_kinds or []:
            for scenario in context.scenarios.values():
                if scenario.kind == kind and scenario not in extra_scenarios:
                    extra_scenarios.append(scenario)
        if include_scenario_kinds and context.active_scenario not in extra_scenarios:
            extra_scenarios.append(context.active_scenario)

        try:
            ret = DimensionalMetric.from_node(root, extra_scenarios=extra_scenarios)
        except Exception:
            context.log.exception('Exception while resolving metric_dim for node %s' % root.id)
            return None
        return ret

    # TODO: input_datasets, baseline_values, context
    @sb.field(graphql_type=list[Annotated['ParameterInterface', sb.lazy('params.schema')]])
    @staticmethod
    def parameters(root: 'Node') -> list[Parameter[Any]]:
        return [param for param in root.parameters.values() if param.is_visible]

    # These are potentially plucked from nodes.models.NodeConfig
    @grapple_field
    @staticmethod
    def short_description(root: 'Node') -> WagtailRichText | None:
        nc = root.db_obj
        if nc is not None and nc.short_description_i18n:
            return expand_db_html(nc.short_description_i18n)
        if root.description:
            desc = str(root.description)
            if desc:
                html = markdown.render(desc)
                return html
        return None

    @sb.field
    @pass_context
    @staticmethod
    def description(root: 'Node', context: 'Context') -> str | None:
        nc = root.db_obj
        if nc is None or not nc.description_i18n:
            if context.instance.features.show_explanations:
                return root.get_explanation()
            return None
        return expand_db_html(nc.description_i18n)

    @sb.field(graphql_type=list[StreamFieldInterface] | None)
    @staticmethod
    def body(root: 'Node') -> StreamValue | None:
        nc = root.db_obj
        if nc is None or not nc.body:
            return None
        return nc.body


@register_strawberry_type
@sb.type(name='Node')
class NodeType(NodeInterface):
    is_outcome: bool

    @classmethod
    def is_type_of(cls, obj: Any, _info: gql.Info) -> bool:
        from .actions.action import ActionNode
        from .node import Node

        return isinstance(obj, Node) and not isinstance(obj, ActionNode)

    @sb.field(graphql_type=list[Annotated['ActionNodeType', sb.lazy('nodes.schema')]])
    @staticmethod
    def upstream_actions(
        root: 'Node',
        only_root: bool = False,
        decision_level: DecisionLevel | None = None,
    ) -> list['Node']:
        from nodes.actions.action import ActionNode

        def filter_action(n: Node) -> bool:
            if not isinstance(n, ActionNode):
                return False
            if only_root and n.parent_action is not None:
                return False
            if decision_level is not None and n.decision_level != decision_level:
                return False
            return True

        return root.get_upstream_nodes(filter_func=filter_action)


@register_strawberry_type
@sb.type(name='ActionNode')
class ActionNodeType(NodeInterface):
    decision_level: DecisionLevel | None

    @classmethod
    def is_type_of(cls, obj: Any, _info: gql.Info) -> bool:
        from .actions.action import ActionNode

        return isinstance(obj, ActionNode)

    @sb.field
    @staticmethod
    def is_enabled(root: 'ActionNode') -> bool:
        return bool(root.is_enabled())

    @sb.field(graphql_type='ActionNodeType | None')
    @staticmethod
    def parent_action(root: 'ActionNode') -> 'ActionNode | None':
        return root.parent_action

    @sb.field(graphql_type=list[Annotated['ActionNodeType', sb.lazy('nodes.schema')]])
    @staticmethod
    def subactions(root: 'ActionNode') -> list['ActionNode']:
        from nodes.actions.parent import ParentActionNode

        if not isinstance(root, ParentActionNode):
            return []
        return root.subactions

    @sb.field(graphql_type=ActionGroupType | None)
    @staticmethod
    def group(root: 'ActionNode') -> ActionGroup | None:
        return root.group

    @grapple_field
    @staticmethod
    def goal(root: 'ActionNode') -> WagtailRichText | None:
        nc = root.db_obj
        if nc is None:
            return None
        val = nc.goal_i18n
        if val:
            return expand_db_html(val)
        return None

    @sb.field(graphql_type='NodeType | None')
    @staticmethod
    def indicator_node(root: 'ActionNode', info: gql.Info) -> 'Node | None':
        nc = root.db_obj
        if nc is None:
            return None
        if nc.indicator_node is None:
            return None
        return nc.indicator_node.get_node(visible_for_user=info.context.user)


AnyNodeType = Annotated[ActionNodeType | NodeType, sb.union('AnyNodeType')]


@sb.type
class ActionImpact:
    action: 'ActionNode' = sb.field(graphql_type=ActionNodeType)
    cost_values: list[YearlyValue] | None = sb.field(deprecation_reason='Use costDim instead.')
    impact_values: list[YearlyValue | None] | None = sb.field(deprecation_reason='Use effectDim instead.')
    cost_dim: DimensionalMetric | None = sb.field(graphql_type=DimensionalMetricType | None)
    effect_dim: DimensionalMetric = sb.field(graphql_type=DimensionalMetricType)
    unit_adjustment_multiplier: float | None


@sb.type
class ImpactOverviewType:
    cost_node: NodeType | None
    effect_node: NodeType

    @sb.field
    @staticmethod
    def id(root: 'ImpactOverview') -> sb.ID:
        cost_id = root.cost_node.id if root.cost_node else 'None'
        return sb.ID('%s:%s' % (cost_id, root.effect_node.id))

    @sb.field
    @staticmethod
    def graph_type(root: 'ImpactOverview') -> str | None:
        if root.spec.graph_type in ['benefit_cost_ratio', 'return_on_investment_gross']:
            return 'return_on_investment'
        return root.spec.graph_type

    @sb.field(graphql_type=UnitType)
    @staticmethod
    def indicator_unit(root: 'ImpactOverview') -> Unit:
        return root.spec.indicator_unit

    @sb.field
    @staticmethod
    def indicator_cutpoint(root: 'ImpactOverview') -> float | None:
        return root.spec.indicator_cutpoint

    @sb.field
    @staticmethod
    def cost_cutpoint(root: 'ImpactOverview') -> float | None:
        return root.spec.cost_cutpoint

    @sb.field
    @staticmethod
    def plot_limit_for_indicator(root: 'ImpactOverview') -> float | None:
        return root.spec.plot_limit_for_indicator

    @sb.field
    @staticmethod
    def label(root: 'ImpactOverview') -> str:
        return str(root.spec.label or '')

    @sb.field
    @staticmethod
    def cost_label(root: 'ImpactOverview') -> str | None:
        return str(root.spec.cost_label) if root.spec.cost_label is not None else None

    @sb.field
    @staticmethod
    def effect_label(root: 'ImpactOverview') -> str | None:
        return str(root.spec.effect_label) if root.spec.effect_label is not None else None

    @sb.field
    @staticmethod
    def indicator_label(root: 'ImpactOverview') -> str | None:
        return str(root.spec.indicator_label) if root.spec.indicator_label is not None else None

    @sb.field
    @staticmethod
    def cost_category_label(root: 'ImpactOverview') -> str | None:
        return str(root.spec.cost_category_label) if root.spec.cost_category_label is not None else None

    @sb.field
    @staticmethod
    def effect_category_label(root: 'ImpactOverview') -> str | None:
        return str(root.spec.effect_category_label) if root.spec.effect_category_label is not None else None

    @sb.field
    @staticmethod
    def description(root: 'ImpactOverview') -> str | None:
        return str(root.spec.description) if root.spec.description is not None else None

    @sb.field
    @pass_context
    @staticmethod
    def actions(root: 'ImpactOverview', context: 'Context') -> list[ActionImpact]:
        all_aes = root.calculate(context)
        out: list[ActionImpact] = []
        for ae in all_aes:
            years = ae.df[YEAR_COLUMN]
            if 'Cost' in ae.df.columns:  # FIXME Deprecated
                cost_values = [
                    YearlyValue(year=year, value=float(val)) for year, val in zip(years, list(ae.df['Cost']), strict=False)
                ]
            else:
                cost_values = None
            effect_dim = DimensionalMetric.from_action_impact(ae, root, 'Effect')
            if effect_dim is None:
                raise ValueError('Effect dimension is None')
            out.append(
                ActionImpact(
                    action=ae.action,
                    cost_values=cost_values,
                    impact_values=[
                        YearlyValue(year=year, value=float(val)) for year, val in zip(years, list(ae.df['Effect']), strict=False)
                    ],
                    cost_dim=DimensionalMetric.from_action_impact(ae, root, 'Cost'),
                    effect_dim=effect_dim,
                    unit_adjustment_multiplier=ae.unit_adjustment_multiplier,
                )
            )
        return out

    @sb.field(graphql_type=UnitType | None)
    @staticmethod
    def cost_unit(root: 'ImpactOverview') -> Unit:
        return root.spec.cost_unit or root.spec.indicator_unit

    @sb.field(graphql_type=UnitType | None)
    @staticmethod
    def effect_unit(root: 'ImpactOverview') -> Unit:
        return root.spec.effect_unit or root.spec.indicator_unit

    @sb.field
    @staticmethod
    def outcome_dimension(root: 'ImpactOverview') -> str | None:
        return root.spec.outcome_dimension_id

    @sb.field
    @staticmethod
    def stakeholder_dimension(root: 'ImpactOverview') -> str | None:
        return root.spec.stakeholder_dimension_id


@sb.type
class InstanceBasicConfiguration:
    default_language: str
    theme_identifier: str
    supported_languages: list[str]

    @sb.field
    @staticmethod
    def identifier(root: Instance) -> str:
        return root.id

    @sb.field
    @staticmethod
    def is_protected(root: Instance) -> bool:
        ic: InstanceConfig = root._config  # type: ignore
        return ic.is_protected

    @sb.field
    @staticmethod
    def requires_authentication(root: Instance) -> bool:
        ic: InstanceConfig = root._config  # type: ignore
        return ic.get_instance().features.requires_authentication

    @sb.field
    @staticmethod
    def hostname(root: Instance) -> InstanceHostname:
        ic: InstanceConfig = root._config  # type: ignore
        hostname: str = root._hostname  # type: ignore
        hn_obj = ic.hostnames.filter(hostname=hostname.lower()).first()
        if not hn_obj:
            return InstanceHostname(hostname=hostname, base_path='')
        return InstanceHostname(hostname=hn_obj.hostname, base_path=hn_obj.base_path)


@sb.type
class NormalizationType:
    @sb.field
    @staticmethod
    def id(root: 'Normalization') -> sb.ID:
        return sb.ID(root.normalizer_node.id)

    @sb.field
    @staticmethod
    def label(root: 'Normalization') -> str:
        return str(root.normalizer_node.name)

    @sb.field(graphql_type=NodeType)
    @staticmethod
    def normalizer(root: 'Normalization') -> 'Node':
        return root.normalizer_node

    @sb.field
    @pass_context
    @staticmethod
    def is_active(root: 'Normalization', context: 'Context') -> bool:
        return context.active_normalization == root


@sb.type
class Query:
    @sb.field(graphql_type=InstanceType)
    @pass_context
    def instance(self, context: 'Context') -> Instance:
        return context.instance

    @sb.field(graphql_type=list[NodeInterface])
    @pass_context
    def nodes(self, context: 'Context') -> list['Node']:
        return list(context.nodes.values())

    @sb.field(graphql_type='NodeInterface | None')
    @ensure_instance
    def node(self, info: gql.InstanceInfo, id: sb.ID) -> 'Node | None':
        instance = info.context.instance
        nodes = instance.context.nodes
        node_id = str(id)
        if node_id.isnumeric():
            for node in nodes.values():
                if node.database_id is not None and node.database_id == int(node_id):
                    return node
            return None

        return instance.context.nodes.get(node_id)

    @sb.field(graphql_type=ActionNodeType | None)
    @pass_context
    def action(self, context: 'Context', id: sb.ID) -> 'ActionNode | None':
        try:
            return context.get_action(str(id))
        except KeyError, TypeError:
            return None

    @sb.field(graphql_type=list[ImpactOverviewType], deprecation_reason='Use impactOverviews instead')
    @pass_context
    def action_efficiency_pairs(self, context: 'Context') -> list['ImpactOverview']:
        return context.impact_overviews

    @sb.field(graphql_type=list[ImpactOverviewType])
    @pass_context
    def impact_overviews(self, context: 'Context') -> list['ImpactOverview']:
        return context.impact_overviews

    @sb.field(graphql_type=list[ScenarioType])
    @pass_context
    def scenarios(self, context: 'Context') -> list[Scenario]:
        return list(context.scenarios.values())

    @sb.field(graphql_type=ScenarioType)
    @pass_context
    def scenario(self, context: 'Context', id: sb.ID) -> Scenario:
        return context.get_scenario(str(id))

    @sb.field(graphql_type=ScenarioType)
    @pass_context
    def active_scenario(self, context: 'Context') -> Scenario:
        return context.active_scenario

    @sb.field(graphql_type=list[NormalizationType])
    @pass_context
    def available_normalizations(self, context: 'Context') -> list['Normalization']:
        return list(context.normalizations.values())

    @sb.field(graphql_type=NormalizationType | None)
    @pass_context
    def active_normalization(self, context: 'Context') -> 'Normalization | None':
        return context.active_normalization


@sb.type
class SBQuery(Query):
    @sb.field(graphql_type=list[NormalizationType])
    @pass_context
    def active_normalizations(self, context: 'Context') -> list['Normalization']:
        return list(context.normalizations.values())

    @sb.field(graphql_type=list[ActionNodeType])
    @pass_context
    @staticmethod
    def actions(context: 'Context', only_root: bool = False) -> list['ActionNode']:
        instance = context.instance
        actions = instance.context.get_actions()
        if only_root:
            actions = list(filter(lambda act: act.parent_action is None, actions))
        return actions

    @sb.field(graphql_type=list[InstanceBasicConfiguration])
    @staticmethod
    def available_instances(info: gql.Info, hostname: str) -> list[Instance]:
        qs = InstanceConfig.objects.get_queryset().for_hostname(hostname, wildcard_domains=info.context.wildcard_domains)
        instances: list[Instance] = []
        for config in qs:
            instance = config.get_instance()
            instance._config = config  # type: ignore
            instance._hostname = hostname  # type: ignore
            instances.append(instance)
        return instances


@register_strawberry_type
@sb.type
class InstanceChange:
    id: sb.ID
    identifier: str
    modified_at: datetime


@sb.type
class Subscription:
    @sb.subscription(graphql_type=InstanceChange)
    async def available_instances(self, info: gql.Info) -> AsyncGenerator[InstanceChange]:
        user = info.context.get_user()
        logger.debug('New available_instances subscription')
        ws = info.context.get_ws_consumer()
        cl = ws.channel_layer
        assert cl is not None
        async with ws.listen_to_channel(INSTANCE_CHANGE_TYPE, groups=[INSTANCE_CHANGE_GROUP]) as channel:
            cl_logger = logger.bind(channel=ws.channel_name)
            cl_logger.debug('Listening to instance_change channel [%s]' % ws.channel_name)
            async for msg in channel:
                cl_logger.debug('Received instance_change message [%s]' % ws.channel_name)
                ic = await InstanceConfig.objects.qs.filter(pk=msg['pk']).viewable_by(user).afirst()
                if ic is None:
                    continue
                yield InstanceChange(id=sb.ID(str(ic.pk)), identifier=ic.identifier, modified_at=ic.modified_at)


@sb.type
class Mutation:
    @sb.type
    class SetNormalizerMutation:
        ok: bool
        active_normalizer: 'Normalization | None' = sb.field(graphql_type=NormalizationType)

    @sb.mutation
    def set_normalizer(self, info: gql.Info, id: sb.ID | None = None) -> 'Mutation.SetNormalizerMutation':
        context = get_instance_context(info)
        default = context.default_normalization
        if id:
            normalizer = context.normalizations.get(id)
            if normalizer is None:
                raise GraphQLError("Normalization '%s' not found" % id)
        else:
            normalizer = None

        assert context.setting_storage is not None

        if normalizer == default:
            context.setting_storage.reset_option('normalizer')
        else:
            context.setting_storage.set_option('normalizer', id)
        context.set_option('normalizer', id)

        return Mutation.SetNormalizerMutation(ok=True, active_normalizer=context.active_normalization)
