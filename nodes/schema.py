# ruff: noqa: N805
from __future__ import annotations

import dataclasses
import re
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Protocol, cast

import graphene
import strawberry
import strawberry as sb
from graphql.error import GraphQLError
from wagtail.blocks.stream_block import StreamValue
from wagtail.rich_text import RichText as WagtailRichText, expand_db_html

import polars as pl
import sentry_sdk
from grapple.types.streamfield import StreamFieldInterface
from loguru import logger
from markdown_it import MarkdownIt

from kausal_common.strawberry.grapple import grapple_field
from kausal_common.strawberry.pydantic import StrawberryPydanticType
from kausal_common.strawberry.registry import register_strawberry_type

from paths import gql
from paths.const import INSTANCE_CHANGE_GROUP, INSTANCE_CHANGE_TYPE
from paths.graphql_helpers import ensure_instance, get_instance_context, pass_context
from paths.graphql_types import UnitType

from nodes.context import Context
from nodes.defs.instance_defs import ActionGroup
from nodes.goals import GoalActualValue, NodeGoalsEntry
from nodes.node import Node
from nodes.normalization import Normalization
from nodes.scenario import Scenario, ScenarioKind
from pages.models import ActionListPage
from params import Parameter

from . import visualizations as viz
from .actions import ActionNode, ImpactOverview
from .actions.parent import ParentActionNode
from .constants import FORECAST_COLUMN, IMPACT_COLUMN, IMPACT_GROUP, YEAR_COLUMN, DecisionLevel
from .instance import Instance, InstanceFeatures
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
from .units import Unit

if TYPE_CHECKING:
    from paths.types import GQLInstanceContext, GQLInstanceInfo

    from common import polars as ppl
    from params.schema import ParameterInterface


logger = logger.bind(name='nodes.schema')
markdown = MarkdownIt('commonmark', {'html': True})


@sb.type
class InstanceHostname:
    hostname: str
    base_path: str


@sb.type
class ActionGroupType:
    id: sb.ID
    name: str
    color: str | None

    @sb.field(graphql_type=list[Annotated['ActionNodeType', sb.lazy('nodes.schema')]])
    @pass_context
    @staticmethod
    def actions(root: ActionGroup, info: gql.Info, context: Context) -> list[ActionNode]:
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
    year: strawberry.auto
    goal: strawberry.auto
    actual: strawberry.auto
    is_interpolated: strawberry.auto
    is_forecast: strawberry.auto


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
    outcome_node: Node = sb.field(graphql_type=Annotated['NodeType', sb.lazy('nodes.schema')])
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
    name: str
    owner: str | None
    default_language: str
    supported_languages: list[str]
    base_path: str
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
    label: strawberry.auto
    color: strawberry.auto
    order: strawberry.auto
    group: strawberry.auto


@register_strawberry_type
@sb.experimental.pydantic.type(model=MetricCategoryGroup)
class MetricDimensionCategoryGroupType:
    id: sb.ID
    original_id: sb.ID
    label: strawberry.auto
    color: strawberry.auto
    order: strawberry.auto


@register_strawberry_type
@sb.experimental.pydantic.type(model=MetricDimension)
class MetricDimensionType:
    id: sb.ID
    original_id: sb.ID | None
    label: strawberry.auto
    help_text: strawberry.auto
    categories: strawberry.auto
    groups: strawberry.auto
    kind: strawberry.auto


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
@sb.experimental.pydantic.type(
    model=DimensionalMetric,
)
class DimensionalMetricType:
    id: sb.ID
    name: strawberry.auto
    dimensions: strawberry.auto
    values: strawberry.auto
    years: strawberry.auto
    stackable: strawberry.auto
    forecast_from: strawberry.auto
    goals: strawberry.auto
    normalized_by: strawberry.auto
    unit: Annotated[UnitType, sb.lazy('paths.graphql_types')]
    measure_datapoint_years: strawberry.auto

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
    action: ActionNodeType
    value: float
    year: int


@sb.type
class ScenarioActionImpacts:
    scenario: ScenarioType
    impacts: list[ActionImpactType]


def _get_impact_metric(source_node: ActionNode, target_node: Node, goal: NodeGoalsEntry | None = None) -> Metric | None:
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


@sb.interface
class NodeInterface:
    id: sb.ID
    short_name: str | None
    order: int | None
    unit: UnitType | None
    quantity: str | None

    input_nodes: list[NodeInterface]
    output_nodes: list[NodeInterface]

    @sb.field
    @staticmethod
    def name(root: Node) -> str:
        nc = root.db_obj
        if nc is not None and nc.name_i18n:
            return nc.name_i18n
        return str(root.name)

    @sb.field
    @staticmethod
    def color(root: Node) -> str | None:
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
    def is_visible(root: Node) -> bool:
        nc = root.db_obj
        if nc and nc.is_visible:
            return nc.is_visible
        return root.is_visible

    @sb.field(deprecation_reason='Replaced by "goals".')
    @staticmethod
    def target_year_goal(root: Node) -> float | None:
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
    def goals(root: Node, info: gql.Info, context: Context, active_goal: sb.ID | None = None) -> list[NodeGoal]:
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
    def is_action(root: Node) -> bool:
        return isinstance(root, ActionNode)

    @sb.field
    @pass_context
    @staticmethod
    def explanation(root: Node, info: gql.Info, context: Context) -> str | None:
        if context.instance.features.show_explanations:
            return None
        return root.get_explanation()

    @sb.field
    @staticmethod
    def node_type(root: Node) -> str:
        typ = str(type(root))
        typstr = re.search(r"'([^']*)'", typ)
        if typstr is not None:
            return typstr.group(1)
        return ''

    @sb.field
    @staticmethod
    def tags(root: Node) -> list[str] | None:
        return list(root.tags)

    @sb.field
    @staticmethod
    def input_dimensions(root: Node) -> list[str] | None:
        return list(root.input_dimensions.keys())

    @sb.field
    @staticmethod
    def output_dimensions(root: Node) -> list[str] | None:
        return list(root.output_dimensions.keys())

    @sb.field(graphql_type=list[VisualizationEntry] | None)
    @staticmethod
    def visualizations(root: Node) -> list[viz.VisualizationEntryType] | None:
        node_viz = root.visualizations
        if not node_viz:
            return None
        return node_viz.root

    @sb.field(graphql_type=list['NodeInterface'])
    @staticmethod
    def downstream_nodes(
        root: Node,
        info: gql.Info,
        max_depth: int | None = None,
        only_outcome: bool = False,
        until_node: sb.ID | None = None,
    ) -> list[Node]:
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
        root: Node,
        same_unit: bool = False,
        same_quantity: bool = False,
        include_actions: bool = True,
    ) -> list[Node]:
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
    def metric(root: Node, goal_id: sb.ID | None = None) -> Metric | None:
        return Metric.from_node(root, goal_id=goal_id)

    @sb.field(graphql_type=DimensionalMetricType | None)
    @staticmethod
    def outcome(root: Node) -> DimensionalMetric | None:
        return getattr(root, 'outcome', None)

    # If resolving through `descendant_nodes`, `impact_metric` will be
    # by default be calculated from the ancestor node.
    @sb.field(graphql_type=ForecastMetricType | None)
    @pass_context
    @staticmethod
    def impact_metric(
        root: Node,
        info: gql.Info,
        context: Context,
        target_node_id: sb.ID | None = None,
        goal_id: sb.ID | None = None,
    ) -> Metric | None:
        instance = context.instance
        upstream_node = getattr(info.context, '_upstream_node', None)

        if goal_id is not None:
            try:
                goal = instance.get_goals(goal_id=goal_id)
            except Exception:
                raise GraphQLError('Goal not found', info.field_nodes) from None
        else:
            goal = None

        target_node: Node
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
    def impact_metrics(root: Node) -> list[Metric]:
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
    def metrics(root: Node) -> list[Metric] | None:
        return getattr(root, 'metrics', None)

    @sb.field(graphql_type=DimensionalFlowType | None)
    @staticmethod
    def dimensional_flow(root: Node) -> DimensionalFlow | None:
        if not isinstance(root, ActionNode):
            return None
        return DimensionalFlow.from_action_node(root)

    @sb.field(graphql_type=DimensionalMetricType | None)
    @staticmethod
    def metric_dim(
        root: Node,
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
                if scenario.kind == kind and scenario.id not in extra_scenarios:
                    extra_scenarios.append(scenario)
        if include_scenario_kinds and context.active_scenario.id not in extra_scenarios:
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
    def parameters(root: Node) -> list[Parameter[Any]]:
        return [param for param in root.parameters.values() if param.is_visible]

    # These are potentially plucked from nodes.models.NodeConfig
    @grapple_field
    @staticmethod
    def short_description(root: Node) -> WagtailRichText | None:
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
    def description(root: Node, info: gql.Info, context: Context) -> str | None:
        nc = root.db_obj
        if nc is None or not nc.description_i18n:
            if context.instance.features.show_explanations:
                return root.get_explanation()
            return None
        return expand_db_html(nc.description_i18n)

    @sb.field(graphql_type=list[StreamFieldInterface] | None)
    @staticmethod
    def body(root: Node) -> StreamValue | None:
        nc = root.db_obj
        if nc is None or not nc.body:
            return None
        return nc.body


@register_strawberry_type
@sb.type(name='Node')
class NodeType(NodeInterface):
    @classmethod
    def is_type_of(cls, obj: Any, _info: gql.Info) -> bool:
        return isinstance(obj, Node) and not isinstance(obj, ActionNode)

    @sb.field(graphql_type=list[Annotated['ActionNodeType', sb.lazy('nodes.schema')]])
    @staticmethod
    def upstream_actions(
        root: Node,
        only_root: bool = False,
        decision_level: DecisionLevel | None = None,
    ) -> list[Node]:
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
        return isinstance(obj, ActionNode)

    @sb.field
    @staticmethod
    def is_enabled(root: ActionNode) -> bool:
        return bool(root.is_enabled())

    @sb.field(graphql_type='ActionNodeType | None')
    @staticmethod
    def parent_action(root: ActionNode) -> ActionNode | None:
        return root.parent_action

    @sb.field(graphql_type=list[Annotated['ActionNodeType', sb.lazy('nodes.schema')]])
    @staticmethod
    def subactions(root: ActionNode) -> list[ActionNode]:
        if not isinstance(root, ParentActionNode):
            return []
        return root.subactions

    @sb.field(graphql_type=ActionGroupType | None)
    @staticmethod
    def group(root: ActionNode) -> ActionGroup | None:
        return root.group

    @grapple_field
    @staticmethod
    def goal(root: ActionNode) -> WagtailRichText | None:
        nc = root.db_obj
        if nc is None:
            return None
        val = nc.goal_i18n
        if val:
            return expand_db_html(val)
        return None

    @sb.field(graphql_type='NodeType | None')
    @staticmethod
    def indicator_node(root: ActionNode, info: gql.Info) -> Node | None:
        nc = root.db_obj
        if nc is None:
            return None
        if nc.indicator_node is None:
            return None
        return nc.indicator_node.get_node(visible_for_user=info.context.user)


@sb.experimental.pydantic.type(model=Scenario)
class ScenarioType:
    id: sb.ID
    name: strawberry.auto
    kind: strawberry.auto
    is_selectable: strawberry.auto
    actual_historical_years: strawberry.auto

    @sb.field
    @pass_context
    @staticmethod
    def is_active(root: Scenario, info: gql.Info, context: Context) -> bool:
        return context.active_scenario == root

    @sb.field
    @staticmethod
    def is_default(root: Scenario) -> bool:
        return root.default


@sb.type
class ActionImpact:
    action: ActionNode = sb.field(graphql_type=ActionNodeType)
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
    def id(root: ImpactOverview) -> sb.ID:
        cost_id = root.cost_node.id if root.cost_node else 'None'
        return sb.ID('%s:%s' % (cost_id, root.effect_node.id))

    @sb.field
    @staticmethod
    def graph_type(root: ImpactOverview) -> str | None:
        if root.spec.graph_type in ['benefit_cost_ratio', 'return_on_investment_gross']:
            return 'return_on_investment'
        return root.spec.graph_type

    @sb.field(graphql_type=UnitType)
    @staticmethod
    def indicator_unit(root: ImpactOverview) -> Unit:
        return root.spec.indicator_unit

    @sb.field
    @staticmethod
    def indicator_cutpoint(root: ImpactOverview) -> float | None:
        return root.spec.indicator_cutpoint

    @sb.field
    @staticmethod
    def cost_cutpoint(root: ImpactOverview) -> float | None:
        return root.spec.cost_cutpoint

    @sb.field
    @staticmethod
    def plot_limit_for_indicator(root: ImpactOverview) -> float | None:
        return root.spec.plot_limit_for_indicator

    @sb.field
    @staticmethod
    def label(root: ImpactOverview) -> str:
        return str(root.spec.label or '')

    @sb.field
    @staticmethod
    def cost_label(root: ImpactOverview) -> str | None:
        return str(root.spec.cost_label) if root.spec.cost_label is not None else None

    @sb.field
    @staticmethod
    def effect_label(root: ImpactOverview) -> str | None:
        return str(root.spec.effect_label) if root.spec.effect_label is not None else None

    @sb.field
    @staticmethod
    def indicator_label(root: ImpactOverview) -> str | None:
        return str(root.spec.indicator_label) if root.spec.indicator_label is not None else None

    @sb.field
    @staticmethod
    def cost_category_label(root: ImpactOverview) -> str | None:
        return str(root.spec.cost_category_label) if root.spec.cost_category_label is not None else None

    @sb.field
    @staticmethod
    def effect_category_label(root: ImpactOverview) -> str | None:
        return str(root.spec.effect_category_label) if root.spec.effect_category_label is not None else None

    @sb.field
    @staticmethod
    def description(root: ImpactOverview) -> str | None:
        return str(root.spec.description) if root.spec.description is not None else None

    @sb.field
    @pass_context
    @staticmethod
    def actions(root: ImpactOverview, info: gql.Info, context: Context) -> list[ActionImpact]:
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
    def cost_unit(root: ImpactOverview) -> Unit:
        return root.spec.cost_unit or root.spec.indicator_unit

    @sb.field(graphql_type=UnitType | None)
    @staticmethod
    def effect_unit(root: ImpactOverview) -> Unit:
        return root.spec.effect_unit or root.spec.indicator_unit


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
    def id(root: Normalization) -> sb.ID:
        return sb.ID(root.normalizer_node.id)

    @sb.field
    @staticmethod
    def label(root: Normalization) -> str:
        return str(root.normalizer_node.name)

    @sb.field(graphql_type=NodeType)
    @staticmethod
    def normalizer(root: Normalization) -> Node:
        return root.normalizer_node

    @sb.field
    @pass_context
    @staticmethod
    def is_active(root: Normalization, info: gql.Info, context: Context) -> bool:
        return context.active_normalization == root


class Query(graphene.ObjectType[Any]):
    instance = graphene.Field(InstanceType, required=True)
    nodes = graphene.List(graphene.NonNull(NodeInterface), required=True)
    node = graphene.Field(
        NodeInterface,
        id=graphene.ID(required=True),
    )
    action = graphene.Field(ActionNodeType, id=graphene.ID(required=True))
    action_efficiency_pairs = graphene.List(
        graphene.NonNull(ImpactOverviewType), required=True, deprecation_reason='Use impactOverviews instead'
    )
    impact_overviews = graphene.List(graphene.NonNull(ImpactOverviewType), required=True)
    scenarios = graphene.List(graphene.NonNull(ScenarioType), required=True)
    scenario = graphene.Field(ScenarioType, id=graphene.ID(required=True))
    active_scenario = graphene.Field(ScenarioType, required=True)
    available_normalizations = graphene.List(graphene.NonNull(NormalizationType), required=True)
    active_normalization = graphene.Field(NormalizationType, required=False)

    @ensure_instance
    def resolve_instance(root: Query, info: GQLInstanceInfo) -> Instance:
        return info.context.instance

    @ensure_instance
    def resolve_scenario(root: Query, info: GQLInstanceInfo, id: str) -> Scenario:
        context = info.context.instance.context
        return context.get_scenario(id)

    @ensure_instance
    def resolve_active_scenario(root: Query, info: GQLInstanceInfo) -> Scenario:
        context = info.context.instance.context
        return context.active_scenario

    @ensure_instance
    def resolve_scenarios(root, info: GQLInstanceInfo) -> list[Scenario]:
        context = info.context.instance.context
        return list(context.scenarios.values())

    @ensure_instance
    def resolve_node(root, info: GQLInstanceInfo, id: str) -> Node | None:
        instance = info.context.instance
        nodes = instance.context.nodes
        if id.isnumeric():
            for node in nodes.values():
                if node.database_id is not None and node.database_id == int(id):
                    return node
            return None

        return instance.context.nodes.get(id)

    @pass_context
    def resolve_nodes(root, info: GQLInstanceInfo, context: Context) -> list[Node]:
        instance = info.context.instance
        return list(instance.context.nodes.values())

    @ensure_instance
    def resolve_action(root, info: GQLInstanceInfo, id: str) -> ActionNode | None:
        instance = info.context.instance
        try:
            action = instance.context.get_action(id)
        except (KeyError, TypeError) as e:
            print(e)
            return None
        return action

    @pass_context
    def resolve_action_efficiency_pairs(root, info: GQLInstanceInfo, context: Context):
        return context.impact_overviews

    @pass_context
    def resolve_impact_overviews(root, info: GQLInstanceInfo, context: Context):
        return context.impact_overviews

    @pass_context
    def resolve_available_normalizations(root, info: GQLInstanceInfo, context: Context):
        return context.normalizations.values()

    @pass_context
    def resolve_active_normalization(root, info: GQLInstanceInfo, context: Context):
        return context.active_normalization


@sb.type
class SBQuery(Query):
    @sb.field(graphql_type=list[NormalizationType])
    @pass_context
    def active_normalizations(self, context: Context) -> list[Normalization]:
        return list(context.normalizations.values())

    @sb.field(graphql_type=list[ActionNodeType])
    @pass_context
    @staticmethod
    def actions(root: SBQuery, info: gql.Info, context: Context, only_root: bool = False) -> list[ActionNode]:
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
        active_normalizer: Normalization | None = sb.field(graphql_type=NormalizationType)

    @sb.mutation
    def set_normalizer(self, info: gql.Info, id: sb.ID | None = None) -> Mutation.SetNormalizerMutation:
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
