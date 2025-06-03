# ruff: noqa: N805
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Annotated, Any, Protocol, cast

import graphene
import strawberry as sb
from graphql.error import GraphQLError
from pydantic import BaseModel
from wagtail.rich_text import expand_db_html

import polars as pl
import sentry_sdk
from grapple.types.rich_text import RichText
from grapple.types.streamfield import StreamFieldInterface
from markdown_it import MarkdownIt

from kausal_common.graphene.utils import create_from_dataclass
from kausal_common.strawberry.registry import register_strawberry_type

from paths.graphql_helpers import ensure_instance, pass_context

from nodes.node import Node
from nodes.scenario import Scenario, ScenarioKind as ScenarioKindEnum

from . import visualizations as viz
from .actions.action import ActionGroup, ActionNode, ImpactOverview
from .actions.parent import ParentActionNode
from .constants import FORECAST_COLUMN, IMPACT_COLUMN, IMPACT_GROUP, YEAR_COLUMN, DecisionLevel
from .instance import Instance, InstanceFeatures
from .metric import (
    DimensionalFlow,
    DimensionalMetric,
    DimensionKind,
    Metric,
    MetricCategory,
    MetricCategoryGroup,
    MetricDimension,
    MetricDimensionGoal,
    MetricYearlyGoal,
    NormalizerNode,
)
from .models import InstanceConfig

strawberry = sb

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from graphql import GraphQLResolveInfo
    from wagtail.blocks.stream_block import StreamValue

    from kausal_common.graphene import GQLInfo

    from paths.graphql_types import UnitType
    from paths.types import GQLInstanceContext, GQLInstanceInfo

    from common import polars as ppl
    from nodes.context import Context
    from nodes.goals import GoalActualValue, NodeGoalsEntry
    from nodes.normalization import Normalization
    from pages.models import ActionListPage
    from params.param import Parameter

    from .node import Node
    from .scenario import Scenario
    from .units import Unit

logger = logging.getLogger(__name__)
markdown = MarkdownIt('commonmark', {'html': True})


class InstanceHostname(BaseModel):
    hostname: str
    base_path: str


class InstanceHostnameType(graphene.ObjectType):
    hostname = graphene.String()
    base_path = graphene.String()

    class Meta:
        name = 'InstanceHostname'


class ActionGroupType(graphene.ObjectType):
    id = graphene.ID(required=True)
    name = graphene.String(required=True)
    color = graphene.String(required=False)
    actions = graphene.List(graphene.NonNull(lambda: ActionNodeType), required=True)

    @staticmethod
    def resolve_actions(root: ActionGroup, info: GQLInstanceInfo) -> list[ActionNode]:
        context = info.context.instance.context
        return [act for act in context.get_actions() if act.group == root]


InstanceFeaturesType = create_from_dataclass(InstanceFeatures)


class InstanceYearlyGoalType(graphene.ObjectType):
    year = graphene.Int(required=True)
    goal = graphene.Float(required=False)
    actual = graphene.Float(required=False)
    is_interpolated = graphene.Boolean(required=False)
    is_forecast = graphene.Boolean(required=True)


class InstanceGoalDimensionProtocol(Protocol):
    dimension: str
    categories: list[str]
    groups: list[str]


class InstanceGoalDimension(graphene.ObjectType[InstanceGoalDimensionProtocol]):
    dimension = graphene.String(required=True)
    categories = graphene.List(graphene.NonNull(graphene.String), required=True)
    groups = graphene.List(graphene.NonNull(graphene.String), required=True)
    category = graphene.String(required=True, deprecation_reason='replaced with categories')

    @staticmethod
    def resolve_category(root: InstanceGoalDimensionProtocol, info):  # noqa: ANN205
        return root.categories[0]


class InstanceGoalEntry(graphene.ObjectType):
    id = graphene.ID(required=True)
    label = graphene.String(required=False)
    disabled = graphene.Boolean(required=True)
    disable_reason = graphene.String(required=False)
    outcome_node: Node = graphene.Field('nodes.schema.NodeType', required=True)  # type: ignore
    dimensions = graphene.List(graphene.NonNull(InstanceGoalDimension), required=True)
    default = graphene.Boolean(required=True)
    values = graphene.List(graphene.NonNull(InstanceYearlyGoalType), required=True)
    unit = graphene.Field('paths.schema.UnitType', required=True)

    _goal: NodeGoalsEntry

    def resolve_values(self, _info) -> list[GoalActualValue]:
        return self._goal.get_actual()

    def resolve_unit(self, _info) -> Unit:
        df = self._goal._get_values_df()
        return df.get_unit(self.outcome_node.get_default_output_metric().column_id)


def get_action_list_page_node():
    from grapple.registry import registry

    from pages.models import ActionListPage

    return registry.pages[ActionListPage]


class InstanceType(graphene.ObjectType):
    id = graphene.ID(required=True)
    name = graphene.String(required=True)
    owner = graphene.String()
    default_language = graphene.String(required=True)
    supported_languages = graphene.List(graphene.NonNull(graphene.String), required=True)
    base_path = graphene.String(required=True)
    target_year = graphene.Int()
    model_end_year = graphene.Int(required=True)
    reference_year = graphene.Int()
    minimum_historical_year = graphene.Int(required=True)
    maximum_historical_year = graphene.Int()

    hostname = graphene.Field(InstanceHostnameType, hostname=graphene.String(required=True))
    lead_title = graphene.String()
    lead_paragraph = graphene.String()
    theme_identifier = graphene.String()
    action_groups = graphene.List(graphene.NonNull(ActionGroupType), required=True)
    features = graphene.Field(InstanceFeaturesType, required=True)
    goals = graphene.List(graphene.NonNull(InstanceGoalEntry), id=graphene.ID(required=False), required=True)
    action_list_page = graphene.Field(get_action_list_page_node, required=False)
    intro_content = graphene.List(graphene.NonNull(StreamFieldInterface))

    @staticmethod
    def resolve_lead_title(root: Instance, _info) -> str:
        return root.config.lead_title_i18n

    @staticmethod
    def resolve_lead_paragraph(root: Instance, _info) -> str | None:
        return root.config.lead_paragraph_i18n

    @staticmethod
    def resolve_hostname(root: Instance, _info: GQLInstanceInfo, hostname: str) -> InstanceHostname | None:
        hn = root.config.hostnames.filter(hostname__iexact=hostname).first()
        if not hn:
            return None
        return InstanceHostname(hostname=hn.hostname, base_path=hn.base_path)

    @staticmethod
    def resolve_goals(root: Instance, _info: GQLInstanceInfo, id: str | None = None) -> list[InstanceGoalEntry]:
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
                id=goal_id,
                label=str(goal.label) if goal.label else str(node.name),
                outcome_node=node,
                dimensions=dims,
                default=goal.default,
                disabled=goal.disabled,
                disable_reason=goal.disable_reason,
            )
            out._goal = goal
            ret.append(out)
        return ret

    @staticmethod
    def resolve_intro_content(root: Instance, _info) -> StreamValue:
        intro_content = root.config.site_content.intro_content
        return intro_content

    @staticmethod
    def resolve_action_list_page(root: Instance, info: GQLInstanceInfo) -> ActionListPage | None:
        return root.config.action_list_page


class YearlyValueProtocol(Protocol):
    year: int
    value: float


class YearlyValue(graphene.ObjectType[YearlyValueProtocol]):
    year = graphene.Int(required=True)
    value = graphene.Float(required=True)


class ForecastMetricType(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    output_node = graphene.Field(lambda: NodeType, description='Will be set if the node outputs multiple time-series')
    unit = graphene.Field('paths.schema.UnitType')
    yearly_cumulative_unit = graphene.Field('paths.schema.UnitType')
    historical_values = graphene.List(graphene.NonNull(YearlyValue), required=True, latest=graphene.Int(required=False))
    forecast_values = graphene.List(graphene.NonNull(YearlyValue), required=True)
    cumulative_forecast_value = graphene.Float()
    baseline_forecast_values = graphene.List(graphene.NonNull(YearlyValue))

    @staticmethod
    def resolve_historical_values(root: Metric, _info, latest: int | None = None):  # noqa: ANN205
        ret = root.get_historical_values()
        if latest:
            if latest >= len(ret):
                return ret
            return ret[-latest:]
        return ret

    @staticmethod
    def resolve_forecast_values(root: Metric, _info) -> Sequence[YearlyValueProtocol]:
        return root.get_forecast_values()

    @staticmethod
    def resolve_baseline_forecast_values(root: Metric, _info) -> Sequence[YearlyValueProtocol]:
        return root.get_baseline_forecast_values()

    @staticmethod
    def resolve_cumulative_forecast_value(root: Metric, _info) -> float | None:
        return root.get_cumulative_forecast_value()


@register_strawberry_type
@sb.experimental.pydantic.type(model=MetricCategory, fields=['label', 'color', 'order', 'group'])
class MetricDimensionCategoryType:
    id: sb.ID
    original_id: sb.ID | None


@register_strawberry_type
@sb.experimental.pydantic.type(model=MetricCategoryGroup, fields=['label', 'color', 'order'])
class MetricDimensionCategoryGroupType:
    id: sb.ID
    original_id: sb.ID


DimensionType = graphene.Enum.from_enum(DimensionKind)


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


ActionDecisionLevel = graphene.Enum.from_enum(DecisionLevel)


class FlowNodeType(graphene.ObjectType):
    id = graphene.String(required=True)
    label = graphene.String(required=True)
    color = graphene.String(required=False)


class FlowLinksType(graphene.ObjectType):
    year = graphene.Int(required=True)
    is_forecast = graphene.Boolean(required=True)
    sources = graphene.List(graphene.NonNull(graphene.String), required=True)
    targets = graphene.List(graphene.NonNull(graphene.String), required=True)
    values = graphene.List(graphene.Float, required=True)
    absolute_source_values = graphene.List(graphene.NonNull(graphene.Float), required=True)


class DimensionalFlowType(graphene.ObjectType):
    id = graphene.String(required=True)
    nodes = graphene.List(graphene.NonNull(FlowNodeType), required=True)
    unit = graphene.Field('paths.schema.UnitType', required=True)
    sources = graphene.List(graphene.NonNull(graphene.String), required=True)
    links = graphene.List(graphene.NonNull(FlowLinksType), required=True)


class NodeGoal(graphene.ObjectType[YearlyValueProtocol]):
    year = graphene.Int(required=True)
    value = graphene.Float(required=True)


@register_strawberry_type
@sb.experimental.pydantic.interface(model=viz.VisualizationEntry)
class VisualizationEntry:
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

    @sb.field
    def metric_dim(self, info: sb.Info) -> DimensionalMetricType | None:
        e = cast('viz.VisualizationNodeOutput', self)
        req: GQLInstanceContext = info.context
        dm = e.get_metric_data(req.instance.context.nodes[self.node_id])
        if dm is None:
            return None
        out: DimensionalMetricType = DimensionalMetricType.from_pydantic(dm)
        setattr(out, 'real_unit', dm.unit)  # noqa: B010
        return out

    def to_pydantic(self) -> viz.VisualizationNodeOutput:
        data = dataclasses.asdict(self)  # type: ignore[call-overload]
        return viz.VisualizationNodeOutput.model_validate(data)


@register_strawberry_type
@sb.experimental.pydantic.type(model=viz.VisualizationGroup)
class VisualizationGroup(VisualizationEntry):  # type: ignore[override]
    children: list[VisualizationEntry]

    def to_pydantic(self) -> viz.VisualizationGroup:
        data = dataclasses.asdict(self)  # type: ignore[call-overload]
        return viz.VisualizationGroup.model_validate(data)


ScenarioKind = graphene.Enum.from_enum(ScenarioKindEnum)


class NodeInterface(graphene.Interface):
    id = graphene.ID(required=True)
    name = graphene.String(required=True)
    short_name = graphene.String(required=False)
    color = graphene.String()
    order = graphene.Int(required=False)
    is_visible = graphene.Boolean(required=True)
    unit = graphene.Field('paths.schema.UnitType')
    quantity = graphene.String()
    target_year_goal = graphene.Float(deprecation_reason='Replaced by "goals".')
    goals = graphene.List(graphene.NonNull(NodeGoal), active_goal=graphene.ID(required=False), required=True)
    is_action = graphene.Boolean(required=True, deprecation_reason='Use __typeName instead')
    explanation = graphene.String(required=False)

    visualizations = graphene.List(graphene.NonNull(VisualizationEntry), required=False)

    input_nodes = graphene.List(graphene.NonNull(lambda: NodeInterface), required=True)
    output_nodes = graphene.List(graphene.NonNull(lambda: NodeInterface), required=True)
    downstream_nodes = graphene.List(
        graphene.NonNull(lambda: NodeInterface),
        max_depth=graphene.Int(required=False),
        only_outcome=graphene.Boolean(required=False),
        until_node=graphene.ID(required=False),
        required=True,
    )
    upstream_nodes = graphene.List(
        graphene.NonNull(lambda: NodeInterface),
        same_unit=graphene.Boolean(),
        same_quantity=graphene.Boolean(),
        include_actions=graphene.Boolean(),
        required=True,
    )

    # TODO: Many nodes will output multiple time series. Remove metric
    # and handle a single-metric node as a special case in the UI??
    metric = graphene.Field(ForecastMetricType, goal_id=graphene.ID(required=False))
    outcome = graphene.Field(DimensionalMetricType)

    # If resolving through `descendant_nodes`, `impact_metric` will be
    # by default be calculated from the ancestor node.
    impact_metric = graphene.Field(
        ForecastMetricType,
        target_node_id=graphene.ID(required=False),
        goal_id=graphene.ID(required=False),
    )
    impact_metrics = graphene.List(graphene.NonNull(ForecastMetricType), required=True)

    metrics = graphene.List(graphene.NonNull(ForecastMetricType))
    dimensional_flow = graphene.Field(DimensionalFlowType, required=False)
    # metric_dim = graphene.Field(
    #    DimensionalMetricType, include_impact=graphene.Boolean(default=False), impact_action_node=graphene.ID(required=False),
    #    required=False
    # )
    metric_dim = graphene.Field(
        DimensionalMetricType,
        with_scenarios=graphene.List(graphene.NonNull(graphene.String), required=False),
        include_scenario_kinds=graphene.List(graphene.NonNull(ScenarioKind), required=False),
    )

    # TODO: input_datasets, baseline_values, context
    parameters = graphene.List(graphene.NonNull('params.schema.ParameterInterface'), required=True)

    # These are potentially plucked from nodes.models.NodeConfig
    short_description = RichText()
    description = graphene.String()
    body = graphene.List(graphene.NonNull(StreamFieldInterface))

    @classmethod
    def resolve_type(cls, instance: Node, info: GQLInstanceInfo) -> type[ActionNodeType | NodeType]:  # noqa: ARG003
        if isinstance(instance, ActionNode):
            return ActionNodeType
        return NodeType

    @staticmethod
    def resolve_color(root: Node, _info) -> str | None:
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

    @staticmethod
    def resolve_is_visible(root: Node, _info) -> bool:
        nc = root.db_obj
        if nc and nc.is_visible:
            return nc.is_visible
        return root.is_visible

    @staticmethod
    def resolve_is_action(root: Node, _info) -> bool:
        return isinstance(root, ActionNode)

    @staticmethod
    def resolve_downstream_nodes(
        root: Node, info: GQLInstanceInfo, max_depth: int | None = None, only_outcome: bool = False, until_node: str | None = None
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

    @staticmethod
    def resolve_upstream_nodes(
        root: Node,
        _info: GQLInstanceInfo,
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

    @staticmethod
    def resolve_metric(root: Node, _info: GQLInstanceInfo, goal_id: str | None = None) -> None | Metric:
        return Metric.from_node(root, goal_id=goal_id)

    @staticmethod
    def resolve_dimensional_flow(root: Node, _info: GraphQLResolveInfo) -> None | DimensionalFlow:
        if not isinstance(root, ActionNode):
            return None
        return DimensionalFlow.from_action_node(root)

    @staticmethod
    def resolve_metric_dim(
        root: Node,
        info: GQLInstanceInfo,
        with_scenarios: list[str] | None = None,
        include_scenario_kinds: list[ScenarioKindEnum] | None = None,
    ) -> None | DimensionalMetric:
        context = info.context.instance.context
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
            logging.exception('Exception while resolving metric_dim for node %s' % root.id)  # noqa: LOG015
            return None
        return ret

    @staticmethod
    def resolve_parameters(root: Node, info) -> list[Parameter[Any]]:
        return [param for param in root.parameters.values() if param.is_visible]

    @staticmethod
    def resolve_name(root: Node, info: GQLInstanceInfo) -> str | None:
        nc = root.db_obj
        if nc is not None and nc.name_i18n:
            return nc.name_i18n
        return str(root.name)

    @staticmethod
    def resolve_short_description(root: Node, info: GQLInstanceInfo) -> str | None:
        nc = root.db_obj
        if nc is not None and nc.short_description_i18n:
            return expand_db_html(nc.short_description_i18n)
        if root.description:
            desc = str(root.description)
            if desc:
                html = markdown.render(desc)
                return html
        return None

    @staticmethod
    def resolve_description(root: Node, info: GQLInstanceInfo) -> str | None:
        nc = root.db_obj
        if nc is None or not nc.description_i18n:
            if info.context.instance.features.show_explanations:
                return root.get_explanation()
            return None
        return expand_db_html(nc.description_i18n)

    @staticmethod
    def resolve_explanation(root: Node, info: GQLInstanceInfo) -> str | None:
        # nc = root.db_obj
        # if nc is None or not nc.description_i18n:
        #     return None
        # return expand_db_html(nc.description_i18n)
        return root.get_explanation()

    @staticmethod
    def resolve_body(root: Node, _info: GQLInstanceInfo) -> StreamValue | None:
        nc = root.db_obj
        if nc is None or not nc.body:
            return None
        return nc.body

    @staticmethod
    def resolve_goals(root: Node, info: GQLInstanceInfo, active_goal: str | None = None) -> Sequence[YearlyValueProtocol]:
        if root.goals is None:
            return []
        goal = None
        if active_goal:
            agoal = info.context.instance.get_goals(active_goal)
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
        return goal.get_values()

    @staticmethod
    def resolve_target_year_goal(root: Node, info: GQLInstanceInfo) -> None | float:
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

    @staticmethod
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

    @staticmethod
    def resolve_impact_metric(
        root: Node,
        info: GQLInstanceInfo,
        target_node_id: str | None = None,
        goal_id: str | None = None,
    ) -> None | Metric:
        instance = info.context.instance
        context = instance.context
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

        return NodeInterface._get_impact_metric(source_node, target_node, goal)

    @staticmethod
    def resolve_impact_metrics(root: Node, info: GQLInstanceInfo) -> list[Metric]:
        if not isinstance(root, ActionNode):
            return []
        metrics = []
        for outcome_node in root.get_downstream_nodes(only_outcome=True):
            metric = NodeInterface._get_impact_metric(root, outcome_node)
            if metric is not None:
                metrics.append(metric)
        return metrics

    @staticmethod
    def resolve_visualizations(root: Node, info: GQLInstanceInfo) -> list[viz.VisualizationEntryType] | None:
        viz = root.visualizations
        if not viz:
            return None
        return viz.root


class NodeType(graphene.ObjectType):
    class Meta:
        name = 'Node'
        interfaces = (NodeInterface,)

    upstream_actions = graphene.List(
        graphene.NonNull(lambda: ActionNodeType, required=True),
        only_root=graphene.Boolean(required=False, default_value=False),
        decision_level=ActionDecisionLevel(required=False),
    )

    @staticmethod
    def resolve_upstream_actions(
        root: Node,
        info: GQLInstanceInfo,
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


class ActionNodeType(graphene.ObjectType):
    class Meta:
        interfaces = (NodeInterface,)
        name = 'ActionNode'

    parent_action = graphene.Field(lambda: ActionNodeType, required=False)
    subactions = graphene.List(graphene.NonNull(lambda: ActionNodeType), required=True)

    group = graphene.Field(ActionGroupType, required=False)
    decision_level = graphene.Field(ActionDecisionLevel)
    goal = RichText(required=False)
    indicator_node = graphene.Field(NodeType, required=False)

    is_enabled = graphene.Boolean(required=True)

    @staticmethod
    def resolve_group(root: ActionNode, info: GQLInstanceInfo) -> ActionGroup | None:
        return root.group

    @staticmethod
    def resolve_parent_action(root: ActionNode, info: GQLInstanceInfo) -> ActionNode | None:
        return root.parent_action

    @staticmethod
    def resolve_subactions(root: ActionNode, info: GQLInstanceInfo) -> list[ActionNode]:
        if not isinstance(root, ParentActionNode):
            return []
        return root.subactions

    @staticmethod
    def resolve_is_enabled(root: ActionNode, info: GQLInstanceInfo) -> bool:
        return bool(root.is_enabled())

    @staticmethod
    def resolve_goal(root: ActionNode, info: GQLInstanceInfo) -> str | None:
        nc = root.db_obj
        if nc is None:
            return None
        val = nc.goal_i18n
        if val:
            return expand_db_html(val)
        return None

    @staticmethod
    def resolve_indicator_node(root: ActionNode, info: GQLInstanceInfo) -> Node | None:
        nc = root.db_obj
        if nc is None:
            return None
        if nc.indicator_node is None:
            return None
        return nc.indicator_node.get_node(visible_for_user=info.context.user)


class ScenarioType(graphene.ObjectType):
    id = graphene.ID(required=True)
    name = graphene.String(required=True)
    kind = ScenarioKind(required=False)
    is_active = graphene.Boolean(required=True)
    is_default = graphene.Boolean(required=True)
    is_selectable = graphene.Boolean(required=True)
    actual_historical_years = graphene.List(graphene.NonNull(graphene.Int), required=False)

    @staticmethod
    def resolve_is_active(root: Scenario, info: GQLInstanceInfo) -> bool:
        context = info.context.instance.context
        return context.active_scenario == root

    @staticmethod
    def resolve_is_default(root: Scenario, info: GQLInfo) -> bool:
        return root.default


class ActionImpact(graphene.ObjectType):
    action = graphene.Field(ActionNodeType, required=True)
    cost_values = graphene.List(YearlyValue, required=True, deprecation_reason="Use costDim instead.")
    impact_values = graphene.List(YearlyValue, required=True, deprecation_reason="Use effectDim instead.")
    cost_dim = graphene.Field(DimensionalMetricType, required=True)
    impact_dim = graphene.Field(DimensionalMetricType, required=True, deprecation_reason="Use effectDim instead.")
    effect_dim = graphene.Field(DimensionalMetricType, required=True)
    efficiency_divisor = graphene.Float()  # FIXME AEP depreciated
    unit_adjustment_multiplier = graphene.Float()  # To replace efficiency_divisor


class ImpactOverviewType(graphene.ObjectType):
    id = graphene.ID(required=True)
    graph_type = graphene.String()
    cost_node = graphene.Field(NodeType, required=True)
    effect_node = graphene.Field(NodeType, required=True)
    impact_node = graphene.Field(
        NodeType,
        required=True,
        deprecation_reason="Use effectNode instead."
    )
    efficiency_unit = graphene.Field(
        'paths.schema.UnitType',
        required=True,
        deprecation_reason="Use indicatorUnit instead"
    )  # FIXME depreciated
    indicator_unit = graphene.Field('paths.schema.UnitType', required=True)
    cost_unit = graphene.Field('paths.schema.UnitType', required=True)
    effect_unit = graphene.Field('paths.schema.UnitType', required=True)
    impact_unit = graphene.Field(
        'paths.schema.UnitType',
        required=True,
        deprecation_reason="Use indicatorUnit instead")
    indicator_cutpoint = graphene.Float()  # For setting decision criterion on the indicator. Uses indicator units
    cost_cutpoint = graphene.Float()  # For setting decision criterion on the cost. Uses cost units
    plot_limit_efficiency = graphene.Float( # FIXME Remove from UI and here.
        deprecation_reason="Use plot_limit_indicator instead"
    )
    plot_limit_indicator = graphene.Float() # FIXME Depreciated
    plot_limit_for_indicator = graphene.Float()
    invert_cost = graphene.Boolean(required=True, deprecation_reason="Not needed") # FIXME Depreciated
    invert_effect = graphene.Boolean(required=True, deprecation_reason="Not needed") # FIXME Depreciated
    invert_impact = graphene.Boolean(required=True, deprecation_reason="Not needed")
    label = graphene.String(required=True)
    actions = graphene.List(graphene.NonNull(ActionImpact), required=True)

    @staticmethod
    def resolve_id(root: ImpactOverview, info: GQLInstanceInfo) -> str:
        cost_id = root.cost_node.id if root.cost_node else 'None'
        return '%s:%s' % (cost_id, root.effect_node.id)

    @staticmethod
    def resolve_plot_limit_efficiency(root: ImpactOverview, info: GQLInstanceInfo) -> float | None:
        return root.plot_limit_for_indicator

    @staticmethod
    def resolve_graph_type(root: ImpactOverview, info: GQLInstanceInfo) -> str:
        if root.graph_type == 'return_on_investment':
            graph_type = 'return_of_investment' # FIXME Depreciated, remove when UI accepts 'on'
        else:
            graph_type = root.graph_type
        return graph_type

    @staticmethod
    def resolve_actions(root: ImpactOverview, info: GQLInstanceInfo) -> list[dict[str, Any]]:
        all_aes = root.calculate(info.context.instance.context)
        out: list[dict] = []
        for ae in all_aes:
            if ae.unit_adjustment_multiplier is not None:
                ed = 1 / ae.unit_adjustment_multiplier
            else:
                ed = None
            years = ae.df[YEAR_COLUMN]
            d = dict(
                action=ae.action,
                cost_values=[YearlyValue(year, float(val)) for year, val in zip(years, list(ae.df['Cost']), strict=False)],
                impact_values=[YearlyValue(year, float(val)) for year, val in zip(years, list(ae.df['Effect']), strict=False)],
                cost_dim=DimensionalMetric.from_action_impact(ae, root, 'Cost'),
                impact_dim=DimensionalMetric.from_action_impact(ae, root, 'Effect'), # FIXME Deprecated
                effect_dim=DimensionalMetric.from_action_impact(ae, root, 'Effect'),
                efficiency_divisor= ed, # FIXME Depreciated
                unit_adjustment_multiplier=ae.unit_adjustment_multiplier,
            )
            out.append(d)
        return out

    @staticmethod
    def resolve_efficiency_unit(root: ImpactOverview, info: GQLInstanceInfo) -> Unit:  # FIXME depreciated.
        return root.indicator_unit

    @staticmethod
    def resolve_indicator_unit(root: ImpactOverview, info: GQLInstanceInfo) -> Unit:
        return root.indicator_unit

    @staticmethod
    def resolve_cost_unit(root: ImpactOverview, info: GQLInstanceInfo) -> Unit:
        return root.cost_unit or root.indicator_unit

    @staticmethod
    def resolve_effect_unit(root: ImpactOverview, info: GQLInstanceInfo) -> Unit:
        return root.effect_unit or root.indicator_unit

    @staticmethod
    def resolve_impact_unit(root: ImpactOverview, info: GQLInstanceInfo) -> Unit:
        return root.effect_unit or root.indicator_unit

    @staticmethod
    def resolve_impact_node(root: ImpactOverview, info: GQLInstanceInfo) -> Node:
        return root.effect_node

    @staticmethod
    def resolve_invert_impact(root: ImpactOverview, info: GQLInstanceInfo) -> bool:
        return False

    @staticmethod
    def resolve_invert_effect(root: ImpactOverview, info: GQLInstanceInfo) -> bool:
        return False

    @staticmethod
    def resolve_invert_cost(root: ImpactOverview, info: GQLInstanceInfo) -> bool:
        return False


class InstanceBasicConfiguration(graphene.ObjectType):
    identifier = graphene.String(required=True)
    is_protected = graphene.Boolean(required=True)
    requires_authentication = graphene.Boolean(required=True)
    default_language = graphene.String(required=True)
    theme_identifier = graphene.String(required=True)
    supported_languages = graphene.List(graphene.NonNull(graphene.String), required=True)
    hostname = graphene.Field(InstanceHostnameType, required=True)

    @staticmethod
    def resolve_identifier(root: Instance, info: GQLInfo) -> str:
        return root.id

    @staticmethod
    def resolve_is_protected(root: Instance, info: GQLInfo) -> bool:
        ic: InstanceConfig = root._config  # type: ignore
        return ic.is_protected

    @staticmethod
    def resolve_requires_authentication(root: Instance, info: GQLInfo) -> bool:
        ic: InstanceConfig = root._config  # type: ignore
        return ic.get_instance().features.requires_authentication

    @staticmethod
    def resolve_hostname(root: Instance, info: GQLInfo) -> InstanceHostname:
        ic: InstanceConfig = root._config  # type: ignore
        hostname: str = root._hostname  # type: ignore
        hn_obj = ic.hostnames.filter(hostname=hostname.lower()).first()
        if not hn_obj:
            return InstanceHostname(hostname=hostname, base_path='')
        return InstanceHostname(hostname=hn_obj.hostname, base_path=hn_obj.base_path)


class NormalizationType(graphene.ObjectType):
    id = graphene.ID(required=True)
    label = graphene.String(required=True)
    normalizer = graphene.Field(NodeType, required=True)
    is_active = graphene.Boolean(required=True)

    @staticmethod
    def resolve_is_active(root: Normalization, info: GQLInstanceInfo) -> bool:
        return info.context.instance.context.active_normalization == root

    @staticmethod
    def resolve_label(root: Normalization, info: GQLInstanceInfo) -> str:
        return str(root.normalizer_node.name)

    @staticmethod
    def resolve_normalizer(root: Normalization, info: GQLInstanceInfo) -> Node:
        return root.normalizer_node

    @staticmethod
    def resolve_id(root: Normalization, info: GQLInstanceInfo) -> str:
        return root.normalizer_node.id


class Query(graphene.ObjectType):
    available_instances = graphene.List(
        graphene.NonNull(InstanceBasicConfiguration),
        hostname=graphene.String(),
        required=True,
    )
    instance = graphene.Field(InstanceType, required=True)
    nodes = graphene.List(graphene.NonNull(NodeInterface), required=True)
    node = graphene.Field(
        NodeInterface,
        id=graphene.ID(required=True),
    )
    actions = graphene.List(
        graphene.NonNull(ActionNodeType),
        only_root=graphene.Boolean(required=False, default_value=False),
        required=True,
    )
    action = graphene.Field(ActionNodeType, id=graphene.ID(required=True))
    action_efficiency_pairs = graphene.List(
        graphene.NonNull(ImpactOverviewType),
        required=True,
        deprecation_reason="Use impactOverviews instead"
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

    @ensure_instance
    def resolve_nodes(root, info: GQLInstanceInfo) -> Iterable[Node]:
        instance = info.context.instance
        return instance.context.nodes.values()

    @pass_context
    def resolve_actions(root, info: GQLInstanceInfo, context: Context, only_root: bool):
        instance = info.context.instance
        actions = instance.context.get_actions()
        if only_root:
            actions = list(filter(lambda act: act.parent_action is None, actions))
        return actions

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

    @staticmethod
    def resolve_available_instances(root, info: GQLInfo, hostname: str) -> list[Instance]:
        qs = InstanceConfig.objects.get_queryset().for_hostname(hostname, request=info.context)
        instances: list[Instance] = []
        for config in qs:
            instance = config.get_instance()
            instance._config = config  # type: ignore
            instance._hostname = hostname  # type: ignore
            instances.append(instance)
        return instances


class SetNormalizerMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=False)

    ok = graphene.Boolean(required=True)
    active_normalization = graphene.Field(NormalizationType, required=False)

    @pass_context
    def mutate(root, info: GQLInstanceInfo, context: Context, id: str | None = None):
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

        return dict(ok=True, active_normalizer=context.active_normalization)


class Mutations(graphene.ObjectType):
    set_normalizer = SetNormalizerMutation.Field()
