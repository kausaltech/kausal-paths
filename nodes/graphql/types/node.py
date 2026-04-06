import re
from typing import TYPE_CHECKING, Annotated, Any, Optional  # pyright: ignore[reportDeprecated]

import strawberry as sb
from graphql.error import GraphQLError
from wagtail.blocks.stream_block import StreamValue
from wagtail.rich_text import RichText as WagtailRichText, expand_db_html

import sentry_sdk
from grapple.types.streamfield import StreamFieldInterface
from markdown_it import MarkdownIt

from kausal_common.strawberry.grapple import grapple_field
from kausal_common.strawberry.pydantic import StrawberryPydanticType, pydantic_type
from kausal_common.strawberry.registry import register_strawberry_type

from paths import gql
from paths.graphql_helpers import get_instance_context, pass_context
from paths.graphql_types import UnitType

from nodes import visualizations as viz
from nodes.actions.action import ActionNode
from nodes.constants import DecisionLevel
from nodes.defs import SimpleConfig
from nodes.defs.node_defs import ActionConfig, FormulaConfig, NodeKind, NodeSpec, PipelineConfig
from nodes.graph_layout import NodeGraphLayoutMeta
from nodes.graphql.types import DatasetPortType
from nodes.graphql.types.impact import get_impact_metric
from nodes.quantities import get_registry as get_quantity_registry
from nodes.scenario import Scenario, ScenarioKind
from params import Parameter

from .graph import ActionGroupType, NodeEdgeType
from .metric import (
    DimensionalFlowType,
    DimensionalMetricType,
    ForecastMetricType,
    NodeGoal,
    VisualizationEntry,
)
from .spec import InputPortType, OutputPortType

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.defs.instance_defs import ActionGroup
    from nodes.metric import DimensionalFlow, DimensionalMetric, Metric
    from nodes.models import NodeConfig
    from nodes.node import Node
    from nodes.quantities import QuantityKind
    from params.schema import ParameterInterface


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


@pydantic_type(model=ActionConfig)
class ActionConfigType:
    node_class: sb.auto
    decision_level: sb.auto
    group: sb.auto
    parent: sb.auto
    no_effect_value: sb.auto


@pydantic_type(model=SimpleConfig)
class SimpleConfigType:
    node_class: sb.auto


@pydantic_type(model=FormulaConfig)
class FormulaConfigType:
    formula: str


@pydantic_type(model=PipelineConfig)
class PipelineConfigType:
    operations: sb.scalars.JSON


def _require_nc(spec_type: 'NodeSpecType') -> NodeConfig:  # noqa: UP037
    if spec_type._node is None:
        raise ValueError('NodeSpecType has no Node instance')
    nc = spec_type._node.db_obj
    if nc is None:
        raise ValueError('NodeSpecType has no NodeConfig instance')
    return nc


@pydantic_type(model=NodeSpec)
class NodeSpecType(StrawberryPydanticType[NodeSpec]):
    type_config: Annotated[
        ActionConfigType | SimpleConfigType | FormulaConfigType | PipelineConfigType,
        sb.union('NodeConfigUnion'),
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
        edges_by_id: dict[Any, list[NodeEdgeType | DatasetPortType]] = {}
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
                node=root._node,
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
    def spec(root: 'Node', info: gql.Info) -> NodeSpecType | None:
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
        info.context._upstream_node = root  # type: ignore[attr-defined]
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

    @sb.field(graphql_type=ForecastMetricType | None)
    @staticmethod
    def metric(root: 'Node', goal_id: sb.ID | None = None) -> 'Metric | None':
        from nodes.metric import Metric

        return Metric.from_node(root, goal_id=goal_id)

    @sb.field(graphql_type=DimensionalMetricType | None)
    @staticmethod
    def outcome(root: 'Node') -> 'DimensionalMetric | None':
        return getattr(root, 'outcome', None)

    @sb.field(graphql_type=ForecastMetricType | None)
    @pass_context
    @staticmethod
    def impact_metric(
        root: 'Node',
        info: gql.Info,
        context: 'Context',
        target_node_id: sb.ID | None = None,
        goal_id: sb.ID | None = None,
    ) -> 'Metric | None':
        from nodes.actions.action import ActionNode

        instance = context.instance
        upstream_node: 'Node | None' = getattr(info.context, '_upstream_node', None)

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
            outcome_nodes = context.get_outcome_nodes()
            if not len(outcome_nodes):
                raise GraphQLError('No default target node available', info.field_nodes)
            source_node = root
            target_node = outcome_nodes[0]

        if not isinstance(source_node, ActionNode):
            return None

        return get_impact_metric(source_node, target_node, goal)

    @sb.field(ForecastMetricType, graphql_type=list[ForecastMetricType])
    @staticmethod
    def impact_metrics(root: 'Node') -> list['Metric']:
        from nodes.actions.action import ActionNode

        if not isinstance(root, ActionNode):
            return []
        metrics = []
        for outcome_node in root.get_downstream_nodes(only_outcome=True):
            metric_val = get_impact_metric(root, outcome_node)
            if metric_val is not None:
                metrics.append(metric_val)
        return metrics

    @sb.field(graphql_type=list[ForecastMetricType] | None)
    @staticmethod
    def metrics(root: 'Node') -> list['Metric'] | None:
        return getattr(root, 'metrics', None)

    @sb.field(graphql_type=DimensionalFlowType | None)
    @staticmethod
    def dimensional_flow(root: 'Node') -> 'DimensionalFlow | None':
        from nodes.actions.action import ActionNode
        from nodes.metric import DimensionalFlow

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
    ) -> 'DimensionalMetric | None':
        from nodes.metric import DimensionalMetric

        context = get_instance_context(info)
        extra_scenarios: list[Scenario] = []
        for scenario_id in with_scenarios or []:
            if scenario_id not in context.scenarios:
                sentry_sdk.capture_message('Scenario %s not found' % scenario_id, level='error')
                continue
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

    @sb.field(graphql_type=list[Annotated['ParameterInterface', sb.lazy('params.schema')]])
    @staticmethod
    def parameters(root: 'Node') -> list[Parameter[Any]]:
        return [param for param in root.parameters.values() if param.is_visible]

    @grapple_field
    @staticmethod
    def short_description(root: 'Node') -> WagtailRichText | None:
        nc = root.db_obj
        if nc is not None and nc.short_description_i18n:
            return expand_db_html(nc.short_description_i18n)
        if root.description:
            desc = str(root.description)
            if desc:
                return markdown.render(desc)
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
        from nodes.actions.action import ActionNode
        from nodes.node import Node

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
        from nodes.actions.action import ActionNode

        return isinstance(obj, ActionNode)

    @sb.field
    @staticmethod
    def is_enabled(root: ActionNode) -> bool:
        return bool(root.is_enabled())

    @sb.field(graphql_type=Optional[Annotated['ActionNodeType', sb.lazy('nodes.schema')]])  # noqa: UP045  # pyright: ignore[reportDeprecated]
    @staticmethod
    def parent_action(root: ActionNode) -> ActionNode | None:
        return root.parent_action

    @sb.field(graphql_type=list[Annotated['ActionNodeType', sb.lazy('nodes.schema')]])
    @staticmethod
    def subactions(root: ActionNode) -> list[ActionNode]:
        from nodes.actions.parent import ParentActionNode

        if not isinstance(root, ParentActionNode):
            return []
        return root.subactions

    @sb.field(graphql_type=ActionGroupType | None)
    @staticmethod
    def group(root: ActionNode) -> 'ActionGroup | None':
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

    @sb.field(graphql_type=Optional[Annotated['NodeType', sb.lazy('nodes.graphql.types')]])  # noqa: UP045  # pyright: ignore[reportDeprecated]
    @staticmethod
    def indicator_node(root: ActionNode, info: gql.Info) -> 'Node | None':
        nc = root.db_obj
        if nc is None:
            return None
        if nc.indicator_node is None:
            return None
        return nc.indicator_node.get_node(visible_for_user=info.context.user)


AnyNodeType = Annotated[ActionNodeType | NodeType, sb.union('AnyNodeType')]
