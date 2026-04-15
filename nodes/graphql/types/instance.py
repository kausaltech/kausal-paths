from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Protocol
from uuid import UUID

import strawberry as sb
from strawberry import auto
from wagtail.blocks.stream_block import StreamValue

from grapple.types.streamfield import StreamFieldInterface

from kausal_common.strawberry.grapple import grapple_field
from kausal_common.strawberry.pydantic import StrawberryPydanticType

from paths import gql
from paths.graphql_helpers import pass_context
from paths.graphql_types import UnitType

from datasets.graphql import DatasetType
from nodes.defs import InstanceSpec
from nodes.defs.instance_defs import InstanceFeatures
from nodes.goals import GoalActualValue, NodeGoalsEntry
from nodes.graph_layout import GraphLayout
from nodes.graphql.types.dimension import DimensionType
from nodes.instance import Instance
from nodes.models import InstanceConfig
from nodes.node import Node
from nodes.normalization import Normalization
from nodes.units import Unit
from pages.models import ActionListPage

from .graph import (
    ActionGroupType,
    DatasetMetricRefType,
    DatasetPortType,
    InstanceHostname,
    NodeEdgeType,
    NodePortRef,
    _external_dataset_id_from_dataset,
)
from .spec import InstanceSpecType, YearsDefType

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.graphql.types.node import NodeInterface, NodeType


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
        result = []
        for dp in dataset_ports:
            port = DatasetPortType(
                id=sb.ID(str(dp.uuid)),
                node_ref=NodePortRef(node_id=sb.ID(str(dp.node.identifier)), port_id=dp.port_id),
                metric=DatasetMetricRefType.from_model(dp.metric),
                external_dataset_id=_external_dataset_id_from_dataset(dp.dataset),
                external_metric_id=dp.metric.name,
            )
            port._dataset = DatasetType.from_model(dp.dataset)
            result.append(port)
        return result

    @sb.field(graphql_type=list[DatasetType])
    @staticmethod
    def datasets(root: Instance) -> list[DatasetType]:
        """All DB-backed datasets scoped to this instance."""
        from kausal_common.datasets.models import Dataset as DatasetModel

        ic = root.config
        qs = DatasetModel.objects.get_queryset().for_instance_config(ic).select_related('schema')
        # FIXME: Permission checks
        return [DatasetType.from_model(ds) for ds in qs]

    @sb.field(graphql_type=list[DimensionType])
    @staticmethod
    def dimensions(root: Instance) -> list[DimensionType]:
        """All dimensions scoped to this model instance."""
        from kausal_common.datasets.models import DimensionScope

        ic = root.config
        scopes = (
            DimensionScope.objects
            .for_instance_config(ic)
            .select_related('dimension')
            .prefetch_related('dimension__categories')
            .order_by('order')
        )
        return [DimensionType.from_scope(scope) for scope in scopes]

    @sb.field(graphql_type=list[Annotated['NodeInterface', sb.lazy('nodes.schema')]])
    @staticmethod
    def nodes(root: Instance, id: list[sb.ID] | None = None) -> list[Node]:
        if id is not None:
            nodes: list[Node] = []
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
        ic: InstanceConfig = root._config  # type: ignore[attr-defined]
        return ic.is_protected

    @sb.field
    @staticmethod
    def requires_authentication(root: Instance) -> bool:
        ic: InstanceConfig = root._config  # type: ignore[attr-defined]
        return ic.get_instance().features.requires_authentication

    @sb.field
    @staticmethod
    def hostname(root: Instance) -> InstanceHostname:
        ic: InstanceConfig = root._config  # type: ignore[attr-defined]
        hostname: str = root._hostname  # type: ignore[attr-defined]
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

    @sb.field(graphql_type=Annotated['NodeType', sb.lazy('nodes.schema')])
    @staticmethod
    def normalizer(root: Normalization) -> Node:
        return root.normalizer_node

    @sb.field
    @pass_context
    @staticmethod
    def is_active(root: 'Normalization', context: 'Context') -> bool:
        return context.active_normalization == root
