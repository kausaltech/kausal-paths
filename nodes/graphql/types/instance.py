import enum
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Protocol, Self, cast
from uuid import UUID

import strawberry as sb
from django.utils import timezone
from strawberry import auto
from wagtail.blocks.stream_block import StreamValue

from grapple.types.streamfield import StreamFieldInterface

from kausal_common.models.uuid import query_pk_or_uuid_or_identifier
from kausal_common.strawberry.grapple import grapple_field
from kausal_common.strawberry.pydantic import StrawberryPydanticType

from paths import gql
from paths.graphql_helpers import pass_context
from paths.graphql_types import UnitType

from datasets.graphql import DatasetType
from frameworks.models import FrameworkConfig
from nodes.defs import InstanceModelSpec
from nodes.defs.instance_defs import InstanceFeatures
from nodes.goals import GoalActualValue, NodeGoalsEntry
from nodes.graph_layout import GraphLayout
from nodes.graphql.types.dimension import DimensionType
from nodes.instance import Instance
from nodes.models import InstanceConfig
from nodes.node import Node
from nodes.normalization import Normalization
from nodes.quantities import get_registry as get_quantity_registry
from nodes.units import Unit
from pages.models import ActionListPage
from users.models import User

from .graph import (
    ActionGroupType,
    DatasetMetricRefType,
    DatasetPortType,
    InstanceHostname,
    NodeEdgeType,
    NodePortRef,
    _external_dataset_id_from_dataset,
)
from .node import QuantityKindType
from .spec import InstanceSpecType, YearsDefType

if TYPE_CHECKING:
    from datasets.graphql.types import DataSourceType  # used in lazy strawberry annotation
    from frameworks.schema import FrameworkConfigType  # used in lazy strawberry annotation
    from nodes.context import Context
    from nodes.graphql.types.change_history import InstanceChangeOperationType
    from nodes.graphql.types.node import NodeInterface, NodeType
    from nodes.models import InstanceInvitation
    from users.graphql.mutations import InstanceInvitationType  # used in lazy strawberry annotation
    from users.schema import UserType  # used in lazy strawberry annotation


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


def _instance_editor_allowed(ic: InstanceConfig, info: gql.Info) -> bool:
    return ic.gql_action_allowed(info, 'change', raise_on_denied=False)


def _instance_admin_allowed(ic: InstanceConfig, info: gql.Info) -> bool:
    from kausal_common.users import user_or_none

    user = user_or_none(info.context.user)
    if user is None:
        return False
    if user.is_superuser:
        return True
    if ic.owned_by_id == user.pk:
        return True
    return ic.permission_policy().is_admin(user, ic)


def _node_is_publicly_visible(node: Node) -> bool:
    nc = node.db_obj
    if nc is not None:
        return nc.is_visible
    return node.is_visible


@sb.enum(name='InstanceMemberRole')
class InstanceMemberRole(enum.Enum):
    SUPER_ADMIN = 'super_admin'
    ADMIN = 'admin'
    REVIEWER = 'reviewer'
    VIEWER = 'viewer'


@sb.type(name='InstanceMember', description='A user with a role on this instance.')
class InstanceMemberType:
    user: User = sb.field(graphql_type=Annotated['UserType', sb.lazy('users.schema')])
    role: InstanceMemberRole
    is_owner: bool


def _collect_instance_members(ic: InstanceConfig) -> list[InstanceMemberType]:
    from users.models import User as _User

    pp = ic.permission_policy()
    role_priority: list[tuple[InstanceMemberRole, Any]] = [
        (InstanceMemberRole.SUPER_ADMIN, pp.super_admin_role.get_existing_instance_group(ic)),
        (InstanceMemberRole.ADMIN, pp.admin_role.get_existing_instance_group(ic)),
        (InstanceMemberRole.REVIEWER, pp.reviewer_role.get_existing_instance_group(ic)),
        (InstanceMemberRole.VIEWER, pp.viewer_role.get_existing_instance_group(ic)),
    ]
    role_by_user_pk: dict[int, InstanceMemberRole] = {}
    for role, group in role_priority:
        if group is None:
            continue
        for user_pk in group.user_set.values_list('pk', flat=True):
            role_by_user_pk.setdefault(user_pk, role)

    owner_pk = ic.owned_by_id
    if owner_pk is not None:
        role_by_user_pk.setdefault(owner_pk, InstanceMemberRole.ADMIN)

    if not role_by_user_pk:
        return []
    users = _User.objects.filter(pk__in=list(role_by_user_pk.keys()))
    return [
        InstanceMemberType(
            user=u,
            role=role_by_user_pk[u.pk],
            is_owner=(u.pk == owner_pk),
        )
        for u in users
    ]


@sb.type(name='QuantityKindUnitUsage')
class QuantityKindUnitUsageType:
    unit: Unit = sb.field(graphql_type=UnitType)
    count: int


@sb.type(name='InstanceQuantityKind')
class InstanceQuantityKindType:
    kind: QuantityKindType
    used_units: list[QuantityKindUnitUsageType]


def _collect_quantity_kind_unit_usage(instance: Instance) -> dict[str, list[QuantityKindUnitUsageType]]:
    counts_by_quantity: dict[str, Counter[str]] = {}
    units_by_key: dict[tuple[str, str], Unit] = {}
    for node in instance.context.nodes.values():
        for metric in node.output_metrics.values():
            quantity = metric.quantity
            unit_key = str(metric.unit)
            counts_by_quantity.setdefault(quantity, Counter())[unit_key] += 1
            units_by_key[(quantity, unit_key)] = metric.unit

    return {
        quantity: [
            QuantityKindUnitUsageType(
                unit=units_by_key[(quantity, unit_key)],
                count=count,
            )
            for unit_key, count in sorted(unit_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        for quantity, unit_counts in counts_by_quantity.items()
    }


@sb.type(name='InstanceEditor')
class InstanceEditorFields:
    _config: sb.Private[InstanceConfig]
    _instance: sb.Private[Instance | None] = None

    def runtime_instance(self) -> Instance:
        if self._instance is None:
            self._instance = self._config.get_instance()
        return self._instance

    @sb.field
    @staticmethod
    def config_source(root: 'InstanceEditorFields') -> str:
        return root._config.config_source

    @sb.field
    @staticmethod
    def is_locked(root: 'InstanceEditorFields') -> bool:
        return root._config.is_locked

    @sb.field
    @staticmethod
    def live(root: 'InstanceEditorFields') -> bool:
        return root._config.live

    @sb.field
    @staticmethod
    def has_unpublished_changes(root: 'InstanceEditorFields') -> bool:
        return root._config.has_unpublished_changes

    @sb.field(
        description=(
            'Optimistic-locking token for draft edits. Editing mutations must pass '
            'this value via `@instance(version: ...)` or `@context(input: { version: ... })`; '
            'mismatched tokens are rejected with a stale-version error. `null` if no '
            'edits have ever been recorded for this instance.'
        ),
    )
    @staticmethod
    def draft_head_token(root: 'InstanceEditorFields') -> UUID | None:
        return root._config.draft_head_token

    @sb.field
    @staticmethod
    def first_published_at(root: 'InstanceEditorFields') -> datetime | None:
        return root._config.first_published_at

    @sb.field
    @staticmethod
    def last_published_at(root: 'InstanceEditorFields') -> datetime | None:
        return root._config.last_published_at

    @sb.field(graphql_type=Annotated[InstanceSpecType | None, sb.lazy('nodes.schema_spec')])
    @staticmethod
    def spec(root: 'InstanceEditorFields') -> InstanceModelSpec | None:
        return root._config.spec

    @sb.field(graphql_type=list[NodeEdgeType])
    @staticmethod
    def edges(root: 'InstanceEditorFields') -> list[NodeEdgeType]:
        edges = root._config.edges.select_related('from_node', 'to_node')
        return [NodeEdgeType.from_node_edge(edge) for edge in edges]

    @sb.field(
        graphql_type=list[Annotated['InstanceChangeOperationType', sb.lazy('nodes.graphql.types.change_history')]],
        description=(
            'Audit trail of user-facing edits to this instance, newest first. '
            'Each operation bundles one or more row-level entries.'
        ),
    )
    @staticmethod
    def change_history(
        root: 'InstanceEditorFields',
        limit: int = 50,
        before: datetime | None = None,
    ) -> 'list[InstanceChangeOperationType]':
        from nodes.graphql.types.change_history import InstanceChangeOperationType
        from nodes.models import InstanceChangeOperation

        qs = (
            InstanceChangeOperation.objects
            .filter(instance_config=root._config)
            .select_related('user', 'superseded_by')
            .order_by('-created_at')
        )
        if before is not None:
            qs = qs.filter(created_at__lt=before)
        return [InstanceChangeOperationType.from_model(op) for op in qs[:limit]]

    @sb.field(graphql_type=list[DatasetPortType])
    @staticmethod
    def dataset_ports(root: 'InstanceEditorFields') -> list[DatasetPortType]:
        dataset_ports = root._config.dataset_ports.select_related('node', 'dataset', 'metric')
        result = []
        for dp in dataset_ports:
            port = DatasetPortType(
                id=sb.ID(str(dp.uuid)),
                uuid=dp.uuid,
                node_ref=NodePortRef(node_id=sb.ID(str(dp.node.identifier)), port_id=dp.port_id),
                metric=DatasetMetricRefType.from_model(dp.metric),
                external_dataset_id=_external_dataset_id_from_dataset(dp.dataset),
                external_metric_id=dp.metric.name,
            )
            port._dataset = DatasetType.from_model(dp.dataset)
            if port._dataset is not None:
                port._dataset._forecast_from = dp.spec.forecast_from
            result.append(port)
        return result

    @sb.field(graphql_type=list[DatasetType])
    @staticmethod
    def datasets(root: 'InstanceEditorFields', info: gql.Info) -> list[DatasetType]:
        """All DB-backed datasets scoped to this instance."""
        from kausal_common.datasets.models import Dataset as DatasetModel

        ic = root._config
        qs = DatasetModel.objects.get_queryset().for_instance_config(ic).viewable_by(info.context.user).select_related('schema')
        return [DatasetType.from_model(ds) for ds in qs]

    @sb.field(graphql_type=DatasetType | None)
    @staticmethod
    def dataset(
        root: 'InstanceEditorFields', info: gql.Info, id: Annotated[sb.ID, sb.argument(description='Dataset pk/uuid/identifier)')]
    ) -> DatasetType | None:
        """One instance-scoped dataset by id."""
        from kausal_common.datasets.models import Dataset as DatasetModel

        if not id.strip():
            return None
        ic = root._config
        qs = DatasetModel.objects.get_queryset().for_instance_config(ic).viewable_by(info.context.user).select_related('schema')
        qs = qs.filter(query_pk_or_uuid_or_identifier(id))
        try:
            ds = qs.get()
        except DatasetModel.DoesNotExist:
            return None
        return DatasetType.from_model(ds)

    @sb.field(
        graphql_type=list[Annotated['DataSourceType', sb.lazy('datasets.graphql.types')]],
        description='The library of DataSources scoped to this instance.',
    )
    @staticmethod
    def data_sources(root: 'InstanceEditorFields') -> list[Any]:
        from django.contrib.contenttypes.models import ContentType

        from kausal_common.datasets.models import DataSource

        ic = root._config
        ct = ContentType.objects.get_for_model(type(ic))
        return list(
            DataSource.objects
            .filter(scope_content_type=ct, scope_id=ic.pk)
            .select_related('created_by', 'last_modified_by')
            .order_by('name'),
        )

    @sb.field(graphql_type=list[DimensionType])
    @staticmethod
    def dimensions(root: 'InstanceEditorFields') -> list[DimensionType]:
        """All dimensions scoped to this model instance."""
        from kausal_common.datasets.models import DimensionScope

        ic = root._config
        scopes = (
            DimensionScope.objects
            .for_instance_config(ic)
            .select_related('dimension')
            .prefetch_related('dimension__categories')
            .order_by('order')
        )
        return [DimensionType.from_scope(scope) for scope in scopes]

    @sb.field(
        graphql_type=list[InstanceQuantityKindType],
        description='All registered quantity kinds, with units already used in this instance ordered by frequency.',
    )
    @staticmethod
    def quantity_kinds(root: 'InstanceEditorFields') -> list[InstanceQuantityKindType]:
        units_by_quantity = _collect_quantity_kind_unit_usage(root.runtime_instance())
        return [
            InstanceQuantityKindType(
                kind=QuantityKindType.from_kind(kind),
                used_units=units_by_quantity.get(kind.id, []),
            )
            for kind in get_quantity_registry()
        ]

    @sb.field
    @staticmethod
    def graph_layout(root: 'InstanceEditorFields') -> GraphLayout:
        classifier = root.runtime_instance().context.node_graph_classifier
        return GraphLayout(
            thresholds=classifier.thresholds,
            core_node_ids=[sb.ID(node_id) for node_id in classifier.core_nodes],
            ghostable_context_source_ids=[sb.ID(node_id) for node_id in classifier.ghostable_context_sources],
            hub_ids=[sb.ID(node_id) for node_id in classifier.hubs],
            action_ids=[sb.ID(node_id) for node_id in classifier.actions],
            outcome_ids=[sb.ID(node_id) for node_id in classifier.outcomes],
            main_graph_node_ids=[sb.ID(node_id) for node_id in classifier.main_graph_node_ids],
        )


@sb.type(name='InstanceModel')
class InstanceModelType:
    _instance: sb.Private[Instance]

    @sb.field
    def goals(self, id: sb.ID | None = None) -> list[InstanceGoalEntry]:
        ret = []
        for goal in self._instance.get_goals():
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

    @sb.field(graphql_type=list[Annotated['NodeInterface', sb.lazy('nodes.schema')]])
    def nodes(self, info: gql.Info, id: list[sb.ID] | None = None) -> list[Node]:
        can_edit = _instance_editor_allowed(self._instance.config, info)
        if id is not None:
            nodes: list[Node] = []
            for obj_id in id:
                node = self._instance.context.nodes.get(obj_id)
                if node is None:
                    continue
                if not can_edit and not _node_is_publicly_visible(node):
                    continue
                nodes.append(node)
            return nodes
        node_seq: Iterable[Node] = self._instance.context.nodes.values()
        if not can_edit:
            node_seq = filter(_node_is_publicly_visible, node_seq)
        return sorted(node_seq, key=lambda node: (node.order is None, node.order or 0, node.id))


@sb.type
class InstanceType:
    _config: sb.Private[InstanceConfig]
    _instance: sb.Private[Instance | None] = None

    id: sb.ID
    uuid: UUID
    name: str
    owner: str | None
    default_language: str
    supported_languages: list[str]
    base_path: str
    identifier: str
    is_locked: bool
    lead_title: str
    lead_paragraph: str | None

    @classmethod
    def from_model(cls, ic: InstanceConfig, instance: Instance | None = None) -> Self:
        return cls(
            _config=ic,
            _instance=instance,
            id=sb.ID(ic.identifier),
            uuid=ic.uuid,
            name=getattr(ic, 'name_i18n', None) or ic.name,
            owner=ic.owner_i18n or ic.owner or None,
            default_language=ic.default_language,
            supported_languages=ic.supported_languages,
            base_path='',
            identifier=ic.identifier,
            is_locked=ic.is_locked,
            lead_title=ic.lead_title_i18n or '',
            lead_paragraph=ic.lead_paragraph_i18n,
        )

    def runtime_instance(self) -> Instance:
        if self._instance is None:
            self._instance = self._config.get_instance()
        return self._instance

    @property
    def spec(self) -> InstanceModelSpec:
        return self._config.ensure_spec()

    @sb.field
    def years(self) -> YearsDefType:
        return cast('YearsDefType', self.spec.years)

    @sb.field
    def target_year(self) -> int | None:
        if self._instance is not None:
            return self._instance.context.target_year
        return self.spec.years.target

    @sb.field
    def model_end_year(self) -> int:
        if self._instance is not None:
            return self._instance.context.model_end_year
        years = self.spec.years
        return years.model_end or years.target or timezone.now().year

    @sb.field
    def reference_year(self) -> int | None:
        if self._instance is not None:
            return self._instance.reference_year
        return self.spec.years.reference

    @sb.field
    def minimum_historical_year(self) -> int:
        if self._instance is not None:
            return self._instance.minimum_historical_year
        years = self.spec.years
        return years.min_historical or years.reference or timezone.now().year

    @sb.field
    def maximum_historical_year(self) -> int | None:
        if self._instance is not None:
            return self._instance.maximum_historical_year
        return self.spec.years.max_historical

    @sb.field
    def theme_identifier(self) -> str | None:
        return self._config.theme_identifier

    @sb.field
    def action_groups(self) -> list[ActionGroupType]:
        return cast('list[ActionGroupType]', list(self.spec.action_groups))

    @sb.field
    def features(self) -> InstanceFeaturesType:
        return cast('InstanceFeaturesType', self.spec.features)

    @sb.field(
        graphql_type=InstanceModelType,
        description='Runtime computation model for fields that require hydrating the calculation graph.',
    )
    def model(self) -> InstanceModelType:
        return InstanceModelType(_instance=self.runtime_instance())

    @sb.field(graphql_type=InstanceHostname | None)
    def hostname(self, hostname: str) -> InstanceHostname | None:
        hn = self._config.hostnames.filter(hostname__iexact=hostname).first()
        if not hn:
            return None
        return InstanceHostname(hostname=hn.hostname, base_path=hn.base_path)

    @sb.field(
        graphql_type=UUID | None,
        description='UUID of the instance this one was copied from, if any.',
    )
    def copy_of(self) -> UUID | None:
        from nodes.models import InstanceConfig as _InstanceConfig

        if self._config.copy_of_id is None:
            return None
        return _InstanceConfig.objects.filter(pk=self._config.copy_of_id).values_list('uuid', flat=True).first()

    @sb.field(graphql_type=Annotated['FrameworkConfigType', sb.lazy('frameworks.schema')] | None)  # pyright: ignore[reportOperatorIssue]
    def framework_config(self, info: gql.Info) -> FrameworkConfig | None:
        return self._config.cache.framework_config

    @sb.field(description='Active members of this instance. Only visible to instance admins.')
    def users(self, info: gql.Info) -> list[InstanceMemberType]:
        if not _instance_admin_allowed(self._config, info):
            return []
        return _collect_instance_members(self._config)

    @sb.field(
        graphql_type=list[Annotated['InstanceInvitationType', sb.lazy('users.graphql.mutations')]],
        description='Active (not accepted, not expired, not revoked) invitations for this instance.',
    )
    def invitations(self, info: gql.Info) -> list['InstanceInvitation']:
        from nodes.models import InstanceInvitation as _InstanceInvitation

        if not _instance_admin_allowed(self._config, info):
            return []
        return list(
            _InstanceInvitation.objects.filter(
                instance_config=self._config,
                accepted_at__isnull=True,
                expires_at__gt=timezone.now(),
            )
        )

    @sb.field
    def editor(self, info: gql.Info) -> InstanceEditorFields | None:
        if not _instance_editor_allowed(self._config, info):
            return None
        return InstanceEditorFields(_config=self._config, _instance=self._instance)

    @sb.field(deprecation_reason='Use model.goals instead.')
    def goals(self, id: sb.ID | None = None) -> list[InstanceGoalEntry]:
        return self.model().goals(id)

    @grapple_field
    def action_list_page(self) -> ActionListPage | None:
        return self._config.action_list_page

    @sb.field(graphql_type=list[StreamFieldInterface] | None)
    def intro_content(self) -> StreamValue:
        return self._config.site_content.intro_content

    @sb.field(
        graphql_type=list[Annotated['NodeInterface', sb.lazy('nodes.schema')]],
        deprecation_reason='Use model.nodes instead.',
    )
    def nodes(self, info: gql.Info, id: list[sb.ID] | None = None) -> list[Node]:
        return self.model().nodes(info, id)


@sb.type
class InstanceBasicConfiguration:
    default_language: str
    supported_languages: list[str]

    @sb.field
    @staticmethod
    def identifier(root: InstanceConfig) -> str:
        return root.identifier

    @sb.field
    @staticmethod
    def is_protected(root: InstanceConfig) -> bool:
        return root.is_protected

    @sb.field
    @staticmethod
    def requires_authentication(root: InstanceConfig) -> bool:
        return root.ensure_spec().features.requires_authentication

    @sb.field
    @staticmethod
    def theme_identifier(root: InstanceConfig) -> str:
        spec = root.ensure_spec()
        return spec.theme_identifier or 'default'

    @sb.field
    @staticmethod
    def hostname(root: InstanceConfig) -> InstanceHostname:
        gql_context = root.graphql_context
        if gql_context is None or gql_context.matched_hostname is None:
            hostname = gql_context.requested_hostname if gql_context is not None else ''
            return InstanceHostname(hostname=hostname, base_path='')
        return InstanceHostname(
            hostname=gql_context.matched_hostname.hostname,
            base_path=gql_context.matched_hostname.base_path,
        )


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
