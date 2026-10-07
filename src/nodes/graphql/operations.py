from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING

import strawberry as sb
from django.db.models import Prefetch
from graphql.error import GraphQLError

from loguru import logger

from kausal_common.strawberry.permissions import AuthenticatedOnly
from kausal_common.strawberry.registry import register_strawberry_type

from paths import gql
from paths.const import INSTANCE_CHANGE_GROUP, INSTANCE_CHANGE_TYPE
from paths.graphql_helpers import default_instance

from nodes.models import InstanceConfig, InstanceGraphQLContext
from nodes.normalization import Normalization
from nodes.scenario import Scenario

from .types.impact import ImpactOverviewType
from .types.instance import (
    InstanceBasicConfiguration,
    InstanceType,
    NormalizationType,
    find_action,
    find_impact_overview,
    find_node,
    list_actions,
)
from .types.node import ActionNodeType, NodeInterface
from .types.scenario import ScenarioType

if TYPE_CHECKING:
    from nodes.actions.action import ActionNode, ImpactOverview
    from nodes.node import Node

logger = logger.bind(name='nodes.schema')


@sb.type
class Query:
    @sb.field(graphql_type=InstanceType)
    def instance(self, info: gql.Info) -> InstanceType:
        config = info.context.instance_config
        if config is None:
            raise GraphQLError(
                "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            )
        snapshot = info.context.instance_snapshot_for_type(config)
        return InstanceType.from_model(config, snapshot=snapshot)

    @sb.field(
        graphql_type=list[InstanceType],
        permission_classes=[AuthenticatedOnly],
        description=(
            'Instances the signed-in user may view, ordered by identifier. Needs only view permission, '
            'so public instances are included; anonymous requests are refused.'
        ),
    )
    @staticmethod
    def instances(info: gql.Info) -> list[InstanceType]:
        qs = InstanceConfig.objects.qs.viewable_by(info.context.get_user()).order_by('identifier')
        return [InstanceType.from_model(ic) for ic in qs]

    # The fields below read the operation's default model: the instance the
    # `@context` directive names, under the visitor's session settings. They are
    # entry points with no root object, so they name that runtime explicitly;
    # `instance { model(...) { ... } }` reaches the same data, with overrides.

    @sb.field(graphql_type=list[NodeInterface])
    @staticmethod
    def nodes(info: gql.Info) -> list['Node']:
        return list(default_instance(info).context.nodes.values())

    @sb.field(graphql_type=NodeInterface | None)
    @staticmethod
    def node(info: gql.Info, id: sb.ID) -> 'Node | None':
        return find_node(default_instance(info).context, str(id))

    @sb.field(graphql_type=ActionNodeType | None)
    @staticmethod
    def action(info: gql.Info, id: sb.ID) -> 'ActionNode | None':
        return find_action(default_instance(info).context, str(id))

    @sb.field(graphql_type=list[ImpactOverviewType], deprecation_reason='Use impactOverviews instead')
    @staticmethod
    def action_efficiency_pairs(info: gql.Info) -> 'list[ImpactOverview]':
        return default_instance(info).context.impact_overviews

    @sb.field(graphql_type=list[ImpactOverviewType])
    @staticmethod
    def impact_overviews(info: gql.Info) -> 'list[ImpactOverview]':
        return default_instance(info).context.impact_overviews

    @sb.field(graphql_type=ImpactOverviewType | None)
    @staticmethod
    def impact_overview(info: gql.Info, id: sb.ID) -> 'ImpactOverview | None':
        return find_impact_overview(default_instance(info).context, str(id))

    @sb.field(graphql_type=list[ScenarioType])
    @staticmethod
    def scenarios(info: gql.Info) -> list[Scenario]:
        return list(default_instance(info).context.scenarios.values())

    @sb.field(graphql_type=ScenarioType)
    @staticmethod
    def scenario(info: gql.Info, id: sb.ID) -> Scenario:
        return default_instance(info).context.get_scenario(str(id))

    @sb.field(graphql_type=ScenarioType)
    @staticmethod
    def active_scenario(info: gql.Info) -> Scenario:
        return default_instance(info).context.active_scenario

    @sb.field(graphql_type=list[NormalizationType])
    @staticmethod
    def available_normalizations(info: gql.Info) -> list[Normalization]:
        return list(default_instance(info).context.normalizations.values())

    @sb.field(graphql_type=NormalizationType | None)
    @staticmethod
    def active_normalization(info: gql.Info) -> Normalization | None:
        return default_instance(info).context.active_normalization


@sb.type
class SBQuery(Query):
    @sb.field(graphql_type=list[NormalizationType])
    @staticmethod
    def active_normalizations(info: gql.Info) -> list[Normalization]:
        return list(default_instance(info).context.normalizations.values())

    @sb.field(graphql_type=list[ActionNodeType])
    @staticmethod
    def actions(info: gql.Info, only_root: bool = False) -> list['ActionNode']:
        return list_actions(default_instance(info).context, only_root=only_root)

    @sb.field(graphql_type=list[InstanceBasicConfiguration])
    @staticmethod
    def available_instances(info: gql.Info, hostname: str) -> list[InstanceConfig]:
        from nodes.models import InstanceHostname

        normalized_hostname = hostname.lower()
        matched_hostnames_attr = '_available_instances_matched_hostnames'
        qs = (
            InstanceConfig.objects
            .get_queryset()
            .for_hostname(normalized_hostname, wildcard_domains=info.context.wildcard_domains)
            .prefetch_related(
                Prefetch(
                    'hostnames',
                    queryset=InstanceHostname.objects.filter(hostname=normalized_hostname),
                    to_attr=matched_hostnames_attr,
                )
            )
        )
        configs = list(qs)
        if configs:
            from frameworks.models import FrameworkConfig

            path_routed_framework_configs = FrameworkConfig.objects.select_related('framework', 'instance_config').filter(
                framework__root_instance__in=configs,
                framework__use_instance_subdomains=False,
            )
            existing_config_ids = {config.pk for config in configs}
            for framework_config in path_routed_framework_configs:
                config = framework_config.instance_config
                if config.pk in existing_config_ids:
                    continue
                setattr(
                    config,
                    matched_hostnames_attr,
                    [InstanceHostname(instance=config, hostname=normalized_hostname, base_path=f'/{config.uuid}')],
                )
                configs.append(config)
                existing_config_ids.add(config.pk)
        instances: list[InstanceConfig] = []
        for config in configs:
            matched_hostnames: list[InstanceHostname] = getattr(config, matched_hostnames_attr)
            config.graphql_context = InstanceGraphQLContext(
                requested_hostname=normalized_hostname,
                matched_hostname=matched_hostnames[0] if matched_hostnames else None,
            )
            instances.append(config)
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
                yield InstanceChange(id=sb.ID(str(ic.uuid)), identifier=ic.identifier, modified_at=ic.modified_at)


@sb.type
class Mutation:
    @sb.type
    class SetNormalizerMutation:
        ok: bool
        active_normalizer: Normalization | None = sb.field(graphql_type=NormalizationType | None)

    @sb.mutation
    def set_normalizer(self, info: gql.Info, id: sb.ID | None = None) -> 'Mutation.SetNormalizerMutation':
        context = default_instance(info).context
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
