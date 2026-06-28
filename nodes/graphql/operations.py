from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING

import strawberry as sb
from django.db.models import Prefetch
from graphql.error import GraphQLError

from loguru import logger

from kausal_common.strawberry.registry import register_strawberry_type

from paths import gql
from paths.const import INSTANCE_CHANGE_GROUP, INSTANCE_CHANGE_TYPE
from paths.graphql_helpers import ensure_instance, get_instance_context, pass_context

from nodes.models import InstanceConfig, InstanceGraphQLContext
from nodes.normalization import Normalization
from nodes.scenario import Scenario

from .types.impact import ImpactOverviewType
from .types.instance import InstanceBasicConfiguration, InstanceType, NormalizationType
from .types.node import ActionNodeType, NodeInterface
from .types.scenario import ScenarioType

if TYPE_CHECKING:
    from nodes.actions.action import ActionNode
    from nodes.context import Context
    from nodes.node import Node

logger = logger.bind(name='nodes.schema')


@sb.type
class Query:
    @sb.field(graphql_type=InstanceType)
    @pass_context
    def instance(self, context: 'Context') -> InstanceType:
        config = context.instance.config
        return InstanceType.from_model(config, instance=context.instance)

    @sb.field(graphql_type=list[NodeInterface])
    @pass_context
    def nodes(self, context: 'Context') -> list['Node']:
        return list(context.nodes.values())

    @sb.field(graphql_type=NodeInterface | None)
    @ensure_instance
    def node(self, info: gql.InstanceInfo, id: sb.ID):
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
    def action(self, context, id: sb.ID):
        try:
            return context.get_action(str(id))
        except KeyError:
            return None
        except TypeError:
            return None

    @sb.field(graphql_type=list[ImpactOverviewType], deprecation_reason='Use impactOverviews instead')
    @pass_context
    def action_efficiency_pairs(self, context):
        return context.impact_overviews

    @sb.field(graphql_type=list[ImpactOverviewType])
    @pass_context
    def impact_overviews(self, context):
        return context.impact_overviews

    @sb.field(graphql_type=ImpactOverviewType | None)
    @pass_context
    def impact_overview(self, context, id: sb.ID):
        return next((io for io in context.impact_overviews if io.spec.id == str(id)), None)

    @sb.field(graphql_type=list[ScenarioType])
    @pass_context
    def scenarios(self, context) -> list[Scenario]:
        return list(context.scenarios.values())

    @sb.field(graphql_type=ScenarioType)
    @pass_context
    def scenario(self, context, id: sb.ID) -> Scenario:
        return context.get_scenario(str(id))

    @sb.field(graphql_type=ScenarioType)
    @pass_context
    def active_scenario(self, context) -> Scenario:
        return context.active_scenario

    @sb.field(graphql_type=list[NormalizationType])
    @pass_context
    def available_normalizations(self, context):
        return list(context.normalizations.values())

    @sb.field(graphql_type=NormalizationType | None)
    @pass_context
    def active_normalization(self, context):
        return context.active_normalization


@sb.type
class SBQuery(Query):
    @sb.field(graphql_type=list[NormalizationType])
    @pass_context
    @staticmethod
    def active_normalizations(context: 'Context') -> list[Normalization]:
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
