from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from django.conf import settings
from django.utils import translation
from graphql import VariableNode, get_argument_values
from graphql.error import GraphQLError
from strawberry.utils.operation import get_first_operation

import sentry_sdk
from loguru import logger

from kausal_common.strawberry.context import GraphQLContext
from kausal_common.strawberry.schema import AuthenticationExtension, ExecutionCacheExtension, GraphQLPerfNode, SchemaExtension

from paths.context import PathsObjectCache

from nodes.models import InstanceConfig, InstanceConfigQuerySet
from params.storage import SessionStorage

if TYPE_CHECKING:
    from collections.abc import Generator

    from graphql.language import DirectiveNode, OperationDefinitionNode

    from nodes.instance import Instance

logger = logger.bind(markup=True)


def _arg_value(arg, variable_vals) -> Any:
    if isinstance(arg.value, VariableNode):
        return variable_vals.get(arg.value.name.value)
    return arg.value.value


@dataclass
class PathsGraphQLContext[InstanceType: Instance | None = Instance | None](GraphQLContext):
    instance_config: InstanceConfig | None = None
    instance: InstanceType = field(init=False)
    cache: PathsObjectCache = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        user = self.get_user()
        self.cache = PathsObjectCache(user=user)
        self.instance = None  # type: ignore[assignment]


class PathsSchemaExtension(SchemaExtension[PathsGraphQLContext]):
    context_class: type[PathsGraphQLContext[Instance | None]] = PathsGraphQLContext


class DetermineInstanceContextExtension(PathsSchemaExtension):
    def process_locale_directive(self, ic: InstanceConfig, directive: DirectiveNode) -> str:
        from kausal_common.strawberry.schema import locale_directive

        assert locale_directive.graphql_name is not None
        exec_ctx = self.execution_context
        directive_ast = exec_ctx.schema._schema.get_directive(locale_directive.graphql_name)
        assert directive_ast is not None
        lang = get_argument_values(directive_ast, directive, exec_ctx.variables).get('lang')
        if lang is None:
            raise GraphQLError('Locale directive missing lang argument', directive)

        if lang not in ic.supported_languages:
            raise GraphQLError('unsupported language: %s' % lang, directive)
        return lang

    def get_ic_queryset(self) -> InstanceConfigQuerySet:
        return (
            InstanceConfig.objects.get_queryset().select_related('framework_config').select_related('framework_config__framework')
        )

    def get_instance_by_identifier(
        self,
        queryset: InstanceConfigQuerySet,
        identifier: str,
        directive: DirectiveNode | None = None,
    ) -> InstanceConfig:
        try:
            if identifier.isnumeric():
                instance = queryset.get(id=identifier)
            else:
                instance = queryset.get(identifier=identifier)
        except InstanceConfig.DoesNotExist:
            raise GraphQLError('Instance with identifier %s not found' % identifier, directive) from None
        return instance

    def get_instance_by_hostname(
        self,
        queryset: InstanceConfigQuerySet,
        hostname: str,
        directive: DirectiveNode | None = None,
    ) -> InstanceConfig:
        ctx = self.get_context()
        try:
            instance = queryset.for_hostname(hostname, wildcard_domains=ctx.wildcard_domains).get()
        except InstanceConfig.DoesNotExist:
            logger.warning(f'No instance found for hostname {hostname} (wildcard domains: {ctx.wildcard_domains})')
            raise GraphQLError('Instance matching hostname %s not found' % hostname, directive) from None
        return instance

    def process_instance_directive(self, directive: DirectiveNode) -> InstanceConfig:
        qs = self.get_ic_queryset()
        exec_ctx = self.execution_context
        arguments = {arg.name.value: _arg_value(arg, exec_ctx.variables) for arg in directive.arguments}
        identifier = arguments.get('identifier')
        hostname = arguments.get('hostname')
        _token = arguments.get('token')
        if identifier:
            return self.get_instance_by_identifier(qs, identifier, directive)
        if hostname:
            return self.get_instance_by_hostname(qs, hostname, directive)
        raise GraphQLError('Invalid instance directive', directive)

    def process_context_directive(self, directive: DirectiveNode) -> tuple[InstanceConfig | None, str | None]:
        from .schema import context_directive

        assert context_directive.graphql_name is not None
        exec_ctx = self.execution_context
        directive_ast = exec_ctx.schema._schema.get_directive(context_directive.graphql_name)
        assert directive_ast is not None
        ctx = get_argument_values(directive_ast, directive, exec_ctx.variables).get('input')
        if ctx is None:
            return None, None
        # FIXME: Filter by user permissions
        qs = self.get_ic_queryset()
        identifier = ctx.get('identifier')
        hostname = ctx.get('hostname')
        if identifier:
            ic = self.get_instance_by_identifier(qs, identifier)
        elif hostname:
            ic = self.get_instance_by_hostname(qs, hostname)
        else:
            return None, None
        locale = ctx.get('locale')
        if not locale:
            locale = ic.primary_language
        elif locale not in ic.supported_languages:
            raise GraphQLError('unsupported language: %s' % locale, directive)
        return ic, locale

    def process_instance_headers(self) -> InstanceConfig | None:
        headers = self.get_request_headers()
        identifier = headers.get(settings.INSTANCE_IDENTIFIER_HEADER)
        hostname = headers.get(settings.INSTANCE_HOSTNAME_HEADER)

        qs = self.get_ic_queryset()
        if identifier:
            return self.get_instance_by_identifier(qs, identifier)
        if hostname:
            return self.get_instance_by_hostname(qs, hostname)
        return None

    def determine_instance_and_locale(self, operation: OperationDefinitionNode) -> tuple[InstanceConfig, str] | None:
        instance_config: InstanceConfig | None = None
        locale: str | None = None
        user = self.get_user()

        for directive in operation.directives or []:
            directive_name = directive.name.value
            if directive_name == 'context':
                instance_config, locale = self.process_context_directive(directive)
                if instance_config is not None:
                    break
            elif directive_name == 'instance':
                instance_config = self.process_instance_directive(directive)
                break
        else:
            instance_config = self.process_instance_headers()

        if instance_config is None:
            return None

        if instance_config.is_protected and user is None:
            raise GraphQLError('Instance is protected', extensions=dict(code='instance_protected'))

        if locale is None:
            for directive in operation.directives or []:
                directive_name = directive.name.value
                if directive_name != 'locale':
                    continue
                locale = self.process_locale_directive(instance_config, directive)
                break
            else:
                locale = instance_config.primary_language

        ctx = self.get_context()
        ctx.graphql_query_language = locale
        ctx.instance_config = instance_config
        return instance_config, locale

    def on_execute(self) -> Generator[None]:
        doc = self.execution_context.graphql_document
        if doc:
            op = get_first_operation(doc)
        else:
            op = None

        if not op or self.execution_context.result:
            yield
            return

        self.determine_instance_and_locale(op)
        yield


class ActivateInstanceContextExtension(PathsSchemaExtension):
    def activate_language(self, lang: str):
        return cast('AbstractContextManager', translation.override(lang))

    def set_instance_scope(self) -> None:
        scope = sentry_sdk.get_current_scope()
        ic = self.get_context().instance_config
        if ic is None:
            return
        scope.set_tag('instance_id', ic.identifier)
        scope.set_tag('instance_uuid', str(ic.uuid))
        if ic.has_framework_config():
            fw = self.get_context().cache.for_framework_id(ic.framework_config.framework_id)
            if fw is not None:
                scope.set_tag('framework_id', fw.identifier)

    def activate_instance(self, instance: Instance):
        context = instance.context
        session = self.get_session()
        assert session is not None
        context.setting_storage = storage = SessionStorage(instance=instance, session=session)
        active_scenario_id = storage.get_active_scenario()
        scenario = None
        if active_scenario_id:
            try:
                scenario = context.get_scenario(active_scenario_id)
            except KeyError:
                storage.set_active_scenario(None)

        # Tell the custom scenario about the user setting so that
        # it can locate the customized parameters.
        if context.custom_scenario is not None:
            context.custom_scenario.set_storage(storage)

        if scenario is None:
            scenario = context.get_default_scenario()

        # Activate normalization
        if context.setting_storage.has_option('normalizer'):
            val = context.setting_storage.get_option('normalizer')
            context.set_option('normalizer', val)
        else:
            for n in context.normalizations.values():
                if n.default:
                    context.active_normalization = n
                    break
            else:
                context.active_normalization = None

        context.activate_scenario(scenario)

    @contextmanager
    def instance_context(self, operation: OperationDefinitionNode):
        context = None
        ctx = self.get_context()
        perf = ctx.graphql_perf
        ic = ctx.instance_config
        assert ic is not None
        assert ctx.graphql_query_language is not None
        with ExitStack() as stack:
            with perf.exec_node(GraphQLPerfNode('prepare instance "%s"' % ic.identifier)):
                stack.enter_context(self.activate_language(ctx.graphql_query_language))
                with perf.exec_node(GraphQLPerfNode('get instance "%s"' % ic.identifier)):
                    instance = stack.enter_context(ic.enter_instance_context())
                    ctx.instance = instance
                context = instance.context
                stack.enter_context(instance.lock)
                stack.enter_context(context.run())
                self.activate_instance(instance)
            yield
        instance.clean()

    def on_execute(self) -> Generator[None]:
        doc = self.execution_context.graphql_document
        if doc:
            op = get_first_operation(doc)
        else:
            op = None
        exec_ctx = self.get_context()

        if not op or self.execution_context.result or exec_ctx.instance_config is None:
            yield
            return

        with self.instance_context(op):
            yield


class PathsExecutionCacheExtension(ExecutionCacheExtension[PathsGraphQLContext]):
    context_class: type[PathsGraphQLContext[Instance | None]] = PathsGraphQLContext

    def get_cache_key_parts(self) -> list[str] | None:
        exec_ctx = self.get_context()
        ic = exec_ctx.instance_config
        if ic is None:
            self.set_reason('no instance config')
            return None

        parts = [str(ic.uuid), ic.cache_invalidated_at.isoformat()]
        session = self.get_session()
        if session is not None:
            session_key = SessionStorage.get_cache_key(session, ic.identifier)
            if session_key is None:
                self.set_reason('user session has custom parameters')
                return None
            parts.append(session_key)
        return parts


class PathsAuthenticationExtension(AuthenticationExtension[PathsGraphQLContext]):
    context_class: type[PathsGraphQLContext[Instance | None]] = PathsGraphQLContext
