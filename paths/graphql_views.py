from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, cast

from django.conf import settings
from django.contrib.auth.models import User
from django.core.cache import cache
from django.utils import translation
from graphene_django.views import GraphQLView
from graphql import (
    DirectiveNode,
    ExecutionResult,
    GraphQLOutputType,
    OperationType,
    get_argument_values,
)
from graphql.error import GraphQLError
from graphql.execution import ExecutionContext
from graphql.language.ast import VariableNode

import orjson
import sentry_sdk
import sentry_sdk.spotlight
from loguru import logger
from rich.console import Console
from rich.syntax import Syntax

from kausal_common.auth.tokens import authenticate_api_request
from kausal_common.deployment import env_bool
from kausal_common.testing.graphql import capture_query

from paths.const import INSTANCE_HOSTNAME_HEADER, INSTANCE_IDENTIFIER_HEADER, WILDCARD_DOMAINS_HEADER

from nodes.models import InstanceConfig, InstanceConfigQuerySet
from nodes.perf import PerfContext
from params.storage import SessionStorage

from .context import PathsObjectCache
from .graphql_helpers import GraphQLPerfNode

if TYPE_CHECKING:
    from django.http import HttpRequest
    from graphql.language import FieldNode, OperationDefinitionNode
    from graphql.pyutils import AwaitableOrValue, Path
    from graphql.type import GraphQLObjectType

    from sentry_sdk.tracing import Transaction

    from paths.types import GQLInstanceContext

    from nodes.instance import Instance

    from .types import PathsGQLContext

SUPPORTED_LANGUAGES = {x[0] for x in settings.LANGUAGES}


def _arg_value(arg, variable_vals) -> Any:
    if isinstance(arg.value, VariableNode):
        return variable_vals.get(arg.value.name.value)
    return arg.value.value


logger = logger.bind(markup=True)

GRAPHQL_CAPTURE_QUERIES = env_bool('GRAPHQL_CAPTURE_QUERIES', default=False)


class PathsExecutionContext(ExecutionContext):
    context_value: GQLInstanceContext

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def handle_field_error(
        self,
        error: GraphQLError,
        return_type: GraphQLOutputType,
    ) -> None:
        if settings.DEBUG and error.original_error is not None and not getattr(error, '_was_printed', False):
            exc = error.original_error
            logger.opt(exception=exc).error('GraphQL field error at {path}', path=error.path)
            setattr(error, '_was_printed', True)  # noqa: B010
        return super().handle_field_error(error, return_type)

    def process_locale_directive(self, ic: InstanceConfig, directive: DirectiveNode) -> str:
        from paths.schema import LocaleDirective

        lang = get_argument_values(LocaleDirective, directive, self.variable_values).get('lang')
        if lang is None:
            raise GraphQLError('Locale directive missing lang argument', directive)

        if lang not in ic.supported_languages:
            raise GraphQLError('unsupported language: %s' % lang, directive)
        return lang

    def activate_language(self, lang: str):
        self.context_value.graphql_query_language = lang
        return cast('AbstractContextManager', translation.override(lang))

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
        request = self.context_value
        try:
            instance = queryset.for_hostname(hostname, request).get()
        except InstanceConfig.DoesNotExist:
            logger.warning(f'No instance found for hostname {hostname} (wildcard domains: {request.wildcard_domains})')
            raise GraphQLError('Instance matching hostname %s not found' % hostname, directive) from None
        return instance

    def process_instance_directive(self, directive: DirectiveNode) -> InstanceConfig:
        qs = InstanceConfig.objects.get_queryset()
        arguments = {arg.name.value: _arg_value(arg, self.variable_values) for arg in directive.arguments}
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
        directive_ast = self.schema.get_directive(context_directive.graphql_name)
        assert directive_ast is not None
        ctx = get_argument_values(directive_ast, directive, self.variable_values).get('input')
        if ctx is None:
            return None, None
        # FIXME: Filter by user permissions
        qs = InstanceConfig.objects.get_queryset()
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

    def process_instance_headers(self, context: PathsGQLContext) -> InstanceConfig | None:
        identifier = context.headers.get(settings.INSTANCE_IDENTIFIER_HEADER)
        hostname = context.headers.get(settings.INSTANCE_HOSTNAME_HEADER)

        qs = InstanceConfig.objects.get_queryset()
        if identifier:
            return self.get_instance_by_identifier(qs, identifier)
        if hostname:
            return self.get_instance_by_hostname(qs, hostname)
        return None

    def determine_instance_and_locale(self, operation: OperationDefinitionNode) -> tuple[InstanceConfig, str] | None:
        instance_config: InstanceConfig | None = None
        locale: str | None = None

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
            instance_config = self.process_instance_headers(self.context_value)

        if instance_config is None:
            return None

        if instance_config.is_protected and not self.context_value.user.is_active:
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
        return instance_config, locale

    def activate_instance(self, instance: Instance):
        context = instance.context
        context.setting_storage = storage = SessionStorage(instance=instance, session=self.context_value.session)
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
        request = self.context_value
        perf = request.graphql_perf

        instance_ctx = self.determine_instance_and_locale(operation)
        if instance_ctx is None:
            yield
            return

        ic, locale = instance_ctx

        with ExitStack() as stack:
            with perf.exec_node(GraphQLPerfNode('prepare instance "%s"' % ic.identifier)):
                stack.enter_context(self.activate_language(locale))
                with perf.exec_node(GraphQLPerfNode('get instance "%s"' % ic.identifier)):
                    instance = stack.enter_context(ic.enter_instance_context())
                    self.context_value.instance = instance
                context = instance.context
                stack.enter_context(instance.lock)
                stack.enter_context(context.run())
                self.activate_instance(instance)
            yield
        instance.clean()

    def execute_fields(
        self,
        parent_type: GraphQLObjectType,
        source_value: Any,
        path: Path | None,
        fields: dict[str, list[FieldNode]],
    ) -> AwaitableOrValue[dict[str, Any]]:
        path_parts = path.as_list() if path else []
        span_cm: AbstractContextManager
        str_path = '.'.join([str(x) for x in path_parts])
        if path_parts:
            node = GraphQLPerfNode(str_path)
            span_cm = self.context_value.graphql_perf.exec_node(node)
        else:
            span_cm = nullcontext()
        with span_cm:
            _rich_traceback_omit = True
            ret = super().execute_fields(parent_type, source_value, path, fields)
        return ret

    def execute_operation(self, operation: OperationDefinitionNode, root_value: Any) -> AwaitableOrValue[Any] | None:
        op_name = operation.name.value if operation.name else '<unnamed>'
        with sentry_sdk.start_span(op='graphql.execute', name='Query %s' % op_name):
            if operation.operation != OperationType.QUERY or self.context_value.user.is_authenticated:
                self.context_value.graphene_no_cache = True  # type: ignore[attr-defined]
            with self.instance_context(operation):
                ret = super().execute_operation(operation, root_value)
            return ret


class PathsGraphQLView(GraphQLView):
    graphiql_version = '3.0.9'
    graphiql_sri = 'sha256-i8HFOsDB6KaRVstG2LSibODRlHNNA1XLKLnDl7TIAZY='
    graphiql_css_sri = 'sha256-wTzfn13a+pLMB5rMeysPPR1hO7x0SwSeQI+cnw7VdbE='
    graphiql_plugin_explorer_version = '1.0.2'
    graphiql_plugin_explorer_sri = 'sha256-CD435QHT45IKYOYnuCGRrwVgCRJNzoKjMuisdNtso4s='
    graphiql_plugin_explorer_css_sri = 'sha256-G6RZ0ey9eHIvQt0w+zQYVh4Rq1nNneislHMWMykzbLs='

    execution_context_class = PathsExecutionContext

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_reg(self, level: str, operation_name: str | None, msg, *args, depth: int = 0, **kwargs):
        log = logger.opt(depth=1 + depth)
        if operation_name:
            log = log.bind(graphql_operation=operation_name)
        log.log(level, 'GQL request [magenta]%s[/]: %s' % (operation_name, msg), *args, **kwargs)

    def json_encode(self, request: GQLInstanceContext, d, pretty=False) -> str:
        errors = []

        def serialize_unknown(obj) -> str:
            err = TypeError('Unable to serialize value %s with type %s' % (obj, type(obj)))
            errors.append(err)
            return ' '.join(['__INVALID__' * 10])

        if not (self.pretty or pretty) and not request.GET.get('pretty'):
            opts = None
        else:
            opts = orjson.OPT_INDENT_2

        ret = orjson.dumps(d, option=opts, default=serialize_unknown)
        if errors:
            from rich import print_json

            print_json(ret.decode('utf8'))
            raise errors[0]
        op_name = getattr(request, 'graphql_operation_name', None)
        self.log_reg(
            'DEBUG',
            op_name,
            'Response was {} bytes',
            len(ret),
        )
        return ret.decode('utf8')

    def get_ic_from_headers(self, request: HttpRequest):
        identifier = request.headers.get('x-paths-instance-identifier')
        _hostname = request.headers.get('x-paths-instance-hostname')
        if not identifier:
            return None
        return InstanceConfig.objects.filter(identifier=identifier).first()

    def set_instance_scope(self, request: GQLInstanceContext, ic: InstanceConfig) -> None:
        scope = sentry_sdk.get_current_scope()
        scope.set_tag('instance_id', ic.identifier)
        scope.set_tag('instance_uuid', str(ic.uuid))
        if ic.has_framework_config():
            fw = request.cache.for_framework_id(ic.framework_config.framework_id)
            if fw is not None:
                scope.set_tag('framework_id', fw.identifier)

    def get_cache_key(
        self,
        request: GQLInstanceContext,
        query: str | None,
        variables: dict | None,
        operation_name: str | None,
    ):
        def log_reason(reason: str) -> None:
            self.log_reg('DEBUG', operation_name, 'request not cached: [red]{}[/]', reason, depth=1)

        if not query:
            return None

        if os.getenv('DISABLE_GRAPHQL_CACHE', '') == '1':
            log_reason('cache disabled by env variable')
            return None

        if request.user.is_authenticated:
            log_reason('user authenticated')
            return None

        ic = self.get_ic_from_headers(request)
        if ic is None:
            log_reason('no instance config')
            return None

        self.set_instance_scope(request, ic)

        session_key = SessionStorage.get_cache_key(request.session, ic.identifier)
        if session_key is None:
            log_reason('user session has custom parameters')
            return None

        m = hashlib.md5(usedforsecurity=False)
        if operation_name:
            m.update(operation_name.encode('utf8'))
        dt = ic.cache_invalidated_at.isoformat()
        m.update(dt.encode('ascii'))
        if variables:
            m.update(json.dumps(variables, sort_keys=True, ensure_ascii=True).encode('ascii'))
        m.update(query.encode('utf8'))
        key = m.hexdigest()
        parts = [key]
        if session_key:
            parts.append(session_key)
        return ':'.join(parts)

    def get_from_cache(self, key):
        return cache.get(key)

    def store_to_cache(self, key, result):
        return cache.set(key, result, timeout=30 * 60)

    def get_response(self, request: HttpRequest, data, show_graphiql=False) -> tuple[str | None, int]:  # pyright: ignore
        operation_name = request.GET.get('operationName') or data.get('operationName')
        if operation_name == 'null':
            operation_name = None
        request = cast('GQLInstanceContext', request)
        perf: PerfContext = PerfContext(
            supports_cache=False,
            min_ms=10,
            description=operation_name,
        )
        perf.enabled = settings.ENABLE_PERF_TRACING
        request.graphql_perf = perf

        auth_error = authenticate_api_request(request, 'graphql')
        if auth_error:
            logger.warning('Authentication failed: %s (%s)' % (auth_error.get('error'), auth_error.get('error_description')))
            resp = dict(
                errors=[
                    dict(
                        message='Authentication failed',
                        extensions=dict(
                            code=auth_error.get('error'),
                            description=auth_error.get('error_description'),
                        ),
                    ),
                ],
            )
            return self.json_encode(request, resp), 401

        # If the user changed, re-initialize the cache just to be on the safe side.
        if request.cache.user != request.user:
            request.cache = PathsObjectCache(user=request.user)

        user_log_context: dict[str, Any]
        if isinstance(request.user, User):
            user = request.user
            user_log_context = {
                'user.email': user.email,
                'user.id': user.pk,
            }
        else:
            user_log_context = {}

        with perf, logger.contextualize(**user_log_context):
            start = time.time()
            ret = super().get_response(request, data, show_graphiql)
            if GRAPHQL_CAPTURE_QUERIES and data and ret[0]:
                headers = [INSTANCE_IDENTIFIER_HEADER, INSTANCE_HOSTNAME_HEADER, WILDCARD_DOMAINS_HEADER]
                now = time.time()
                logger.info('Capturing GraphQL query and response')
                capture_query(request, headers, data, ret[0], ret[1], (now - start) * 1000)

        return ret

    def _print_req_headers(self, request: HttpRequest) -> None:
        from rich import print

        keys = ['sentry-trace', 'baggage', 'tracecontext', 'tracestate']
        print({k: request.headers.get(k) for k in keys if request.headers.get(k)})
        span = sentry_sdk.get_current_span()
        if span:
            print(span.get_trace_context())

    def graphql_span_name(self, query: str | None, operation_name: str | None) -> str:
        return f'http.graphql.{operation_name or 'request'}'

    def configure_root_transaction(
        self, request: GQLInstanceContext, query: str | None, variables: dict | None, operation_name: str | None
    ) -> None:
        scope = sentry_sdk.get_current_scope()
        transaction: Transaction | None = scope.transaction
        if not transaction:
            return
        possible_types = ('query', 'mutation', 'subscription')
        transaction_type = None
        if query:
            for t in possible_types:
                if not query.startswith(t):
                    continue
                transaction_type = t
                break

        name = operation_name or '<unnamed>'
        if transaction_type:
            name = f'{transaction_type} {name}'

        transaction.op = f'http.graphql.{transaction_type or 'request'}'
        transaction.name = name

        if sentry_sdk.spotlight:
            transaction.set_data('graphql.document', query)
            transaction.set_data('graphql.variables', orjson.dumps(variables or {}).decode('utf8'))

        if request._referer:
            transaction.set_tag('referer', request._referer)

    def execute_graphql_request(
        self,
        request: GQLInstanceContext,
        data: dict,
        query: str | None,
        variables: dict | None,
        operation_name: str | None,
        *args,
        **kwargs,
    ):
        """Set up context for the request execution."""

        # self._print_req_headers(request)

        request._referer = self.request.META.get('HTTP_REFERER')
        request.graphql_operation_name = operation_name
        wildcard_domains = request.headers.get(WILDCARD_DOMAINS_HEADER)
        request.wildcard_domains = [d.lower() for d in wildcard_domains.split(',')] if wildcard_domains else []

        if query is not None:
            query = query.strip()
        if settings.LOG_GRAPHQL_QUERIES and query:
            console = Console()
            syntax = Syntax(query, 'graphql')
            console.print(syntax)
            if variables:
                console.print('Variables:', variables)

        self.configure_root_transaction(request, query, variables, operation_name)

        with (
            sentry_sdk.start_span(op='graphql', name=operation_name),
            request.graphql_perf.exec_node(GraphQLPerfNode('execute %s' % operation_name)),
        ):
            return self._execute_graphql_request(request, data, query, variables, operation_name, *args, **kwargs)

    def _execute_graphql_request(  # noqa: C901, PLR0912, PLR0915
        self,
        request: GQLInstanceContext,
        data: dict,
        query: str | None,
        variables: dict | None,
        operation_name: str | None,
        *args,
        **kwargs,
    ) -> ExecutionResult | None:
        def log(level: str, msg: str, *args, depth: int = 0, **kwargs) -> None:
            self.log_reg(level, operation_name, msg, *args, depth=depth + 1, **kwargs)

        def enter(id: str):  # noqa: ANN202
            return request.graphql_perf.exec_node(GraphQLPerfNode(id))

        span = sentry_sdk.get_current_span()
        assert span is not None
        with enter('get query cache key'):
            cache_key = self.get_cache_key(request, query, variables, operation_name)

        result: ExecutionResult | None = None
        if cache_key is not None:
            result = self.get_from_cache(cache_key)

        if cache_key is None:
            cache_res = 'disabled'
            color = 'magenta'
        elif result is None:
            cache_res = 'miss'
            color = 'yellow'
        else:
            cache_res = 'hit'
            color = 'green'

        span.set_tag('cache', cache_res)

        log('INFO', 'referrer: %s; cache [%s]%s[/]' % (request._referer, color, cache_res))
        # log('DEBUG', 'Cache key %s, %s' % (cache_key, "[green]got result[/]" if result is not None else "[orange]no hit[/]"))
        if result is not None:
            return result

        with enter('resolve query'):
            _rich_traceback_guard = True
            result = super().execute_graphql_request(
                request,
                data,
                query,
                variables,
                operation_name,
                *args,
                **kwargs,
            )
        if result is None:
            return None

        if result.errors:
            if settings.DEBUG or os.getenv('PYTEST_CURRENT_TEST', None):
                from rich.traceback import Traceback

                console = Console()

                def print_error(err: GraphQLError | Exception) -> None:
                    if getattr(err, '_was_printed', False):
                        return
                    if isinstance(err, GraphQLError):
                        orig = err.original_error
                        if orig is None:
                            return
                        err = orig

                    tb = Traceback.from_exception(
                        type(err),
                        err,
                        traceback=err.__traceback__,
                    )
                    console.print(tb)
            else:
                def print_error(err: GraphQLError | Exception) -> None:
                    if isinstance(err, GraphQLError):
                        orig = getattr(err, 'original_error', None)
                    else:
                        orig = err
                    if orig is not None:
                        level = 'ERROR'
                        orig_str = ' (resulting from: %s)' % str(orig)
                    else:
                        level = 'WARNING'
                        orig_str = ''
                    log(level, 'Query error: %s%s' % (str(err), orig_str))

            log('WARNING', 'Query resulted in %d errors' % len(result.errors))

            server_error_count = 0
            invalid_query_count = 0
            for error in result.errors:
                if isinstance(error, GraphQLError):
                    orig_error = getattr(error, 'original_error', None)
                else:
                    orig_error = error
                if orig_error:
                    server_error_count += 1
                    if server_error_count >= 5:
                        if server_error_count == 5:
                            logger.warning('Too many server errors, skipping the rest')
                        continue
                else:
                    invalid_query_count += 1
                    if invalid_query_count >= 5:
                        if invalid_query_count == 5:
                            logger.warning('Too many invalid query errors, skipping the rest')
                        continue
                print_error(error)
                if isinstance(error, GraphQLError) and error.original_error is None:
                    # It's an invalid query
                    logger.warning('GraphQL query error: {error.message} {error.locations}', error=error)
                    continue
                sentry_sdk.capture_exception(orig_error)

        # Check for the reasons why the result might not be cached
        if cache_key:
            def log_reason(msg, *args, okay=False, **kwargs) -> None:
                level = 'DEBUG' if okay else 'WARNING'
                log(level, 'not caching response: %s' % msg, *args, **kwargs)

            if result.errors:
                log_reason('query processing errors')
                cache_key = None
            elif getattr(request, 'graphene_no_cache', False):
                log_reason('not cacheable request', okay=True)
                cache_key = None

        if cache_key:
            log('DEBUG', 'Storing to cache: %s' % cache_key)
            with enter('store query response'):
                self.store_to_cache(cache_key, result)

        return result
