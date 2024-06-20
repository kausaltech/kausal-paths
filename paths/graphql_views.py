from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from typing import Any, Dict, List, Optional, cast

import orjson
import sentry_sdk
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.core.cache import cache
from django.http import HttpRequest
from django.utils import translation
from graphene_django.views import GraphQLView
from graphql import (
    DirectiveNode,
    ExecutionResult,
    GraphQLOutputType,
    OperationType,
)
from graphql.error import GraphQLError
from graphql.execution import ExecutionContext
from graphql.language import FieldNode, OperationDefinitionNode
from graphql.language.ast import VariableNode
from graphql.pyutils import AwaitableOrValue, Path
from graphql.type import GraphQLObjectType
from loguru import logger
from rich.console import Console
from rich.syntax import Syntax
from sentry_sdk.tracing import Transaction

from kausal_common.testing.graphql import capture_query
from nodes.models import Instance, InstanceConfig, InstanceConfigQuerySet
from nodes.perf import PerfContext
from params.storage import SessionStorage
from paths.authentication import IDTokenAuthentication
from paths.const import INSTANCE_HOSTNAME_HEADER, INSTANCE_IDENTIFIER_HEADER, WILDCARD_DOMAINS_HEADER

from .graphql_helpers import GQLContext, GQLInstanceContext, GraphQLPerfNode

SUPPORTED_LANGUAGES = {x[0] for x in settings.LANGUAGES}


def _arg_value(arg, variable_vals):
    if isinstance(arg.value, VariableNode):
        return variable_vals.get(arg.value.name.value)
    return arg.value.value


logger = logger.bind(markup=True)

GRAPHQL_CAPTURE_QUERIES = os.getenv('GRAPHQL_CAPTURE_QUERIES', '0') == '1'


class PathsExecutionContext(ExecutionContext):
    context_value: GQLInstanceContext

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def handle_field_error(
        self, error: GraphQLError, return_type: GraphQLOutputType,
    ) -> None:
        if settings.DEBUG and error.original_error is not None:
            from rich import traceback
            exc = error
            tb = traceback.Traceback.from_exception(
                type(exc), exc, traceback=exc.__traceback__
            )
            console = Console()
            console.print(tb)
            raise error.original_error
        else:
            return super().handle_field_error(error, return_type)

    def process_locale_directive(self, ic: InstanceConfig, directive: DirectiveNode) -> Optional[str]:
        for arg in directive.arguments:
            if arg.name.value == 'lang':
                lang = _arg_value(arg, self.variable_values)
                if lang not in ic.supported_languages:
                    raise GraphQLError("unsupported language: %s" % lang, directive)
                return lang
        return None

    def activate_language(self, instance: InstanceConfig, operation: OperationDefinitionNode):
        # First see if the locale directive is there. If not, fall back to
        # figuring out the locale from the request.
        lang = None
        for directive in operation.directives or []:
            if directive.name.value == 'locale':
                lang = self.process_locale_directive(instance, directive)
                break

        if lang is None:
            lang = instance.primary_language

        self.context_value.graphql_query_language = lang
        return cast(AbstractContextManager, translation.override(lang))

    def get_instance_by_identifier(self, queryset: InstanceConfigQuerySet, identifier: str, directive: DirectiveNode | None = None) -> InstanceConfig:
        try:
            if identifier.isnumeric():
                instance = queryset.get(id=identifier)
            else:
                instance = queryset.get(identifier=identifier)
        except InstanceConfig.DoesNotExist:
            raise GraphQLError("Instance with identifier %s not found" % identifier, directive)
        return instance

    def get_instance_by_hostname(self, queryset: InstanceConfigQuerySet, hostname: str, directive: DirectiveNode | None = None) -> InstanceConfig:
        request = self.context_value
        try:
            instance = queryset.for_hostname(hostname, request).get()
        except InstanceConfig.DoesNotExist:
            logger.warning(f"No instance found for hostname {hostname} (wildcard domains: {request.wildcard_domains})")
            raise GraphQLError("Instance matching hostname %s not found" % hostname, directive)
        return instance

    def process_instance_directive(self, directive: DirectiveNode) -> InstanceConfig:
        qs = InstanceConfig.objects.all()
        arguments = {arg.name.value: _arg_value(arg, self.variable_values) for arg in directive.arguments}
        identifier = arguments.get('identifier')
        hostname = arguments.get('hostname')
        token = arguments.get('token')  # noqa
        if identifier:
            return self.get_instance_by_identifier(qs, identifier, directive)
        if hostname:
            return self.get_instance_by_hostname(qs, hostname, directive)
        raise GraphQLError("Invalid instance directive", directive)

    def process_instance_headers(self, context: GQLContext) -> InstanceConfig | None:
        identifier = context.headers.get(settings.INSTANCE_IDENTIFIER_HEADER)
        hostname = context.headers.get(settings.INSTANCE_HOSTNAME_HEADER)
        auth = IDTokenAuthentication()
        ret = auth.authenticate(context)  # type: ignore
        if ret is not None:
            user, token = ret
            context.user = user

        qs = InstanceConfig.objects.all()
        if identifier:
            return self.get_instance_by_identifier(qs, identifier)
        if hostname:
            return self.get_instance_by_hostname(qs, hostname)
        return None

    def determine_instance(self, operation: OperationDefinitionNode):
        instance_config: Optional[InstanceConfig] = None

        for directive in operation.directives or []:
            if directive.name.value == 'instance':
                instance_config = self.process_instance_directive(directive)
                break
        else:
            instance_config = self.process_instance_headers(self.context_value)

        if instance_config is None:
            return None

        if instance_config.is_protected and not self.context_value.user.is_active:
            raise GraphQLError("Instance is protected", extensions=dict(code='instance_protected'))

        return instance_config

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

        ic = self.determine_instance(operation)
        if ic is None:
            yield
            return

        with perf.exec_node(GraphQLPerfNode('get instance "%s"' % ic.identifier)):
            instance = ic.get_instance(generate_baseline=False)
        self.context_value.instance = instance
        context = instance.context

        with ExitStack() as stack:
            with perf.exec_node(GraphQLPerfNode('prepare instance "%s"' % ic.identifier)):
                stack.enter_context(self.activate_language(ic, operation))
                stack.enter_context(instance.lock)
                stack.enter_context(context.run())
                if not context.baseline_values_generated:
                    with (
                        sentry_sdk.start_span(op='calc', description='Generate baseline'),
                        request.graphql_perf.exec_node(GraphQLPerfNode('generate baseline'))
                    ):
                        context.generate_baseline_values()
                self.activate_instance(instance)
            yield

    def execute_fields(
        self,
        parent_type: GraphQLObjectType,
        source_value: Any,
        path: Path | None,
        fields: Dict[str, List[FieldNode]]
    ) -> AwaitableOrValue[Dict[str, Any]]:
        path_parts = path.as_list() if path else []
        span_cm: AbstractContextManager
        str_path = '.'.join([str(x) for x in path_parts])
        if path_parts:
            node = GraphQLPerfNode(str_path)
            span_cm = self.context_value.graphql_perf.exec_node(node)
        else:
            span_cm = nullcontext()
        #if field_def and not isinstance(field_def.type, GraphQLScalarType):
        #    print(path.as_list(), field_def_type)

        with span_cm:
            #print('-> %s' % str_path)
            ret = super().execute_fields(parent_type, source_value, path, fields)
            #print('<- %s' % str_path)
        return ret

    def execute_operation(self, operation: OperationDefinitionNode, root_value: Any) -> AwaitableOrValue[Any] | None:
        op_name = operation.name.value if operation.name else '<unnamed>'
        with sentry_sdk.start_span(op='graphql_execute', description='Query %s' % op_name):
            if operation.operation != OperationType.QUERY:
                setattr(self.context_value, 'graphene_no_cache', True)
            with self.instance_context(operation):
                ret = super().execute_operation(operation, root_value)
            return ret


class PathsGraphQLView(GraphQLView):
    graphiql_version = "3.0.9"
    graphiql_sri = "sha256-i8HFOsDB6KaRVstG2LSibODRlHNNA1XLKLnDl7TIAZY="
    graphiql_css_sri = "sha256-wTzfn13a+pLMB5rMeysPPR1hO7x0SwSeQI+cnw7VdbE="
    graphiql_plugin_explorer_version = "1.0.2"
    graphiql_plugin_explorer_sri = "sha256-CD435QHT45IKYOYnuCGRrwVgCRJNzoKjMuisdNtso4s="
    graphiql_plugin_explorer_css_sri = "sha256-G6RZ0ey9eHIvQt0w+zQYVh4Rq1nNneislHMWMykzbLs="

    execution_context_class = PathsExecutionContext

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_reg(self, level: str, operation_name: str | None, msg, *args, depth: int = 0, **kwargs):
        log = logger.opt(depth=1 + depth)
        if operation_name:
            log = log.bind(**{'graphql_operation': operation_name}).log(level, 'GQL request [magenta]%s[/]: %s' % (operation_name, msg), *args, **kwargs)

    def json_encode(self, request: GQLInstanceContext, d, pretty=False):
        errors = []
        def serialize_unknown(obj):
            err = TypeError("Unable to serialize value %s with type %s" % (obj, type(obj)))
            errors.append(err)
            return ' '.join(['__INVALID__' * 10])

        if not (self.pretty or pretty) and not request.GET.get("pretty"):
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
            'DEBUG', op_name, 'Response was {} bytes', len(ret)
        )
        return ret

    def get_ic_from_headers(self, request: HttpRequest):
        identifier = request.headers.get('x-paths-instance-identifier')
        hostname = request.headers.get('x-paths-instance-hostname')  # noqa
        if not identifier:
            return None
        return InstanceConfig.objects.filter(identifier=identifier).first()

    def get_cache_key(
        self, request: GQLInstanceContext, query: str | None, variables: dict | None,
        operation_name: str | None
    ):
        def log_reason(reason: str):
            self.log_reg('DEBUG', operation_name, 'request not cached: [red]{}[/]', reason, depth=1)
        if not query:
            return None
        if request.user.is_authenticated:
            log_reason('user authenticated')
            return None

        ic = self.get_ic_from_headers(request)
        if ic is None:
            log_reason('no instance config')
            return None

        session_key = SessionStorage.get_cache_key(request.session, ic.identifier)
        if session_key is None:
            log_reason('user session has custom parameters')
            return None

        m = hashlib.sha1()
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

    def get_response(self, request: HttpRequest, data, show_graphiql=False):
        operation_name = request.GET.get("operationName") or data.get("operationName")
        if operation_name == "null":
            operation_name = None
        request = cast(GQLInstanceContext, request)
        perf: PerfContext = PerfContext(
            supports_cache=False, min_ms=10, description=operation_name
        )
        perf.enabled = settings.ENABLE_PERF_TRACING
        request.graphql_perf = perf
        with perf:
            start = time.time()
            ret = super().get_response(request, data, show_graphiql)
            if GRAPHQL_CAPTURE_QUERIES and data and ret[0]:
                headers = [INSTANCE_IDENTIFIER_HEADER, INSTANCE_HOSTNAME_HEADER, WILDCARD_DOMAINS_HEADER]
                now = time.time()
                logger.info("Capturing GraphQL query and response")
                capture_query(request, headers, data, ret[0], ret[1], (now - start) * 1000)

        return ret

    def execute_graphql_request(
        self, request: GQLInstanceContext, data: dict, query: str | None, variables: dict | None,
        operation_name: str | None, *args, **kwargs
    ):
        "Sets up context for the request execution."

        request._referer = self.request.META.get('HTTP_REFERER')
        request.graphql_operation_name = operation_name
        wildcard_domains = request.headers.get(settings.WILDCARD_DOMAINS_HEADER)
        request.wildcard_domains = [d.lower() for d in wildcard_domains.split(',')] if wildcard_domains else []

        transaction: Transaction | None = sentry_sdk.Hub.current.scope.transaction
        if query is not None:
            query = query.strip()
        if settings.LOG_GRAPHQL_QUERIES and query:
            console = Console()
            syntax = Syntax(query, "graphql")
            console.print(syntax)
            if variables:
                console.print('Variables:', variables)

        with sentry_sdk.push_scope() as scope:  # type: ignore
            scope.set_context('graphql_variables', variables or {})
            scope.set_tag('graphql_operation_name', operation_name)
            scope.set_tag('referer', request._referer)

            if transaction is not None:
                span = transaction.start_child(op='graphql query', description=operation_name)
                span.set_data('graphql_variables', variables)
                span.set_tag('graphql_operation_name', operation_name)
                span.set_tag('referer', request._referer)
            else:
                # No tracing activated, use an inert Span
                span = sentry_sdk.start_span()

            with span, request.graphql_perf.exec_node(GraphQLPerfNode('execute %s' % operation_name)):
                return self._execute_graphql_request(request, data, query, variables, operation_name, *args, **kwargs)

    def _execute_graphql_request(
        self, request: GQLInstanceContext, data: dict, query: str | None, variables: dict | None,
        operation_name: str | None, *args, **kwargs
    ):
        def log(level: str, msg: str, *args, depth: int = 0, **kwargs):
            self.log_reg(level, operation_name, msg, *args, depth=depth + 1, **kwargs)

        def enter(id: str):
            return request.graphql_perf.exec_node(GraphQLPerfNode(id))

        # We currently disable users in the graphql requests
        request.user = AnonymousUser()

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
            result = super().execute_graphql_request(
                request, data, query, variables, operation_name, *args, **kwargs
            )
        if result is None:
            return None

        if result.errors:
            if settings.DEBUG:
                from rich.traceback import Traceback
                console = Console()

                def print_error(err: GraphQLError, orig: Exception | None):
                    oe = getattr(err, 'original_error', err)
                    if oe:
                        if settings.DEBUG:
                            raise oe
                        tb = Traceback.from_exception(
                            type(oe), oe, traceback=oe.__traceback__
                        )
                        console.print(tb)
            else:
                def print_error(err: GraphQLError, orig: Exception | None):
                    if orig is not None:
                        level = 'ERROR'
                        orig_str = ' (resulting from: %s)' % str(orig)
                    else:
                        level = 'WARNING'
                        orig_str = ''
                    log(level, 'Query error: %s%s' % (str(err), orig_str))

            log('WARNING', "Query resulted in %d errors" % len(result.errors))

            server_errors = []
            for error in result.errors:
                err = getattr(error, 'original_error', None)
                print_error(error, orig=err)
                if not err:
                    # It's an invalid query
                    continue
                server_errors.append(error)
                sentry_sdk.capture_exception(err)

            if settings.DEBUG and server_errors:
                raise server_errors[0]

        # Check for the reasons why the result might not be cached
        if cache_key:
            def log_reason(msg, *args, okay=False, **kwargs):
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
