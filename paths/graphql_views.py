import logging
from contextlib import ExitStack, contextmanager
from typing import Any, Optional

import orjson
import sentry_sdk
from django.conf import settings
from django.utils.translation import activate, get_language_from_request
from graphene_django.views import GraphQLView
from graphql import DirectiveNode
from graphql.error import GraphQLError
from graphql.execution import ExecutionContext
from graphql.language import OperationDefinitionNode
from graphql.language.ast import VariableNode
from graphql.pyutils import AwaitableOrValue
from rich.console import Console
from rich.syntax import Syntax

from common.perf import PerfCounter
from nodes.models import Instance, InstanceConfig
from params.storage import SessionStorage
from paths.authentication import IDTokenAuthentication

from .graphql_helpers import GQLContext, GQLInstanceContext

SUPPORTED_LANGUAGES = {x[0] for x in settings.LANGUAGES}
logger = logging.getLogger(__name__)


def _arg_value(arg, variable_vals):
    if isinstance(arg.value, VariableNode):
        return variable_vals.get(arg.value.name.value)
    return arg.value.value


class PathsExecutionContext(ExecutionContext):
    context_value: GQLInstanceContext

    def process_locale_directive(self, directive: DirectiveNode) -> Optional[str]:
        for arg in directive.arguments:
            if arg.name.value == 'lang':
                lang = arg.value.value  # type: ignore
                if lang not in SUPPORTED_LANGUAGES:
                    raise GraphQLError("unsupported language: %s" % lang, directive)
                return lang
        return None

    def activate_language(self, operation: OperationDefinitionNode):
        # First see if the locale directive is there. If not, fall back to
        # figuring out the locale from the request.
        lang = None
        for directive in operation.directives or []:
            if directive.name.value == 'locale':
                lang = self.process_locale_directive(directive)
                break

        if lang is None:
            lang = get_language_from_request(self.context_value)

        self.context_value.graphql_query_language = lang
        activate(lang)

    def get_instance_by_identifier(self, queryset, identifier: str, directive: DirectiveNode | None = None) -> InstanceConfig:
        try:
            if identifier.isnumeric():
                instance = queryset.get(id=identifier)
            else:
                instance = queryset.get(identifier=identifier)
        except InstanceConfig.DoesNotExist:
            raise GraphQLError("Instance with identifier %s not found" % identifier, directive)
        return instance

    def get_instance_by_hostname(self, queryset, hostname: str, directive: DirectiveNode | None = None) -> InstanceConfig:
        try:
            instance = queryset.for_hostname(hostname).get()
        except InstanceConfig.DoesNotExist:
            raise GraphQLError("Instance matching hostname %s not found" % hostname, directive)
        return instance

    def process_instance_directive(self, directive: DirectiveNode) -> InstanceConfig:
        qs = InstanceConfig.objects.all()
        arguments = {arg.name.value: _arg_value(arg, self.variable_values) for arg in directive.arguments}
        identifier = arguments.get('identifier')
        hostname = arguments.get('hostname')
        token = arguments.get('token')
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

        return instance_config.get_instance(generate_baseline=False)

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
        instance = self.determine_instance(operation)
        if instance is None:
            yield
            return

        self.context_value.instance = instance
        context = instance.context

        with ExitStack() as stack:
            stack.enter_context(instance.lock)
            if context.dataset_repo is not None:
                stack.enter_context(context.dataset_repo.lock.lock)

            context.perf_context.start()
            context.cache.start_run()
            context.generate_baseline_values()
            self.activate_instance(instance)

            try:
                yield
            finally:
                context.cache.end_run()
                context.perf_context.stop()
                assert instance is not None

    def execute_operation(self, operation: OperationDefinitionNode, root_value: Any) -> AwaitableOrValue[Any] | None:
        self.activate_language(operation)

        try:
            with self.instance_context(operation):
                ret = super().execute_operation(operation, root_value)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise
        return ret


class PathsGraphQLView(GraphQLView):
    graphiql_version = "2.4.1"
    graphiql_sri = "sha256-s+f7CFAPSUIygFnRC2nfoiEKd3liCUy+snSdYFAoLUc="
    graphiql_css_sri = "sha256-88yn8FJMyGboGs4Bj+Pbb3kWOWXo7jmb+XCRHE+282k="
    execution_context_class = PathsExecutionContext

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def json_encode(self, request, d, pretty=False):
        if not (self.pretty or pretty) and not request.GET.get("pretty"):
            return orjson.dumps(d)
        return orjson.dumps(d, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)

    def execute_graphql_request(self, request: GQLInstanceContext, data, query, variables, operation_name, *args, **kwargs):
        request._referer = self.request.META.get('HTTP_REFERER')
        transaction = sentry_sdk.Hub.current.scope.transaction
        logger.info('GraphQL request %s from %s' % (operation_name, request._referer))
        if settings.LOG_GRAPHQL_QUERIES and query and query.strip():
            console = Console()
            syntax = Syntax(query, "graphql")
            console.print(syntax)
            if variables:
                console.print('Variables:', variables)

        with sentry_sdk.push_scope() as scope:  # type: ignore
            scope.set_context('graphql_variables', variables)
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

            with span:
                pc = PerfCounter('graphql query')
                result = super().execute_graphql_request(
                    request, data, query, variables, operation_name, *args, **kwargs
                )
                query_time = pc.measure()
                logger.debug("GQL response took %.1f ms" % query_time)

            # If 'invalid' is set, it's a bad request
            if result and result.errors:
                if settings.LOG_GRAPHQL_QUERIES:
                    from rich.traceback import Traceback
                    console = Console()

                    def print_error(err: GraphQLError):
                        console.print(err)
                        oe = err.original_error
                        if oe:
                            tb = Traceback.from_exception(
                                type(oe), oe, traceback=oe.__traceback__
                            )
                            console.print(tb)
                else:
                    def print_error(err: GraphQLError):
                        pass

                for error in result.errors:
                    print_error(error)
                    err = getattr(error, 'original_error', None)
                    if not err:
                        # It's an invalid query
                        continue
                    sentry_sdk.capture_exception(err)

        return result
