from contextlib import contextmanager
import logging
from typing import Optional, Tuple
from graphql import DirectiveNode

import orjson
from common.perf import PerfCounter
from paths.authentication import IDTokenAuthentication, InstanceTokenAuthentication
import sentry_sdk
from django.conf import settings
from django.utils.translation import activate, get_language_from_request
from graphene_django.views import GraphQLView
from graphql.language.ast import VariableNode
from graphql.error import GraphQLError
from rich.console import Console
from rich.syntax import Syntax
from rest_framework.authentication import TokenAuthentication

from nodes.models import Instance, InstanceConfig
from params.storage import SessionStorage
from .graphql_helpers import GQLContext, GQLInfo, GQLInstanceContext, GQLInstanceInfo


SUPPORTED_LANGUAGES = {x[0] for x in settings.LANGUAGES}
logger = logging.getLogger(__name__)


def _arg_value(arg, variable_vals):
    if isinstance(arg.value, VariableNode):
        return variable_vals.get(arg.value.name.value)
    return arg.value.value


class LocaleMiddleware:
    def process_locale_directive(self, info: GQLInfo, directive) -> Optional[str]:
        for arg in directive.arguments:
            if arg.name.value == 'lang':
                lang = arg.value.value
                if lang not in SUPPORTED_LANGUAGES:
                    raise GraphQLError("unsupported language: %s" % lang, info.field_nodes)
                return lang
        return None

    def resolve(self, next, root, info: GQLInfo, **kwargs):
        if root is None:
            lang = None
            operation = info.operation
            # First see if the locale directive is there. If not, fall back to
            # figuring out the locale from the request.
            for directive in operation.directives or []:
                if directive.name.value == 'locale':
                    lang = self.process_locale_directive(info, directive)
                    break

            if lang is None:
                lang = get_language_from_request(info.context)

            info.context.graphql_query_language = lang
            activate(lang)

        return next(root, info, **kwargs)


class InstanceMiddleware:
    def get_instance_by_identifier(self, queryset, identifier: str, info: GQLInfo | None = None) -> InstanceConfig:
        try:
            if identifier.isnumeric():
                instance = queryset.get(id=identifier)
            else:
                instance = queryset.get(identifier=identifier)
        except InstanceConfig.DoesNotExist:
            raise GraphQLError("Instance with identifier %s not found" % identifier, info.field_nodes if info else None)
        return instance

    def get_instance_by_hostname(self, queryset, hostname: str, info: GQLInfo | None = None) -> InstanceConfig:
        try:
            instance = queryset.for_hostname(hostname).get()
        except InstanceConfig.DoesNotExist:
            raise GraphQLError("Instance matching hostname %s not found" % hostname, info.field_nodes if info else None)
        return instance

    def process_instance_directive(self, info: GQLInfo, directive) -> InstanceConfig:
        qs = InstanceConfig.objects.all()
        arguments = {arg.name.value: _arg_value(arg, info.variable_values) for arg in directive.arguments}
        identifier = arguments.get('identifier')
        hostname = arguments.get('hostname')
        token = arguments.get('token')
        if identifier:
            return self.get_instance_by_identifier(qs, identifier, info)
        if hostname:
            return self.get_instance_by_hostname(qs, hostname, info)
        raise GraphQLError("Invalid instance directive", info.field_nodes)

    def process_instance_headers(self, context: GQLContext) -> InstanceConfig | None:
        identifier = context.headers.get(settings.INSTANCE_IDENTIFIER_HEADER)
        hostname = context.headers.get(settings.INSTANCE_HOSTNAME_HEADER)
        auth = IDTokenAuthentication()
        ret = auth.authenticate(context)
        if ret is not None:
            user, token = ret
            context.user = user

        qs = InstanceConfig.objects.all()
        if identifier:
            return self.get_instance_by_identifier(qs, identifier)
        if hostname:
            return self.get_instance_by_hostname(qs, hostname)
        return None

    def determine_instance(self, info: GQLInfo):
        instance_config: Optional[InstanceConfig] = None

        for directive in info.operation.directives or []:
            if directive.name.value == 'instance':
                instance_config = self.process_instance_directive(info, directive)
                break
        else:
            instance_config = self.process_instance_headers(info.context)

        if instance_config is None:
            return None

        if instance_config.is_protected and not info.context.user.is_active:
            raise GraphQLError("Instance is protected", extensions=dict(code='instance_protected'))

        return instance_config.get_instance(generate_baseline=True)

    def activate_instance(self, instance: Instance, info: GQLInfo):
        context = instance.context
        context.setting_storage = storage = SessionStorage(instance=instance, session=info.context.session)
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
    def instance_context(self, info: GQLInfo):
        context = None
        instance = self.determine_instance(info)
        if instance is not None:
            self.activate_instance(instance, info)
            info.context.instance = instance  # type: ignore
            context = instance.context
            context.perf_context.start()
            context.cache.start_run()

        try:
            yield
        finally:
            if context is not None:
                context.cache.end_run()
                context.perf_context.stop()

    def resolve(self, next, root, info: GQLInfo, **kwargs):
        if root is None:
            with self.instance_context(info):
                return next(root, info, **kwargs)
        else:
            return next(root, info, **kwargs)

class PathsGraphQLView(GraphQLView):
    graphiql_version = "2.2.0"
    graphiql_sri = "sha256-fLMqXdOkS8Q7/tzR9a511DnNqR1Z9sk2hPOBXMVrCnY="
    graphiql_css_sri = "sha256-JZnrtAzCwc6VWlHZwhlFS77c7Jv8tcD2jGd3vdjMfWU="

    def __init__(self, *args, **kwargs):
        if 'middleware' not in kwargs:
            kwargs['middleware'] = (LocaleMiddleware, InstanceMiddleware)
        super().__init__(*args, **kwargs)

    def json_encode(self, request, d, pretty=False):
        if not (self.pretty or pretty) and not request.GET.get("pretty"):
            return orjson.dumps(d)
        return orjson.dumps(d, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)

    def execute_graphql_request(self, request: GQLInstanceContext, data, query, variables, operation_name, *args, **kwargs):
        request._referer = self.request.META.get('HTTP_REFERER')
        transaction = sentry_sdk.Hub.current.scope.transaction
        logger.info('GraphQL request %s from %s' % (operation_name, request._referer))
        if settings.LOG_GRAPHQL_QUERIES:
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
