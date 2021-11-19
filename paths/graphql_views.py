from typing import Optional
import orjson
from django.conf import settings
from django.utils.translation import activate, get_language_from_request
from graphene_django.views import GraphQLView
from graphql.language.ast import Variable
from graphql.error import GraphQLError

from nodes.models import Instance, InstanceConfig, InstanceHostname
from params.storage import SessionStorage
from .graphql_helpers import GQLContext, GQLInfo, GQLInstanceInfo


SUPPORTED_LANGUAGES = {x[0] for x in settings.LANGUAGES}


class LocaleMiddleware:
    def process_locale_directive(self, info: GQLInfo, directive) -> Optional[str]:
        for arg in directive.arguments:
            if arg.name.value == 'lang':
                lang = arg.value.value
                if lang not in SUPPORTED_LANGUAGES:
                    raise GraphQLError("unsupported language: %s" % lang, [info])
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
    def get_instance_by_identifier(self, queryset, identifier: str, info: GQLInfo = None) -> InstanceConfig:
        try:
            if identifier.isnumeric():
                instance = queryset.get(id=identifier)
            else:
                instance = queryset.get(identifier=identifier)
        except InstanceConfig.DoesNotExist:
            raise GraphQLError("Instance with identifier %s not found" % identifier, [info] if info else None)
        return instance

    def get_instance_by_hostname(self, queryset, hostname: str, info: GQLInfo = None) -> InstanceConfig:
        try:
            instance = queryset.for_hostname(hostname).get()
        except InstanceConfig.DoesNotExist:
            print('not found')
            raise GraphQLError("Instance matching hostname %s not found" % hostname, [info] if info else None)
        return instance

    def process_instance_directive(self, info: GQLInfo, directive) -> InstanceConfig:
        qs = InstanceConfig.objects.all()
        variable_vals = info.variable_values
        for arg in directive.arguments:
            if isinstance(arg.value, Variable):
                val = variable_vals.get(arg.value.name.value)
            else:
                val = arg.value.value
            if arg.name.value == 'identifier' and val:
                return self.get_instance_by_identifier(qs, val, info)
            if arg.name.value == 'hostname' and val:
                return self.get_instance_by_hostname(qs, val, info)
        raise GraphQLError("Invalid instance directive", [info])

    def process_instance_headers(self, context: GQLContext) -> Optional[InstanceConfig]:
        identifier = context.headers.get(settings.INSTANCE_IDENTIFIER_HEADER)
        hostname = context.headers.get(settings.INSTANCE_HOSTNAME_HEADER)
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

        return instance_config.get_instance()

    def activate_instance(self, instance: Instance, info: GQLInstanceInfo):
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

        context.activate_scenario(scenario)

        info.context.instance = instance

    def resolve(self, next, root, info: GQLInfo, **kwargs):
        if root is None:
            instance = self.determine_instance(info)
            if instance is not None:
                self.activate_instance(instance, info)
            else:
                info.context.instance = None
        return next(root, info, **kwargs)


class PathsGraphQLView(GraphQLView):
    def __init__(self, *args, **kwargs):
        if 'middleware' not in kwargs:
            kwargs['middleware'] = (LocaleMiddleware, InstanceMiddleware)
        super().__init__(*args, **kwargs)

    def json_encode(self, request, d, pretty=False):
        if not (self.pretty or pretty) and not request.GET.get("pretty"):
            return orjson.dumps(d)
        return orjson.dumps(d, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)

    def execute_graphql_request(self, request, data, query, variables, operation_name, *args, **kwargs):
        if settings.DEBUG:
            from rich.console import Console
            from rich.syntax import Syntax

            console = Console()
            syntax = Syntax(query, "graphql")
            console.print(syntax)

        ret = super().execute_graphql_request(request, data, query, variables, operation_name, *args, **kwargs)
        return ret
