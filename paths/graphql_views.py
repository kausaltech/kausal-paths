import orjson
from django.conf import settings
from django.utils.translation import activate
from graphene_django.views import GraphQLView
from graphql.error import GraphQLError


SUPPORTED_LANGUAGES = {x[0] for x in settings.LANGUAGES}


class LocaleMiddleware:
    def process_locale_directive(self, info, directive):
        for arg in directive.arguments:
            if arg.name.value == 'lang':
                lang = arg.value.value
                if lang not in SUPPORTED_LANGUAGES:
                    raise GraphQLError("unsupported language: %s" % lang, [info])
                info.context._graphql_query_language = lang
                activate(lang)

    def resolve(self, next, root, info, **kwargs):
        if root is None:
            info.context._graphql_query_language = None
            operation = info.operation
            for directive in operation.directives:
                if directive.name.value == 'locale':
                    self.process_locale_directive(info, directive)
        return next(root, info, **kwargs)


class InstanceMiddleware:
    def resolve(self, next, root, info, **kwargs):
        if root is None:
            from pages.global_instance import instance

            instance.refresh()
            context = instance.context
            session = info.context.session

            scenario = None
            if 'active_scenario' in session:
                try:
                    scenario = context.get_scenario(session['active_scenario'])
                except KeyError:
                    del session['active_scenario']

            # Tell the custom scenario about the user setting so that
            # it can locate the customized parameters.
            if context.custom_scenario is not None:
                context.custom_scenario.set_session(session)

            if scenario is None:
                scenario = context.get_default_scenario()

            context.activate_scenario(scenario)

            info.context.instance = instance
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
