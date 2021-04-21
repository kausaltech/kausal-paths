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


class PathsGraphQLView(GraphQLView):
    def __init__(self, *args, **kwargs):
        if 'middleware' not in kwargs:
            kwargs['middleware'] = (LocaleMiddleware,)
        super().__init__(*args, **kwargs)
