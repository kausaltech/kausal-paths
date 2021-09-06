from django.utils.translation import gettext_lazy, gettext, get_language
from django.core.exceptions import ImproperlyConfigured
from django.conf import settings


try:
    SUPPORTED_LANGUAGES = set([x[0] for x in settings.LANGUAGES])
    DEFAULT_LANGUAGE = settings.LANGUAGE_CODE
except ImproperlyConfigured:
    SUPPORTED_LANGUAGES = None
    DEFAULT_LANGUAGE = None


class TranslatedString:
    def __init__(self, *args, **kwargs):
        self.i18n = {}

        if len(args) > 1:
            raise Exception('You can supply at most one default translation')
        if len(args) == 1:
            if not DEFAULT_LANGUAGE:
                raise Exception('No default language found')
            self.i18n[DEFAULT_LANGUAGE] = args[0]
        self.i18n.update(kwargs)

    def __str__(self):
        if DEFAULT_LANGUAGE is None:
            return list(self.i18n.values())[0]
        lang = get_language()
        if lang not in self.i18n:
            if not DEFAULT_LANGUAGE:
                return self.i18n.values()[0]
            return self.i18n[DEFAULT_LANGUAGE]
        return self.i18n[lang]
