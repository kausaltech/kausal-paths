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
    i18n: dict[str, str]
    default_language: str

    def __init__(self, *args, **kwargs):
        self.i18n = {}
        if 'default_language' in kwargs:
            self.default_language = kwargs.pop('default_language')
        else:
            self.default_language = None

        if len(args) > 1:
            raise Exception('You can supply at most one default translation')
        if len(args) == 1:
            if not self.default_language:
                raise Exception('No default language found')
            self.i18n[self.default_language] = args[0]

        if len(kwargs) == 1:
            self.default_language = list(kwargs.keys())[0]

        self.i18n.update(kwargs)

    def __str__(self):
        try:
            lang = get_language()
        except:
            lang = None
        if lang not in self.i18n:
            return self.i18n[self.default_language]
        return self.i18n[lang]
