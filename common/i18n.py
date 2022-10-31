from __future__ import annotations
import typing

from django.utils.translation import gettext_lazy, gettext, get_language  # noqa
from django.core.exceptions import ImproperlyConfigured
from django.conf import settings


if typing.TYPE_CHECKING:
    from django.db.models import Model


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
            self.default_language = DEFAULT_LANGUAGE

        if len(args) > 1:
            raise Exception('You can supply at most one default translation')
        if len(args) == 1:
            if not self.default_language:
                raise Exception('No default language found')
            self.i18n[self.default_language] = args[0]

        if len(kwargs) == 1:
            self.default_language = list(kwargs.keys())[0]

        self.i18n.update(kwargs)

    def get_fallback(self) -> str:
        dl = self.default_language
        if dl in self.i18n:
            return self.i18n[dl]
        elif '_' in dl:
            lang, _ = dl.split('_')
            if lang in self.i18n:
                return self.i18n[lang]
        raise Exception("Default translation not available for: %s" % self.default_language)

    def __str__(self):
        try:
            lang = get_language()
        except Exception:
            lang = None
        if lang not in self.i18n:
            return self.get_fallback()
            return self.i18n[self.default_language]
        return self.i18n[lang]

    def __repr__(self):
        return "[i18n]'%s'" % str(self)


def get_modeltrans_attrs_from_str(
    s: str | TranslatedString, field_name: str, default_lang: str
) -> typing.Tuple[str, dict[str, str]]:
    i18n = {}
    if isinstance(s, TranslatedString):
        i18n.update({f'{field_name}_{lang}': v for lang, v in s.i18n.items()})
        if default_lang not in s.i18n:
            raise Exception("Field %s does not have a value in language %s" % (field_name, default_lang))
        field_val = s.i18n[default_lang]
    else:
        field_val = s
        i18n[f'{field_name}_{default_lang}'] = s

    return field_val, i18n
