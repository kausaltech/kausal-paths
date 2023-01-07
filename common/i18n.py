from __future__ import annotations
import typing
import threading
from contextlib import contextmanager

from django.utils.translation import gettext_lazy, gettext, get_language  # noqa
from django.core.exceptions import ImproperlyConfigured
from django.conf import settings


if typing.TYPE_CHECKING:
    from django.db.models import Model
    from django.utils.functional import _StrPromise as StrPromise  # type: ignore


SUPPORTED_LANGUAGES: typing.Set[str] | None
DEFAULT_LANGUAGE: str | None

try:
    SUPPORTED_LANGUAGES = set([x[0] for x in settings.LANGUAGES])
    DEFAULT_LANGUAGE = settings.LANGUAGE_CODE
except ImproperlyConfigured:
    SUPPORTED_LANGUAGES = None
    DEFAULT_LANGUAGE = None


local = threading.local()
local.default_language = DEFAULT_LANGUAGE

@contextmanager
def set_default_language(lang: str):
    old = local.default_language
    local.default_language = lang
    try:
        yield
    finally:
        local.default_language = old


class TranslatedString:
    i18n: dict[str, str]
    default_language: str

    def __init__(self, *args: str, **kwargs: str):
        self.i18n = {}
        default_language: str | None
        if 'default_language' in kwargs:
            default_language = kwargs.pop('default_language')
        else:
            default_language = local.default_language

        if len(args) > 1:
            raise Exception('You can supply at most one default translation')
        if len(args) == 1:
            if not default_language:
                raise Exception('No default language found')
            self.i18n[default_language] = args[0]
        elif len(kwargs) == 1:
            default_language = list(kwargs.keys())[0]

        assert default_language is not None
        self.default_language = default_language
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

    def all(self) -> list[str]:
        unique_vals = set(self.i18n.values())
        return list(unique_vals)

    def __str__(self):
        try:
            lang = get_language()
        except Exception:
            lang = None
        if lang not in self.i18n:
            return self.get_fallback()
        return self.i18n[lang]

    def __repr__(self):
        return "[i18n]'%s'" % str(self)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, str):
            return cls(v)
        if not isinstance(v, dict):
            raise ValueError('expecting a dict')
        languages = list(v.keys())
        if 'default_language' in languages:
            languages.pop('default_language')
        if SUPPORTED_LANGUAGES:
            for lang in languages:
                if lang not in SUPPORTED_LANGUAGES:
                    raise ValueError('unsupported language: %s' % lang)
        return cls(**v)


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


I18nString = typing.Union[TranslatedString, str, 'StrPromise']
