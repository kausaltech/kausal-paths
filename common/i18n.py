from __future__ import annotations

import abc
import threading
import types
import typing
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeAliasType

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.translation import (
    get_language as get_language,  # noqa: PLC0414
    gettext as gettext,  # noqa: PLC0414
    gettext_lazy as gettext_lazy,  # noqa: PLC0414
)
from pydantic import BaseModel, GetCoreSchemaHandler, model_validator
from pydantic_core import CoreSchema, core_schema

from kausal_common.i18n.helpers import convert_language_code

if TYPE_CHECKING:
    from django.db.models import Model
    from django_stubs_ext import StrPromise


SUPPORTED_LANGUAGES: set[str] | None
DEFAULT_LANGUAGE: str | None

try:
    SUPPORTED_LANGUAGES = {x[0] for x in settings.LANGUAGES}
    DEFAULT_LANGUAGE = settings.LANGUAGE_CODE
except ImproperlyConfigured:
    SUPPORTED_LANGUAGES = None
    DEFAULT_LANGUAGE = None


local = threading.local()
local.default_language = DEFAULT_LANGUAGE


@contextmanager
def set_default_language(lang: str):
    old = getattr(local, 'default_language', DEFAULT_LANGUAGE)
    local.default_language = lang
    try:
        yield
    finally:
        local.default_language = old


def get_default_language() -> str | None:
    return getattr(local, 'default_language', DEFAULT_LANGUAGE)


class TranslatedString:
    i18n: dict[str, str]
    default_language: str

    def __init__(self, *args: str, default_language: str | None = None, **kwargs: str):
        self.i18n = {}
        if default_language is None:
            default_language = get_default_language()

        if len(args) > 1:
            raise Exception('You can supply at most one default translation')
        if len(args) == 1:
            if not default_language:
                raise Exception('No default language found')
            default_language = convert_language_code(default_language, 'kausal')
            self.i18n[default_language] = args[0]
        elif len(kwargs) == 1:
            default_language = next(iter(kwargs.keys()))

        kwargs = {convert_language_code(key, 'kausal'): value for key, value in kwargs.items()}

        assert default_language is not None
        default_language = convert_language_code(default_language, 'kausal')
        self.default_language = default_language

        self.i18n.update(kwargs)

    def get_fallback(self) -> str:
        dl = self.default_language
        if dl in self.i18n:
            return self.i18n[dl]
        if '-' in dl:
            lang, _ = dl.split('-')
            if lang in self.i18n:
                return self.i18n[lang]
        raise Exception('Default translation not available for: %s' % self.default_language)

    def all(self) -> list[str]:
        unique_vals = set(self.i18n.values())
        return list(unique_vals)

    def set_modeltrans_field(self, obj: Model, field_name: str, default_language: str):
        field_val, i18n = get_modeltrans_attrs_from_str(self, field_name, default_lang=default_language)
        setattr(obj, field_name, field_val)

        old_i18n: dict[str, str] = dict(getattr(obj, 'i18n') or {})  # noqa: B009
        old_i18n.update(i18n)
        setattr(obj, 'i18n', old_i18n)  # noqa: B010

    def __str__(self):
        try:
            lang = get_language()
        except Exception:
            lang = None

        if lang:
            lang = convert_language_code(lang, 'kausal')

        if lang not in self.i18n:
            return self.get_fallback()

        return self.i18n[lang]

    def __repr__(self):
        return "[i18n]'%s'" % str(self)

    @classmethod
    def __get_validators__(cls):  # noqa: ANN206
        yield cls.validate

    @classmethod
    def validate(cls, v) -> typing.Self:
        if isinstance(v, str):
            return cls(v)
        if isinstance(v, TranslatedString):
            return cls(default_language=v.default_language, **v.i18n)
        if not isinstance(v, dict):
            raise TypeError('TranslatedString expects a dict or str, not %s' % type(v))
        languages = list(v.keys())
        if 'default_language' in languages:
            languages.remove('default_language')
        if SUPPORTED_LANGUAGES:
            for lang in languages:
                if lang not in SUPPORTED_LANGUAGES:
                    raise ValueError('unsupported language: %s' % lang)
        return cls(**v)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        def validate(v) -> TranslatedString:
            return cls.validate(v)

        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(validate),
            ],
        )
        return core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(cls),
                    from_str_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance),
            ),
        )


def get_modeltrans_attrs_from_str(
    s: str | TranslatedString,
    field_name: str,
    default_lang: str,
) -> tuple[str, dict[str, str]]:
    i18n = {}
    default_lang = convert_language_code(default_lang, 'kausal')

    if isinstance(s, TranslatedString):
        translations = {
            f'{field_name}_{convert_language_code(lang, "modeltrans")}': v for lang, v in s.i18n.items() if lang != default_lang
        }
        i18n.update(translations)

        if default_lang not in s.i18n:
            fallbacks = settings.MODELTRANS_FALLBACK.get(convert_language_code(default_lang, 'django'), ())
            for lang in fallbacks:
                lang_kausal = convert_language_code(lang, 'kausal')
                if lang_kausal in s.i18n:
                    key = f'{convert_language_code(default_lang, "modeltrans")}_{convert_language_code(lang, "modeltrans")}'
                    i18n[key] = s.i18n[lang_kausal]
                    break
            else:
                raise Exception("Field '%s' does not have a value in language %s" % (field_name, default_lang))

        field_val = s.i18n[default_lang]
    else:
        field_val = s

    return field_val, i18n


def get_translated_string_from_modeltrans(
    obj: Model,
    field_name: str,
    primary_language: str,
) -> TranslatedString:
    val = getattr(obj, field_name)
    langs = {}
    langs[primary_language] = val
    i18n: dict[str, str] = obj.i18n or {}  # type: ignore
    for key, val in i18n.items():
        parts = key.split('_')
        lang = parts.pop(-1)
        field = '_'.join(parts)
        if field != field_name:
            continue
        langs[lang] = val
    return TranslatedString(default_language=primary_language, **langs)


type I18nStringInstance = TranslatedString | str
type I18nString = I18nStringInstance | StrPromise


def validate_translated_string(cls: type[BaseModel], field_name: str, obj: dict) -> TranslatedString | None:  # noqa: C901, PLR0912
    f = cls.model_fields[field_name]
    field_val = obj.get(field_name)
    langs: dict[str, str] = {}
    default_language = get_default_language()

    if isinstance(field_val, TranslatedString):
        return field_val

    if isinstance(field_val, str):
        assert default_language is not None
        langs[default_language] = field_val
    elif isinstance(field_val, dict):
        return TranslatedString(**field_val)
    else:
        if default_language is None:
            raise Exception('default_language is None')
        assert default_language is not None
        if field_val is not None:
            raise TypeError('%s: Invalid type: %s' % (field_name, type(field_val)))

    base_default = default_language.split('-')[0]

    # FIXME: how to get default language?
    for key, val in list(obj.items()):
        if '_' not in key or not key.startswith(field_name):
            continue
        parts = key.split('_')
        lang = parts.pop(-1)
        fn = '_'.join(parts)
        if fn != field_name:
            continue
        if not isinstance(val, str):
            raise TypeError('%s: Expecting str, got %s' % (key, type(val)))
        obj.pop(key)
        if lang == base_default:
            lang = default_language
        langs[lang] = val

    if not langs:
        if not f.is_required():
            return None
        raise KeyError('%s: Value missing' % field_name)
    ts = TranslatedString(default_language=default_language, **langs)
    return ts


def is_i18n_field(type_: type[Any] | TypeAliasType | types.UnionType | None) -> bool:
    if type_ is TranslatedString:
        return True
    if isinstance(type_, TypeAliasType):
        type_ = type_.__value__
    if isinstance(type_, types.UnionType):
        for arg in typing.get_args(type_):
            if is_i18n_field(arg):
                return True
    return False


class I18nBaseModel(BaseModel, abc.ABC):
    @model_validator(mode='before')
    @classmethod
    def validate_translated_fields(cls, val: dict) -> dict[str, Any]:
        val = val.copy()
        for fn, f in cls.model_fields.items():
            if is_i18n_field(f.annotation):
                val[fn] = validate_translated_string(cls, fn, val)
        return val
