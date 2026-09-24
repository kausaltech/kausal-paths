"""Shared snapshot helpers independent of Django model modules."""

from typing import Any, Self, cast

from django.db.models import Model
from modeltrans.translator import get_i18n_field

from kausal_common.i18n.pydantic import (
    I18nBaseModel,
    ModeltransModelProtocol,
    TranslatedString,
    get_modeltrans_attrs_from_str,
    get_translated_string_from_modeltrans,
)


class ModelSnapshot[ModelT: Model](I18nBaseModel):
    """
    Base for Pydantic types that mirror ORM-row state of editable children.

    Subclasses declare their fields; ``from_model`` maps an ORM instance to
    this snapshot shape (default: attribute access via
    ``model_validate(obj, from_attributes=True)``). Override when a field
    needs dereferencing (e.g. FK → string identifier).

    Inherits ``I18nBaseModel`` so ``TranslatedString``-typed fields are
    handled uniformly; snapshots without i18n fields pay no runtime cost.
    """

    @classmethod
    def from_model(cls, obj: ModelT) -> Self:
        return cls.model_validate(obj, from_attributes=True)


def translated_string_from_model(obj: Model, field_name: str, primary_language: str) -> TranslatedString | None:
    """
    Read a modeltrans-backed field into a ``TranslatedString``.

    Returns ``None`` when the field is empty across all languages.
    """
    val = getattr(obj, field_name, None)
    i18n_field = get_i18n_field(obj)
    assert i18n_field is not None
    assert i18n_field.attname == 'i18n'
    mt_obj = cast('ModeltransModelProtocol', obj)
    i18n = cast('dict[str, str]', mt_obj.i18n or {})
    has_translation = any(k.startswith(f'{field_name}_') and v for k, v in i18n.items())
    if not val and not has_translation:
        return None
    return get_translated_string_from_modeltrans(mt_obj, field_name, primary_language)


def apply_translated(
    fields: dict[str, Any],
    i18n: dict[str, str],
    ts: TranslatedString | None,
    field_name: str,
    default_lang: str,
) -> None:
    """
    Split a TranslatedString into its modeltrans parts.

    The primary-language value goes into ``fields[field_name]`` and the
    non-primary translations into ``i18n`` (modeltrans keys like
    ``{field}_{lang}``). No-op on ``None``.
    """
    if ts is None:
        fields[field_name] = None
        return
    primary_val, translations = get_modeltrans_attrs_from_str(ts, field_name, default_lang, strict=False)
    fields[field_name] = primary_val
    i18n.update(translations)
