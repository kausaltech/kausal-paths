"""
Write a dataset schema's translated name the way `modeltrans` reads it back.

The value for `settings.LANGUAGE_CODE` goes in the plain `name` column and the other languages
under `name_<lang>` in `i18n`. Storing the instance's primary language in the column instead
is the mistake this exists to prevent: on a German instance the German text then sits in a
column that is read as English (see `docs/trailhead/tools.md`, *rename_dataset*).
"""

from typing import TYPE_CHECKING

from django.conf import settings

from kausal_common.i18n.pydantic import TranslatedString, get_modeltrans_attrs_from_str

if TYPE_CHECKING:
    from kausal_common.datasets.models import DatasetSchema


def default_language() -> str:
    """
    Return the language whose value lives in the model's own column rather than in ``i18n``.

    ``modeltrans`` reads the plain field for the active language when it *is* the default,
    so this is the one label that cannot be left behind.
    """
    return settings.LANGUAGE_CODE.split('-')[0].lower()


def set_schema_name(schema: DatasetSchema, labels: dict[str, str]) -> bool:
    """
    Store `labels` (language -> text) as the schema's name; return whether anything changed.

    The labels must include `default_language()`, or the column would keep a stale value while
    the translations move on. Other translated fields in `i18n` are kept.
    """
    lang = default_language()
    translated = TranslatedString(**labels, default_language=lang)
    name, i18n = get_modeltrans_attrs_from_str(translated, 'name', lang)
    current = schema.i18n or {}
    merged = {**current, **i18n}
    if schema.name == name and merged == current:
        return False
    schema.name = name
    schema.i18n = merged
    schema.save(update_fields=['name', 'i18n'])
    return True
