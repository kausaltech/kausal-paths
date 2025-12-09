from __future__ import annotations

from typing import TYPE_CHECKING, cast

from modeltrans.translator import get_i18n_field
from wagtail.admin.panels import FieldPanel, Panel

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from django.db.models import Model
    from modeltrans.fields import TranslatedVirtualField

    from nodes.models import InstanceConfig


def insert_model_translation_panels(model: Model, panels: Sequence[Panel], instance: InstanceConfig) -> list[Panel]:
    """Return a list of panels containing all of `panels` and language-specific panels for fields with i18n."""
    i18n_field = get_i18n_field(model)
    if not i18n_field:
        return list(panels)

    out = []

    field_map: dict[str, dict[str, TranslatedVirtualField]] = {}
    for f in cast('Iterator[TranslatedVirtualField]', i18n_field.get_translated_fields()):
        lang = cast('str', f.language)
        field_map.setdefault(f.original_name, {})[lang] = f # type: ignore

    for p in panels:
        out.append(p)
        if not isinstance(p, FieldPanel):
            continue
        t_fields = field_map.get(p.field_name)
        if not t_fields:
            continue

        # TODO: Use instance-specific default language, but our own modeltrans fork requires the default language to be
        # in a database field, whereas it's just in YAML so far.
        for lang_code in (lang for lang in (instance.other_languages or [])):
            tf = t_fields.get(lang_code)
            if not tf:
                continue
            out.append(type(p)(tf.name)) # type: ignore
    return out
