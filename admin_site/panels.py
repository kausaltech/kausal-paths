from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack, cast

from django.conf import settings
from django.db.models import Model
from modeltrans.fields import TranslatedVirtualField
from modeltrans.translator import get_i18n_field
from modeltrans.utils import build_localized_fieldname
from wagtail.admin.panels import FieldPanel, FieldRowPanel, MultiFieldPanel, Panel

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from wagtail.admin.panels.field_panel import WidgetOverrideType
    from wagtail.admin.panels.group import PanelGroupInitArgs

    from nodes.models import InstanceConfig


def insert_model_translation_panels(model: Model, panels: Sequence[Panel], instance: InstanceConfig) -> list[Panel]:
    """Return a list of panels containing all of `panels` and language-specific panels for fields with i18n."""
    i18n_field = get_i18n_field(model)
    if not i18n_field:
        return list(panels)

    out = []

    field_map: dict[str, dict[str, TranslatedVirtualField]] = {}
    for f in cast('Iterator[TranslatedVirtualField]', i18n_field.get_translated_fields()):
        lang = cast(str, f.language)
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


class TranslatedLanguagePanel(FieldPanel):
    main_field_name: str
    language: str

    def __init__(self, field_name: str, language: str, **kwargs):
        self.main_field_name = field_name
        self.language = language
        field_name = build_localized_fieldname(field_name, language.lower(), default_language='')
        super().__init__(field_name, **kwargs)

    def clone_kwargs(self):
        ret = super().clone_kwargs()
        ret['field_name'] = self.main_field_name
        ret['language'] = self.language
        return ret

    class BoundPanel(FieldPanel.BoundPanel):
        panel: TranslatedLanguagePanel

        def is_shown(self):
            from paths.context import realm_context
            instance = realm_context.get().realm
            ret = super().is_shown()
            if not ret:
                return False
            is_other_lang = self.panel.language in (instance.other_languages or [])
            return is_other_lang


class TranslatedFieldRowPanel[M: Model, P: Any](FieldRowPanel[M, P]):
    def __init__(self, field_name: str, widget: WidgetOverrideType | None = None, **kwargs: Unpack[PanelGroupInitArgs]):
        self.field_name = field_name
        self.widget = widget
        primary_panel = FieldPanel(field_name, widget=widget, **kwargs)
        lang_panels = [TranslatedLanguagePanel(
            field_name=field_name,
            language=lang[0],
            widget=widget,
            **kwargs
        ) for lang in settings.LANGUAGES]
        super().__init__(children=[primary_panel, *lang_panels], **kwargs)

    def clone_kwargs(self):
        ret = super().clone_kwargs()
        del ret['children']
        ret['field_name'] = self.field_name
        ret['widget'] = self.widget
        return ret


class TranslatedFieldPanel(MultiFieldPanel):
    def __init__(self, field_name: str, widget: Any = None, **kwargs):
        self.field_name = field_name
        self.widget = widget
        primary_panel = FieldPanel(field_name, widget=widget, **kwargs)
        lang_panels = [TranslatedLanguagePanel(
            field_name=field_name,
            language=lang[0],
            widget=widget,
            **kwargs
        ) for lang in settings.LANGUAGES]
        super().__init__(children=[primary_panel, *lang_panels], **kwargs)

    def clone_kwargs(self):
        ret = super().clone_kwargs()
        del ret['children']
        ret['field_name'] = self.field_name
        ret['widget'] = self.widget
        return ret
