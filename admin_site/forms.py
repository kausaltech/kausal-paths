from django.db.models import Model
from modeltrans.conf import get_available_languages
from modeltrans.translator import get_i18n_field
from modeltrans.utils import build_localized_fieldname
from wagtail.admin.forms import WagtailAdminModelForm

from nodes.models import InstanceConfig


class PathsAdminModelForm(WagtailAdminModelForm):
    admin_instance: InstanceConfig | None = None

    def prune_i18n_fields(self):
        model: type[Model] = self._meta.model
        i18n_field = get_i18n_field(model)
        if not i18n_field:
            return
        other_langs = self.admin_instance.other_languages if self.admin_instance is not None else []
        for base_field_name in i18n_field.fields:
            langs = list(get_available_languages(include_default=True))
            for lang in langs:
                fn = build_localized_fieldname(base_field_name, lang)
                if fn in self.fields and lang not in other_langs:
                    del self.fields[fn]

    def save(self, commit=True):
        obj = super().save(commit)
        return obj

    def __init__(self, *args, **kwargs):
        self.admin_instance = kwargs.pop("admin_instance", None)
        super().__init__(*args, **kwargs)
        self.prune_i18n_fields()
