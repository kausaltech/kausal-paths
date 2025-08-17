from __future__ import annotations

from typing import TYPE_CHECKING, override

from modeltrans.conf import get_available_languages
from modeltrans.translator import get_i18n_field
from modeltrans.utils import build_localized_fieldname
from wagtail.admin.forms import WagtailAdminModelForm

from kausal_common.i18n.helpers import convert_language_code
from kausal_common.i18n.forms import LanguageAwareAdminModelForm

from users.models import User

if TYPE_CHECKING:
    from django.db.models import Model

    from nodes.models import InstanceConfig


class PathsAdminModelForm[M: Model](LanguageAwareAdminModelForm[M]):
    admin_instance: InstanceConfig | None = None

    @override
    def get_primary_realm_language(self) -> str:
         if self.admin_instance is None:
             raise ValueError('Cannot get instance languages without instance.')
         return convert_language_code(self.admin_instance.primary_language, 'django')

    @override
    def get_all_realm_languages(self) -> set[str]:
         if self.admin_instance is None:
             raise ValueError('Cannot get instance languages without instance.')
         result = set(self.admin_instance.other_languages).union({ self.admin_instance.primary_language })
         return {convert_language_code(lang, 'django') for lang in result}

    def __init__(self, *args, **kwargs):
        self.admin_instance = kwargs.pop("admin_instance", None)
        if self.admin_instance:
            self.realm_initialized = True
        super().__init__(*args, **kwargs)
