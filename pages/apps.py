from __future__ import annotations

from django.apps import AppConfig
from django.conf import settings
from django.db.models.fields import BLANK_CHOICE_DASH
from django.utils.translation import get_language_info
from wagtail.admin.localization import get_available_admin_languages


def _get_language_choices():
    language_choices = []
    for lang_code, lang_name in get_available_admin_languages():
        lang_info = get_language_info(lang_code.lower())
        if lang_info['code'] == lang_code.lower():
            lang_name_local = lang_info['name_local']
        else:
            # get_language_info probably fell back to a variant of lang_code, so we don't know what the variant is
            # called in its own language. We expect that the local names of such languages are specified manually in
            # the Django settings.
            lang_name_local = settings.LOCAL_LANGUAGE_NAMES[lang_code]
        language_choices.append((lang_code, lang_name_local))
    return sorted(BLANK_CHOICE_DASH + language_choices, key=lambda l: l[1].lower())


class PagesConfig(AppConfig):
    name = 'pages'

    def ready(self):
        global _wagtail_preferred_language_choices_func

        # Monkey-patch Wagtail's _get_language_choices to transform language codes to lower case. See the comment above
        # LANGUAGES in settings.py for details about this.
        from wagtail.admin.forms import account

        account.LocalePreferencesForm.base_fields['preferred_language']._choices.choices_func = _get_language_choices
