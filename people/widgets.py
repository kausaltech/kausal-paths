from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django import forms
from django.utils.translation import gettext_lazy as _

from kausal_common.models.roles import role_registry

from paths.const import NONE_ROLE

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django.forms.utils import ErrorList

    from django_stubs_ext import StrOrPromise


class RoleSelectionWidget(forms.RadioSelect):
    """Custom widget for selecting instance roles using radio buttons."""

    template_name = 'people/widgets/role_selection.html'
    option_template_name = 'people/widgets/role_selection_option.html'

    def __init__(
        self,
        choices: Iterable[tuple[Any, Any]] | None = None,
        help_text: StrOrPromise | None = None,
        disable_role_options: bool = False,
        disable_reason: StrOrPromise | None = None,
        label: StrOrPromise | None = None,
        errors: ErrorList | None = None,
        attrs=None
    ):
        assert choices is not None
        self.disable_role_options = disable_role_options
        self.disable_reason = disable_reason
        self.help_text = help_text
        self.label = label
        self.errors = errors
        super().__init__({'help_text': help_text}, choices)

    class Media:
        css = {
            'all': ('people/css/role_selection.css',)
        }

    def get_context(self, name, value, attrs):
        help_text = self.help_text
        if self.disable_reason:
            help_text = self.disable_reason

        widget_opts = dict(
            help_text=help_text,
            label=self.label,
            errors=self.errors,
            show_options=not self.disable_role_options,
            emphasize_help_text=self.disable_reason is not None
        )

        context = super().get_context(name, value, attrs)
        context['widget'].update(widget_opts)
        return context

    def create_option(
        self, name, value, label, selected, index, subindex=None, attrs=None
    ):
        if self.disable_role_options:
            selected = (value == NONE_ROLE)
        option = super().create_option(name, value, label, selected, index, subindex, attrs)
        try:
            role = role_registry.get_role(option['value'])
        except KeyError:
            if option['value'] != NONE_ROLE:
                raise
            option['help_text'] = _("The person has no access")
        else:
            option['help_text'] = role.description
        return option
