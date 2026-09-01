from __future__ import annotations

from django import forms
from django.utils.translation import gettext_lazy as _
from wagtail.users.forms import UserCreationForm as WagtailUserCreationForm, UserEditForm as WagtailUserEditForm


class CustomUserFieldsMixin:
    is_staff = forms.BooleanField(
        label=_('staff status'),
        required=False,
        help_text=_('Designates whether the user can log into this admin site.'),
    )


class UserEditForm(CustomUserFieldsMixin, WagtailUserEditForm):
    class Meta(WagtailUserEditForm.Meta):
        fields = WagtailUserEditForm.Meta.fields | {'is_staff'}


class UserCreationForm(CustomUserFieldsMixin, WagtailUserCreationForm):
    class Meta(WagtailUserCreationForm.Meta):
        fields = WagtailUserCreationForm.Meta.fields | {'is_staff'}
