from __future__ import annotations

import typing

from django import forms
from django.utils.text import format_lazy
from django.utils.translation import gettext_lazy as _
from django_stubs_ext import StrOrPromise

from kausal_common.models.roles import role_registry

from paths.const import NONE_ROLE
from paths.context import realm_context

from admin_site.forms import PathsAdminModelForm
from nodes.models import InstanceConfig

from .models import Person
from .widgets import RoleSelectionWidget

if typing.TYPE_CHECKING:
    from django.utils.functional import _StrPromise
    from django_stubs_ext import StrOrPromise


class PersonForm(PathsAdminModelForm):
    """Custom form for Person instances with role selection."""

    role = forms.CharField(
        label=_('Permissions'),
        help_text=_('Select the role for this person in the current instance'),
        required=True,
    )
    class Meta:
        model = Person
        fields = ['first_name', 'last_name', 'email', 'title', 'image', 'organization', 'role']

    def save(self, commit=True) -> Person:
        """Save the person and process role assignment."""
        role = self.cleaned_data.pop('role')
        instance = super().save(commit=commit)
        if commit and role is not None:
            self.sync_user_groups_with_role(instance, role, self.active_instance)

        return instance

    def sync_user_groups_with_role(self, person: Person, role_id: str, active_instance: InstanceConfig) -> None:
        """Process the role assignment by managing group memberships."""
        if not active_instance or not person.user:
            return
        person.user.sync_instance_groups_with_role(role_id, active_instance)

    def get_current_role(self, person, active_instance) -> str:
        """Determine the person's current role in the active instance."""
        if not person.user or not active_instance:
            return NONE_ROLE
        role = person.user.get_role_for_instance(active_instance)
        if not role:
            return NONE_ROLE
        return role.id

    def __init__(self, *args, **kwargs):
        self.active_instance = realm_context.get().realm

        super().__init__(*args, **kwargs)
        is_superuser = False
        if self.instance and self.instance.pk:
            current_role = self.get_current_role(self.instance, self.active_instance)
            self.initial['role'] = current_role
            if self.instance.user.is_superuser:
                is_superuser = True

        assert self.active_instance
        instance_name = self.active_instance.name
        help_text: StrOrPromise = format_lazy(_(
            'Select a role for this person within {instance_name}'
        ), instance_name=instance_name)
        field_label: StrOrPromise = self.fields['role'].label or ''
        roles = role_registry.get_all_roles()
        choices: list[tuple[str, _StrPromise]] = [
            (r.id, r.name) for r in roles if r.model == InstanceConfig
        ] + [(NONE_ROLE, _('None'))]
        if is_superuser:
            disable_role_options = True
            disable_reason = _(
                'The person has global superuser rights. '
                'No individual permissions within an instance are required '
                'since the person can access all data within all instances.'
            )
        else:
            disable_role_options = False
            disable_reason = None
        self.fields['role'].widget = RoleSelectionWidget(
            help_text=help_text,
            label=field_label,
            choices=choices,
            disable_role_options=disable_role_options,
            disable_reason=disable_reason,
        )
