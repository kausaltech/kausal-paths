from __future__ import annotations

import typing

from django import forms
from django.contrib.admin.widgets import AdminFileWidget
from django.utils.text import format_lazy
from django.utils.translation import gettext_lazy as _
from django_stubs_ext import StrOrPromise

from dal import autocomplete

from kausal_common.models.roles import InstanceSpecificRole, role_registry

from paths.const import INSTANCE_SUPER_ADMIN_ROLE, NONE_ROLE
from paths.context import realm_context

from admin_site.forms import PathsAdminModelForm
from nodes.models import InstanceConfig
from orgs.models import Organization

from .models import Person
from .widgets import RoleSelectionWidget

if typing.TYPE_CHECKING:
    from django.utils.functional import _StrPromise
    from django_stubs_ext import StrOrPromise


class AvatarWidget(AdminFileWidget):
    template_name = 'kausal_common/people/avatar_widget.html'


class PersonForm(PathsAdminModelForm):
    """Custom form for Person instances with role selection."""

    active_instance: InstanceConfig

    role = forms.CharField(
        label=_('Permissions'),
        help_text=_('Select the role for this person in the current instance'),
        required=True,
    )

    class Meta:
        model = Person
        fields = ['first_name', 'last_name', 'email', 'title', 'image', 'organization', 'role']

    def clean_role(self):
        role_id = self.cleaned_data['role']
        if self.instance is None or self.instance.user is None:
            return role_id
        if role_id == INSTANCE_SUPER_ADMIN_ROLE:
            return role_id

        # We are changing to some other role than instance super admin.
        # Verify we are not removing the last instance super admin.
        super_admin_role = role_registry.get_role(INSTANCE_SUPER_ADMIN_ROLE)
        super_admin_group = super_admin_role.get_existing_instance_group(self.active_instance)
        if (
            super_admin_group in self.instance.user.groups.all() and
            super_admin_group.user_set.count() == 1
        ):
            raise forms.ValidationError(
                _('Removing the last Super Admin of an instance is not permitted')
            )
        return role_id

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

    def get_role_choices(self) -> list[tuple[str, _StrPromise]]:
        def group_exists(role: InstanceSpecificRole[typing.Any]) -> bool:
            return (
                role.model == InstanceConfig and
                role.get_existing_instance_group(self.active_instance) is not None
            )

        roles = role_registry.get_all_roles()
        choices: list[tuple[str, _StrPromise]] = [
            (r.id, r.name) for r in roles if group_exists(r)
        ]
        return choices + [(NONE_ROLE, _('None'))]

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

        if is_superuser:
            disable_role_options = True
            disable_reason = _(
                'The person is a superuser in the system. '
                'No individual permissions within an instance are required '
                'since the person can access all data within all instances.'
            )
        else:
            disable_role_options = False
            disable_reason = None

        self.fields['role'].widget = RoleSelectionWidget(
            help_text=help_text,
            label=field_label,
            errors=self.errors.get('role'),
            choices=self.get_role_choices(),
            disable_role_options=disable_role_options,
            disable_reason=disable_reason,
        )

        self.fields['organization'].widget = autocomplete.ModelSelect2(
            url='organization-autocomplete',
            choices=self.fields['organization'].choices
        )
        self.fields['image'].widget = AvatarWidget()
