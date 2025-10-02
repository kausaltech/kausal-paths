from __future__ import annotations

from typing import TYPE_CHECKING

from kausal_common.organizations.views import (
    CreateChildNodeView as BaseCreateChildNodeView,
    OrganizationCreateView as BaseOrganizationCreateView,
    OrganizationIndexView as BaseOrganizationIndexView,
)

from paths.context import realm_context

from admin_site.viewsets import PathsCreateView, PathsIndexView, admin_req
from orgs.models import Organization

if TYPE_CHECKING:
    from django.db.models import Model


class OrganizationCreateView(BaseOrganizationCreateView, PathsCreateView):

    def initialize_instance(self, instance: Model) -> None:
        """
        Initialize the instance with plan defaults.

        Override this in subclasses to implement custom initialization logic.
        """
        if isinstance(instance, Organization):
            instance_config = self.admin_instance
            instance.initialize_instance_defaults(instance_config)

    def get_initial_form_instance(self):
        instance = super().get_initial_form_instance()
        if instance is None:
            instance = self.model()

        self.initialize_instance(instance)
        return instance

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['admin_instance'] = self.admin_instance
        return kwargs


class CreateChildNodeView(BaseCreateChildNodeView):
    """Override to initialize instance defaults for child organizations."""

    def get_initial_form_instance(self):
        """Get the initial form instance with initialized defaults."""
        instance = super().get_initial_form_instance()
        if instance is None:
            instance = self.model()

        # Initialize the instance with defaults
        if hasattr(instance, 'initialize_instance_defaults'):
            from paths.context import realm_context
            instance_config = realm_context.get().realm
            instance.initialize_instance_defaults(instance_config)

        return instance

class OrganizationIndexView(BaseOrganizationIndexView, PathsIndexView):
    def get_list_more_buttons(self, instance: Organization):
        assert self.view_set is not None
        user = admin_req(self.request).user
        active_instance = realm_context.get().realm

        return self.view_set.get_index_view_buttons(user, instance, active_instance)
