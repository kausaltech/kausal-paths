from __future__ import annotations

from admin_site.viewsets import PathsCreateView
from kausal_common.organizations.views import OrganizationCreateView as BaseOrganizationCreateView
from orgs.models import Organization
from django.db.models import Model
from django.contrib.auth.models import User



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
