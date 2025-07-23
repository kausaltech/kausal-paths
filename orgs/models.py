from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

from django.db import models
from django.utils.translation import gettext_lazy as _

from paths.context import realm_context

from kausal_common.models.types import MLModelManager
from kausal_common.organizations.models import (
    BaseNamespace,
    BaseOrganization,
    BaseOrganizationClass,
    BaseOrganizationIdentifier,
    BaseOrganizationMetadataAdmin,
    BaseOrganizationQuerySet,
    Node,
)

from users.models import User

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


class OrganizationClass(BaseOrganizationClass):
    class Meta:
        verbose_name = _('Organization class')
        verbose_name_plural = _('Organization classes')

class OrganizationMetadataAdmin(BaseOrganizationMetadataAdmin):
    class Meta:
        verbose_name = _('Organization metadata admin')
        verbose_name_plural = _('Organization metadata admins')

class Namespace(BaseNamespace):
    class Meta:
        verbose_name = _('Namespace')
        verbose_name_plural = _('Namespaces')

class OrganizationIdentifier(BaseOrganizationIdentifier):
    class Meta:
        verbose_name = _('Organization identifier')
        verbose_name_plural = _('Organization identifiers')

class OrganizationQuerySet(BaseOrganizationQuerySet):
    def editable_by_user(self, user):
        if not user.is_authenticated:
            return self.none()

        # Superusers can edit all organizations
        if user.is_superuser:
            return self.all()

        # Users can edit organizations they are metadata admins for
        return self.filter(metadata_admins__user=user)

    def available_for_instance(self, instance: InstanceConfig):
        if not hasattr(instance, 'organization') or not instance.organization:
            return self.none()

        return self.filter(
            path__startswith=instance.organization.path
        )

_OrganizationManager = models.Manager.from_queryset(OrganizationQuerySet)


class OrganizationManager(MLModelManager['Organization', OrganizationQuerySet], _OrganizationManager): ...


del _OrganizationManager

class Organization(BaseOrganization, Node[OrganizationQuerySet]):
    objects: ClassVar[OrganizationManager] = OrganizationManager()  # type: ignore[assignment]
    VIEWSET_CLASS = 'orgs.wagtail_hooks.OrganizationViewSet'
    class Meta:
        verbose_name = _('Organization')
        verbose_name_plural = _('Organizations')
        ordering = ['name']

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('organization-detail', kwargs={'pk': self.pk})

    def initialize_instance_defaults(self, instance: InstanceConfig):
        assert not self.primary_language
        self.primary_language = instance.primary_language
        self.primary_language_lowercase = instance.primary_language.lower()

    @classmethod
    def get_parent_choices(cls, user: User, obj: Self | None = None) -> OrganizationQuerySet:
        instance = realm_context.get().realm
        parent_choices = Organization.objects.qs.available_for_instance(instance).editable_by_user(user)

        # If the parent is not editable, the form would display an empty parent,
        # leading to the org becoming a root when saved. Prevent this by adding
        # the parent to the queryset.
        if obj and (parent := obj.get_parent()):
            parent_choices |= Organization.objects.filter(pk=parent.pk)

        return parent_choices

