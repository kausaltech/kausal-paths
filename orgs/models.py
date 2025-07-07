from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from django.db import models
from django.utils.translation import gettext_lazy as _
from modelcluster.fields import ParentalKey
from modelcluster.models import ClusterableModel

from treebeard.mp_tree import MP_Node

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
    node_order_by = ['name']

    objects: ClassVar[OrganizationManager] = OrganizationManager()  # type: ignore[assignment]

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
