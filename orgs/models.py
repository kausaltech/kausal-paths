from django.utils.translation import gettext_lazy as _
from kausal_common.organizations.models import (
    BaseOrganization,
    BaseOrganizationClass,
    BaseOrganizationMetadataAdmin,
    BaseNamespace,
    BaseOrganizationIdentifier,
    BaseOrganizationQuerySet
)
from treebeard.mp_tree import MP_Node
from modelcluster.models import ClusterableModel
from modelcluster.fields import ParentalKey
from django.db import models
from users.models import User

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

class Organization(BaseOrganization, MP_Node, ClusterableModel):
    node_order_by = ['name']

    class Meta:
        verbose_name = _('Organization')
        verbose_name_plural = _('Organizations')
        ordering = ['name']

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('organization-detail', kwargs={'pk': self.pk})