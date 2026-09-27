"""Administrative coverage and delegated access for a framework."""

from django.conf import settings
from django.db import models
from django.db.models import Q

from kausal_common.people.models import ObjectRole


class FrameworkOrganizationRoot(models.Model):
    framework = models.ForeignKey('frameworks.Framework', on_delete=models.CASCADE, related_name='organization_roots')
    organization = models.ForeignKey('orgs.Organization', on_delete=models.PROTECT, related_name='framework_roots')

    class Meta:
        constraints = [models.UniqueConstraint(fields=['framework', 'organization'], name='unique_framework_organization_root')]

    def __str__(self) -> str:
        return f'{self.framework_id}: {self.organization_id}'


class OrganizationAccessGrant(models.Model):
    """Grant access to an organization and its descendants within one framework."""

    framework = models.ForeignKey('frameworks.Framework', on_delete=models.CASCADE, related_name='organization_grants')
    organization = models.ForeignKey('orgs.Organization', on_delete=models.PROTECT, related_name='access_grants')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='organization_grants')
    role = models.CharField(max_length=10, choices=ObjectRole.choices)
    suspended_at = models.DateTimeField(null=True, blank=True)
    retention_until = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='created_organization_grants'
    )
    last_modified_at = models.DateTimeField(auto_now=True)
    last_modified_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='modified_organization_grants'
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['framework', 'organization', 'user'], name='unique_organization_access_grant'),
            models.CheckConstraint(
                condition=Q(suspended_at__isnull=True, retention_until__isnull=True)
                | Q(suspended_at__isnull=False, retention_until__isnull=False),
                name='organization_grant_suspension_has_retention',
            ),
        ]

    def __str__(self) -> str:
        return f'{self.framework_id}: {self.user_id} on {self.organization_id} ({self.role})'


class OrganizationAccessGrantEvent(models.Model):
    grant = models.ForeignKey(OrganizationAccessGrant, on_delete=models.CASCADE, related_name='events')
    action = models.CharField(max_length=20)
    old_role = models.CharField(max_length=10, choices=ObjectRole.choices, null=True, blank=True)
    new_role = models.CharField(max_length=10, choices=ObjectRole.choices, null=True, blank=True)
    suspended_at = models.DateTimeField(null=True, blank=True)
    retention_until = models.DateField(null=True, blank=True)
    changed_at = models.DateTimeField(auto_now_add=True)
    changed_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self) -> str:
        return f'{self.grant_id}: {self.action}'
