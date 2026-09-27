"""Keep cached instance reachability in sync with organization grants."""

from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from frameworks.models import OrganizationAccessGrant
from users.models import User


@receiver(post_save, sender=OrganizationAccessGrant)
@receiver(post_delete, sender=OrganizationAccessGrant)
def invalidate_grantee_instances(
    sender: type[OrganizationAccessGrant], instance: OrganizationAccessGrant, **kwargs: object
) -> None:
    User.objects.filter(pk=instance.user_id).update(cached_adminable_instances=None)
