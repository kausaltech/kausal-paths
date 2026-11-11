"""Keep what is derived from grants and from instance drafts in step with them."""

from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from frameworks.indicators import materialise_open_submissions
from frameworks.models import OrganizationAccessGrant, Submission
from frameworks.models.submission import OPEN_STATUSES
from nodes.change_ops import instance_draft_changed
from nodes.models import InstanceConfig
from users.models import User


@receiver(post_save, sender=OrganizationAccessGrant)
@receiver(post_delete, sender=OrganizationAccessGrant)
def invalidate_grantee_instances(
    sender: type[OrganizationAccessGrant], instance: OrganizationAccessGrant, **kwargs: object
) -> None:
    User.objects.filter(pk=instance.user_id).update(cached_adminable_instances=None)


@receiver(instance_draft_changed)
def rewrite_open_submission_indicators(sender: type[InstanceConfig], instance_config: InstanceConfig, **kwargs: object) -> None:
    """Every edit of a draft moves the figures of the years being entered; see `frameworks.indicators`."""
    if not Submission.objects.filter(instance_config=instance_config, status__in=OPEN_STATUSES).exists():
        return
    # The caller's object may predate what the operation wrote.
    materialise_open_submissions(InstanceConfig.objects.get(pk=instance_config.pk))
