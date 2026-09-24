from __future__ import annotations

from typing import TYPE_CHECKING

from django.db.models.signals import m2m_changed, post_delete, post_save
from django.dispatch import receiver

from people.models import (
    DatasetGroupPermission,
    DatasetPersonPermission,
    PersonGroup,
    PersonGroupMember,
)
from users.models import User
from users.signals import user_permissions_changed

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset
    from kausal_common.people.models import ObjectGroupPermissionBase, ObjectPersonPermissionBase


@receiver(m2m_changed, sender=PersonGroupMember)
def handle_person_group_member_saved(sender, instance: PersonGroup, action: str, **kwargs):
    """Handle person added to a PersonGroup."""
    if action not in ('post_add', 'post_remove', 'post_clear'):
        return
    for user in User.objects.filter(person__in=instance.persons.all()):
        user_permissions_changed.send(sender=sender, user=user, add_permission_group=True)


@receiver(post_save, sender=PersonGroup)
def handle_person_group_saved(sender, instance: PersonGroup, created: bool, **kwargs):
    """Handle PersonGroup creation - invalidate cache for all members."""
    if created:
        for person in instance.persons.all():
            if person.user:
                user_permissions_changed.send(sender=sender, user=person.user)


@receiver(post_save, sender=DatasetGroupPermission)
def handle_dataset_group_permission_saved(sender, instance: ObjectGroupPermissionBase[Dataset], **kwargs):
    """Handle DatasetGroupPermission added or modified."""
    for person in instance.group.persons.all():
        if person.user_id:
            user_permissions_changed.send(sender=sender, user=person.user)


@receiver(post_delete, sender=DatasetGroupPermission)
def handle_dataset_group_permission_deleted(sender, instance: ObjectGroupPermissionBase[Dataset], **kwargs):
    """Handle DatasetGroupPermission removed."""
    for person in instance.group.persons.all():
        if person.user_id:
            user_permissions_changed.send(sender=sender, user=person.user)


@receiver(post_save, sender=DatasetPersonPermission)
def handle_dataset_person_permission_saved(sender, instance: ObjectPersonPermissionBase[Dataset], **kwargs):
    """Handle DatasetPersonPermission added or modified."""
    if instance.person.user_id:
        user_permissions_changed.send(sender=sender, user=instance.person.user)


@receiver(post_delete, sender=DatasetPersonPermission)
def handle_dataset_person_permission_deleted(sender, instance: ObjectPersonPermissionBase[Dataset], **kwargs):
    """Handle DatasetPersonPermission removed."""
    if instance.person.user_id:
        user_permissions_changed.send(sender=sender, user=instance.person.user)
