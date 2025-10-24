from __future__ import annotations

from typing import TYPE_CHECKING

from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from people.models import (
    DatasetSchemaGroupPermission,
    DatasetSchemaPersonPermission,
    PersonGroup,
    PersonGroupMember,
)
from users.signals import user_permissions_changed

if TYPE_CHECKING:
    from kausal_common.datasets.models import DatasetSchema
    from kausal_common.people.models import ObjectGroupPermissionBase, ObjectPersonPermissionBase


@receiver(post_save, sender=PersonGroupMember)
def handle_person_group_member_saved(sender, instance: PersonGroupMember, created: bool, **kwargs):
    """Handle person added to a PersonGroup."""
    if instance.person.user_id:
        user_permissions_changed.send(sender=sender, user=instance.person.user)


@receiver(post_delete, sender=PersonGroupMember)
def handle_person_group_member_deleted(sender, instance: PersonGroupMember, **kwargs):
    """Handle person removed from a PersonGroup."""
    if instance.person.user_id:
        user_permissions_changed.send(sender=sender, user=instance.person.user)


@receiver(post_save, sender=PersonGroup)
def handle_person_group_saved(sender, instance: PersonGroup, created: bool, **kwargs):
    """Handle PersonGroup creation - invalidate cache for all members."""
    if created:
        for person in instance.persons.all():
            if person.user:
                user_permissions_changed.send(sender=sender, user=person.user)


@receiver(post_save, sender=DatasetSchemaGroupPermission)
def handle_dataset_schema_group_permission_saved(sender, instance: ObjectGroupPermissionBase[DatasetSchema], **kwargs):
    """Handle DatasetSchemaGroupPermission added or modified."""
    for person in instance.group.persons.all():
        if person.user_id:
            user_permissions_changed.send(sender=sender, user=person.user)


@receiver(post_delete, sender=DatasetSchemaGroupPermission)
def handle_dataset_schema_group_permission_deleted(sender, instance: ObjectGroupPermissionBase[DatasetSchema], **kwargs):
    """Handle DatasetSchemaGroupPermission removed."""
    for person in instance.group.persons.all():
        if person.user_id:
            user_permissions_changed.send(sender=sender, user=person.user)


@receiver(post_save, sender=DatasetSchemaPersonPermission)
def handle_dataset_schema_person_permission_saved(sender, instance: ObjectPersonPermissionBase[DatasetSchema], **kwargs):
    """Handle DatasetSchemaPersonPermission added or modified."""
    if instance.person.user_id:
        user_permissions_changed.send(sender=sender, user=instance.person.user)


@receiver(post_delete, sender=DatasetSchemaPersonPermission)
def handle_dataset_schema_person_permission_deleted(sender, instance: ObjectPersonPermissionBase[DatasetSchema], **kwargs):
    """Handle DatasetSchemaPersonPermission removed."""
    if instance.person.user_id:
        user_permissions_changed.send(sender=sender, user=instance.person.user)


# TODO add as a kwargs parameter, whether to add the subsector admin group to the user
