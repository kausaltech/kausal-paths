from __future__ import annotations

from typing import TYPE_CHECKING

from django.dispatch import Signal, receiver

if TYPE_CHECKING:
    from users.models import User


user_permissions_changed = Signal()

@receiver(user_permissions_changed)
def invalidate_user_cache(sender, user: User, **kwargs):
    """Invalidate the cached adminable instances when user permissions change."""
    if not user.pk:
        return

    from nodes.roles import SubsectorAdminRole
    from people.models import DatasetSchemaGroupPermission, DatasetSchemaPersonPermission
    if (
        kwargs.get('add_permission_group')
        or DatasetSchemaPersonPermission.objects.filter(person=user.get_corresponding_person()).exists()
        or DatasetSchemaGroupPermission.objects.filter(group__persons=user.get_corresponding_person()).exists()
    ):
        role = SubsectorAdminRole()
        role.refresh()
        group = role.get_group()
        user.groups.add(group)
    else:
        user.groups.remove(SubsectorAdminRole().get_group())

    user.invalidate_adminable_instances_cache()
