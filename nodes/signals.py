from __future__ import annotations

from typing import TYPE_CHECKING

from django.db.models import Model, Q
from django.db.models.signals import m2m_changed, post_save, pre_save
from django.dispatch import receiver

from kausal_common.models.roles import InstanceFieldGroupRole, role_registry

from nodes.models import InstanceConfig
from pages.models import InstanceSiteContent
from users.models import User
from users.signals import user_permissions_changed

if TYPE_CHECKING:
    class InstanceConfigWithTempVar(InstanceConfig):
        _old_role_groups: dict[str, int | None] | None


def get_instance_config_role_group_fields() -> list[str]:
    """
    Get all InstanceConfig role group field names from the role registry.

    Returns a list of field names with _id suffix (e.g., ['viewer_group_id', 'admin_group_id', ...])
    for all registered roles that are InstanceFieldGroupRole instances
    targeting InstanceConfig.
    """
    return [
        f'{role.instance_group_field_name}_id'
        for role in role_registry.get_all_roles()
        if isinstance(role, InstanceFieldGroupRole) and role.model == InstanceConfig
    ]


@receiver(post_save, sender=InstanceConfig)
def create_instance_site_content(sender, instance: InstanceConfig, created: bool, **kwargs):
    if created:
        InstanceSiteContent.objects.create(instance=instance)


@receiver(pre_save, sender=InstanceConfig)
def store_instance_config_old_groups(sender, instance: InstanceConfigWithTempVar, **kwargs):
    """Store old role group ID values before they're changed."""
    if not instance.pk:
        instance._old_role_groups = None
        return
    field_names = get_instance_config_role_group_fields()
    instance._old_role_groups = InstanceConfig.objects.filter(pk=instance.pk).values(*field_names).first()


@receiver(post_save, sender=InstanceConfig)
def handle_instance_config_role_groups_changed(sender, instance: InstanceConfigWithTempVar, created: bool, **kwargs):
    """Handle changes to InstanceConfig role groups."""
    if created:
        return

    old_groups = getattr(instance, '_old_role_groups', None)
    if not old_groups:
        return

    affected_group_ids = set()

    for field_name_id in get_instance_config_role_group_fields():
        old_group_id = old_groups.get(field_name_id)
        new_group_id = getattr(instance, field_name_id)

        if old_group_id != new_group_id:
            if old_group_id:
                affected_group_ids.add(old_group_id)
            if new_group_id:
                affected_group_ids.add(new_group_id)

    if affected_group_ids:
        for user in User.objects.filter(groups__id__in=affected_group_ids).distinct():
            user_permissions_changed.send(sender=sender, user=user)


@receiver(m2m_changed, sender=User.groups.through)
def handle_user_groups_changed(sender: Model, instance: User, action: str, pk_set: set[int], **kwargs):
    """Handle changes to User.groups to detect role group membership changes."""
    if action not in ('post_add', 'post_remove', 'post_clear'):
        return

    if action == 'post_clear':
        user_permissions_changed.send(sender=sender, user=instance)
        return

    if not pk_set:
        return

    # Build Q objects dynamically for all InstanceConfig role group fields
    q_filters = Q()
    for field_name_id in get_instance_config_role_group_fields():
        q_filters |= Q(**{f'{field_name_id}__in': pk_set})

    if q_filters and InstanceConfig.objects.filter(q_filters).exists():
        user_permissions_changed.send(sender=sender, user=instance)
