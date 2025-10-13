from __future__ import annotations

from typing import TYPE_CHECKING, override

from django.db.models import Q

from paths.dataset_permission_policy import InstanceConfigScopedPermissionPolicy

from users.models import User, UserGroup

if TYPE_CHECKING:
    from kausal_common.models.permission_policy import BaseObjectAction


class UserGroupPermissionPolicy(InstanceConfigScopedPermissionPolicy[UserGroup]):
    """Permission policy for UserGroup, based on its scope (InstanceConfig)."""

    def __init__(self):
        super().__init__(UserGroup)  # type: ignore[type-abstract]

    @override
    def get_instance_configs_for_obj(self, obj: UserGroup) -> list[int]:
        """Get IDs of all InstanceConfigs the given UserGroup is scoped for."""
        return [obj.instance.id]

    @override
    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        q = Q(instance_id__in=self.instance_admin_role.get_instances_for_user(user))
        if action == 'view':
            q |= Q(instance_id__in=self.instance_viewer_role.get_instances_for_user(user))
        return q
