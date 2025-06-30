from __future__ import annotations

from typing import TYPE_CHECKING, override

from django.db.models import Q

from paths.dataset_permission_policy import InstanceConfigScopedPermissionPolicy

from nodes.models import InstanceRoleGroup

if TYPE_CHECKING:
    from kausal_common.models.permission_policy import BaseObjectAction

    from users.models import User


class InstanceRoleGroupPermissionPolicy(InstanceConfigScopedPermissionPolicy[InstanceRoleGroup]):
    """Permission policy for InstanceRoleGroup, based on its scope (InstanceConfig)."""

    def __init__(self):
        super().__init__(InstanceRoleGroup)

    @override
    def get_instance_configs_for_obj(self, obj: InstanceRoleGroup) -> list[int]:
        """Get IDs of all InstanceConfigs the given InstanceRoleGroup is scoped for."""
        return [obj.instance.id]

    @override
    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        q = Q(instance_id__in=self.instance_admin_role.get_instances_for_user(user))
        if action == 'view':
            q |= Q(instance_id__in=self.instance_viewer_role.get_instances_for_user(user))
        return q
