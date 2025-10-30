from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard, override

from django.db.models import Q

from paths.dataset_permission_policy import InstanceConfigScopedPermissionPolicy

from nodes.models import InstanceConfig
from people.models import PersonGroup

if TYPE_CHECKING:
    from kausal_common.models.permission_policy import BaseObjectAction

    from users.models import User


class PersonGroupPermissionPolicy(InstanceConfigScopedPermissionPolicy[PersonGroup, InstanceConfig]):
    """Permission policy for PersonGroup, based on its scope (InstanceConfig)."""

    def __init__(self):
        super().__init__(PersonGroup)  # type: ignore[type-abstract]

    @override
    def get_instance_configs_for_obj(self, obj: PersonGroup) -> list[int]:
        """Get IDs of all InstanceConfigs the given PersonGroup is scoped for."""
        return [obj.instance.pk]

    @override
    def is_create_context_valid(self, context: Any) -> TypeGuard[InstanceConfig]:
        return isinstance(context, InstanceConfig)

    @override
    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        q = Q(instance_id__in=self.get_role('instance-super-admin').get_instances_for_user(user))
        return q
