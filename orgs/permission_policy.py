from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard, cast, override

from django.db.models import Q

from kausal_common.models.permission_policy import ModelPermissionPolicy, ObjectSpecificAction

from nodes.models import InstanceConfig
from orgs.models import Organization

if TYPE_CHECKING:
    from paths.permissions import CreateContext

    from users.models import User


class OrganizationPermissionPolicy(ModelPermissionPolicy[Organization]):
    def __init__(self):
        org_class: type[Organization] = Organization
        from nodes.roles import instance_super_admin_role

        self.super_admin_role = instance_super_admin_role
        super().__init__(org_class)

    @override
    def construct_perm_q(self, user: User, action: ObjectSpecificAction) -> Q | None:
        """
        Construct a Q object for determining the organizations for which the user has the permissions for the action.

        Returns None if no objects are allowed, and Q otherwise.
        """
        is_super_admin = self.super_admin_role.role_q(user)
        instance_configs = InstanceConfig.objects.filter(is_super_admin)
        q = Q()
        for ic in instance_configs:
            q |= Q(path__startswith=ic.organization.path)
        print('returning q')
        return q

    @override
    def construct_perm_q_anon(self, action: ObjectSpecificAction) -> Q | None:
        """
        Construct a Q object for determining the permissions for the action for anonymous users.

        Returns None no objects are allowed, and Q otherwise.
        """
        return None

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: Organization) -> bool:
        """Check if user has permission to perform an action on an instance."""
        if user.is_superuser:
            return True
        ancestors = obj.get_ancestors()
        instance_configs_with_organization = InstanceConfig.objects.filter(organization__in=[obj] + list(ancestors))
        return any(user.has_instance_role(self.super_admin_role, instance) for instance in instance_configs_with_organization)

    @override
    def anon_has_perm(self, action: ObjectSpecificAction, obj: Organization) -> bool:
        """Check if an unauthenticated user has permission to perform an action on an instance."""
        return False

    @override
    def is_create_context_valid(self, context: CreateContext) -> TypeGuard[InstanceConfig]:
        from nodes.models import InstanceConfig

        return isinstance(context, InstanceConfig)

    @override
    def user_can_create(self, user: User, context: CreateContext) -> bool:
        """Check if user can create a new object."""
        active_instance = cast('InstanceConfig', context)
        return user.is_superuser or user.has_instance_role(self.super_admin_role, active_instance)
