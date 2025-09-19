from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard, cast, override

from django.db.models import Q

from kausal_common.models.permission_policy import ModelPermissionPolicy, ObjectSpecificAction

from nodes.models import InstanceConfig
from orgs.models import Organization
from people.models import Person

if TYPE_CHECKING:
    from paths.permissions import CreateContext

    from users.models import User


class PersonPermissionPolicy(ModelPermissionPolicy[Person]):
    def __init__(self):
        from nodes.roles import instance_super_admin_role

        self.super_admin_role = instance_super_admin_role
        super().__init__(Person)

    @override
    def construct_perm_q(self, user: User, action: ObjectSpecificAction) -> Q | None:
        """
        Construct a Q object for determining the persons for which the user has the permissions for the action.

        Returns None if no objects are allowed, and Q otherwise.
        """
        if user.is_superuser:
            return Q()
        is_super_admin = self.super_admin_role.role_q(user)
        instance_configs = InstanceConfig.objects.filter(is_super_admin)
        if not instance_configs:
            return None
        organizations = None
        for ic in instance_configs:
            organizations = Organization.objects.filter(path__startswith=ic.organization.path)
        if not organizations:
            return None
        q = Q()
        for org in organizations:
            q |= Q(organization_id=org.pk)
        return q

    @override
    def construct_perm_q_anon(self, action: ObjectSpecificAction) -> Q | None:
        """
        Construct a Q object for determining the permissions for the action for anonymous users.

        Returns None no objects are allowed, and Q otherwise.
        """
        return None

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: Person) -> bool:
        """Check if user has permission to perform an action on an instance."""
        if user.is_superuser:
            return True
        organization = obj.organization
        ancestors = organization.get_ancestors()
        instance_configs_with_organization = InstanceConfig.objects.filter(organization__in=[organization] + list(ancestors))
        return any(user.has_instance_role(self.super_admin_role, instance) for instance in instance_configs_with_organization)

    @override
    def anon_has_perm(self, action: ObjectSpecificAction, obj: Person) -> bool:
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
