from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar, override

from django.contrib.contenttypes.models import ContentType
from django.db.models import Q, QuerySet

from kausal_common.datasets.models import (
    DataPoint,
    Dataset,
    DatasetQuerySet,
    DatasetSchema,
    DataSource,
)
from kausal_common.models.permission_policy import (
    BaseObjectAction,
    ModelPermissionPolicy,
    ObjectSpecificAction,
    ParentInheritedPolicy,
)
from kausal_common.models.permissions import PermissionedQuerySet

from paths.context import realm_context

from nodes.roles import instance_admin_role, instance_viewer_role, instance_reviewer_role

if TYPE_CHECKING:
    from collections.abc import Sequence

    from django.contrib.auth.models import AnonymousUser

    from kausal_common.models.permissions import PermissionedModel
    from kausal_common.models.roles import InstanceSpecificRole

    from nodes.models import InstanceConfig
    from users.models import User

_M = TypeVar('_M', bound='PermissionedModel')
_QS = TypeVar('_QS', bound=QuerySet, default=QuerySet[_M])


class InstanceConfigScopedPermissionPolicy(ModelPermissionPolicy[_M, 'InstanceConfig', _QS]):
    """Permission policy for models that have one or many InstanceConfig objects as scope."""

    def __init__(self, model: type[_M]):
        from nodes.models import InstanceConfig

        self.model = model
        self.instance_admin_role = instance_admin_role
        self.instance_viewer_role = instance_viewer_role
        self.instance_reviewer_role = instance_reviewer_role
        self.ic_model = InstanceConfig
        super().__init__(model)

    @override
    def is_create_context_valid(self, context: Any) -> TypeGuard[InstanceConfig]:
        from nodes.models import InstanceConfig
        return isinstance(context, InstanceConfig)

    @abstractmethod
    def get_instance_configs_for_obj(self, obj: _M) -> list[int]:
        """Get IDs of all InstanceConfigs this obj is scoped for."""

    @override
    def user_has_perm(self, user: User, action: BaseObjectAction, obj: _M) -> bool:
        active_instance = realm_context.get().realm
        instance_ids = self.get_instance_configs_for_obj(obj)
        if active_instance is None or active_instance.pk not in instance_ids:
            return False
        # TODO HERE
        # Check if user is admin for any of the instances
        if user.has_instance_role(self.instance_admin_role, active_instance):
            return True
        # For view permission, check if user is a viewer or reviewer for any of the instances
        if action == 'view':
            return any((
                user.has_instance_role(self.instance_viewer_role, active_instance),
                user.has_instance_role(self.instance_reviewer_role, active_instance),
            ))
        return False

    @override
    def anon_has_perm(self, action: BaseObjectAction, obj: _M) -> bool:
        return False

    @override
    def user_can_create(self, user: User, context: InstanceConfig) -> bool:
        return user.has_instance_role(self.instance_admin_role, context)

    @override
    def construct_perm_q_anon(self, action: BaseObjectAction) -> Q | None:
        return None

    def user_has_any_role_in_active_instance(self, user: User, roles: Sequence[InstanceSpecificRole[InstanceConfig]]) -> bool:
        if user.is_superuser:
            return True

        active_instance = realm_context.get().realm

        if active_instance is None:
            return False

        return any(user.has_instance_role(role, active_instance) for role in roles)


class DatasetSchemaPermissionPolicy(InstanceConfigScopedPermissionPolicy[DatasetSchema]):
    """Permission policy for DatasetSchema, based on its scope (InstanceConfig)."""

    def __init__(self):
        from kausal_common.datasets.models import DatasetSchema  # TODO why import here?
        super().__init__(DatasetSchema)

    @override
    def get_instance_configs_for_obj(self, obj: DatasetSchema) -> list[int]:
        """Get IDs of all InstanceConfigs this schema is scoped for."""
        from nodes.models import InstanceConfig
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)
        return list(obj.scopes.filter(
            scope_content_type=ic_content_type
        ).values_list('scope_id', flat=True))

    @override
    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        from nodes.models import InstanceConfig
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)

        admin_q = Q(
            scopes__scope_content_type=ic_content_type,
            scopes__scope_id__in=self.instance_admin_role.get_instances_for_user(user)
        )

        if action == 'view':
            viewer_q = Q(
                scopes__scope_content_type=ic_content_type,
                scopes__scope_id__in=self.instance_viewer_role.get_instances_for_user(user)
            )
            reviewer_q = Q(
                scopes__scope_content_type=ic_content_type,
                scopes__scope_id__in=self.instance_reviewer_role.get_instances_for_user(user)
            )
            return admin_q | viewer_q | reviewer_q

        return admin_q

    def user_has_permission(self, user: User | AnonymousUser, action: str) -> bool:
        if not self.user_is_authenticated(user):
            return False

        allowed_roles: list[InstanceSpecificRole[InstanceConfig]] = [self.instance_admin_role]
        if action == 'view':
            allowed_roles.append(self.instance_viewer_role)
            allowed_roles.append(self.instance_reviewer_role)
        if action == 'review':
            allowed_roles.append(self.instance_reviewer_role)

        return self.user_has_any_role_in_active_instance(user, allowed_roles)

    def user_has_any_permission(self, user: User | AnonymousUser, actions: Sequence[str]) -> bool:
        return any(self.user_has_permission(user, action) for action in actions)


class DatasetPermissionPolicy(ParentInheritedPolicy[Dataset, DatasetSchema, DatasetQuerySet]):
    """Permission policy for Dataset, inheriting from its schema."""

    def __init__(self):
        from kausal_common.datasets.models import Dataset, DatasetSchema
        super().__init__(Dataset, DatasetSchema, 'schema')

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: Dataset) -> bool:
        parent_obj = self.get_parent_obj(obj)
        return self.parent_policy.user_has_perm(user, action, parent_obj)

    @override
    def anon_has_perm(self, action: BaseObjectAction, obj: Dataset) -> bool:
        return False

    @override
    def user_can_create(self, user: User, context: DatasetSchema) -> bool:
        return self.parent_policy.user_has_perm(user, 'change', context)

    def user_can_review(self, user: User) -> bool:
        return self.parent_policy.user_has_permission(user, 'review')


class DataPointPermissionPolicy(ParentInheritedPolicy[DataPoint, Dataset, PermissionedQuerySet[DataPoint]]):
    """Permission policy for DataPoint, inheriting from Dataset."""

    def __init__(self):
        from kausal_common.datasets.models import DataPoint, Dataset
        super().__init__(DataPoint, Dataset, 'dataset')

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: DataPoint) -> bool:
        parent_obj = self.get_parent_obj(obj)
        return self.parent_policy.user_has_perm(user, action, parent_obj)

    @override
    def anon_has_perm(self, action: BaseObjectAction, obj: DataPoint) -> bool:
        return False

    @override
    def user_can_create(self, user: User, context: Dataset) -> bool:
        return self.parent_policy.user_has_perm(user, 'change', context)


class DataSourcePermissionPolicy(InstanceConfigScopedPermissionPolicy[DataSource]):
    """Permission policy for DataSource, based on its scope (InstanceConfig)."""

    def __init__(self):
        super().__init__(DataSource)

    @override
    def get_instance_configs_for_obj(self, obj: DataSource) -> list[int]:
        from nodes.models import InstanceConfig
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)
        if obj.scope_content_type != ic_content_type:
            return []
        return [obj.scope_id]

    @override
    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        from nodes.models import InstanceConfig
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)

        admin_q = Q(
            scope_content_type=ic_content_type,
            scope_id__in=self.instance_admin_role.get_instances_for_user(user)
        )

        if action == 'view':
            viewer_q = Q(
                scope_content_type=ic_content_type,
                scope_id__in=self.instance_viewer_role.get_instances_for_user(user)
            )
            reviewer_q = Q(
                scope_content_type=ic_content_type,
                scope_id__in=self.instance_reviewer_role.get_instances_for_user(user)
            )
            return admin_q | viewer_q | reviewer_q

        return admin_q
