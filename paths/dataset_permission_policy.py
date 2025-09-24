from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar, override

from django.contrib.contenttypes.models import ContentType
from django.db.models import Q, QuerySet
from rest_framework.views import APIView

from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
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
from kausal_common.models.roles import role_registry

from paths.context import realm_context

from nodes.models import InstanceConfig
from nodes.roles import (
    InstanceGroupMembershipRole,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from django.contrib.auth.models import AnonymousUser

    from kausal_common.models.permissions import PermissionedModel
    from kausal_common.models.roles import InstanceSpecificRole

    from paths.const import InstanceRoleIdentifier

    from users.models import User

_M = TypeVar('_M', bound='PermissionedModel')
_QS = TypeVar('_QS', bound=QuerySet, default=QuerySet[_M])


def get_instance_config_content_type() -> ContentType:
    """Get the ContentType for InstanceConfig, cached for performance."""
    from nodes.models import InstanceConfig
    return ContentType.objects.get_for_model(InstanceConfig)


class InstanceConfigScopedPermissionPolicy(ModelPermissionPolicy[_M, 'InstanceConfig', _QS], metaclass=ABCMeta):
    """Permission policy for models that have one or many InstanceConfig objects as scope."""

    roles: dict[str, InstanceGroupMembershipRole]
    models: type[_M]

    def __init__(self, model: type[_M]):
        self.model = model
        self._role_registry = role_registry
        super().__init__(model)

    def get_role(self, role_id: InstanceRoleIdentifier) -> InstanceGroupMembershipRole:
        role = self._role_registry.get_role(role_id)
        if not isinstance(role, InstanceGroupMembershipRole):
            raise TypeError('Currently only InstanceGroupMembershipRoles supported')
        return role

    def get_instanceconfig_scope_q_for_role(self, user: User, role_id: InstanceRoleIdentifier) -> Q:
        ic_content_type = get_instance_config_content_type()
        return Q(
            scope_content_type=ic_content_type,
            scope_id__in=self.get_role(role_id).get_instances_for_user(user)
        )

    @override
    def is_create_context_valid(self, context: Any) -> TypeGuard[InstanceConfig]:
        from nodes.models import InstanceConfig
        return isinstance(context, InstanceConfig)

    @abstractmethod
    def get_instance_configs_for_obj(self, obj: _M) -> list[int]:
        """Get IDs of all InstanceConfigs this obj is scoped for."""

    @override
    def user_has_perm(self, user: User, action: BaseObjectAction, obj: _M) -> bool:
        try:
            # Realm context only works in admin context, not for REST API
            active_instance = realm_context.get().realm
        except LookupError:
            active_instance = None
        instance_ids = self.get_instance_configs_for_obj(obj)
        if active_instance is None:
            instances = InstanceConfig.objects.filter(pk__in=instance_ids)
        else:
            if active_instance.pk not in instance_ids:
                return False
            instances = [active_instance]
        for instance in instances:
            # Check if user is admin for any of the instances
            if user.has_instance_role_with_id('instance-admin', instance):
                return True
            if user.has_instance_role_with_id('instance-super-admin', instance):
               return True
            # For view permission, check if user is a viewer or reviewer for any of the instances
            if action == 'view':
               return any((
                   user.has_instance_role_with_id('instance-viewer', instance),
                   user.has_instance_role_with_id('instance-reviewer', instance),
               ))
        return False

    @override
    def anon_has_perm(self, action: BaseObjectAction, obj: _M) -> bool:
        return False

    @override
    def user_can_create(self, user: User, context: InstanceConfig) -> bool:
        return (
            user.is_superuser or
            user.has_instance_role_with_id('instance-admin', context) or
            user.has_instance_role_with_id('instance-super-admin', context)
        )


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
        ic_content_type = get_instance_config_content_type()
        return list(obj.scopes.filter(
            scope_content_type=ic_content_type
        ).values_list('scope_id', flat=True))

    @override
    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        from nodes.models import InstanceConfig
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)
        def make_q(role: InstanceRoleIdentifier):
            return Q(
                scopes__scope_content_type=ic_content_type,
                scopes__scope_id__in=self.get_role(role).get_instances_for_user(user)

            )
        super_admin_q = make_q('instance-super-admin')
        admin_q = make_q('instance-admin')
        reviewer_q = make_q('instance-reviewer')
        viewer_q = make_q('instance-viewer')

        if action == 'view':
            return super_admin_q | admin_q | viewer_q | reviewer_q
        return super_admin_q | admin_q

    def user_has_permission(self, user: User | AnonymousUser, action: str) -> bool:
        if not self.user_is_authenticated(user):
            return False

        allowed_roles: list[InstanceSpecificRole[InstanceConfig]] = [
            self.get_role('instance-admin'),
            self.get_role('instance-super-admin')
        ]
        if action == 'view':
            allowed_roles.append(self.get_role('instance-viewer'))
            allowed_roles.append(self.get_role('instance-reviewer'))
        if action == 'review':
            allowed_roles.append(self.get_role('instance-reviewer'))

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
        ic_content_type = get_instance_config_content_type()
        if obj.scope_content_type != ic_content_type:
            return []
        return [obj.scope_id]

    @override
    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        admin_q = self.get_instanceconfig_scope_q_for_role(user, 'instance-admin')
        super_admin_q = self.get_instanceconfig_scope_q_for_role(user, 'instance-super-admin')
        viewer_q = self.get_instanceconfig_scope_q_for_role(user, 'instance-viewer')
        reviewer_q = self.get_instanceconfig_scope_q_for_role(user, 'instance-reviewer')

        if action == 'view':
            return super_admin_q | admin_q | viewer_q | reviewer_q
        return super_admin_q | admin_q


class DataPointCommentPermissionPolicy(ParentInheritedPolicy[DataPointComment, Dataset, PermissionedQuerySet[DataPointComment]]):
    """Permission policy for DataPointComment, inheriting from Dataset."""

    def __init__(self):
        from kausal_common.datasets.models import DataPointComment, Dataset
        super().__init__(DataPointComment, Dataset, 'dataset')

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: DataPointComment) -> bool:
        parent_obj = self.get_parent_obj(obj)
        return self.parent_policy.user_has_perm(user, action, parent_obj)

    @override
    def anon_has_perm(self, action: BaseObjectAction, obj: DataPointComment) -> bool:
        return False

    @override
    def user_can_create(self, user: User, context: APIView) -> bool:
        view = context
        dataset_uuid = view.kwargs['dataset_uuid']
        dataset = Dataset.objects.get(uuid=dataset_uuid)
        user_can_change_dataset = self.parent_policy.user_has_perm(user, 'change', dataset)
        instance_config_in_scope = dataset.scope
        if not isinstance(instance_config_in_scope, InstanceConfig):
            raise TypeError('Only InstanceConfigs supported as Dataset scopes in paths.')
        user_is_reviewer_in_instance = user.has_instance_role_with_id('instance-reviewer', instance_config_in_scope)
        return user_can_change_dataset or user_is_reviewer_in_instance
