from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar, cast, final, override

from django.contrib.contenttypes.models import ContentType
from django.db.models import Q, QuerySet

from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
    Dataset,
    DatasetMetric,
    DatasetQuerySet,
    DatasetSchema,
    DatasetSourceReference,
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
from kausal_common.people.models import ObjectRole

from paths.context import realm_context

from nodes.models import InstanceConfig
from nodes.roles import (
    InstanceGroupMembershipRole,
)
from people.models import DatasetSchemaGroupPermission, DatasetSchemaPersonPermission, PersonGroupMember

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from django.contrib.auth.models import AnonymousUser
    from django.db.models import Model

    from kausal_common.models.permissions import PermissionedModel
    from kausal_common.models.roles import InstanceSpecificRole

    from paths.const import InstanceRoleIdentifier
    from paths.permissions import CreateContext

    from users.models import User

_M = TypeVar('_M', bound='PermissionedModel')
_CCTX = TypeVar('_CCTX', bound='Model | None')  # create context
_QS = TypeVar('_QS', bound=QuerySet, default=QuerySet[_M])


class InstanceConfigScopedPermissionPolicy(ModelPermissionPolicy[_M, _CCTX, _QS], metaclass=ABCMeta):
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
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)
        return Q(
            scope_content_type=ic_content_type,
            scope_id__in=self.get_role(role_id).get_instances_for_user(user)
        )

    @override
    @abstractmethod
    def is_create_context_valid(self, context: Any) -> TypeGuard[_CCTX]:
        pass

    @abstractmethod
    def get_instance_configs_for_obj(self, obj: _M) -> list[int]:
        """Get IDs of all InstanceConfigs this obj is scoped for."""

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: _M) -> bool:
        if user.is_superuser:
            return True
        try:
            # Realm context only works in admin context, not for REST API
            active_instance = realm_context.get().realm
        except LookupError:
            active_instance = None
        instance_ids = self.get_instance_configs_for_obj(obj)
        instances: Iterable[InstanceConfig]
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
    def user_can_create(self, user: User, context: _CCTX) -> bool:
        return (
            user.is_superuser or
            user.has_instance_role_in_any_instance('instance-admin') or
            user.has_instance_role_in_any_instance('instance-super-admin')
        )


    @override
    def construct_perm_q_anon(self, action: BaseObjectAction) -> Q | None:
        return None

    def user_has_any_role_in_active_instance(self, user: User, roles: Sequence[InstanceSpecificRole[InstanceConfig]]) -> bool:
        if user.is_superuser:
            return True

        active_instance = realm_context.get().realm
        return any(user.has_instance_role(role, active_instance) for role in roles)


@final
class DatasetSchemaPermissionPolicy(InstanceConfigScopedPermissionPolicy[DatasetSchema, None]):
    """Permission policy for DatasetSchema, based on its scope (InstanceConfig)."""

    def __init__(self):
        from kausal_common.datasets.models import DatasetSchema  # TODO why import here?
        super().__init__(DatasetSchema)

    def is_create_context_valid(self, context: Any) -> TypeGuard[None]:
        return context is None

    @override
    def get_instance_configs_for_obj(self, obj: DatasetSchema) -> list[int]:
        """Get IDs of all InstanceConfigs this schema is scoped for."""
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)
        return list(obj.scopes.filter(
            scope_content_type=ic_content_type
        ).values_list('scope_id', flat=True))

    @override
    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        from nodes.models import InstanceConfig
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)
        def make_q(role: InstanceRoleIdentifier) -> Q:
            return Q(
                scopes__scope_content_type=ic_content_type,
                scopes__scope_id__in=self.get_role(role).get_instances_for_user(user)

            )
        super_admin_q = make_q('instance-super-admin')
        admin_q = make_q('instance-admin')
        reviewer_q = make_q('instance-reviewer')
        viewer_q = make_q('instance-viewer')

        q = super_admin_q | admin_q

        if action == 'view':
            viewer_q = Q(
                scopes__scope_content_type=ic_content_type,
                scopes__scope_id__in=self.get_role('instance-viewer').get_instances_for_user(user)
            )
            reviewer_q = Q(
                scopes__scope_content_type=ic_content_type,
                scopes__scope_id__in=self.get_role('instance-reviewer').get_instances_for_user(user)
            )
            q |= viewer_q | reviewer_q

        if not hasattr(user, 'person'):
            return q

        privileged_roles = ObjectRole.get_roles_for_action(action)
        if not privileged_roles:
            return q

        group_ids = PersonGroupMember.objects.filter(person=user.person).values_list('group_id', flat=True)
        group_object_ids = DatasetSchemaGroupPermission.objects.filter(
            group_id__in=group_ids,
            role__in=privileged_roles
        ).values_list('object_id', flat=True)
        individual_object_ids = DatasetSchemaPersonPermission.objects.filter(
            person=user.person,
            role__in=privileged_roles
        ).values_list('object_id', flat=True)
        object_ids = set(group_object_ids) | set(individual_object_ids)
        if object_ids:
            return q | Q(pk__in=object_ids)
        return q

    @override
    def user_can_create(self, user: User, context: None) -> bool:
        return super().user_can_create(user, context)

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: DatasetSchema) -> bool:
        if hasattr(user, 'person'):
            # Check dataset schema's person / group permissions first
            privileged_roles = ObjectRole.get_roles_for_action(action)
            if obj.person_permissions.filter(person=user.person, role__in=privileged_roles).exists():
                return True

            if obj.group_permissions.filter(group__persons=user.person, role__in=privileged_roles).exists():
                return True

        return super().user_has_perm(user, action, obj)

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

    def is_create_context_valid(self, context: Any) -> TypeGuard[DatasetSchema]:
        return isinstance(context, DatasetSchema)


class DatasetMetricPermissionPolicy(ParentInheritedPolicy[DatasetMetric, DatasetSchema, PermissionedQuerySet[DatasetMetric]]):
    """Permission policy for DatasetMetric, inheriting from its schema."""

    def __init__(self):
        from kausal_common.datasets.models import DatasetMetric, DatasetSchema
        super().__init__(DatasetMetric, DatasetSchema, 'schema')

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: DatasetMetric) -> bool:
        parent_obj = self.get_parent_obj(obj)
        return self.parent_policy.user_has_perm(user, action, parent_obj)

    @override
    def anon_has_perm(self, action: BaseObjectAction, obj: DatasetMetric) -> bool:
        return False

    @override
    def user_can_create(self, user: User, context: DatasetSchema) -> bool:
        return self.parent_policy.user_has_perm(user, 'change', context)

    def is_create_context_valid(self, context: Any) -> TypeGuard[DatasetSchema]:
        return isinstance(context, DatasetSchema)


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

    def is_create_context_valid(self, context: Any) -> TypeGuard[Dataset]:
        return isinstance(context, Dataset)


class DataSourcePermissionPolicy(InstanceConfigScopedPermissionPolicy[DataSource, None]):
    """Permission policy for DataSource, based on its scope (InstanceConfig)."""

    def __init__(self):
        super().__init__(DataSource)

    @override
    def get_instance_configs_for_obj(self, obj: DataSource) -> list[int]:
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)
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

    def is_create_context_valid(self, context: Any) -> TypeGuard[None]:
        return context is None


class DataPointCommentPermissionPolicy(
        ParentInheritedPolicy[DataPointComment, DataPoint, PermissionedQuerySet[DataPointComment]]
):
    """Permission policy for DataPointComment, delegating to DataPoint."""

    def __init__(self):
        super().__init__(DataPointComment, DataPoint, 'data_point')

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: DataPointComment) -> bool:
        parent_obj = self.get_parent_obj(obj)
        return self.parent_policy.user_has_perm(user, action, parent_obj)

    @override
    def anon_has_perm(self, action: BaseObjectAction, obj: DataPointComment) -> bool:
        return False

    @override
    def is_create_context_valid(self, context: Any) -> TypeGuard[DataPoint]:
        return isinstance(context, DataPoint)

    @override
    def user_can_create(self, user: User, context: CreateContext) -> bool:
        data_point: DataPoint = cast('DataPoint', context)
        dataset = data_point.dataset
        instance_config_in_scope = dataset.scope
        if not isinstance(instance_config_in_scope, InstanceConfig):
            raise TypeError('Only InstanceConfigs supported as Dataset scopes in Paths.')
        user_has_reviewer_role_in_instance = user.has_instance_role_with_id('instance-reviewer', instance_config_in_scope)
        user_can_create_datapoint = self.parent_policy.user_can_create(user, dataset)
        return user_can_create_datapoint or user_has_reviewer_role_in_instance


class DatasetSourceReferencePermissionPolicy(
        ParentInheritedPolicy[DatasetSourceReference, Dataset, PermissionedQuerySet[DatasetSourceReference]]
):
    """Permission policy for DatasetSourceReference, delegating to DataSet."""

    def __init__(self):
        super().__init__(DatasetSourceReference, Dataset, 'dataset')

    @override
    def get_parent_obj(self, obj: DatasetSourceReference) -> Dataset:
        if obj.data_point:
            return obj.data_point.dataset
        if obj.dataset is None:
            raise ValueError('Invalid dataset source reference')
        return obj.dataset

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: DatasetSourceReference) -> bool:
        parent_obj = self.get_parent_obj(obj)
        return self.parent_policy.user_has_perm(user, action, parent_obj)

    @override
    def anon_has_perm(self, action: BaseObjectAction, obj: DatasetSourceReference) -> bool:
        return False

    @override
    def is_create_context_valid(self, context: Any) -> TypeGuard[Dataset]:
        return isinstance(context, Dataset)

    @override
    def user_can_create(self, user: User, context: CreateContext) -> bool:
        dataset: Dataset = cast('Dataset', context)
        instance_config_in_scope = dataset.scope
        if not isinstance(instance_config_in_scope, InstanceConfig):
            raise TypeError('Only InstanceConfigs supported as Dataset scopes in Paths.')
        return self.parent_policy.user_has_perm(user, 'change', dataset)
