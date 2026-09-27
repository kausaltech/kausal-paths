from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, final, override

from django.contrib.contenttypes.models import ContentType
from django.db.models import Model, Q

from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
    Dataset,
    DatasetMetric,
    DatasetQuerySet,
    DatasetSchema,
    DatasetSchemaQuerySet,
    DatasetSourceReference,
    DataSource,
)
from kausal_common.models.permission_policy import (
    ModelPermissionPolicy,
    ParentInheritedPolicy,
    PermissionBlock,
)
from kausal_common.models.permissions import PermissionedQuerySet
from kausal_common.models.roles import role_registry
from kausal_common.people.models import ObjectRole

from paths.context import realm_context

from frameworks.models import Framework
from frameworks.roles import framework_admin_role
from nodes.models import InstanceConfig, InstanceConfigPermissionPolicy, NodeConfig
from nodes.roles import (
    InstanceGroupMembershipRole,
)
from people.models import DatasetGroupPermission, DatasetPersonPermission

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from django.contrib.auth.models import AnonymousUser
    from django.db.models import QuerySet

    from kausal_common.models.permission_policy import (
        BaseObjectAction,
        ObjectSpecificAction,
    )
    from kausal_common.models.permissions import PermissionedModel
    from kausal_common.models.roles import InstanceSpecificRole

    from paths.const import InstanceRoleIdentifier

    from users.models import User


def _instance_role_ids(user: User, action: BaseObjectAction) -> QuerySet[InstanceConfig]:
    policy_action: Literal['view', 'change'] = 'view' if action == 'view' else 'change'
    return InstanceConfig.objects.filter(InstanceConfigPermissionPolicy().role_q(user, policy_action))


class InstanceConfigScopedPermissionPolicy[
    M: PermissionedModel,
    CreateCtx: Model | None = None,
    QS: 'PermissionedQuerySet[Any]' = 'PermissionedQuerySet[M]',
](ModelPermissionPolicy[M, CreateCtx, QS], ABC):
    """Permission policy for models that have one or many InstanceConfig objects as scope."""

    roles: dict[str, InstanceGroupMembershipRole]
    models: type[M]

    def __init__(self, model: type[M]):
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
        return Q(scope_content_type=ic_content_type, scope_id__in=self.get_role(role_id).get_instances_for_user(user))

    @override
    @abstractmethod
    def is_create_context_valid(self, context: Any) -> TypeGuard[CreateCtx]: ...

    @abstractmethod
    def get_instance_configs_for_obj(self, obj: M) -> list[int]:
        """Get IDs of all InstanceConfigs this obj is scoped for."""

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: M) -> bool:
        if self.get_permission_block(action, obj=obj) is not None:
            return False
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
        return any(InstanceConfigPermissionPolicy().user_has_perm(user, action, instance) for instance in instances)

    def get_permission_block(
        self,
        action: BaseObjectAction,
        *,
        obj: M | None = None,
        context: CreateCtx | None = None,
    ) -> PermissionBlock | None:
        if action == 'add':
            try:
                active_instance = realm_context.get().realm
            except LookupError:
                return None
            if active_instance.is_locked:
                return PermissionBlock('Instance is locked', code='instance_locked')
            return None

        if obj is None or action not in ('change', 'delete'):
            return None
        instance_ids = self.get_instance_configs_for_obj(obj)
        if not instance_ids:
            return None
        try:
            active_instance = realm_context.get().realm
        except LookupError:
            active_instance = None
        if active_instance is not None:
            is_locked = active_instance.pk in instance_ids and active_instance.is_locked
        else:
            is_locked = InstanceConfig.objects.filter(pk__in=instance_ids, is_locked=True).exists()
        if is_locked:
            return PermissionBlock('Instance is locked', code='instance_locked')
        return None

    @override
    def anon_has_perm(self, action: BaseObjectAction, obj: M) -> bool:
        return False

    @override
    def user_can_create(self, user: User, context: CreateCtx) -> bool:
        return (
            user.is_superuser
            or user.has_instance_role_in_any_instance('instance-admin')
            or user.has_instance_role_in_any_instance('instance-editor')
            or user.has_instance_role_in_any_instance('instance-super-admin')
            or InstanceConfig.objects.filter(InstanceConfigPermissionPolicy().role_q(user, 'change')).exists()
        )

    @override
    def construct_perm_q_anon(self, action: BaseObjectAction) -> Q | None:
        return None

    def user_has_any_role_in_active_instance(self, user: User, roles: Sequence[InstanceSpecificRole[InstanceConfig]]) -> bool:
        if user.is_superuser:
            return True

        active_instance = realm_context.get().realm
        return any(user.has_instance_role(role, active_instance) for role in roles)


def _dataset_grant_q(user: User, action: BaseObjectAction) -> Q | None:
    """Match the datasets an explicit person or group grant gives the user `action` on."""
    roles = ObjectRole.get_roles_for_action(action)
    if not roles:
        return None
    return Q(pk__in=DatasetPersonPermission.objects.filter(person__user=user, role__in=roles).values('object_id')) | Q(
        pk__in=DatasetGroupPermission.objects.filter(group__persons__user=user, role__in=roles).values('object_id')
    )


@final
class DatasetSchemaPermissionPolicy(InstanceConfigScopedPermissionPolicy[DatasetSchema, None, DatasetSchemaQuerySet]):
    """Permission policy for DatasetSchema, based on its scope (InstanceConfig)."""

    def __init__(self):
        super().__init__(DatasetSchema)

    def is_create_context_valid(self, context: Any) -> TypeGuard[None]:
        return context is None

    @override
    def get_instance_configs_for_obj(self, obj: DatasetSchema) -> list[int]:
        """Get IDs of all InstanceConfigs this schema is scoped for."""
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)
        local = obj.scopes.filter(scope_content_type=ic_content_type).values_list('scope_id', flat=True)
        node_ids = obj.scopes.filter(scope_content_type=ContentType.objects.get_for_model(NodeConfig)).values('scope_id')
        framework_ids = obj.scopes.filter(scope_content_type=ContentType.objects.get_for_model(Framework)).values('scope_id')
        return list(
            InstanceConfig.objects
            .filter(Q(pk__in=local) | Q(nodes__in=node_ids) | Q(framework_config__framework_id__in=framework_ids))
            .distinct()
            .values_list('pk', flat=True)
        )

    @override
    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        def make_q(role: InstanceRoleIdentifier) -> Q:
            return Dataset.instance_scope_q(self.get_role(role).get_instances_for_user(user), prefix='scopes__')

        q = make_q('instance-super-admin') | make_q('instance-admin') | make_q('instance-editor')
        if action != 'delete':
            from frameworks.organization_access import instance_grant_q

            granted_instances = InstanceConfig.objects.filter(
                instance_grant_q(user, action='view' if action == 'view' else 'change')
            )
            q |= Dataset.instance_scope_q(granted_instances, prefix='scopes__')
        if action == 'view':
            q |= make_q('instance-viewer') | make_q('instance-reviewer')
            # A dataset grant lets its holder read the schema's structure, never change it.
            grant_q = _dataset_grant_q(user, 'view')
            if grant_q is not None:
                q |= Q(pk__in=Dataset.objects.filter(grant_q).values('schema_id'))

        def apply_editable_filter(value: Q) -> Q:
            framework_ct = ContentType.objects.get_for_model(Framework)
            if action == 'view':
                value |= Q(
                    scopes__scope_content_type=framework_ct,
                    scopes__scope_id__in=InstanceConfig.objects.filter(pk__in=_instance_role_ids(user, action)).values(
                        'framework_config__framework_id'
                    ),
                )
                return value
            shared_ids = DatasetSchema.objects.filter(scopes__scope_content_type=framework_ct).values('pk')
            administered = Framework.objects.filter(framework_admin_role.role_q(user)).values('pk')
            # FIXME: `is_editable` is deprecated; see DatasetSchema.is_editable.
            return (value & ~Q(pk__in=shared_ids) & Q(is_editable=True)) | Q(
                scopes__scope_content_type=framework_ct,
                scopes__scope_id__in=administered,
            )

        return apply_editable_filter(q)

    def construct_state_perm_q(self, action: ObjectSpecificAction) -> Q:
        if action not in ('change', 'delete'):
            return Q()
        unlocked_instances = InstanceConfig.objects.filter(is_locked=False)
        return Dataset.instance_scope_q(unlocked_instances, prefix='scopes__') | Q(
            scopes__scope_content_type=ContentType.objects.get_for_model(Framework),
        )

    @override
    def user_can_create(self, user: User, context: None) -> bool:
        return super().user_can_create(user, context)

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: DatasetSchema) -> bool:
        framework_ids = obj.scopes.filter(scope_content_type=ContentType.objects.get_for_model(Framework)).values('scope_id')
        if framework_ids.exists():
            if user.is_superuser:
                return True
            if Framework.objects.filter(framework_admin_role.role_q(user), pk__in=framework_ids).exists():
                return True
            if action != 'view':
                return False
        if self.get_permission_block(action, obj=obj) is not None:
            return False
        # FIXME: `is_editable` is deprecated; see DatasetSchema.is_editable.
        if action in ('change', 'delete') and not obj.is_editable and not user.is_superuser:
            return False
        if action == 'view':
            grant_q = _dataset_grant_q(user, 'view')
            if grant_q is not None and obj.datasets.filter(grant_q).exists():
                return True
        return super().user_has_perm(user, action, obj)

    def user_has_permission(self, user: User | AnonymousUser, action: str) -> bool:
        if not self.user_is_authenticated(user):
            return False

        allowed_roles: list[InstanceSpecificRole[InstanceConfig]] = [
            self.get_role('instance-admin'),
            self.get_role('instance-editor'),
            self.get_role('instance-super-admin'),
        ]
        if action == 'view':
            allowed_roles.append(self.get_role('instance-viewer'))
            allowed_roles.append(self.get_role('instance-reviewer'))
        if action == 'review':
            allowed_roles.append(self.get_role('instance-reviewer'))

        return self.user_has_any_role_in_active_instance(user, allowed_roles)

    def user_has_any_permission(self, user: User | AnonymousUser, actions: Sequence[str]) -> bool:
        return any(self.user_has_permission(user, action) for action in actions)


class DatasetPermissionPolicy(ParentInheritedPolicy[Dataset, DatasetSchema, DatasetQuerySet, DatasetSchema]):
    """
    A dataset's permissions are its scope's, plus explicit grants on the dataset.

    An instance dataset follows InstanceConfigPermissionPolicy and a node-owned one
    NodeConfigPermissionPolicy, which adds the node's edit lock. Viewing needs an explicit
    instance or framework role: a public instance does not make its data rows public. The
    schema governs structure only and grants nothing here.
    """

    def __init__(self):
        super().__init__(Dataset, DatasetSchema, 'schema', create_context_type=DatasetSchema)

    @staticmethod
    def _scope_q(user: User, action: BaseObjectAction) -> Q:
        ic_policy = InstanceConfigPermissionPolicy()
        if action == 'view':
            return Dataset.instance_scope_q(InstanceConfig.objects.filter(ic_policy.role_q(user, 'view')))
        # Editing or deleting a dataset changes its scope object; it never deletes it.
        instances = InstanceConfig.objects.filter(ic_policy.role_q(user, 'change'))
        node_q = NodeConfig.permission_policy().construct_perm_q(user, 'change')
        nodes = NodeConfig.objects.filter(node_q) if node_q is not None else NodeConfig.objects.none()
        return Q(scope_content_type=ContentType.objects.get_for_model(InstanceConfig), scope_id__in=instances.values('pk')) | Q(
            scope_content_type=ContentType.objects.get_for_model(NodeConfig), scope_id__in=nodes.values('pk')
        )

    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        q = self._scope_q(user, action)
        grant_q = _dataset_grant_q(user, action)
        if grant_q is not None:
            q |= grant_q
        if action in ('change', 'delete'):
            # FIXME: `is_editable` is deprecated; see DatasetSchema.is_editable. It locks the data
            # of legacy local schemas; shared schemas protect only their structure.
            shared_schemas = DatasetSchema.objects.for_scope_type(Framework).values('pk')
            q &= Q(schema__is_editable=True) | Q(schema_id__in=shared_schemas)
        return q

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: Dataset) -> bool:
        if self.get_permission_block(action, obj=obj) is not None:
            return False
        if user.is_superuser:
            return True
        query = self.construct_perm_q(user, action)
        return query is not None and Dataset.objects.filter(query, self.construct_state_perm_q(action), pk=obj.pk).exists()

    def get_permission_block(
        self,
        action: BaseObjectAction,
        *,
        obj: Dataset | None = None,
        context: DatasetSchema | None = None,
    ) -> PermissionBlock | None:
        if obj is not None and action in ('change', 'delete'):
            return InstanceConfig.permission_policy().get_permission_block('change', obj=obj.scope_instance)
        return super().get_permission_block(action, obj=obj, context=context)

    def construct_state_perm_q(self, action: ObjectSpecificAction) -> Q:
        if action not in ('change', 'delete'):
            return Q()
        return Dataset.instance_scope_q(InstanceConfig.objects.filter(is_locked=False))

    @override
    def anon_has_perm(self, action: BaseObjectAction, obj: Dataset) -> bool:
        return False

    @override
    def user_can_create(self, user: User, context: DatasetSchema) -> bool:
        return self.parent_policy.user_has_perm(user, 'change', context)

    def user_can_review(self, user: User) -> bool:
        return self.parent_policy.user_has_permission(user, 'review')

    @override
    def user_has_permission(self, user: User | AnonymousUser, action: str) -> bool:
        if self.parent_policy.user_has_permission(user, action):
            return True
        if not self.user_is_authenticated(user) or action == 'review':
            return False
        grant_action: BaseObjectAction = 'view' if action == 'view' else 'delete' if action == 'delete' else 'change'
        grant_q = _dataset_grant_q(user, grant_action)
        return grant_q is not None and Dataset.objects.filter(grant_q).exists()

    @override
    def user_has_any_permission(self, user: User | AnonymousUser, actions: Sequence[str]) -> bool:
        return any(self.user_has_permission(user, action) for action in actions)


class DatasetMetricPermissionPolicy(
    ParentInheritedPolicy[DatasetMetric, DatasetSchema, PermissionedQuerySet[DatasetMetric], DatasetSchema]
):
    """Permission policy for DatasetMetric, inheriting from its schema."""

    def __init__(self):
        super().__init__(DatasetMetric, DatasetSchema, 'schema', create_context_type=DatasetSchema)

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


class DataPointPermissionPolicy(ParentInheritedPolicy[DataPoint, Dataset, PermissionedQuerySet[DataPoint], Dataset]):
    """Permission policy for DataPoint, inheriting from Dataset."""

    def __init__(self):
        super().__init__(DataPoint, Dataset, 'dataset', create_context_type=Dataset)

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


class DataSourcePermissionPolicy(InstanceConfigScopedPermissionPolicy[DataSource, None, PermissionedQuerySet[DataSource]]):
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
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)
        admin_q = self.get_instanceconfig_scope_q_for_role(user, 'instance-admin')
        editor_q = self.get_instanceconfig_scope_q_for_role(user, 'instance-editor')
        super_admin_q = self.get_instanceconfig_scope_q_for_role(user, 'instance-super-admin')
        viewer_q = self.get_instanceconfig_scope_q_for_role(user, 'instance-viewer')
        reviewer_q = self.get_instanceconfig_scope_q_for_role(user, 'instance-reviewer')

        q = super_admin_q | admin_q | editor_q
        if action != 'delete':
            from frameworks.organization_access import instance_grant_q

            granted_instances = InstanceConfig.objects.filter(
                instance_grant_q(user, action='view' if action == 'view' else 'change')
            )
            q |= Q(scope_content_type=ic_content_type, scope_id__in=granted_instances.values('pk'))
        if action == 'view':
            q |= viewer_q | reviewer_q

            # Also allow users who have schema-level permissions via InstanceConfig permissions
            subsector_q = InstanceConfigPermissionPolicy().construct_perm_q(user, 'view', include_implicit_public=False)
            if subsector_q is None:
                return q

            instance_ids = InstanceConfig.objects.filter(subsector_q).values_list('pk', flat=True)
            if not instance_ids:
                return q
            schema_perm_q = Q(scope_content_type=ic_content_type, scope_id__in=instance_ids)
            q |= schema_perm_q

        return q

    def construct_state_perm_q(self, action: ObjectSpecificAction) -> Q:
        if action not in ('change', 'delete'):
            return Q()
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)
        unlocked_instances = InstanceConfig.objects.filter(is_locked=False).values_list('pk', flat=True)
        return Q(scope_content_type=ic_content_type, scope_id__in=unlocked_instances)

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: DataSource) -> bool:
        if super().user_has_perm(user, action, obj):
            return True

        if action == 'view':
            ic_policy = InstanceConfigPermissionPolicy()
            subsector_q = ic_policy.construct_perm_q(user, 'view', include_implicit_public=False)
            if subsector_q is not None:
                ic_content_type = ContentType.objects.get_for_model(InstanceConfig)
                if obj.scope_content_type == ic_content_type:
                    return InstanceConfig.objects.filter(pk=obj.scope_id).filter(subsector_q).exists()

        return False

    @override
    def is_create_context_valid(self, context: Any) -> TypeGuard[None]:
        return context is None


class DataPointCommentPermissionPolicy(
    ParentInheritedPolicy[DataPointComment, DataPoint, PermissionedQuerySet[DataPointComment], DataPoint]
):
    """Permission policy for DataPointComment, delegating to DataPoint."""

    def __init__(self):
        super().__init__(DataPointComment, DataPoint, 'data_point', create_context_type=DataPoint)

    @override
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: DataPointComment) -> bool:
        parent_obj = self.get_parent_obj(obj)
        return self.parent_policy.user_has_perm(user, action, parent_obj)

    @override
    def anon_has_perm(self, action: BaseObjectAction, obj: DataPointComment) -> bool:
        return False

    @override
    def user_can_create(self, user: User, context: DataPoint) -> bool:
        data_point = context
        dataset = data_point.dataset
        instance_config_in_scope = dataset.scope_instance
        user_has_reviewer_role_in_instance = user.has_instance_role_with_id('instance-reviewer', instance_config_in_scope)
        user_can_create_datapoint = self.parent_policy.user_can_create(user, dataset)
        return user_can_create_datapoint or user_has_reviewer_role_in_instance


class DatasetSourceReferencePermissionPolicy(
    ParentInheritedPolicy[DatasetSourceReference, Dataset, PermissionedQuerySet[DatasetSourceReference], Dataset]
):
    """Permission policy for DatasetSourceReference, delegating to DataSet."""

    def __init__(self):
        super().__init__(DatasetSourceReference, Dataset, 'dataset', create_context_type=Dataset)

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
    def user_can_create(self, user: User, context: Dataset) -> bool:
        dataset = context
        return self.parent_policy.user_has_perm(user, 'change', dataset)
