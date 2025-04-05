from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

from django.contrib.contenttypes.models import ContentType
from django.db.models import Q

from kausal_common.datasets.models import (
    DataPoint,
    Dataset,
    DatasetQuerySet,
    DatasetSchema,
)
from kausal_common.models.permission_policy import (
    BaseObjectAction,
    ModelPermissionPolicy,
    ObjectSpecificAction,
    ParentInheritedPolicy,
)
from kausal_common.models.permissions import PermissionedQuerySet

from paths.context import realm_context

from nodes.roles import instance_admin_role, instance_viewer_role

if TYPE_CHECKING:
    from nodes.models import InstanceConfig
    from users.models import User


class DatasetSchemaPermissionPolicy(ModelPermissionPolicy[DatasetSchema, 'InstanceConfig', PermissionedQuerySet[DatasetSchema]]):
    """Permission policy for DatasetSchema, based on its scope (InstanceConfig)."""

    def __init__(self):
        from kausal_common.datasets.models import DatasetSchema  # TODO why import here?

        from nodes.models import InstanceConfig

        self.model = DatasetSchema
        self.instance_admin_role = instance_admin_role
        self.instance_viewer_role = instance_viewer_role
        self.ic_model = InstanceConfig
        super().__init__(DatasetSchema)

    def is_create_context_valid(self, context: Any) -> TypeGuard[InstanceConfig]:
        from nodes.models import InstanceConfig
        return isinstance(context, InstanceConfig)

    def get_instance_configs_for_schema(self, schema: DatasetSchema) -> list[int]:
        """Get IDs of all InstanceConfigs this schema is scoped for."""
        from nodes.models import InstanceConfig
        ic_content_type = ContentType.objects.get_for_model(InstanceConfig)
        return list(schema.scopes.filter(
            scope_content_type=ic_content_type
        ).values_list('scope_id', flat=True))

    def user_has_perm(self, user: User, action: BaseObjectAction, obj: DatasetSchema) -> bool:
        active_instance = realm_context.get().realm
        instance_ids = self.get_instance_configs_for_schema(obj)
        if active_instance is None or active_instance.pk not in instance_ids:
            return False
        # TODO HERE
        # Check if user is admin for any of the instances
        if user.has_instance_role(self.instance_admin_role, active_instance):
            return True
        # For view permission, check if user is a viewer for any of the instances
        if action == 'view':
            return user.has_instance_role(self.instance_viewer_role, active_instance)
        return False

    def anon_has_perm(self, action: BaseObjectAction, obj: DatasetSchema) -> bool:
        return False

    def user_can_create(self, user: User, context: InstanceConfig) -> bool:
        return user.has_instance_role(self.instance_admin_role, context)

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
            return admin_q | viewer_q

        return admin_q

    def construct_perm_q_anon(self, action: BaseObjectAction) -> Q | None:
        return None


class DatasetPermissionPolicy(ParentInheritedPolicy[Dataset, DatasetSchema, DatasetQuerySet]):
    """Permission policy for Dataset, inheriting from its schema."""

    def __init__(self):
        from kausal_common.datasets.models import Dataset, DatasetSchema
        super().__init__(Dataset, DatasetSchema, 'schema')

    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: Dataset) -> bool:
        parent_obj = self.get_parent_obj(obj)
        return self.parent_policy.user_has_perm(user, action, parent_obj)

    def anon_has_perm(self, action: BaseObjectAction, obj: Dataset) -> bool:
        return False

    def user_can_create(self, user: User, context: DatasetSchema) -> bool:
        return self.parent_policy.user_has_perm(user, 'change', context)


class DataPointPermissionPolicy(ParentInheritedPolicy[DataPoint, Dataset, PermissionedQuerySet[DataPoint]]):
    """Permission policy for DataPoint, inheriting from Dataset."""

    def __init__(self):
        from kausal_common.datasets.models import DataPoint, Dataset
        super().__init__(DataPoint, Dataset, 'dataset')

    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: DataPoint) -> bool:
        parent_obj = self.get_parent_obj(obj)
        return self.parent_policy.user_has_perm(user, action, parent_obj)

    def anon_has_perm(self, action: BaseObjectAction, obj: DataPoint) -> bool:
        return False

    def user_can_create(self, user: User, context: Dataset) -> bool:
        return self.parent_policy.user_has_perm(user, 'change', context)
