from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Self, overload

from django.db import models
from django.db.models import Model
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from modelcluster.fields import ParentalKey
from modelcluster.models import ClusterableModel
from pydantic import BaseModel, Field

from django_pydantic_field import SchemaField

from kausal_common.datasets.models import DatasetSchema
from kausal_common.models.roles import role_registry
from kausal_common.models.types import ModelManager
from kausal_common.users.models import create_permission_membership_models

from paths.const import NONE_ROLE, PathsRoleIdentifier
from paths.types import PathsModel, PathsQuerySet

from .base import AbstractUser, UserManager

if TYPE_CHECKING:
    from django.contrib.auth.models import Group

    from kausal_common.models.roles import InstanceSpecificRole, UserPermissionCache
    from kausal_common.models.types import FK, M2M, QS

    from frameworks.roles import FrameworkRoleDef
    from nodes.models import InstanceConfig, InstanceConfigQuerySet
    from orgs.models import Organization
    from people.models import Person
    from users.permissions import UserGroupPermissionPolicy


class UserFrameworkRole(BaseModel):
    framework_id: str
    role_id: str


class UserExtra(BaseModel):
    framework_roles: Sequence[FrameworkRoleDef] = Field(default_factory=list)

    def set_framework_role(self, role: FrameworkRoleDef):
        self.framework_roles = list(filter(
            lambda role: role.framework_id != role.framework_id,
            self.framework_roles,
        ))
        self.framework_roles.append(role)

    @classmethod
    def get_default(cls) -> Self:
        from frameworks.roles import FrameworkRoleDef  # noqa: F401
        cls.model_rebuild()
        return cls()


class User(AbstractUser):
    selected_instance: FK[InstanceConfig | None] = models.ForeignKey(
        'nodes.InstanceConfig', null=True, blank=True, on_delete=models.SET_NULL,
    )
    email = models.EmailField(_('email address'), unique=True)
    extra: UserExtra = SchemaField(schema=UserExtra, default=UserExtra.get_default)

    objects: ClassVar[UserManager[User]]

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    autocomplete_search_field = 'email'

    person: Person

    def natural_key(self) -> tuple[str]:
        # If we don't override this, it will use `get_username()`, which may not always return the email field. The
        # manager's `get_by_natural_key()`, on the other hand, will expect that the natural key is the email field since
        # we specified `USERNAME_FIELD = 'email'`. We can't just override `get_by_natural_key()` because, if I remember
        # correctly, in some places, Django expects this to actually match with field specified in `USERNAME_FIELD`.
        return (self.email,)

    def get_adminable_instances(self) -> InstanceConfigQuerySet:
        from nodes.models import InstanceConfig
        return InstanceConfig.permission_policy().adminable_instances(self)

    def user_is_admin_for_instance(self, instance_config: InstanceConfig) -> bool:
        from nodes.models import InstanceConfig
        return InstanceConfig.permission_policy().user_has_permission_for_instance(self, 'change', instance_config)

    def get_corresponding_person(self) -> Person | None:
        # Copied from KW. We don't have a cache here yet.
        # cache = self.get_cache()
        # if hasattr(cache, '_corresponding_person'):
        #     return cache._corresponding_person

        from people.models import Person

        try:
            person = self.person
        except Person.DoesNotExist:
            person = None

        if person is None:
            person = Person.objects.filter(email__iexact=self.email).first()
        # cache._corresponding_person = person
        return person

    @cached_property
    def cgroups(self) -> QS[Group]:
        return self.groups.all()

    @cached_property
    def perms(self) -> UserPermissionCache:
        from kausal_common.models.roles import UserPermissionCache
        return UserPermissionCache(self)

    @overload
    def has_instance_role[M: Model](self, role: InstanceSpecificRole[M], obj: M) -> bool: ...

    @overload
    def has_instance_role(self, role: PathsRoleIdentifier, obj: Model) -> bool: ...

    def has_instance_role(self, role: PathsRoleIdentifier | InstanceSpecificRole[Any], obj: Model) -> bool:
        return self.perms.has_instance_role(role, obj)

    def has_instance_role_in_any_instance(self, role_id: PathsRoleIdentifier) -> bool:
        role = role_registry.get_role(role_id)
        return self.perms.has_instance_role_in_any_instance(role)

    def has_instance_role_with_id(self, role_id: PathsRoleIdentifier, obj: Model) -> bool:
        role = role_registry.get_role(role_id)
        return self.has_instance_role(role, obj)

    def get_role_for_instance(self, active_instance: InstanceConfig) -> InstanceSpecificRole[InstanceConfig] | None:
        user_groups = set(self.groups.all())
        for role in role_registry.get_all_roles():
            model_meta = role.model._meta
            app_label, object_name = (model_meta.app_label, model_meta.object_name)
            if (app_label, object_name) != ('nodes', 'InstanceConfig'):
                continue
            group = role.get_existing_instance_group(active_instance)
            if group and group in user_groups:
                return role
        return None

    def sync_instance_groups_with_role(self, role_id: PathsRoleIdentifier, instance: InstanceConfig) -> None:
        """
        Verify that the user has exactly the instance role group corresponding to this role_id.

        Only modifies group memberships of groups connected to active_instance.
        """
        for role_obj in role_registry.get_all_roles():
            group = role_obj.get_existing_instance_group(instance)
            if group:
                self.groups.remove(group)

        if role_id == NONE_ROLE:
            return

        role_obj = role_registry.get_role(role_id)
        group = role_obj.get_existing_instance_group(instance)
        if group:
            self.groups.add(group)

    def can_access_admin(self) -> bool:
        if not self.is_active:
            return False
        if not self.is_staff:
            return False
        return True

    def deactivate(self, admin_user):
        self.is_active = False
        self.deactivated_by = admin_user
        self.deactivated_at = timezone.now()
        self.save()

    def can_create_organization(self) -> bool:
        return self.is_superuser

    def can_modify_organization(self, organization: Organization) -> bool:
        return self.is_superuser

    def can_delete_organization(self, organization: Organization) -> bool:
        return self.is_superuser

    def can_edit_or_delete_person_within_instance(
            self, person: Person, instance_config: InstanceConfig) -> bool:
        return self.is_superuser

    def can_create_person(self) -> bool:
        return self.is_superuser

    def can_modify_person(self, person: Person) -> bool:
        return self.is_superuser


class UserGroupQuerySet(PathsQuerySet['UserGroup']):
    pass


_UserGroupManager = models.Manager.from_queryset(UserGroupQuerySet)
class UserGroupManager(ModelManager['UserGroup', UserGroupQuerySet], _UserGroupManager):
    """Model manager for UserGroup."""
del _UserGroupManager


class UserGroup(PathsModel, ClusterableModel):
    """
    Group of users for various purposes such as assigning permissions on certain models or model instances.

    In contrast to Django groups, names don't have to be globally unique.
    """

    instance: FK[InstanceConfig] = models.ForeignKey(
        'nodes.InstanceConfig', on_delete=models.CASCADE, related_name='user_groups'
    )
    name = models.CharField(max_length=200)
    users: M2M[User, UserGroupMember] = models.ManyToManyField(
        User,
        through='UserGroupMember',
        related_name='user_groups',
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['instance', 'name'], name='unique_user_group_name_per_instance'),
        ]

    @classmethod
    def permission_policy(cls) -> UserGroupPermissionPolicy:
        from .permissions import UserGroupPermissionPolicy
        return UserGroupPermissionPolicy()

    def __str__(self) -> str:
        return self.name


class UserGroupMember(models.Model):
    group = ParentalKey(UserGroup, on_delete=models.CASCADE, related_name='users_edges')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='groups_edges')

    class Meta:
        verbose_name = _('Group member')
        verbose_name_plural = _('Group members')

    def __str__(self) -> str:
        return f'{self.user} ∈ {self.group}'


# Create permission membership models here, in the `users` app, since they will be part of this app. If you call
# `create_permission_membership_models` in a different app, `shell_plus` will get confused.
DatasetSchemaGroupPermission, DatasetSchemaUserPermission = create_permission_membership_models(DatasetSchema)
