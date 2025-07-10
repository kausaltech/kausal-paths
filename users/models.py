from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Self, overload

from django.db import models
from django.db.models import Model
from django.utils.translation import gettext_lazy as _
from modelcluster.fields import ParentalKey
from modelcluster.models import ClusterableModel
from pydantic import BaseModel, Field

from django_pydantic_field import SchemaField

from kausal_common.datasets.models import DatasetSchema
from kausal_common.models.types import ModelManager
from kausal_common.users.models import create_permission_membership_models

from paths.types import PathsModel, PathsQuerySet

from .base import AbstractUser, UserManager

if TYPE_CHECKING:
    from django.contrib.auth.models import Group

    from kausal_common.models.roles import InstanceSpecificRole, UserPermissionCache
    from kausal_common.models.types import FK, M2M, QS

    from frameworks.roles import FrameworkRoleDef
    from nodes.models import InstanceConfig, InstanceConfigQuerySet
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

    def natural_key(self) -> tuple[str]:
        # If we don't override this, it will use `get_username()`, which may not always return the email field. The
        # manager's `get_by_natural_key()`, on the other hand, will expect that the natural key is the email field since
        # we specified `USERNAME_FIELD = 'email'`. We can't just override `get_by_natural_key()` because, if I remember
        # correctly, in some places, Django expects this to actually match with field specified in `USERNAME_FIELD`.
        return (self.email,)

    def get_active_instance(self) -> InstanceConfig | None:
        # TODO
        return self.selected_instance

    def get_adminable_instances(self) -> InstanceConfigQuerySet:
        from nodes.models import InstanceConfig
        return InstanceConfig.permission_policy().adminable_instances(self)

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
    def has_instance_role(self, role: str, obj: Model) -> bool: ...

    def has_instance_role(self, role: str | InstanceSpecificRole[Any], obj: Model) -> bool:
        return self.perms.has_instance_role(role, obj)

    def can_access_admin(self) -> bool:
        if not self.is_active:
            return False
        if not self.is_staff:
            return False
        return True


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
