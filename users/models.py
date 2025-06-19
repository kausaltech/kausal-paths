from __future__ import annotations

from collections.abc import Sequence  # noqa: TCH003
from django.utils import timezone
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Self, overload

from django.db import models
from django.db.models import Model
from django.utils.translation import gettext_lazy as _
from pydantic import BaseModel, Field

from django_pydantic_field import SchemaField

from .base import AbstractUser, UserManager

if TYPE_CHECKING:
    from django.contrib.auth.models import Group

    from kausal_common.models.roles import InstanceSpecificRole, UserPermissionCache
    from kausal_common.models.types import FK, QS

    from frameworks.roles import FrameworkRoleDef
    from nodes.models import InstanceConfig, InstanceConfigQuerySet


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

    def deactivate(self, admin_user):
        self.is_active = False
        self.deactivated_by = admin_user
        self.deactivated_at = timezone.now()
        self.save()

