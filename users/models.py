from __future__ import annotations

from collections.abc import Iterable, Sequence  # noqa: TC003
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Self, overload

from django.core.validators import validate_comma_separated_integer_list
from django.db import models
from django.db.models import Model
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from pydantic import BaseModel, Field

from django_pydantic_field import SchemaField

from kausal_common.models.roles import role_registry

from paths.const import NONE_ROLE, PathsRoleIdentifier

from .base import AbstractUser, UserManager

if TYPE_CHECKING:

    from django.contrib.auth.models import Group

    from kausal_common.models.roles import InstanceSpecificRole, UserPermissionCache
    from kausal_common.models.types import FK, QS, RevOne

    from frameworks.roles import FrameworkRoleDef
    from nodes.models import InstanceConfig, InstanceConfigQuerySet
    from orgs.models import Organization
    from people.models import Person


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

    # Used for quickly retrieving the instances the user can administer
    cached_adminable_instances = models.CharField(
        _('adminable instances'),
        validators=[validate_comma_separated_integer_list],
        null=True,
        blank=True,
    )

    objects: ClassVar[UserManager[User]]
    person: RevOne[User, Person] | None

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
        if self.is_superuser:
            return InstanceConfig.objects.qs
        return self.get_cached_adminable_instances()

    def set_cached_adminable_instances(self, instance_configs: InstanceConfigQuerySet, save: bool = True) -> str | None:
        if not instance_configs.exists():
            cached_value = None
        else:
            cached_value = ','.join(str(pk) for pk in instance_configs.distinct().values_list('pk', flat=True))
        self.cached_adminable_instances = cached_value
        if save:
            self.save(update_fields=('cached_adminable_instances',))
        return cached_value

    def get_cached_adminable_instances(self) -> InstanceConfigQuerySet:
        from nodes.models import InstanceConfig
        return InstanceConfig.objects.qs.filter(pk__in=self.get_cached_adminable_instance_pks())

    def user_is_admin_for_instance(self, instance_config: InstanceConfig) -> bool:
        if self.is_superuser:
            return True
        return instance_config.pk in self.get_cached_adminable_instance_pks()

    def get_cached_adminable_instance_pks(self) -> Iterable[int]:
        if getattr(self, 'cached_adminable_instances', None)  is not None:
            value = self.cached_adminable_instances
        else:
            value = self.refresh_adminable_instances(save=True)
        if value is None:
            return []
        return [int(x) for x in value.split(',')]

    def refresh_adminable_instances(self, save: bool = True) -> str | None:
        from nodes.models import InstanceConfig
        if self.is_superuser:
            # No sense in storing all of the instances; the get_adminable_instances and user_is_admin_for_instance
            # methods handle superusers as a special case
            self.cached_adminable_instances = ''
            return ''
        cached_value = self.set_cached_adminable_instances(
            InstanceConfig.permission_policy().adminable_instances(self),
            save=save
        )
        return cached_value

    def invalidate_adminable_instances_cache(self) -> None:
        self.cached_adminable_instances = None
        self.save(update_fields=['cached_adminable_instances'])

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

        Only modifies group memberships of groups connected to instance, leaving
        other groups intact.
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
        return self.get_adminable_instances().exists()

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
