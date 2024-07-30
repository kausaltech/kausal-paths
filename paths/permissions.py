from __future__ import annotations

import abc
from typing import TYPE_CHECKING, TypeGuard, TypeVar
from django.contrib.auth.models import AnonymousUser
from django.db.models import Model, QuerySet
from wagtail.permission_policies.base import ModelPermissionPolicy
from rest_framework.permissions import DjangoModelPermissions

from kausal_common.users import UserOrAnon

if TYPE_CHECKING:
    from users.models import User
    from nodes.models import InstanceConfig


M = TypeVar('M', bound=Model)
QS = TypeVar('QS', bound=QuerySet)

ModelPermissionPolicy.__class_getitem__ = classmethod(lambda cls, *args, **kwargs: cls)  # type: ignore

class PathsPermissionPolicy(ModelPermissionPolicy[M, QS, 'User', QuerySet['User']]):
    public_fields: list[str]

    def __init__(self, model: type[M]):
        super().__init__(model)
        pf = getattr(model, 'public_fields', None)
        self_pf = getattr(self, 'public_fields', None)
        if self_pf is None:
            if pf is not None:
                self_pf = list(pf)
            else:
                self_pf = []
        self.public_fields = self_pf

    def user_is_authenticated(self, user: UserOrAnon | None) -> TypeGuard[User]:
        if user is None or isinstance(user, AnonymousUser):
            return False
        return user.is_authenticated and user.is_active

    def is_field_visible(self, instance: M, field_name: str, user: UserOrAnon | None) -> bool:
        if field_name in self.public_fields:
            return True
        if not self.user_is_authenticated(user):
            return False
        return self.user_has_any_permission_for_instance(user, ['change', 'add', 'delete'], instance)


class InstanceConfigRelatedPermissionPolicy(PathsPermissionPolicy[M, QS], abc.ABC):
    instance_field_name: str = 'instance'

    @abc.abstractmethod
    def get_instance_config(self, obj: M) -> InstanceConfig: ...


class PathsAPIPermission(DjangoModelPermissions):
    pass
