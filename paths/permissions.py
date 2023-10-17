from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Sequence, TypeVar, cast
from django.db.models import Model, QuerySet
from wagtail.permission_policies import ModelPermissionPolicy

if TYPE_CHECKING:
    from users.models import User


M = TypeVar('M', bound=Model)
QS = TypeVar('QS', bound=QuerySet, covariant=True)


class PathsPermissionPolicy(ModelPermissionPolicy, Generic[M, QS]):
    def __init__(self, model: type[M], auth_model=None):
        super().__init__(model, auth_model)

    def instances_user_has_any_permission_for(self, user: User, actions: Sequence[str]) -> QS:
        qs = cast(QS, super().instances_user_has_any_permission_for(user, actions))
        return qs
