from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeGuard, cast, overload
from typing_extensions import TypeVar

from django.contrib.auth.models import AnonymousUser
from django.db.models import Q, QuerySet
from rest_framework.permissions import DjangoModelPermissions
from wagtail.permission_policies.base import ModelPermissionPolicy

from loguru import logger
from social_core.utils import user_is_authenticated

from users.models import User

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kausal_common.graphene import GQLInfo
    from kausal_common.users import UserOrAnon

    from paths.types import PathsModel


_M = TypeVar('_M', bound='PathsModel')
_QS = TypeVar('_QS', bound=QuerySet, default=QuerySet[_M])
CreateContext = TypeVar('CreateContext', default=Any)

type BaseObjectAction = Literal['view', 'add', 'change', 'delete']
type ObjectSpecificAction = Literal['view', 'change', 'delete']

def is_base_action(action: str) -> TypeGuard[ObjectSpecificAction]:
    return action in ('view', 'change', 'delete')


class PathsPermissionPolicy(Generic[_M, _QS, CreateContext], ABC, ModelPermissionPolicy[_M, User, Any]):
    public_fields: list[str]

    def __init__(self, model: type[_M]):
        super().__init__(model)
        pf = getattr(model, 'public_fields', None)
        self_pf = getattr(self, 'public_fields', None)
        if self_pf is None:
            if pf is not None:
                self_pf = list(pf)
            else:
                self_pf = []
        self.public_fields = self_pf

    def is_create_context_valid(self, context: Any) -> TypeGuard[CreateContext]:  # noqa: ANN401
        return False

    @staticmethod
    def user_is_authenticated(user: UserOrAnon | None) -> TypeGuard[User]:
        if user is None or isinstance(user, AnonymousUser):
            return False
        return user.is_authenticated and user.is_active

    @abstractmethod
    def construct_perm_q(self, user: User, action: ObjectSpecificAction) -> Q | None:
        """
        Construct a Q object for determining the permissions for the action for a user.

        Returns None if no objects are allowed, and Q otherwise.
        """

    @abstractmethod
    def construct_perm_q_anon(self, action: ObjectSpecificAction) -> Q | None:
        """
        Construct a Q object for determining the permissions for the action for anonymous users.

        Returns None no objects are allowed, and Q otherwise.
        """

    @abstractmethod
    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: _M) -> bool:
        """Check if user has permission to perform an action on an instance."""

    @abstractmethod
    def anon_has_perm(self, action: ObjectSpecificAction, obj: _M) -> bool:
        """Check if an unauthenticated user has permission to perform an action on an instance."""

    @abstractmethod
    def user_can_create(self, user: User, context: CreateContext) -> bool:
        """Check if user can create a new object."""

    def anon_can_create(self, context: CreateContext) -> bool:
        """Check if an unauthenticated user can create a new object."""
        return False

    def get_queryset(self) -> _QS:
        mgr = getattr(self.model, 'objects', self.model._default_manager)
        return cast(_QS, mgr.get_queryset())

    @overload
    def gql_action_allowed(
        self, info: GQLInfo, action: Literal['add'], obj: None = ..., context: CreateContext = ...,
    ) -> bool: ...

    @overload
    def gql_action_allowed(
        self, info: GQLInfo, action: ObjectSpecificAction, obj: _M = ..., context: None = ...,
    ) -> bool: ...

    def gql_action_allowed(
        self, info: GQLInfo, action: BaseObjectAction, obj: _M | None = None, context: CreateContext | None = None,
    ) -> bool:
        user = info.context.user
        if action == 'add':
            if not self.is_create_context_valid(context):
                raise TypeError("Invalid create context type for %s: %s" % (
                    type(self), context,
                ))
            if not self.user_is_authenticated(user):
                return self.anon_can_create(context)
            return self.user_can_create(user, context)

        if obj is None:
            return False
        if not self.user_is_authenticated(user):
            return self.anon_has_perm(action, obj)
        return self.user_has_perm(user, action, obj)

    def user_has_permission(self, user: UserOrAnon, action: str) -> bool:
        return super().user_has_permission(user, action)

    def user_has_any_permission(self, user: UserOrAnon, actions: Sequence[str]) -> bool:
        return any(self.user_has_permission(user, action) for action in actions)

    def _construct_q(self, user: UserOrAnon, action: ObjectSpecificAction) -> Q | None:
        if not self.user_is_authenticated(user):
            return self.construct_perm_q_anon(action)
        if user.is_superuser:
            return Q()
        return self.construct_perm_q(user, action)

    def filter_by_perm(self, qs: _QS, user: UserOrAnon, action: ObjectSpecificAction) -> _QS:
        q = self._construct_q(user, action)
        if q is None:
            return qs.none()
        return qs.filter(q).distinct()

    def instances_user_has_permission_for(self, user: UserOrAnon, action: str) -> _QS:
        return self.instances_user_has_any_permission_for(user, [action])

    def instances_user_has_any_permission_for(self, user: UserOrAnon, actions: Sequence[str]) -> _QS:
        qs = self.get_queryset()
        if user_is_authenticated(user) and user.is_superuser:
            return qs
        filters = None
        for action in actions:
            if not is_base_action(action):
                logger.error("Unknown action: %s" % action)
                return qs.none()
            q = self._construct_q(user, action)
            if q is None:
                continue
            if filters is None:
                filters = q
            else:
                filters |= q
        if filters is None:
            return qs.none()
        return qs.filter(filters).distinct()

    def user_has_permission_for_instance(self, user: UserOrAnon, action: str, instance: _M) -> bool:
        if not is_base_action(action):
            logger.error("Unknown action: %s" % action)
            return False
        if not self.user_is_authenticated(user):
            return self.anon_has_perm(action, instance)
        return self.user_has_perm(user, action, instance)

    def is_field_visible(self, instance: _M, field_name: str, user: UserOrAnon | None) -> bool:
        if field_name in self.public_fields:
            return True
        if not self.user_is_authenticated(user):
            return False
        return self.user_has_any_permission_for_instance(user, ['change', 'add', 'delete'], instance)


class PathsReadOnlyPolicy(PathsPermissionPolicy[_M, _QS, CreateContext]):
    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        if action == 'view':
            return Q()
        return None

    def construct_perm_q_anon(self, action: BaseObjectAction) -> Q | None:
        if action == 'view':
            return Q()
        return None

    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: _M) -> bool:
        return action == 'view'

    def anon_has_perm(self, action: ObjectSpecificAction, obj: _M) -> bool:
        return action == 'view'

    def user_can_create(self, user: User, context: CreateContext) -> bool:
        return False


class PathsAPIPermission(DjangoModelPermissions):
    pass
