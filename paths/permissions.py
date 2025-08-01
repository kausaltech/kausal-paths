from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import TypeVar

from django.db.models import QuerySet
from rest_framework import permissions

from kausal_common.models.permission_policy import ParentInheritedPolicy

from paths.context import realm_context

from orgs.models import Organization
from people.models import Person

if TYPE_CHECKING:
    from paths.types import PathsAuthenticatedRequest, PathsModel

    from nodes.models import InstanceConfig
    from users.models import User

_M = TypeVar('_M', bound='PathsModel')
_QS = TypeVar('_QS', bound=QuerySet, default=QuerySet[_M])
CreateContext = TypeVar('CreateContext', default=Any)


_ParentM = TypeVar('_ParentM', bound='PathsModel')


class PathsParentPolicy(ParentInheritedPolicy[_M, _ParentM, _QS]):
    pass


class PathsAPIPermission(permissions.DjangoModelPermissions):
    pass

class ReadOnly(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.method in permissions.SAFE_METHODS


class OrganizationPermission(permissions.DjangoObjectPermissions):
    def check_permission(self, user: User, perm: str, organization: Organization | None = None):
        # Check for object permissions first
        if not user.has_perms([perm]):
            return False
        if perm == 'orgs.change_organization':
            if not organization or not user.can_modify_organization(organization=organization):
                return False
        elif perm == 'orgs.add_organization':
            if not user.can_create_organization():
                return False
        elif perm == 'orgs.delete_organization':
            if not organization or not user.can_delete_organization(organization=organization):
                return False
        else:
            return False
        return True

    def has_permission(self, request: PathsAuthenticatedRequest, view):
        perms = self.get_required_permissions(request.method, Organization)
        return all(self.check_permission(request.user, perm) for perm in perms)

    def has_object_permission(self, request, view, obj):
        perms = self.get_required_object_permissions(request.method, Organization)
        if not perms and request.method in permissions.SAFE_METHODS:
            return True
        return all(self.check_permission(request.user, perm, obj) for perm in perms)

class PersonPermission(permissions.DjangoObjectPermissions):
    def check_permission(
            self, user: User, perm: str, person: Person = None, instance_config: InstanceConfig = None):
        # Check for object permissions first
        if not user.has_perms([perm]):
            return False
        if perm == 'people.change_person':
            if not user.can_modify_person(person=person):
                return False
        elif perm == 'people.add_person':
            if not user.can_create_person():
                return False
        elif perm == 'people.delete_person':
            if person is None:
                #  Does the user have deletion rights in general
                if not user.is_superuser:
                    return False
            # Does the user have deletion rights to this person in this plan
            elif not user.can_edit_or_delete_person_within_instance(person, instance_config=instance_config):
                return False
        else:
            return False
        return True

    def has_permission(self, request: PathsAuthenticatedRequest, view):
        perms = self.get_required_permissions(request.method, Person)
        instance_config = realm_context.get().realm
        return all(self.check_permission(request.user, perm, instance_config=instance_config) for perm in perms)

    def has_object_permission(self, request: PathsAuthenticatedRequest, view, obj):
        perms = self.get_required_object_permissions(request.method, Person)
        instance_config = realm_context.get().realm
        if not perms and request.method in permissions.SAFE_METHODS:
            return True
        return all(self.check_permission(request.user, perm, person=obj, instance_config=instance_config) for perm in perms)
