from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import TypeVar

from django.db.models import QuerySet
from rest_framework.permissions import DjangoModelPermissions

from kausal_common.models.permission_policy import ParentInheritedPolicy

if TYPE_CHECKING:
    from paths.types import PathsModel


_M = TypeVar('_M', bound='PathsModel')
_QS = TypeVar('_QS', bound=QuerySet, default=QuerySet[_M])
CreateContext = TypeVar('CreateContext', default=Any)


_ParentM = TypeVar('_ParentM', bound='PathsModel')


class PathsParentPolicy(ParentInheritedPolicy[_M, _ParentM, _QS]):
    pass


class PathsAPIPermission(DjangoModelPermissions):
    pass
