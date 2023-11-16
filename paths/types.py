from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
import typing

from django.db import models
from django.http import HttpRequest
from django.contrib.auth.models import AnonymousUser

from paths.permissions import PathsPermissionPolicy

if TYPE_CHECKING:
    from users.models import User
    from nodes.models import InstanceConfig
    from wagtail.models import Site
    from paths.cache import PathsObjectCache


UserOrAnon: typing.TypeAlias = 'User | AnonymousUser'


class PathsRequest(HttpRequest):
    user: UserOrAnon
    cache: PathsObjectCache


class PathsAuthenticatedRequest(PathsRequest):
    user: 'User'


class PathsAdminRequest(PathsAuthenticatedRequest):
    admin_instance: InstanceConfig
    _wagtail_site: Site


class PathsAPIRequest(PathsAuthenticatedRequest):
    pass


class PathsModel(models.Model):
    permission_policy: ClassVar[PathsPermissionPolicy]

    class Meta:
        abstract = True
