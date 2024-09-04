from __future__ import annotations

import abc
import typing
from typing import TYPE_CHECKING, Self

from django.db import models
from django.http import HttpRequest

if TYPE_CHECKING:
    from django.contrib.auth.models import AnonymousUser
    from wagtail.models import Site

    from paths.cache import PathsObjectCache
    from paths.permissions import PathsPermissionPolicy

    from nodes.models import InstanceConfig
    from users.models import User


type UserOrAnon = 'User | AnonymousUser'


class PathsRequest(HttpRequest):
    user: UserOrAnon
    cache: PathsObjectCache
    correlation_id: str
    """Randomly generated ID for correlation."""


class PathsAuthenticatedRequest(PathsRequest):
    user: 'User'


class PathsAdminRequest(PathsAuthenticatedRequest):
    admin_instance: InstanceConfig
    _wagtail_site: Site | None


class PathsAPIRequest(PathsAuthenticatedRequest):
    wildcard_domains: list[str] | None


class PathsModel(models.Model):
    class Meta:
        abstract = True

    @classmethod
    @abc.abstractmethod
    def permission_policy(cls) -> PathsPermissionPolicy[Self]: ...
