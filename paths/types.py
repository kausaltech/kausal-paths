from __future__ import annotations

from typing import TYPE_CHECKING
import typing

from django.http import HttpRequest
from django.contrib.auth.models import AnonymousUser

if TYPE_CHECKING:
    from users.models import User
    from nodes.models import InstanceConfig
    from wagtail.models import Site


UserOrAnon: typing.TypeAlias = 'User | AnonymousUser'


class PathsRequest(HttpRequest):
    user: UserOrAnon


class PathsAuthenticatedRequest(HttpRequest):
    user: 'User'


class PathsAdminRequest(PathsAuthenticatedRequest):
    admin_instance: InstanceConfig
    _wagtail_site: Site


class PathsAPIRequest(PathsAuthenticatedRequest):
    pass
