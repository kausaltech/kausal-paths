import typing

from django.http import HttpRequest
from nodes.models import InstanceConfig

if typing.TYPE_CHECKING:
    from users.models import User


class PathsRequest(HttpRequest):
    admin_instance: InstanceConfig


class PathsAuthenticatedRequest(HttpRequest):
    user: 'User'


class APIRequest(PathsAuthenticatedRequest):
    pass
