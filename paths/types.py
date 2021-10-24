from django.http import HttpRequest
from nodes.models import InstanceConfig


class PathsRequest(HttpRequest):
    admin_instance: InstanceConfig
