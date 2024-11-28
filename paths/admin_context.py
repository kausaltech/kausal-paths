from __future__ import annotations

import typing
from contextvars import ContextVar

if typing.TYPE_CHECKING:
    from paths.types import PathsAdminRequest

    from nodes.models import InstanceConfig


def set_admin_instance(ic: InstanceConfig, request: PathsAdminRequest | None = None):
    if request:
        assert not hasattr(request, 'admin_instance')
        request.admin_instance = ic
