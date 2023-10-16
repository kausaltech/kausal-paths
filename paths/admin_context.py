from __future__ import annotations

import typing
from contextvars import ContextVar

from paths.types import PathsAdminRequest

if typing.TYPE_CHECKING:
    from nodes.models import InstanceConfig


admin_instance: ContextVar[InstanceConfig] = ContextVar('admin_context')


def get_admin_instance() -> InstanceConfig:
    return admin_instance.get()


def set_admin_instance(ic: InstanceConfig, request: PathsAdminRequest | None = None):
    admin_instance.set(ic)
    if request:
        assert not hasattr(request, 'admin_instance')
        request.admin_instance = ic
