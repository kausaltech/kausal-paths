from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Parameter

param_type_registry: set[type[Parameter]] = set()


def register_parameter_type(cls: type[Parameter]) -> None:
    if cls in param_type_registry:
        raise Exception('Parameter class %s already registered', str(cls))
    param_type_registry.add(cls)
