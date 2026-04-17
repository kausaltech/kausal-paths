from __future__ import annotations

from typing import Any

__all__ = [
    'DataPointType',
    'DatasetDimensionCategoryType',
    'DatasetDimensionType',
    'DatasetMetricType',
    'DatasetType',
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import types

        return getattr(types, name)
    raise AttributeError(name)
