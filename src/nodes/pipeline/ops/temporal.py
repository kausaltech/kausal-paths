"""
Operations over the year axis.

These create values (they fill years), so they are pipeline operations rather
than binding transformations. They share their implementation with the
binding transformations of the same name (``nodes.transforms``): one
implementation per operation.
"""

from __future__ import annotations

from typing import Literal

from .base import InputOperationSpec


class InterpolateOperationSpec(InputOperationSpec):
    """Linearly fill the missing years between the first and last year of each series."""

    kind: Literal['interpolate'] = 'interpolate'


class ExtendOperationSpec(InputOperationSpec):
    """Carry the last historical value forward to the model end year."""

    kind: Literal['extend'] = 'extend'


class BackfillOperationSpec(InputOperationSpec):
    """Copy each category's first known value backwards over leading gaps."""

    kind: Literal['backfill'] = 'backfill'
