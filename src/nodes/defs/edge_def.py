"""
The legacy edge transformation vocabulary.

Kept as a module so existing imports keep working; the types themselves live
with the rest of the vocabulary in `transform_def`.
"""

from __future__ import annotations

from .transform_def import (
    AssignCategoryTransformation,
    EdgeTransformation,
    FlattenTransformation,
    SelectCategoriesTransformation,
)

__all__ = [
    'AssignCategoryTransformation',
    'EdgeTransformation',
    'FlattenTransformation',
    'SelectCategoriesTransformation',
]
