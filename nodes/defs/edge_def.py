from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SelectCategoriesTransformation(BaseModel):
    """Filter or select categories within a dimension."""

    kind: Literal['select_categories'] = 'select_categories'
    dimension: str
    categories: list[str] = Field(default_factory=list)
    flatten: bool = False
    exclude: bool = False


class AssignCategoryTransformation(BaseModel):
    """Assign a category to a dimension (adds a dimension if not present)."""

    kind: Literal['assign_category'] = 'assign_category'
    dimension: str
    category: str


class FlattenTransformation(BaseModel):
    """Flatten (sum over) a dimension."""

    kind: Literal['flatten'] = 'flatten'
    dimension: str


EdgeTransformation = SelectCategoriesTransformation | AssignCategoryTransformation | FlattenTransformation
