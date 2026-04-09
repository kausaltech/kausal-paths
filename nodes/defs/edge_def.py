from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from paths.refs import DimensionCategoryRef, DimensionRef


class SelectCategoriesTransformation(BaseModel):
    """Filter or select categories within a dimension, optionally flattening afterward."""

    kind: Literal['select_categories'] = 'select_categories'
    dimension: DimensionRef
    categories: list[DimensionCategoryRef] = Field(default_factory=list)
    flatten: bool = False
    exclude: bool = False


class AssignCategoryTransformation(BaseModel):
    """Assign a fixed category to a (possibly new) dimension."""

    kind: Literal['assign_category'] = 'assign_category'
    dimension: DimensionRef
    category: DimensionCategoryRef


class FlattenTransformation(BaseModel):
    """Flatten (sum over) a dimension."""

    kind: Literal['flatten'] = 'flatten'
    dimension: DimensionRef


EdgeTransformation = SelectCategoriesTransformation | AssignCategoryTransformation | FlattenTransformation
