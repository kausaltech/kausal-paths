"""Strawberry GraphQL types for instance-scoped dimensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import strawberry as sb

from kausal_common.strawberry.ordering import with_sibling_ids

if TYPE_CHECKING:
    from kausal_common.datasets.models import (
        DimensionCategory as DimensionCategoryModel,
        DimensionScope,
    )


@sb.type(name='InstanceDimensionCategory')
class DimensionCategoryType:
    """A category within an instance dimension."""

    id: sb.ID
    identifier: str | None
    label: str
    order: int
    previous_sibling: sb.ID | None
    next_sibling: sb.ID | None

    @classmethod
    def from_model(
        cls,
        cat: DimensionCategoryModel,
        *,
        previous_sibling: sb.ID | None = None,
        next_sibling: sb.ID | None = None,
    ) -> DimensionCategoryType:
        return cls(
            id=sb.ID(str(cat.uuid)),
            identifier=cat.identifier,
            label=cat.label_i18n or str(cat.uuid),
            order=cat.order,
            previous_sibling=previous_sibling,
            next_sibling=next_sibling,
        )


@sb.type(name='InstanceDimension')
class DimensionType:
    """A dimension scoped to a model instance."""

    id: sb.ID
    identifier: str
    name: str
    categories: list[DimensionCategoryType]

    @classmethod
    def from_scope(cls, scope: DimensionScope) -> DimensionType:
        dim = scope.dimension
        models = list(dim.categories.all())

        def _cat_id(c: DimensionCategoryModel) -> sb.ID:
            return sb.ID(str(c.uuid))

        cats = [
            DimensionCategoryType.from_model(cat, previous_sibling=prev_id, next_sibling=next_id)
            for cat, prev_id, next_id in with_sibling_ids(models, _cat_id)
        ]
        return cls(
            id=sb.ID(str(dim.uuid)),
            identifier=scope.identifier or '',
            name=dim.name_i18n or str(dim.uuid),
            categories=cats,
        )
