"""
Pydantic specs for extra fields in `Dimension.spec` / `DimensionCategory.spec`.

They're stored as JSONFields on the shared ORM models in `kausal_common/datasets/models.py`.

The ORM models hold what KP and KW agree on (identifier, label, ordering,
categories, scopes). These specs carry the KP-only gap: category groups,
regex-based group matching, aliases, colors, help text, and the
`is_internal` flag used by runtime-generated dimensions.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from kausal_common.i18n.pydantic import I18nBaseModel, I18nStringInstance

from paths.identifiers import Identifier

if TYPE_CHECKING:
    from nodes.dimensions import Dimension as RuntimeDimension, DimensionCategory as RuntimeDimensionCategory


class DimensionCategoryGroupSpec(I18nBaseModel):
    id: Identifier
    label: I18nStringInstance
    color: str | None = None
    order: int | None = None
    id_match: str | None = None

    @field_validator('id_match')
    @classmethod
    def validate_id_match(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not len(v):
            raise ValueError('zero length regex supplied')
        _ = re.match(v, '')
        return v


class DimensionSpec(I18nBaseModel):
    help_text: I18nStringInstance | None = None
    is_internal: bool = False
    groups: list[DimensionCategoryGroupSpec] = Field(default_factory=list)

    @classmethod
    def from_runtime(cls, dim: RuntimeDimension) -> DimensionSpec:
        return cls.model_validate(dim.model_dump(include={'help_text', 'is_internal', 'groups'}))

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode='json', exclude_defaults=True)


class DimensionCategorySpec(I18nBaseModel):
    color: str | None = None
    group: str | None = None
    aliases: list[str] = Field(default_factory=list)

    @classmethod
    def from_runtime(cls, cat: RuntimeDimensionCategory) -> DimensionCategorySpec:
        return cls.model_validate(cat.model_dump(include={'color', 'group', 'aliases'}))

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode='json', exclude_defaults=True)
