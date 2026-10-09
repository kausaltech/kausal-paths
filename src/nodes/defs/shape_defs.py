"""
Declared category shapes: named sets of category combinations and the groups among them that are required.

See `docs/architecture/shapes.md`. These are the stored declarations of the instance spec;
`nodes.shapes` resolves their inheritance into the effective shapes the runtime uses.
"""

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kausal_common.i18n.pydantic import I18nBaseModel, I18nStringInstance

from paths.identifiers import (
    DimensionCategoryIdentifier,
    DimensionIdentifier,
    ShapeCombinationId,
    ShapeId,
    ShapeIdentifier,
    ShapeRequiredGroupId,
)
from paths.refs import ShapeCombinationRef, ShapeRef

from nodes.value_validation import QualifierRequirement

if TYPE_CHECKING:
    from uuid import UUID

type ShapeOwner = Literal['framework', 'instance']
"""
Who may change a shape that a framework template declares.

`framework` shapes are read-only in dependent instances. An `instance` shape is an extension
point: every dependent instance holds its own record of it and adds its own combinations.
"""


class ShapeCombinationSpec(BaseModel):
    """One category tuple of a shape, with its own identity for required groups and entry rows."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    uuid: ShapeCombinationId
    identifier: str | None = None
    categories: dict[DimensionIdentifier, DimensionCategoryIdentifier]


class ShapeRequiredGroupSpec(BaseModel):
    """A requirement satisfied when any one of its combinations has a value meeting the qualifier bounds."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    uuid: ShapeRequiredGroupId
    identifier: str | None = None
    combinations: list[ShapeCombinationRef] = Field(min_length=1)
    qualifiers: dict[str, QualifierRequirement] = Field(default_factory=dict)


class ShapeSpec(I18nBaseModel):
    """A shape as one instance declares it; inherited content is not copied in."""

    model_config = ConfigDict(extra='forbid')

    uuid: ShapeId
    identifier: ShapeIdentifier | None = None
    name: I18nStringInstance | None = None
    owner: ShapeOwner = 'framework'
    dimensions: list[DimensionIdentifier] = Field(default_factory=list)
    """Required unless the shape inherits; a child constrains the same dimensions as its parents."""
    inherits: list[ShapeRef] = Field(default_factory=list)
    closed: bool = False
    combinations: list[ShapeCombinationSpec] = Field(default_factory=list)
    required: list[ShapeRequiredGroupSpec] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_own_content(self) -> ShapeSpec:
        if not self.dimensions and not self.inherits:
            raise ValueError(f'Shape {self.label}: declare its dimensions or the shapes it inherits')
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError(f'Shape {self.label}: dimensions repeat')
        if self.uuid in self.inherits:
            raise ValueError(f'Shape {self.label} inherits from itself')
        uuids = [combination.uuid for combination in self.combinations]
        if len(set(uuids)) != len(uuids):
            raise ValueError(f'Shape {self.label}: combination identities repeat')
        tuples = [tuple(sorted(combination.categories.items())) for combination in self.combinations]
        if len(set(tuples)) != len(tuples):
            raise ValueError(f'Shape {self.label}: a category combination is declared twice')
        if self.dimensions:
            for combination in self.combinations:
                if set(combination.categories) != set(self.dimensions):
                    name = combination.identifier or combination.uuid
                    msg = f'Shape {self.label}: combination {name} must name exactly the dimensions {sorted(self.dimensions)}'
                    raise ValueError(msg)
        groups = [group.uuid for group in self.required]
        if len(set(groups)) != len(groups):
            raise ValueError(f'Shape {self.label}: required group identities repeat')
        return self

    @property
    def label(self) -> str:
        return self.identifier or str(self.uuid)

    def fixed_fields(self) -> tuple[tuple[str, ...], tuple[UUID, ...], bool]:
        """Return the fields an instance-owned record keeps as the template declares them."""
        return tuple(self.dimensions), tuple(self.inherits), self.closed
