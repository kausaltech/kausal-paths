"""
A binding's transformations, resolved to the steps that change what a value's shape can be.

The constraint solver propagates facts through them, the data-entry resolver translates
requirements back through them, and the shape check projects declared combinations forward.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from nodes.units import Unit


@dataclass(frozen=True, slots=True)
class FilterStep:
    dimension_id: UUID
    selection: frozenset[UUID] | None
    """Selected category UUIDs; ``None`` when the selection is unresolvable (e.g. groups)."""
    exclude: bool
    flatten: bool
    index: int


@dataclass(frozen=True, slots=True)
class AssignStep:
    dimension_id: UUID
    category_id: UUID | None
    index: int


@dataclass(frozen=True, slots=True)
class UnitStep:
    unit: Unit
    index: int


@dataclass(frozen=True, slots=True)
class OpaqueStep:
    reason: str
    index: int


type TransformStep = FilterStep | AssignStep | UnitStep | OpaqueStep
