"""
Resolve declared shapes into the effective shapes the runtime checks against.

Inheritance is resolved here and nowhere else: a stored shape names the shapes it inherits,
and its effective combinations and required groups are computed from whatever those resolve
to in the composed instance. See `docs/architecture/shapes.md`.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from nodes.defs.shape_defs import ShapeSpec
    from nodes.value_validation import QualifierRequirement


class _CategorySource(Protocol):
    def get_cat_ids(self) -> set[str]: ...


@dataclass(frozen=True, slots=True)
class EffectiveCombination:
    uuid: UUID
    identifier: str | None
    categories: Mapping[str, str]
    origin: UUID
    """The shape that declared this combination."""

    @property
    def key(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.categories.items()))


@dataclass(frozen=True, slots=True)
class EffectiveRequiredGroup:
    uuid: UUID
    identifier: str | None
    combinations: tuple[UUID, ...]
    qualifiers: Mapping[str, QualifierRequirement]
    origin: UUID


@dataclass(frozen=True, slots=True)
class EffectiveShape:
    spec: ShapeSpec
    dimensions: tuple[str, ...]
    combinations: tuple[EffectiveCombination, ...]
    required: tuple[EffectiveRequiredGroup, ...]
    redundant: tuple[UUID, ...]
    """Own combinations that an inherited shape already declares; kept out of `combinations`."""

    @property
    def uuid(self) -> UUID:
        return self.spec.uuid

    @property
    def closed(self) -> bool:
        return self.spec.closed

    def combination_keys(self) -> set[tuple[tuple[str, str], ...]]:
        return {combination.key for combination in self.combinations}


class ShapeResolutionError(ValueError):
    pass


def resolve_shapes(
    shapes: Sequence[ShapeSpec], dimensions: Mapping[str, _CategorySource] | None = None
) -> dict[UUID, EffectiveShape]:
    """
    Resolve every shape's inheritance; raise on the first inconsistency.

    With `dimensions`, every category a combination names must exist in its dimension.
    """
    by_uuid: dict[UUID, ShapeSpec] = {}
    for shape in shapes:
        if shape.uuid in by_uuid:
            raise ShapeResolutionError(f'Shape {shape.label} is declared twice')
        by_uuid[shape.uuid] = shape
    resolved: dict[UUID, EffectiveShape] = {}
    for shape in shapes:
        _resolve(shape.uuid, by_uuid, resolved, (), dimensions)
    return resolved


def _resolve(  # noqa: C901, PLR0912
    uuid: UUID,
    by_uuid: Mapping[UUID, ShapeSpec],
    resolved: dict[UUID, EffectiveShape],
    chain: tuple[UUID, ...],
    dimensions: Mapping[str, _CategorySource] | None,
) -> EffectiveShape:
    if uuid in resolved:
        return resolved[uuid]
    shape = by_uuid[uuid]
    if uuid in chain:
        names = ' -> '.join(by_uuid[item].label for item in (*chain, uuid))
        raise ShapeResolutionError(f'Shape inheritance forms a cycle: {names}')
    parents: list[EffectiveShape] = []
    for parent_uuid in shape.inherits:
        if parent_uuid not in by_uuid:
            raise ShapeResolutionError(f'Shape {shape.label} inherits from {parent_uuid}, which is not declared')
        parents.append(_resolve(parent_uuid, by_uuid, resolved, (*chain, uuid), dimensions))

    shape_dimensions = tuple(shape.dimensions) or parents[0].dimensions
    for parent in parents:
        if set(parent.dimensions) != set(shape_dimensions):
            msg = (
                f'Shape {shape.label} constrains {sorted(shape_dimensions)} '
                f'but inherits {parent.spec.label}, which constrains {sorted(parent.dimensions)}'
            )
            raise ShapeResolutionError(msg)

    combinations: dict[tuple[tuple[str, str], ...], EffectiveCombination] = {}
    aliases: dict[UUID, UUID] = {}
    for parent in parents:
        for combination in parent.combinations:
            combinations.setdefault(combination.key, combination)
    redundant: list[UUID] = []
    for own in shape.combinations:
        if set(own.categories) != set(shape_dimensions):
            raise ShapeResolutionError(
                f'Shape {shape.label}: combination {own.identifier or own.uuid} must name exactly {sorted(shape_dimensions)}'
            )
        if dimensions is not None:
            for dimension_id, category_id in own.categories.items():
                dimension = dimensions.get(dimension_id)
                if dimension is None:
                    raise ShapeResolutionError(f'Shape {shape.label}: unknown dimension {dimension_id}')
                if category_id not in dimension.get_cat_ids():
                    raise ShapeResolutionError(f'Shape {shape.label}: dimension {dimension_id} has no category {category_id}')
        effective = EffectiveCombination(own.uuid, own.identifier, dict(own.categories), shape.uuid)
        existing = combinations.get(effective.key)
        if existing is not None:
            # An inherited shape declares the tuple too; its entry wins.
            redundant.append(own.uuid)
            aliases[own.uuid] = existing.uuid
            continue
        combinations[effective.key] = effective

    known = {combination.uuid for combination in combinations.values()}
    required: list[EffectiveRequiredGroup] = [group for parent in parents for group in parent.required]
    seen_groups = {group.uuid for group in required}
    for group in shape.required:
        if group.uuid in seen_groups:
            continue
        members: list[UUID] = []
        for member in group.combinations:
            canonical = aliases.get(member, member)
            if canonical not in members:
                members.append(canonical)
        unknown = [str(member) for member in members if member not in known]
        if unknown:
            raise ShapeResolutionError(
                f'Shape {shape.label}: required group {group.identifier or group.uuid} names unknown combinations {unknown}'
            )
        required.append(EffectiveRequiredGroup(group.uuid, group.identifier, tuple(members), dict(group.qualifiers), shape.uuid))
        seen_groups.add(group.uuid)

    result = EffectiveShape(
        spec=shape,
        dimensions=shape_dimensions,
        combinations=tuple(combinations.values()),
        required=tuple(required),
        redundant=tuple(redundant),
    )
    resolved[uuid] = result
    return result
