"""
GraphQL projection of an instance's declared category shapes; see docs/architecture/shapes.md.

A shape is listed with its own declaration and with its effective content, inheritance resolved:
each effective combination names the shape that declared it, so a client can tell a combination
the standard defines from one the municipality added.
"""

from enum import Enum
from typing import TYPE_CHECKING, Self
from uuid import UUID

import strawberry as sb

from nodes.models import PreferredInstanceSource

if TYPE_CHECKING:
    from collections.abc import Mapping

    from paths import gql

    from nodes.defs.shape_defs import ShapeSpec
    from nodes.instance_graph import InstanceGraph
    from nodes.models import InstanceConfig
    from nodes.shapes import EffectiveShape
    from nodes.value_validation import QualifierRequirement


@sb.enum(name='ShapeOwner', description='Who may change a shape that a framework template declares.')
class ShapeOwnerEnum(Enum):
    FRAMEWORK = 'framework'
    """Read-only in the instances that inherit it."""
    INSTANCE = 'instance'
    """An extension point: each inheriting instance keeps its own record of it and adds combinations."""


@sb.type(name='ShapeCoordinate', description='One dimension and category of a shape combination.')
class ShapeCoordinateType:
    dimension: str = sb.field(description='Dimension identifier.')
    category: str = sb.field(description='Category identifier.')
    dimension_id: UUID | None = sb.field(description='The dimension in this instance; null when it has no catalogue row.')
    category_id: UUID | None = sb.field(description='The category in this instance; null when it has no catalogue row.')


@sb.type(name='ShapeCombination', description='One category tuple of a shape.')
class ShapeCombinationType:
    id: UUID
    identifier: str | None
    coordinates: list[ShapeCoordinateType]
    origin_shape_id: UUID = sb.field(description='The shape that declares this combination.')


@sb.type(name='ShapeQualifierRequirement', description='A bound a required value must meet, on one qualifier path.')
class ShapeQualifierRequirementType:
    path: str
    min: float | None
    max: float | None
    present: bool


@sb.type(
    name='ShapeRequiredGroup',
    description='A requirement met in a year when one of its combinations has values that meet its qualifiers.',
)
class ShapeRequiredGroupType:
    id: UUID
    identifier: str | None
    combination_ids: list[UUID]
    qualifiers: list[ShapeQualifierRequirementType]
    origin_shape_id: UUID = sb.field(description='The shape that declares this group.')


@sb.type(name='Shape', description='A named set of category combinations, and the groups among them that are required.')
class ShapeType:
    id: UUID
    identifier: str | None
    name: str | None
    owner: ShapeOwnerEnum
    closed: bool = sb.field(description='Whether its combinations are the only ones allowed; an open shape is a minimum.')
    dimensions: list[str] = sb.field(description='Identifiers of the dimensions it constrains.')
    inherits: list[UUID] = sb.field(description='The shapes it extends, as declared.')
    is_editable: bool = sb.field(
        description=(
            "Whether the current user can change this instance's own declaration of it. An inherited "
            'shape is read-only; an extension point is editable, except for its dimensions, '
            'inheritance and closedness.'
        )
    )
    combinations: list[ShapeCombinationType] = sb.field(description='Its own combinations, inheritance not resolved.')
    required: list[ShapeRequiredGroupType] = sb.field(description='Its own required groups, inheritance not resolved.')
    effective_combinations: list[ShapeCombinationType] = sb.field(
        description='Its combinations with inheritance resolved, each with the shape that declares it.'
    )
    effective_required: list[ShapeRequiredGroupType] = sb.field(
        description='Its required groups with inheritance resolved, each with the shape that declares it.'
    )

    @classmethod
    def from_graph(cls, graph: InstanceGraph, *, editable: frozenset[UUID]) -> list[Self]:
        """List the graph's shapes; `editable` holds those the user may change."""
        coordinates = _Coordinates(graph)
        return [
            cls._build(spec, graph.shapes[spec.uuid], coordinates, is_editable=spec.uuid in editable)
            for spec in graph.spec.shapes
            if spec.uuid in graph.shapes
        ]

    @classmethod
    def for_instance(
        cls, info: gql.Info, config: InstanceConfig, source: PreferredInstanceSource | None = None
    ) -> tuple[InstanceGraph, frozenset[UUID]]:
        """Return the request's graph of the instance and the shapes the user may change in it."""
        resources = info.context.instance_resources
        assert resources is not None
        config, source = resources.resolve_source(config, source)
        graph = info.context.require_instance_graph(config, source=source)
        editable: frozenset[UUID] = frozenset()
        if source != PreferredInstanceSource.PUBLISHED and config.gql_action_allowed(info, 'change', raise_on_denied=False):
            editable = frozenset(shape.uuid for shape in config.ensure_spec().shapes)
        return graph, editable

    @classmethod
    def resolve(
        cls, info: gql.Info, config: InstanceConfig, shape_id: UUID | None, source: PreferredInstanceSource | None = None
    ) -> Self | None:
        """Build one shape from the request's graph, for a field that refers to it."""
        if shape_id is None:
            return None
        graph, editable = cls.for_instance(info, config, source)
        spec = next((spec for spec in graph.spec.shapes if spec.uuid == shape_id), None)
        if spec is None or shape_id not in graph.shapes:
            return None
        return cls._build(spec, graph.shapes[shape_id], _Coordinates(graph), is_editable=shape_id in editable)

    @classmethod
    def _build(cls, spec: ShapeSpec, effective: EffectiveShape, coordinates: _Coordinates, *, is_editable: bool) -> Self:
        return cls(
            id=spec.uuid,
            identifier=spec.identifier,
            name=str(spec.name) if spec.name is not None else None,
            owner=ShapeOwnerEnum(spec.owner),
            closed=spec.closed,
            dimensions=list(effective.dimensions),
            inherits=list(spec.inherits),
            is_editable=is_editable,
            combinations=[
                ShapeCombinationType(
                    id=combination.uuid,
                    identifier=combination.identifier,
                    coordinates=coordinates.of(combination.categories),
                    origin_shape_id=spec.uuid,
                )
                for combination in spec.combinations
            ],
            required=[
                ShapeRequiredGroupType(
                    id=group.uuid,
                    identifier=group.identifier,
                    combination_ids=list(group.combinations),
                    qualifiers=_qualifiers(group.qualifiers),
                    origin_shape_id=spec.uuid,
                )
                for group in spec.required
            ],
            effective_combinations=[
                ShapeCombinationType(
                    id=combination.uuid,
                    identifier=combination.identifier,
                    coordinates=coordinates.of(combination.categories),
                    origin_shape_id=combination.origin,
                )
                for combination in effective.combinations
            ],
            effective_required=[
                ShapeRequiredGroupType(
                    id=group.uuid,
                    identifier=group.identifier,
                    combination_ids=list(group.combinations),
                    qualifiers=_qualifiers(group.qualifiers),
                    origin_shape_id=group.origin,
                )
                for group in effective.required
            ],
        )


class _Coordinates:
    def __init__(self, graph: InstanceGraph) -> None:
        self.catalogue = {
            dimension.identifier: (dimension.id, {category.identifier: category.id for category in dimension.categories})
            for dimension in graph.dimensions
            if dimension.identifier
        }

    def of(self, categories: Mapping[str, str]) -> list[ShapeCoordinateType]:
        result: list[ShapeCoordinateType] = []
        for dimension, category in categories.items():
            dimension_id, category_ids = self.catalogue.get(dimension, (None, {}))
            result.append(
                ShapeCoordinateType(
                    dimension=dimension,
                    category=category,
                    dimension_id=dimension_id,
                    category_id=category_ids.get(category),
                )
            )
        return result


def _qualifiers(qualifiers: Mapping[str, QualifierRequirement]) -> list[ShapeQualifierRequirementType]:
    return [
        ShapeQualifierRequirementType(path=path, min=requirement.min, max=requirement.max, present=requirement.present)
        for path, requirement in qualifiers.items()
    ]
