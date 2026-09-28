"""Stable identities for dataset dimension coordinates."""

from typing import TYPE_CHECKING
from uuid import UUID

from django.contrib.contenttypes.models import ContentType
from pydantic import BaseModel

from kausal_common.datasets.models import DatasetSchemaDimension, DimensionCategory

from frameworks.catalogue import dimension_scopes

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset


class DatasetCoordinate(BaseModel):
    """Both the legacy column identifiers and the stable dimension/category UUIDs."""

    dimension: str
    category: str
    dimension_uuid: UUID
    category_uuid: UUID


class DatasetCoordinateIndex:
    def __init__(self, dataset: Dataset) -> None:
        schema = dataset.schema
        if schema is None:
            raise ValueError('A dataset without a schema has no coordinates')
        instance = dataset.scope_instance
        local_type_id = ContentType.objects.get_for_model(instance).pk
        scopes = {
            scope.dimension_id: scope
            for scope in sorted(
                dimension_scopes(instance).filter(dimension_id__in=schema.dimensions.values_list('dimension_id', flat=True)),
                key=lambda scope: scope.scope_content_type_id == local_type_id,
            )
        }
        dimensions = list(DatasetSchemaDimension.objects.filter(schema=schema).select_related('dimension'))
        dimension_ids = [dimension.dimension_id for dimension in dimensions]
        self._by_identifier: dict[tuple[str, str], DatasetCoordinate] = {}
        self._by_uuid: dict[tuple[str, str], DatasetCoordinate] = {}
        columns = {
            dimension.dimension_id: dimension.column_name
            or (scopes[dimension.dimension_id].identifier if dimension.dimension_id in scopes else None)
            or str(dimension.dimension.uuid)
            for dimension in dimensions
        }
        dimension_uuids = {dimension.dimension_id: dimension.dimension.uuid for dimension in dimensions}
        for category in DimensionCategory.objects.filter(dimension_id__in=dimension_ids):
            column = columns[category.dimension_id]
            identifier = category.identifier or str(category.uuid)
            coordinate = DatasetCoordinate(
                dimension=column,
                category=identifier,
                dimension_uuid=dimension_uuids[category.dimension_id],
                category_uuid=category.uuid,
            )
            self._by_identifier[column, identifier] = coordinate
            self._by_uuid[str(coordinate.dimension_uuid), str(coordinate.category_uuid)] = coordinate

    def resolve(self, categories: dict[str, str]) -> list[DatasetCoordinate]:
        try:
            return [self._by_identifier[dimension, category] for dimension, category in categories.items()]
        except KeyError as exc:
            raise ValueError(f'Unknown dataset coordinate: {exc.args[0]}') from exc

    def resolve_uuids(self, coordinates: dict[str, str]) -> list[DatasetCoordinate]:
        try:
            return [self._by_uuid[dimension, category] for dimension, category in coordinates.items()]
        except KeyError as exc:
            raise ValueError(f'Unknown dataset coordinate UUID pair: {exc.args[0]}') from exc

    def resolve_selection(self, selection: dict[str, list[str]]) -> list[DatasetCoordinate]:
        """Every (dimension, category) pair of a multi-category selection, in selection order."""
        try:
            return [self._by_uuid[dimension, category] for dimension, categories in selection.items() for category in categories]
        except KeyError as exc:
            raise ValueError(f'Unknown dataset coordinate UUID pair: {exc.args[0]}') from exc
