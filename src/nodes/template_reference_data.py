"""Pin an explicitly selected historical reference-data edition under the method's catalogue identities."""

from typing import TYPE_CHECKING

from nodes.dataset_materialization import ensure_dataset_materializations

if TYPE_CHECKING:
    from pydantic import JsonValue

    from kausal_common.datasets.models import Dataset, DatasetSchemaDimension

    from nodes.models import DatasetMaterialization


def release_reference_materializations(
    datasets: list[Dataset],
    materializations: dict[int, DatasetMaterialization],
    reference_data: dict[str, Dataset],
) -> dict[int, DatasetMaterialization]:
    selected = ensure_dataset_materializations([reference_data[d.identifier] for d in datasets if d.identifier in reference_data])
    result = dict(materializations)
    for target in datasets:
        source = reference_data.get(target.identifier or '')
        if source is None:
            continue
        for field in ('repo_url', 'dataset_id'):
            if (
                not source.external_ref
                or not target.external_ref
                or source.external_ref.get(field) != target.external_ref.get(field)
            ):
                raise ValueError('Reference editions must identify the same upstream dataset')
        identities = _catalogue_identities(target, source)
        # This object is an in-memory publication payload; do not save it over
        # either the template's or the dependent instance's current materialization.
        materialization = type(selected[source.pk]).objects.get(pk=selected[source.pk].pk)
        materialization.shape_profiles = remap_json(materialization.shape_profiles, identities)
        result[target.pk] = materialization
    return result


def _catalogue_identities(target: Dataset, source: Dataset) -> dict[str, str]:
    assert target.schema is not None
    assert source.schema is not None
    metrics = {metric.name: metric for metric in target.schema.metrics.all()}
    source_metrics = list(source.schema.metrics.all())
    if set(metrics) != {metric.name for metric in source_metrics}:
        raise ValueError('Reference editions have different metrics')
    identities: dict[str, str] = {}
    for metric in source_metrics:
        counterpart = metrics[metric.name]
        if counterpart.unit != metric.unit:
            raise ValueError('Reference editions have different metric units')
        identities[str(metric.uuid)] = str(counterpart.uuid)
    target_dimensions = {_dimension_identifier(item): item.dimension for item in target.schema.dimensions.all()}
    for item in source.schema.dimensions.all():
        dimension = target_dimensions.get(_dimension_identifier(item))
        if dimension is None:
            raise ValueError('Reference editions have different dimensions')
        identities[str(item.dimension.uuid)] = str(dimension.uuid)
        categories = {cat.identifier: cat.uuid for cat in dimension.categories.all()}
        for category in item.dimension.categories.all():
            if category.identifier not in categories:
                raise ValueError(
                    f'Reference edition {source.identifier} contains undeclared category {item.column_name}/{category.identifier}'
                )
            identities[str(category.uuid)] = str(categories[category.identifier])
    return identities


def _dimension_identifier(item: DatasetSchemaDimension) -> str:
    if item.column_name:
        return item.column_name
    scope = item.dimension.scopes.exclude(identifier=None).first()
    if scope is None or scope.identifier is None:
        raise ValueError('Reference dimension has no identifier')
    return scope.identifier


def remap_json(value: JsonValue, identities: dict[str, str]) -> JsonValue:
    if isinstance(value, dict):
        return {identities.get(key, key): remap_json(item, identities) for key, item in value.items()}
    if isinstance(value, list):
        return [remap_json(item, identities) for item in value]
    if isinstance(value, str):
        return identities.get(value, value)
    return value
