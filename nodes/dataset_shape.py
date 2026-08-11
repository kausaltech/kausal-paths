"""Payload-light observed shape metadata for dataset metrics."""

from collections import defaultdict
from typing import TYPE_CHECKING
from uuid import UUID

from django.core.cache import cache
from pydantic import Field

from nodes.defs.binding_def import DatasetBindingDef
from nodes.defs.graph import FrozenGraphModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from kausal_common.datasets.models import Dataset

    from nodes.instance_graph import InstanceGraph
    from nodes.instance_graph_cache import ResolvedInstanceSource
    from nodes.models import InstanceConfig


DATASET_SHAPE_FORMAT_VERSION = 1
DATASET_SHAPE_CACHE_TIMEOUT = 7 * 24 * 60 * 60


class ObservedMetricShape(FrozenGraphModel):
    """Observed facts stored beside one immutable dataset payload."""

    metric_id: UUID
    categories_by_dimension: dict[UUID, frozenset[UUID]] = Field(default_factory=dict)
    has_datapoints: bool


class DatasetShapeProfile(FrozenGraphModel):
    """Observed shape of one dataset metric at an identified source version."""

    dataset_id: UUID
    metric_id: UUID
    categories_by_dimension: dict[UUID, frozenset[UUID] | None]
    has_datapoints: bool | None
    source_version: str


type DatasetMetricPair = tuple[UUID, UUID]


def build_observed_metric_shapes(dataset: Dataset) -> tuple[ObservedMetricShape, ...]:
    """Collect all metric/category facts for one materialized dataset in one datapoint query."""
    from kausal_common.datasets.models import DataPoint

    schema = dataset.schema
    if schema is None:
        return ()
    metric_ids = tuple(schema.metrics.order_by('order').values_list('uuid', flat=True))
    categories: defaultdict[UUID, defaultdict[UUID, set[UUID]]] = defaultdict(lambda: defaultdict(set))
    metrics_with_datapoints: set[UUID] = set()
    rows = (
        DataPoint.objects
        .filter(dataset=dataset)
        .order_by()
        .values_list(
            'metric__uuid',
            'dimension_categories__dimension__uuid',
            'dimension_categories__uuid',
        )
        .distinct()
    )
    for metric_id, dimension_id, category_id in rows:
        metrics_with_datapoints.add(metric_id)
        if dimension_id is not None and category_id is not None:
            categories[metric_id][dimension_id].add(category_id)

    return tuple(
        ObservedMetricShape(
            metric_id=metric_id,
            categories_by_dimension={
                dimension_id: frozenset(category_ids) for dimension_id, category_ids in categories[metric_id].items()
            },
            has_datapoints=metric_id in metrics_with_datapoints,
        )
        for metric_id in metric_ids
    )


def dump_observed_metric_shapes(shapes: Iterable[ObservedMetricShape]) -> list[dict[str, object]]:
    return [shape.model_dump(mode='json') for shape in shapes]


def load_observed_metric_shapes(value: object) -> dict[UUID, ObservedMetricShape] | None:
    """Return ``None`` for legacy payload metadata whose coverage was never recorded."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError('Stored dataset shape profiles must be a list')
    shapes = (ObservedMetricShape.model_validate(item) for item in value)
    return {shape.metric_id: shape for shape in shapes}


def bound_dataset_metric_pairs(graph: InstanceGraph) -> frozenset[DatasetMetricPair]:
    return frozenset(
        (binding.dataset_uuid, binding.metric_uuid)
        for binding in graph.bindings
        if isinstance(binding, DatasetBindingDef) and binding.dataset_uuid is not None and binding.metric_uuid is not None
    )


def load_dataset_shape_profiles(
    config: InstanceConfig,
    graph: InstanceGraph,
    source: ResolvedInstanceSource,
    *,
    pairs: Iterable[DatasetMetricPair] | None = None,
) -> dict[DatasetMetricPair, DatasetShapeProfile]:
    """Bulk-load only requested metric profiles without reading dataframe payloads."""
    requested = frozenset(pairs if pairs is not None else bound_dataset_metric_pairs(graph))
    if not requested:
        return {}
    unknown = _external_profiles(graph, requested)
    local_pairs = requested - unknown.keys()
    if source.kind == 'database-published':
        loaded = _load_published_profiles(config, graph, source, local_pairs)
    else:
        loaded = _load_current_profiles(graph, local_pairs)
    return {**unknown, **loaded}


def _external_profiles(
    graph: InstanceGraph,
    requested: frozenset[DatasetMetricPair],
) -> dict[DatasetMetricPair, DatasetShapeProfile]:
    profiles: dict[DatasetMetricPair, DatasetShapeProfile] = {}
    for pair in requested:
        dataset = graph.dataset_by_id[pair[0]]
        if not dataset.is_external_placeholder:
            continue
        profiles[pair] = DatasetShapeProfile(
            dataset_id=pair[0],
            metric_id=pair[1],
            categories_by_dimension=dict.fromkeys(dataset.declared_dimension_ids),
            has_datapoints=None,
            source_version='external-placeholder',
        )
    return profiles


def _load_current_profiles(
    graph: InstanceGraph,
    requested: frozenset[DatasetMetricPair],
) -> dict[DatasetMetricPair, DatasetShapeProfile]:
    from kausal_common.datasets.models import Dataset

    from nodes.dataset_materialization import ensure_dataset_materializations

    dataset_ids = {dataset_id for dataset_id, _metric_id in requested}
    datasets = list(Dataset.objects.filter(uuid__in=dataset_ids).select_related('schema'))
    materializations = ensure_dataset_materializations(datasets)
    datasets_by_uuid = {dataset.uuid: dataset for dataset in datasets}
    profiles: dict[DatasetMetricPair, DatasetShapeProfile] = {}
    for pair in requested:
        dataset = datasets_by_uuid.get(pair[0])
        if dataset is None:
            raise RuntimeError(f'Dataset {pair[0]} is unavailable for current shape validation')
        materialization = materializations[dataset.pk]
        profiles[pair] = _profile_from_stored_shape(
            graph,
            pair,
            materialization.shape_profiles,
            f'materialization:{materialization.generation}:{materialization.content_hash}',
        )
    return _cache_profiles(profiles)


def _load_published_profiles(
    config: InstanceConfig,
    graph: InstanceGraph,
    source: ResolvedInstanceSource,
    requested: frozenset[DatasetMetricPair],
) -> dict[DatasetMetricPair, DatasetShapeProfile]:
    from nodes.models import InstanceRevisionDatasetPin

    if source.revision_id is None:
        raise RuntimeError('Published dataset profiles require an instance revision')
    dataset_ids = {dataset_id for dataset_id, _metric_id in requested}
    pins = InstanceRevisionDatasetPin.objects.filter(
        instance_config=config,
        instance_revision_id=source.revision_id,
        dataset_uuid__in=dataset_ids,
    ).only('dataset_uuid', 'dataset_revision_id', 'shape_profiles')
    pins_by_dataset = {pin.dataset_uuid: pin for pin in pins}
    profiles: dict[DatasetMetricPair, DatasetShapeProfile] = {}
    for pair in requested:
        pin = pins_by_dataset.get(pair[0])
        if pin is None:
            raise RuntimeError(f'Published dataset {pair[0]} has no retained revision pin')
        profiles[pair] = _profile_from_stored_shape(
            graph,
            pair,
            pin.shape_profiles,
            f'revision:{pin.dataset_revision_id}',
        )
    return _cache_profiles(profiles)


def _profile_from_stored_shape(
    graph: InstanceGraph,
    pair: DatasetMetricPair,
    stored: object,
    source_version: str,
) -> DatasetShapeProfile:
    cache_key = _profile_cache_key(pair, source_version)
    if (cached := cache.get(cache_key)) is not None:
        return DatasetShapeProfile.model_validate(cached)

    dataset = graph.dataset_by_id[pair[0]]
    shapes = load_observed_metric_shapes(stored)
    shape = shapes.get(pair[1]) if shapes is not None else None
    categories: dict[UUID, frozenset[UUID] | None]
    if shapes is None:
        categories = dict.fromkeys(dataset.declared_dimension_ids)
        has_datapoints = None
    elif shape is None:
        categories = {dimension_id: frozenset() for dimension_id in dataset.declared_dimension_ids}
        has_datapoints = False
    else:
        dimension_ids = dict.fromkeys((*dataset.declared_dimension_ids, *shape.categories_by_dimension))
        categories = {
            dimension_id: shape.categories_by_dimension.get(dimension_id, frozenset()) for dimension_id in dimension_ids
        }
        has_datapoints = shape.has_datapoints
    return DatasetShapeProfile(
        dataset_id=pair[0],
        metric_id=pair[1],
        categories_by_dimension=categories,
        has_datapoints=has_datapoints,
        source_version=source_version,
    )


def _cache_profiles(
    profiles: Mapping[DatasetMetricPair, DatasetShapeProfile],
) -> dict[DatasetMetricPair, DatasetShapeProfile]:
    cache.set_many(
        {_profile_cache_key(pair, profile.source_version): profile.model_dump(mode='json') for pair, profile in profiles.items()},
        timeout=DATASET_SHAPE_CACHE_TIMEOUT,
    )
    return dict(profiles)


def _profile_cache_key(pair: DatasetMetricPair, source_version: str) -> str:
    return f'dataset-shape:v{DATASET_SHAPE_FORMAT_VERSION}:{pair[0]}:{pair[1]}:{source_version}'
