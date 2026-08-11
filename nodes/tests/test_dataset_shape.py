from datetime import date
from decimal import Decimal
from uuid import uuid4

from django.core.cache import cache
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

import pytest

from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DimensionCategoryFactory,
)

from nodes.dataset_materialization import materialize_dataset
from nodes.dataset_shape import load_dataset_shape_profiles
from nodes.defs.graph import DatasetMeta, DatasetMetricMeta
from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec
from nodes.instance_graph import InstanceGraph
from nodes.instance_graph_cache import ResolvedInstanceSource
from nodes.models import DatasetMaterialization
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

pytestmark = pytest.mark.django_db


def _config():
    instance = InstanceFactory.create()
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        spec=InstanceModelSpec(),
    )


def _graph(config, datasets: list[DatasetMeta]) -> InstanceGraph:
    return InstanceGraph(
        instance_id=config.uuid,
        metadata=InstanceMetadata(uuid=config.uuid, identifier=config.identifier),
        spec=InstanceModelSpec(),
        datasets=tuple(datasets),
    )


def _dataset_meta(dataset, metrics, dimension_id) -> DatasetMeta:
    return DatasetMeta(
        id=dataset.uuid,
        identifier=dataset.identifier,
        schema_id=dataset.schema.uuid,
        metrics=tuple(DatasetMetricMeta(id=metric.uuid, identifier=metric.name) for metric in metrics),
        declared_dimension_ids=(dimension_id,),
    )


def _draft_source(config) -> ResolvedInstanceSource:
    return ResolvedInstanceSource(str(config.uuid), 'database-draft', config.cache_invalidated_at.isoformat())


def test_current_profiles_distinguish_observed_empty_and_categories() -> None:
    config = _config()
    category = DimensionCategoryFactory.create(identifier='first')
    dataset = DatasetFactory.create(identifier='observed', scope=config)
    populated = DatasetMetricFactory.create(schema=dataset.schema, name='populated')
    empty = DatasetMetricFactory.create(schema=dataset.schema, name='empty')
    DataPointFactory.create(dataset=dataset, metric=populated, dimension_categories=[category])
    materialization = materialize_dataset(dataset)
    graph = _graph(config, [_dataset_meta(dataset, [populated, empty], category.dimension.uuid)])

    profiles = load_dataset_shape_profiles(
        config,
        graph,
        _draft_source(config),
        pairs={(dataset.uuid, populated.uuid), (dataset.uuid, empty.uuid)},
    )

    populated_profile = profiles[dataset.uuid, populated.uuid]
    assert populated_profile.has_datapoints is True
    assert populated_profile.categories_by_dimension == {category.dimension.uuid: frozenset({category.uuid})}
    assert populated_profile.source_version == f'materialization:{materialization.generation}:{materialization.content_hash}'
    empty_profile = profiles[dataset.uuid, empty.uuid]
    assert empty_profile.has_datapoints is False
    assert empty_profile.categories_by_dimension == {category.dimension.uuid: frozenset()}


def test_external_placeholder_profile_is_unknown_without_queries() -> None:
    config = _config()
    dimension_id = uuid4()
    dataset_id = uuid4()
    metric_id = uuid4()
    graph = _graph(
        config,
        [
            DatasetMeta(
                id=dataset_id,
                identifier='external',
                schema_id=uuid4(),
                metrics=(DatasetMetricMeta(id=metric_id, identifier='value'),),
                declared_dimension_ids=(dimension_id,),
                is_external_placeholder=True,
            )
        ],
    )

    with CaptureQueriesContext(connection) as queries:
        profiles = load_dataset_shape_profiles(
            config,
            graph,
            _draft_source(config),
            pairs={(dataset_id, metric_id)},
        )

    assert len(queries) == 0
    profile = profiles[dataset_id, metric_id]
    assert profile.has_datapoints is None
    assert profile.categories_by_dimension == {dimension_id: None}


def test_missing_or_stale_materialization_is_repaired() -> None:
    config = _config()
    category = DimensionCategoryFactory.create(identifier='fresh')
    dataset = DatasetFactory.create(identifier='repair', scope=config)
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='value')
    graph = _graph(config, [_dataset_meta(dataset, [metric], category.dimension.uuid)])

    first = load_dataset_shape_profiles(
        config,
        graph,
        _draft_source(config),
        pairs={(dataset.uuid, metric.uuid)},
    )[dataset.uuid, metric.uuid]
    materialization = DatasetMaterialization.objects.get(dataset=dataset)
    assert first.has_datapoints is False

    DataPointFactory.create(dataset=dataset, metric=metric, dimension_categories=[category])
    dataset.last_modified_at = timezone.now()
    dataset.save(update_fields=['last_modified_at'])
    cache.clear()
    second = load_dataset_shape_profiles(
        config,
        graph,
        _draft_source(config),
        pairs={(dataset.uuid, metric.uuid)},
    )[dataset.uuid, metric.uuid]

    materialization.refresh_from_db()
    assert materialization.generation == 2
    assert second.has_datapoints is True
    assert second.categories_by_dimension == {category.dimension.uuid: frozenset({category.uuid})}


def test_profile_query_count_is_constant_with_dataset_count() -> None:
    config = _config()
    dimension_id = uuid4()
    metas: list[DatasetMeta] = []
    pairs = set()
    for index in range(3):
        dataset = DatasetFactory.create(identifier=f'dataset-{index}', scope=config)
        metric = DatasetMetricFactory.create(schema=dataset.schema, name='value')
        materialize_dataset(dataset)
        metas.append(_dataset_meta(dataset, [metric], dimension_id))
        pairs.add((dataset.uuid, metric.uuid))
    graph = _graph(config, metas)
    cache.clear()

    with CaptureQueriesContext(connection) as queries:
        load_dataset_shape_profiles(config, graph, _draft_source(config), pairs=pairs)

    assert len(queries) == 2


def test_published_profile_uses_pinned_shape_not_live_rows() -> None:
    from nodes.models import DatasetPort, InstanceRevisionDatasetPin, NodeConfig

    config = _config()
    category = DimensionCategoryFactory.create(identifier='pinned')
    dataset = DatasetFactory.create(identifier='published', scope=config)
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='value')
    DataPointFactory.create(
        dataset=dataset,
        metric=metric,
        date=date(2020, 1, 1),
        value=Decimal(1),
        dimension_categories=[category],
    )
    materialize_dataset(dataset)
    node = NodeConfig.objects.create(instance=config, identifier='owner', name='Owner')
    DatasetPort.objects.create(instance=config, node=node, port_id=uuid4(), dataset=dataset, metric=metric)
    config.publish_instance()
    config.refresh_from_db()
    graph = _graph(config, [_dataset_meta(dataset, [metric], category.dimension.uuid)])
    source = ResolvedInstanceSource(
        str(config.uuid),
        'database-published',
        str(config.live_revision_id),
        revision_id=config.live_revision_id,
    )

    dataset.data_points.all().delete()
    dataset.last_modified_at = timezone.now()
    dataset.save(update_fields=['last_modified_at'])
    cache.clear()
    with CaptureQueriesContext(connection) as queries:
        profile = load_dataset_shape_profiles(
            config,
            graph,
            source,
            pairs={(dataset.uuid, metric.uuid)},
        )[dataset.uuid, metric.uuid]

    assert len(queries) == 1
    assert profile.has_datapoints is True
    assert profile.categories_by_dimension == {category.dimension.uuid: frozenset({category.uuid})}
    assert profile.source_version.startswith('revision:')

    InstanceRevisionDatasetPin.objects.filter(instance_revision_id=config.live_revision_id).update(shape_profiles=None)
    cache.clear()
    legacy_profile = load_dataset_shape_profiles(
        config,
        graph,
        source,
        pairs={(dataset.uuid, metric.uuid)},
    )[dataset.uuid, metric.uuid]
    assert legacy_profile.has_datapoints is None
    assert legacy_profile.categories_by_dimension == {category.dimension.uuid: None}
