"""Payload-light observed shape metadata for dataset metrics."""

from collections import defaultdict
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import Field

from nodes.defs.graph import FrozenGraphModel

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kausal_common.datasets.models import Dataset


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
