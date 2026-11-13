"""
Creating a node's own dataset from the shape of its ports.

A node-owned dataset is scoped to its `NodeConfig`, has no identifier, and is
deleted, copied and exported with the node (docs/plans/node-owned-datasets.md).
This module creates one: a schema with one metric per port, the union of the
ports' dimensions, and optionally the cells of one year. Binding the ports to
the metrics is the binding editor's business, so that the constraint check runs
as for any other binding.

A metric need not use every dimension of the dataset. Its cells name
categories only for the dimensions it has; the dimensions it lacks are left
empty, and the binding drops them, as for any multi-metric dataset whose
columns span different dimensions.
"""

import itertools
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError

from kausal_common.datasets.models import (
    DataPoint,
    DataPointDimensionCategory,
    Dataset,
    DatasetMetric,
    DatasetSchema,
    DatasetSchemaDimension,
    DatasetSchemaScope,
    Dimension,
    DimensionCategory,
)
from kausal_common.i18n.pydantic import get_modeltrans_attrs_from_str

from datasets.graphql.editor import CreateDatasetMetricInput, create_metric_row
from nodes.change_ops import record_change
from nodes.models import NodeConfig

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from nodes.units import Unit
    from users.models import User


class OwnedDatasetError(ValueError):
    """The requested dataset cannot be created; the message is for the modeller."""


@dataclass(frozen=True)
class OwnedMetric:
    """One metric of a node-owned dataset, and the input port it is for."""

    port_id: UUID
    label: str
    unit: Unit
    quantity: str | None


@dataclass(frozen=True)
class OwnedDataset:
    dataset: Dataset
    metrics: dict[UUID, DatasetMetric]
    """The metric for each port, by port id."""


def dataset_change_snapshot(dataset: Dataset) -> dict[str, object]:
    """Return the change-log snapshot of a dataset, as `createDataset` records it."""
    schema = dataset.schema
    return {
        'uuid': str(dataset.uuid),
        'identifier': dataset.identifier,
        'schema_uuid': str(schema.uuid) if schema is not None else None,
        'name': schema.name if schema is not None else None,
        'i18n': dict(schema.i18n or {}) if schema is not None else {},
    }


def create_owned_dataset(
    nc: NodeConfig,
    *,
    name: str,
    metrics: Sequence[OwnedMetric],
    dimension_ids: Sequence[UUID],
    user: User | None,
) -> OwnedDataset:
    """
    Create a dataset owned by `nc`, with one metric per entry of `metrics`, inside an open change operation.

    Metric labels must be distinct, as the editor requires of sibling metrics
    (decision 13 of the node-owned datasets plan).
    """
    if not metrics:
        raise OwnedDatasetError('A dataset needs at least one metric')
    labels = [metric.label for metric in metrics]
    if len(set(labels)) != len(labels):
        raise OwnedDatasetError('The metric labels must be distinct: %s' % ', '.join(labels))
    dimensions = {dimension.uuid: dimension for dimension in Dimension.objects.filter(uuid__in=dimension_ids)}
    missing = [str(dimension_id) for dimension_id in dimension_ids if dimension_id not in dimensions]
    if missing:
        raise OwnedDatasetError('Unknown dimensions: %s' % ', '.join(missing))

    language = nc.instance.primary_language
    node_ct = ContentType.objects.get_for_model(NodeConfig)
    schema_name, schema_i18n = get_modeltrans_attrs_from_str(name, 'name', language)
    schema = DatasetSchema.objects.create(name=schema_name, i18n=schema_i18n)
    DatasetSchemaScope.objects.create(schema=schema, scope_content_type=node_ct, scope_id=nc.pk)
    for order, dimension_id in enumerate(dimension_ids):
        DatasetSchemaDimension.objects.create(schema=schema, dimension=dimensions[dimension_id], order=order)
    created: dict[UUID, DatasetMetric] = {}
    try:
        for metric in metrics:
            created[metric.port_id] = create_metric_row(
                schema,
                CreateDatasetMetricInput(label=metric.label, unit=str(metric.unit), quantity=metric.quantity),  # type: ignore[arg-type]
                language,
            )
    except ValidationError as exc:
        raise OwnedDatasetError('; '.join(exc.messages)) from exc
    dataset = Dataset.objects.create(
        schema=schema,
        uuid=uuid4(),
        identifier=None,
        scope_content_type=node_ct,
        scope_id=nc.pk,
        created_by=user,
        last_modified_by=user,
    )
    record_change(dataset, action='dataset.create', before=None, after=dataset_change_snapshot(dataset))
    return OwnedDataset(dataset=dataset, metrics=created)


def write_year(
    dataset: Dataset,
    metric: DatasetMetric,
    year: int,
    categories: Mapping[UUID, Sequence[UUID]],
    *,
    value: float | None,
) -> int:
    """
    Write one cell of `metric` for each combination of `categories` in `year`.

    `categories` gives the categories to create cells for, per dimension the
    metric has. Those must be dimensions of the dataset; a dimension of the
    dataset that is not named stays empty in these cells. Returns the number of
    cells written. The caller refreshes the materialized copy.
    """
    assert dataset.schema is not None
    schema_dimensions = list(dataset.schema.dimensions.order_by('order').values_list('dimension__uuid', flat=True))
    if not set(categories) <= set(schema_dimensions):
        raise OwnedDatasetError('Cells may name categories only for dimensions of the dataset')
    category_pks = dict(
        DimensionCategory.objects.filter(uuid__in=[uuid for uuids in categories.values() for uuid in uuids]).values_list(
            'uuid', 'pk'
        )
    )
    per_dimension = [
        [category_pks[uuid] for uuid in categories[dimension]] for dimension in schema_dimensions if dimension in categories
    ]
    combinations = list(itertools.product(*per_dimension))
    points = DataPoint.objects.bulk_create([
        DataPoint(dataset=dataset, metric=metric, date=date(year, 1, 1), value=value) for _combination in combinations
    ])
    DataPointDimensionCategory.objects.bulk_create([
        DataPointDimensionCategory(data_point=point, dimension_category_id=category_pk)
        for point, combination in zip(points, combinations, strict=True)
        for category_pk in combination
    ])
    return len(points)
