"""Editing a local input selection without exposing the shared node to mutation."""

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import strawberry as sb
from django.core.exceptions import ValidationError

from kausal_common.datasets.models import Dataset

from nodes.defs.node_defs import InputDatasetDef
from nodes.graphql.types.transformations import DatasetTransformationInput, dataset_transformations_from_input
from nodes.instance_serialization import DatasetMetricSource, InputBindingSnapshot, NodePortSource

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


@sb.input
class InputPortBindingInput:
    source_node_id: UUID | None = None
    source_port_id: UUID | None = None
    dataset_id: UUID | None = None
    metric_id: UUID | None = None
    transformations: list[DatasetTransformationInput] | None = None

    def to_snapshot(self, instance: InstanceConfig, node_id: UUID, port_id: UUID, position: int) -> InputBindingSnapshot:
        edge = self.source_node_id is not None and self.source_port_id is not None
        dataset = self.dataset_id is not None and self.metric_id is not None
        if (
            edge == dataset
            or (edge and (self.dataset_id or self.metric_id))
            or (dataset and (self.source_node_id or self.source_port_id))
        ):
            raise ValidationError('Supply exactly one source: node and output port, or dataset and metric')
        source: NodePortSource | DatasetMetricSource
        if edge:
            assert self.source_node_id is not None
            assert self.source_port_id is not None
            source = NodePortSource(node_id=self.source_node_id, port_id=self.source_port_id)
            default = []
        else:
            # Dataset scope is checked by the graph editing service, including released defaults.
            assert self.dataset_id is not None
            assert self.metric_id is not None
            row = Dataset.objects.select_related('schema').get(uuid=self.dataset_id)
            if row.schema is None:
                raise ValidationError('Dataset has no schema')
            metric = row.schema.metrics.get(uuid=self.metric_id)
            source = DatasetMetricSource(
                dataset=row.identifier or str(row.uuid),
                dataset_uuid=row.uuid,
                metric=metric.name or str(metric.uuid),
                metric_uuid=metric.uuid,
            )
            default = InputDatasetDef(id=source.dataset, column=source.metric).to_transformations()
        return InputBindingSnapshot(
            uuid=uuid4(),
            node_id=node_id,
            port_id=port_id,
            position=position,
            source=source,
            transformations=default
            if self.transformations is None
            else dataset_transformations_from_input(
                self.transformations,
            ),
        )
