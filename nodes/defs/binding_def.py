from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from paths.identifiers import DatasetIdentifier, NodeIdentifier, NodePortIdentifier

from .graph import InstanceGraphBoundModel
from .transform_def import PortTransformOp, forecast_from_transformations, modernized_transformations

if TYPE_CHECKING:
    from nodes.defs.graph import DatasetMeta, DatasetMetricMeta
    from nodes.defs.port_def import InputPortDef, OutputPortDef
    from nodes.instance_graph import NodeMeta


class NodePortRef(BaseModel):
    node_uuid: UUID | None = None
    """
    Canonical node identity.

    Optional only while the pre-InstanceGraph ORM projection still supplies
    identifier-only references in a few internal callers. Graph construction
    requires this value and emits UUID-only references.
    """
    node_id: NodeIdentifier
    """Deprecated compatibility label; never use as durable graph identity."""
    port_id: NodePortIdentifier


class PortBindingDef(InstanceGraphBoundModel):
    """
    Something bound to one node input port, independent of what is on the other end.

    Bindings are the ORM-free view of ``NodeEdge`` and ``DatasetPort`` rows.
    Consumers that only care that a port *has* an input — validation, the
    editor, dimension-constraint propagation — work against this base.

    ``transformations`` is presented in the current vocabulary regardless of
    what the underlying row stores: legacy edge kinds are rewritten on
    construction, so consumers of a binding never see two names for one idea.
    """

    id: UUID = Field(description='Globally unique identifier of the binding.')
    port_ref: NodePortRef = Field(description='Reference to the node and the input port this binds to.')
    position: int = Field(default=0, ge=0, description='Stable order among values delivered to the input port.')
    tags: list[str] = Field(default_factory=list)
    transformations: list[PortTransformOp] = Field(
        default_factory=list,
        description="The binding's transform pipeline, in execution order.",
    )

    @field_validator('transformations', mode='after')
    @classmethod
    def _modernize_transformations(cls, value: list[PortTransformOp]) -> list[PortTransformOp]:
        return modernized_transformations(value)

    @field_validator('tags', mode='before')
    @classmethod
    def _null_tags_as_empty(cls, value: object) -> object:
        """Treat a missing JSON key, which surfaces as null in DB annotations, as an empty list."""
        return [] if value is None else value

    @property
    def target_node(self) -> NodeMeta:
        node_uuid = self.port_ref.node_uuid
        if node_uuid is None:
            raise ValueError(f'Binding {self.id} has no target node UUID')
        try:
            return self.graph.node_by_id[node_uuid]
        except KeyError:
            raise ValueError(f'Binding {self.id} targets unknown node {node_uuid}') from None

    @property
    def target_port(self) -> InputPortDef:
        try:
            return self.graph.input_port_by_id[(self.target_node.id, self.port_ref.port_id)]
        except KeyError:
            raise ValueError(f'Binding {self.id} targets unknown input port {self.port_ref.port_id}') from None


class EdgeBindingDef(PortBindingDef):
    """A source-node binding to one input port on a node."""

    kind: Literal['edge'] = 'edge'
    from_ref: NodePortRef = Field(description='Reference to the source node and output port.')

    @property
    def source_node(self) -> NodeMeta:
        node_uuid = self.from_ref.node_uuid
        if node_uuid is None:
            raise ValueError(f'Binding {self.id} has no source node UUID')
        try:
            return self.graph.node_by_id[node_uuid]
        except KeyError:
            raise ValueError(f'Binding {self.id} references unknown source node {node_uuid}') from None

    @property
    def source_port(self) -> OutputPortDef:
        try:
            return self.graph.output_port_by_id[(self.source_node.id, self.from_ref.port_id)]
        except KeyError:
            raise ValueError(f'Binding {self.id} references unknown output port {self.from_ref.port_id}') from None


class DatasetBindingDef(PortBindingDef):
    """A dataset-metric binding to one input port on a node."""

    kind: Literal['dataset'] = 'dataset'
    dataset_uuid: UUID | None = Field(default=None, description='Globally unique identifier of the bound dataset object.')
    metric_uuid: UUID | None = Field(default=None, description='Globally unique identifier of the bound dataset metric object.')
    dataset_is_external_placeholder: bool = Field(
        default=False,
        description='Whether the bound dataset object is only a placeholder without imported datapoints.',
    )
    dataset_external_ref: dict[str, str | None] | None = Field(
        default=None,
        description='External source reference for the bound dataset object.',
    )
    external_dataset_id: DatasetIdentifier | None = Field(
        default=None,
        description='Stable identifier of the external dataset, typically the dataset repo path without extension.',
    )
    external_metric_id: str | None = Field(
        default=None,
        description='Stable identifier of the external metric within the dataset.',
    )

    @property
    def dataset(self) -> DatasetMeta:
        if self.dataset_uuid is None:
            raise ValueError(f'Binding {self.id} has no dataset UUID')
        try:
            return self.graph.dataset_by_id[self.dataset_uuid]
        except KeyError:
            raise ValueError(f'Binding {self.id} references unknown dataset {self.dataset_uuid}') from None

    @property
    def metric(self) -> DatasetMetricMeta:
        if self.metric_uuid is None:
            raise ValueError(f'Binding {self.id} has no metric UUID')
        try:
            return self.dataset.metric_by_id[self.metric_uuid]
        except KeyError:
            raise ValueError(f'Binding {self.id} references unknown metric {self.metric_uuid}') from None

    @property
    def forecast_from(self) -> int | None:
        """The year from which the time series becomes a forecast. Derived from the pipeline."""
        return forecast_from_transformations(self.transformations)


type AnyPortBindingDef = EdgeBindingDef | DatasetBindingDef
