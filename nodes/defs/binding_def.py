from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from paths.identifiers import DatasetIdentifier, NodeIdentifier, NodePortIdentifier

from .edge_def import EdgeTransformation


class NodePortRef(BaseModel):
    node_id: NodeIdentifier
    port_id: NodePortIdentifier


class EdgeBindingDef(BaseModel):
    """A source-node binding to one input port on a node."""

    id: UUID = Field(description='Globally unique identifier of the node-edge binding.')
    from_ref: NodePortRef = Field(description='Reference to the source node and output port.')
    to_ref: NodePortRef = Field(description='Reference to the target node and input port.')
    transformations: list[EdgeTransformation] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class DatasetPortBindingDef(BaseModel):
    """A dataset-metric binding to one input port on a node."""

    id: UUID = Field(description='Globally unique identifier of the dataset-port binding.')
    node_ref: NodePortRef = Field(description='Reference to the node and its bound input port.')
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
