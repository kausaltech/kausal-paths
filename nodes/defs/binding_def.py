from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from paths.identifiers import DatasetIdentifier, NodeIdentifier, NodePortIdentifier

from .edge_def import EdgeTransformation
from .transform_def import PortTransformOp, forecast_from_operations


class NodePortRef(BaseModel):
    node_id: NodeIdentifier
    port_id: NodePortIdentifier


class PortBindingDef(BaseModel):
    """
    Something bound to one node input port, independent of what is on the other end.

    Bindings are the ORM-free view of ``NodeEdge`` and ``DatasetPort`` rows.
    Consumers that only care that a port *has* an input — validation, the
    editor, dimension-constraint propagation — work against this base.

    ``transformations`` is deliberately not here yet. Dataset bindings will
    carry ``list[PortTransformOp]``, but edge transformations still use the
    legacy ``EdgeTransformation`` vocabulary, which has no home for the port
    shape declarations that ``FlattenTransformation`` actually encodes. The
    field moves onto this base once those declarations move onto
    ``InputPortDef``. See `docs/architecture/dimension-constraints.md`.
    """

    id: UUID = Field(description='Globally unique identifier of the binding.')
    port_ref: NodePortRef = Field(description='Reference to the node and the input port this binds to.')
    tags: list[str] = Field(default_factory=list)


class EdgeBindingDef(PortBindingDef):
    """A source-node binding to one input port on a node."""

    kind: Literal['edge'] = 'edge'
    from_ref: NodePortRef = Field(description='Reference to the source node and output port.')
    transformations: list[EdgeTransformation] = Field(default_factory=list)


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
    operations: list[PortTransformOp] = Field(
        default_factory=list,
        description="The binding's transform pipeline, in execution order.",
    )

    @property
    def forecast_from(self) -> int | None:
        """The year from which the time series becomes a forecast. Derived from the pipeline."""
        return forecast_from_operations(self.operations)


type AnyPortBindingDef = EdgeBindingDef | DatasetBindingDef
