from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from paths.identifiers import DatasetIdentifier, NodeIdentifier, NodePortIdentifier

from .transform_def import PortTransformOp, forecast_from_transformations, modernized_transformations


class NodePortRef(BaseModel):
    node_id: NodeIdentifier
    port_id: NodePortIdentifier


class PortBindingDef(BaseModel):
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


class EdgeBindingDef(PortBindingDef):
    """A source-node binding to one input port on a node."""

    kind: Literal['edge'] = 'edge'
    from_ref: NodePortRef = Field(description='Reference to the source node and output port.')


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
    def forecast_from(self) -> int | None:
        """The year from which the time series becomes a forecast. Derived from the pipeline."""
        return forecast_from_transformations(self.transformations)


type AnyPortBindingDef = EdgeBindingDef | DatasetBindingDef
