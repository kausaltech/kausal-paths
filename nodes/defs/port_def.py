from __future__ import annotations

from uuid import UUID

from pydantic import Field, PrivateAttr

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString

from paths.identifiers import MixedCaseIdentifier
from paths.refs import DimensionRef, NodeRef, QuantityKindRef, UniqueList

from nodes.units import Unit


class InputPortDef(I18nBaseModel):
    """Definition of a node input port (stored in NodeConfig.input_ports JSONField)."""

    id: UUID
    label: I18nString | None = None
    quantity: QuantityKindRef | None = None
    unit: Unit | None = None
    multi: bool = False
    """When True, the port accepts multiple connections (aggregated by the node's computation)."""
    required_dimensions: UniqueList[DimensionRef] = Field(default_factory=list)
    supported_dimensions: UniqueList[DimensionRef] = Field(default_factory=list)

    # This is used only temporarily at export time to store the node reference.
    _from_node: NodeRef | None = PrivateAttr(default=None)


class OutputPortDef(I18nBaseModel):
    """
    Definition of a node output port.

    This is the canonical representation of what a node produces.
    Each output port maps 1:1 to a runtime ``NodeMetric`` and can
    be connected to zero or more edges.
    """

    id: UUID
    label: I18nString | None = None
    quantity: QuantityKindRef | None = None
    unit: Unit
    column_id: MixedCaseIdentifier | None = None
    """DataFrame column name for this port. When None, inferred by the loader."""
    is_editable: bool = True
    """Whether the user can modify this port in the model editor."""
    dimensions: UniqueList[DimensionRef] = Field(default_factory=list)
