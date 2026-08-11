from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import Field, PrivateAttr

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString

from paths.identifiers import MixedCaseIdentifier
from paths.refs import DimensionRef, NodeRef, QuantityKindRef, UniqueList

from nodes.units import Unit

if TYPE_CHECKING:
    from nodes.node import Node


@dataclass(frozen=True, slots=True, kw_only=True)
class InputPortDeclaration:
    """Class-level semantic input role shared by computation and shape rules."""

    role: MixedCaseIdentifier
    identifier: MixedCaseIdentifier | None = None
    multi: bool = False
    required: bool = True

    @property
    def instance_identifier(self) -> MixedCaseIdentifier:
        return self.identifier or self.role


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputPortDeclaration:
    """Class-level semantic output role mapped to an instance port identifier."""

    role: MixedCaseIdentifier
    identifier: MixedCaseIdentifier


class InputPortDef(I18nBaseModel):
    """Definition of a node input port (stored in NodeConfig.input_ports JSONField)."""

    id: UUID
    identifier: MixedCaseIdentifier | None = None
    """
    Optional human-readable name for the port, unique among the node's input
    ports. This is the name the port is addressed by outside the API — most
    importantly in formulas, where a port identifier is a variable. Optional
    because ports synced from YAML often have no name worth keeping (their
    key was an index or a label with spaces).
    """
    label: I18nString | None = None
    quantity: QuantityKindRef | None = None
    unit: Unit | None = None
    multi: bool = False
    """When True, the port accepts multiple connections (aggregated by the node's computation)."""
    required_dimensions: UniqueList[DimensionRef] = Field(default_factory=list)
    supported_dimensions: UniqueList[DimensionRef] = Field(default_factory=list)

    # These are used only temporarily at export time to store the node reference and metric ID.
    _from_node: NodeRef | None = PrivateAttr(default=None)
    _edge_metric_id: str | None = PrivateAttr(default=None)


class OutputPortDef(I18nBaseModel):
    """
    Definition of a node output port.

    This is the canonical representation of what a node produces.
    Each output port maps 1:1 to a runtime ``NodeMetric`` and can
    be connected to zero or more edges.
    """

    id: UUID
    identifier: MixedCaseIdentifier | None = None
    """
    Optional human-readable name for the port, unique among the node's output
    ports. Distinct from ``column_id``: this names the port in the node's
    namespace, while ``column_id`` names the physical metric column in the
    output frame. They usually agree, and ``identifier`` defaults to
    ``column_id`` when only the latter is known.
    """
    label: I18nString | None = None
    quantity: QuantityKindRef | None = None
    unit: Unit
    column_id: MixedCaseIdentifier | None = None
    """DataFrame column name for this port. When None, inferred by the loader."""
    is_editable: bool = True
    """Whether the user can modify this port in the model editor."""
    dimensions: UniqueList[DimensionRef] = Field(default_factory=list)

    _metric_id: str | None = PrivateAttr(default=None)

    _node: 'Node | None' = PrivateAttr(default=None)
