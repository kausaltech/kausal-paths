from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, PrivateAttr

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString

from paths.identifiers import MixedCaseIdentifier
from paths.refs import DimensionRef, NodeRef, QuantityKindRef, UniqueList

from nodes.units import Unit

if TYPE_CHECKING:
    from django.utils.functional import Promise

    from nodes.node import Node


@dataclass(frozen=True, slots=True, kw_only=True)
class InputPortDeclaration:
    """
    Class-level semantic input role shared by computation and shape rules.

    Two kinds of multiplicity, each meaning exactly one thing:

    * ``multi`` — one port instance accepting many bindings. The values are
      delivered individually unless ``aggregation`` names an operation.
    * ``repeatable`` — many port instances of this role. Each instance is
      *heterogeneous*: it carries its own unit, quantity and dimension
      expectations (e.g. each factor of a product).

    ``label`` is the default UI presentation for the role ("Factor",
    "Additive inputs") — what an add-port affordance shows before a port
    exists. An instantiated port's own label/identifier specializes it.
    """

    role: MixedCaseIdentifier
    identifier: MixedCaseIdentifier | None = None
    label: str | Promise | None = None
    multi: bool = False
    repeatable: bool = False
    required: bool = True
    """Whether computation requires at least one binding for this role."""
    aggregation: Literal['sum'] | None = None
    """Optional operation that combines a multi port into one delivered value."""
    min_count: int = 1
    """Minimum number of port instances of this role for a valid node."""
    default_count: int | None = None
    """Port instances created by default at node creation; defaults to ``min_count``."""

    def __post_init__(self) -> None:
        if self.multi and self.repeatable:
            raise ValueError(f'Port role {self.role!r}: multi and repeatable are mutually exclusive')
        if self.min_count < 0:
            raise ValueError(f'Port role {self.role!r}: min_count must be non-negative')
        if self.default_count is not None and self.default_count < self.min_count:
            raise ValueError(f'Port role {self.role!r}: default_count may not be below min_count')
        if not self.repeatable and max(self.min_count, self.default_count or 0) > 1:
            raise ValueError(f'Port role {self.role!r}: only a repeatable role may have more than one instance')
        if self.aggregation is not None and not self.multi:
            raise ValueError(f'Port role {self.role!r}: aggregation requires a multi port')

    @property
    def instance_identifier(self) -> MixedCaseIdentifier:
        return self.identifier or self.role

    @property
    def effective_default_count(self) -> int:
        return self.default_count if self.default_count is not None else self.min_count


class InputPort:
    """Concise constructors for class-level input port declarations."""

    @staticmethod
    def one(role: MixedCaseIdentifier, *, label: str | Promise | None = None) -> InputPortDeclaration:
        return InputPortDeclaration(role=role, label=label)

    @staticmethod
    def optional(role: MixedCaseIdentifier, *, label: str | Promise | None = None) -> InputPortDeclaration:
        return InputPortDeclaration(role=role, label=label, required=False)

    @staticmethod
    def multi(
        role: MixedCaseIdentifier,
        *,
        required: bool = True,
        aggregation: Literal['sum'] | None = None,
        label: str | Promise | None = None,
    ) -> InputPortDeclaration:
        return InputPortDeclaration(
            role=role,
            label=label,
            multi=True,
            required=required,
            aggregation=aggregation,
        )

    @staticmethod
    def repeatable(
        role: MixedCaseIdentifier,
        *,
        min_count: int = 1,
        default_count: int | None = None,
        label: str | Promise | None = None,
    ) -> InputPortDeclaration:
        return InputPortDeclaration(
            role=role,
            label=label,
            repeatable=True,
            min_count=min_count,
            default_count=default_count,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputPortDeclaration:
    """Class-level semantic output role mapped to an instance port identifier."""

    role: MixedCaseIdentifier
    identifier: MixedCaseIdentifier
    label: str | Promise | None = None


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
    role: MixedCaseIdentifier | None = None
    """
    Semantic role linking this port to its class-level ``InputPortDeclaration``.

    Unlike ``identifier`` (a human/formula name, freely renameable), the role is
    fixed class vocabulary: shape rules and computation resolve ports through it.
    Not unique within the node — every instance of a repeatable role carries the
    same role string.
    """
    quantity: QuantityKindRef | None = None
    unit: Unit | None = None
    multi: bool = False
    """When True, the port accepts multiple connections (aggregated by the node's computation)."""
    paired_output_port_id: UUID | None = None
    """Output port produced from this input by nodes with paired metric impacts."""
    is_editable: bool = True
    """Whether the port definition itself may be edited; bindings remain independently editable."""
    required_dimensions: UniqueList[DimensionRef] = Field(default_factory=list)

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
    role: MixedCaseIdentifier | None = None
    """Semantic role linking this port to its class-level ``OutputPortDeclaration``."""
    quantity: QuantityKindRef | None = None
    unit: Unit
    column_id: MixedCaseIdentifier | None = None
    """DataFrame column name for this port. When None, inferred by the loader."""
    is_editable: bool = True
    """Whether the user can modify this port in the model editor."""
    dimensions: UniqueList[DimensionRef] = Field(default_factory=list)

    _metric_id: str | None = PrivateAttr(default=None)

    _node: 'Node | None' = PrivateAttr(default=None)


def pair_input_ports_to_outputs(
    input_ports: list[InputPortDef],
    output_ports: list[OutputPortDef],
    *,
    role: MixedCaseIdentifier,
    keep_unpaired: bool = True,
) -> list[InputPortDef]:
    """Pair legacy or newly generated single-metric inputs with output ports."""
    remaining = list(input_ports)
    paired: list[InputPortDef] = []

    explicit = {port.paired_output_port_id: port for port in remaining if port.paired_output_port_id is not None}
    used: set[UUID] = set()
    for index, output in enumerate(output_ports):
        port = explicit.get(output.id)
        if port is None:
            output_names = {str(value) for value in (output.identifier, output.column_id) if value is not None}
            port = next(
                (
                    candidate
                    for candidate in remaining
                    if candidate.id not in used and candidate.identifier is not None and str(candidate.identifier) in output_names
                ),
                None,
            )
        if port is None and len(input_ports) == len(output_ports) and index < len(input_ports):
            candidate = input_ports[index]
            if candidate.id not in used:
                port = candidate
        if port is None and len(input_ports) == len(output_ports) == 1:
            port = input_ports[0]
        if port is None:
            port = InputPortDef(
                id=uuid5(NAMESPACE_URL, f'kausal-paths:paired-input-port:{output.id}'),
            )

        used.add(port.id)
        identifier = output.identifier or output.column_id
        paired.append(
            port.model_copy(
                update={
                    'identifier': identifier,
                    'label': output.label,
                    'role': role,
                    'quantity': output.quantity,
                    'unit': output.unit,
                    'multi': False,
                    'paired_output_port_id': output.id,
                    'is_editable': False,
                }
            )
        )

    if keep_unpaired:
        paired.extend(port for port in remaining if port.id not in used)
    return paired
