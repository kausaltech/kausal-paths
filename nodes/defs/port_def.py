from __future__ import annotations

from pydantic import Field

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString

from paths.identifiers import NodePortIdentifier


class InputPortDef(I18nBaseModel):
    """Definition of a node input port (stored in NodeConfig.input_ports JSONField)."""

    id: NodePortIdentifier
    label: I18nString | None = None
    quantity: str = ''
    unit: str = ''
    required_dimensions: list[str] = Field(default_factory=list)
    supported_dimensions: list[str] = Field(default_factory=list)


class OutputPortDef(I18nBaseModel):
    """Definition of a node output port (stored in NodeConfig.output_ports JSONField)."""

    id: NodePortIdentifier
    label: I18nString | None = None
    quantity: str = ''
    unit: str = ''
    dimensions: list[str] = Field(default_factory=list)
