from __future__ import annotations

from pydantic import BaseModel, Field

from kausal_common.i18n.pydantic import TranslatedString

from common.types import Identifier


class InputPortDef(BaseModel):
    """Definition of a node input port (stored in NodeConfig.input_ports JSONField)."""

    id: Identifier
    label: TranslatedString | None = None
    quantity: str = ''
    unit: str = ''
    required_dimensions: list[str] = Field(default_factory=list)
    supported_dimensions: list[str] = Field(default_factory=list)


class OutputPortDef(BaseModel):
    """Definition of a node output port (stored in NodeConfig.output_ports JSONField)."""

    id: Identifier
    label: TranslatedString | None = None
    quantity: str = ''
    unit: str = ''
    dimensions: list[str] = Field(default_factory=list)
