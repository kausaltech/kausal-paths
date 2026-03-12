from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ParameterDef(BaseModel):
    """Definition of a parameter (used in both InstanceConfig.parameters and NodeConfig.params)."""

    id: str
    label: dict[str, str] | None = None
    type: Literal['number', 'bool', 'enum'] = 'number'
    unit: str | None = None
    value: float | bool | str | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    is_visible: bool = True
    is_customizable: bool = True
    enum_options: list[EnumOption] | None = None


class EnumOption(BaseModel):
    """An option for an enum parameter."""

    id: str
    label: dict[str, str] | None = None


# Rebuild ParameterDef to resolve forward reference to EnumOption
ParameterDef.model_rebuild()
