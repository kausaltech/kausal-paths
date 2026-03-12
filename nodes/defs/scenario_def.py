from __future__ import annotations

from pydantic import BaseModel


class ScenarioParameterOverride(BaseModel):
    """A single parameter override within a scenario (stored in Scenario.parameter_overrides)."""

    parameter_id: str
    value: float | bool | str
    node_id: str | None = None
