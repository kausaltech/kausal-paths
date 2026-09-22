"""The local settings that do not change a shared node's calculation definition."""

from uuid import UUID

from pydantic import BaseModel, Field

from nodes.goals import NodeGoals
from nodes.instance_serialization import NodeLayoutSnapshot


class InheritedNodeSettings(BaseModel):
    node_uuid: UUID
    goals: NodeGoals | None = None
    layout: NodeLayoutSnapshot | None = None
    parameter_values: dict[str, bool | float | str | None] = Field(default_factory=dict)
    parameter_sources: dict[str, str] = Field(default_factory=dict)
