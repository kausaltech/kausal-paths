from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from kausal_common.i18n.pydantic import TranslatedString

from common.types import ActionGroupIdentifier, NodeIdentifier, ParameterGlobalId, ScenarioIdentifier
from nodes.scenario import ScenarioKind
from params.discover import AnyParameter


class YearsDef(BaseModel):
    """Year boundaries for the model instance."""

    reference: int | None = None
    min_historical: int | None = None
    max_historical: int | None = None
    target: int | None = None
    model_end: int | None = None


class DatasetRepoDef(BaseModel):
    """Pointer to the DVC dataset repository."""

    url: str = ''
    commit: str | None = None
    dvc_remote: str | None = None


class ActionGroupDef(BaseModel):
    """A named group of action nodes."""

    id: ActionGroupIdentifier
    name: TranslatedString | None = None
    color: str = ''
    order: int = 0


class ScenarioParameterOverrideDef(BaseModel):
    """A single parameter override within a scenario."""

    parameter_id: ParameterGlobalId
    value: float | bool | str
    node_id: NodeIdentifier | None = None


class ScenarioDef(BaseModel):
    """A named scenario with parameter overrides."""

    id: ScenarioIdentifier
    name: TranslatedString | None = None
    description: TranslatedString | None = None
    kind: ScenarioKind | None = None
    all_actions_enabled: bool = False
    params: list[ScenarioParameterOverrideDef] = Field(default_factory=list)


class InstanceSpec(BaseModel):
    """Computation schema for a model instance, stored as a SchemaField on InstanceConfig."""

    years: YearsDef = YearsDef()
    dataset_repo: DatasetRepoDef = DatasetRepoDef()
    features: dict[str, object] = Field(default_factory=dict)
    params: list[AnyParameter] = Field(default_factory=list)
    action_groups: list[ActionGroupDef] = Field(default_factory=list)
    scenarios: list[ScenarioDef] = Field(default_factory=list)
    # Raw dimension configs — will be properly modeled later
    dimensions: list[dict[str, Any]] = Field(default_factory=list)
