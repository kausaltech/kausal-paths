from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from kausal_common.i18n.pydantic import TranslatedString

from common.types import ActionGroupIdentifier
from nodes.instance import InstanceFeatures
from nodes.scenario import Scenario
from params.discover import AnyParameter


class YearsSpec(BaseModel):
    """Year boundaries for the model instance."""

    reference: int | None = None
    min_historical: int | None = None
    """The earliest year for which historical data is available.

    This is commonly also the minimum year for which outputs will be generated.
    """

    max_historical: int | None = None
    """The most recent year for which historical data is available.

    The years after this year are generally considered forecasted years.
    """

    target: int | None = None
    """The target year for the model (e.g. the year for the most important future goal)."""

    model_end: int | None = None
    """The last year for which the model will be run.

    This is typically the same as the target year, but may be different if computation
    results are needed beyond the target year.
    """


class DatasetRepoSpec(BaseModel):
    """Pointer to the DVC dataset repository."""

    url: str = ''
    commit: str | None = None
    dvc_remote: str | None = None


class ActionGroup(BaseModel):
    """A named group of action nodes."""

    id: ActionGroupIdentifier
    name: TranslatedString | None = None
    color: str = ''
    order: int = 0


class InstanceSpec(BaseModel):
    """Computation schema for a model instance, stored as a SchemaField on InstanceConfig."""

    years: YearsSpec = YearsSpec()
    dataset_repo: DatasetRepoSpec | None = None
    features: InstanceFeatures = Field(default_factory=InstanceFeatures)
    params: list[AnyParameter] = Field(default_factory=list)
    action_groups: list[ActionGroup] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(default_factory=list)
    # Raw dimension configs — will be properly modeled later
    dimensions: list[dict[str, Any]] = Field(default_factory=list)
