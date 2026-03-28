from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString, set_i18n_context

from paths.identifiers import ActionGroupIdentifier

from nodes.actions.action import ImpactOverviewSpec
from nodes.excel_results import InstanceResultExcel
from nodes.instance import InstanceFeatures, InstanceTerms
from nodes.normalization import NormalizationSpec
from nodes.scenario import Scenario
from pages.config import OutcomePage
from params.discover import AnyParameter


class YearsSpec(I18nBaseModel):
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


class DatasetRepoSpec(I18nBaseModel):
    """Pointer to the DVC dataset repository."""

    url: str = ''
    commit: str | None = None
    dvc_remote: str | None = None


class ActionGroup(I18nBaseModel):
    """A named group of action nodes."""

    id: ActionGroupIdentifier
    name: I18nString | None = None
    color: str | None = None
    order: int = 0


class InstanceSpec(I18nBaseModel):
    """Computation schema for a model instance, stored as a SchemaField on InstanceConfig."""

    primary_language: str
    other_languages: list[str] = Field(default_factory=list)

    years: YearsSpec = YearsSpec()
    dataset_repo: DatasetRepoSpec | None = None
    features: InstanceFeatures = Field(default_factory=InstanceFeatures)
    terms: InstanceTerms = Field(default_factory=InstanceTerms)
    result_excels: list[InstanceResultExcel] = Field(default_factory=list)
    pages: list[OutcomePage] = Field(default_factory=list)
    impact_overviews: list[ImpactOverviewSpec] = Field(default_factory=list)
    normalizations: list[NormalizationSpec] = Field(default_factory=list)
    params: list[AnyParameter] = Field(default_factory=list)
    action_groups: list[ActionGroup] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(default_factory=list)
    # Raw dimension configs — will be properly modeled later
    dimensions: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode='wrap')
    @classmethod
    def _with_i18n_context(cls, values: Any, handler: Any) -> InstanceSpec:
        if isinstance(values, dict):
            lang = values.get('primary_language')
            others = values.get('other_languages', [])
            if lang is not None:
                with set_i18n_context(lang, others):
                    return handler(values)
        return handler(values)


InstanceSpec.model_rebuild()
