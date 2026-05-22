from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString, I18nStringInstance, TranslatedString, set_i18n_context

from paths.identifiers import ActionGroupIdentifier

from nodes.constants import KNOWN_QUANTITIES
from nodes.scenario import Scenario
from nodes.units import Unit
from pages.config import OutcomePage
from params.discover import AnyParameter

from .action_def import ImpactOverviewSpec


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

    url: str
    commit: str | None = None
    dvc_remote: str | None = None


class ActionGroup(I18nBaseModel):
    """A named group of action nodes."""

    id: ActionGroupIdentifier
    name: I18nString | None = None
    color: str | None = None
    order: int = 0


class InstanceResultExcelSpec(I18nBaseModel):
    name: I18nStringInstance
    base_excel_url: str | None
    node_ids: list[str] | None = None
    action_ids: list[str] | None = None
    format: str = 'long'


class InstanceTerms(I18nBaseModel):
    """
    The specific terms to be used for different concepts in the UI (e.g. "action").

    If a term is not set, the default term for the concept will be used.
    """

    action: TranslatedString | None = None
    enabled_label: TranslatedString | None = None


class InstanceFeatures(BaseModel):
    """
    Features available for the instance.

    Used mostly by the UI to customize the display of the results.
    """

    baseline_visible_in_graphs: bool = True
    """Whether to display the baseline data in graphs and visualizations."""

    show_accumulated_effects: bool = True
    """Whether to display accumulated effects over time in the UI."""

    show_significant_digits: int | None = 3
    """Number of significant digits to display in numerical results. None means no limit."""

    maximum_fraction_digits: int | None = None
    """Maximum number of decimal places to display after the decimal point. None means no limit."""

    hide_node_details: bool = False
    """Whether to hide detailed node information in the UI."""

    show_refresh_prompt: bool = False
    """Whether to show a prompt to refresh data when it might be outdated."""

    requires_authentication: bool = False
    """Whether authentication is required to access this instance."""

    use_datasets_from_db: bool = False
    """Whether to use datasets from the database instead of the .parquet files."""

    show_explanations: bool = False
    """Whether to show node explanation in the slot for description (under the graph)."""

    show_category_warnings: bool = False
    """Whether to show category warnings in the node explanation."""

    model_config = ConfigDict(use_attribute_docstrings=True)


class NormalizationQuantitySpec(BaseModel):
    """A quantity that can be normalized by a specific normalizer node."""

    id: str
    """The quantity identifier affected by this normalization."""

    unit: Unit
    """The unit expected after normalization has been applied."""

    @field_validator('id')
    @classmethod
    def validate_quantity_id(cls, value: str) -> str:
        """Ensure the normalization only references known quantity identifiers."""
        if value not in KNOWN_QUANTITIES:
            raise ValueError(f'Unknown quantity: {value}')
        return value


class NormalizationSpec(BaseModel):
    """Serialized normalization configuration stored in instance specs."""

    normalizer_node_id: str
    """The node id whose output is used as the normalization divisor."""

    quantities: list[NormalizationQuantitySpec]
    """Quantities that this normalization can be applied to."""

    default: bool = False
    """Whether this normalization should be activated by default for the instance."""


class InstanceSpec(I18nBaseModel):
    """Computation schema for a model instance, stored as a SchemaField on InstanceConfig."""

    uuid: UUID = Field(default_factory=uuid4)
    identifier: str = ''
    name: I18nString = ''
    owner: I18nString | None = None
    primary_language: str = 'en'
    other_languages: list[str] = Field(default_factory=list)

    years: YearsSpec = YearsSpec()
    dataset_repo: DatasetRepoSpec | None = None
    features: InstanceFeatures = Field(default_factory=InstanceFeatures)
    terms: InstanceTerms = Field(default_factory=InstanceTerms)
    result_excels: list[InstanceResultExcelSpec] = Field(default_factory=list)
    pages: list[OutcomePage] = Field(default_factory=list)
    impact_overviews: list[ImpactOverviewSpec] = Field(default_factory=list)
    normalizations: list[NormalizationSpec] = Field(default_factory=list)
    params: list[AnyParameter] = Field(default_factory=list)
    action_groups: list[ActionGroup] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(default_factory=list)
    theme_identifier: str | None = None
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
