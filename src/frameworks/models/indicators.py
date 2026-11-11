"""The figures stored on a submission: what its model gave for its year, and what they rest on."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from nodes.defs.indicator_defs import IndicatorComposition

type SettingValue = bool | float | str


class IndicatorCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    categories: dict[str, str]
    """Category identifier by dimension identifier."""
    value: float


class IndicatorValue(BaseModel):
    """
    One declared figure for one year.

    It carries its own composition rule, so that a sum over submissions frozen
    under different template revisions reads each one's rule from the submission
    itself and not from today's template.
    """

    unit: str
    dimensions: list[str] = Field(default_factory=list)
    composition: IndicatorComposition
    weight: str | None = None
    total: float | None
    """The figure over all cells; null when the model gave no value for the year."""
    cells: list[IndicatorCell] = Field(default_factory=list)
    """The figure by the kept dimensions; empty for a figure without dimensions."""


class Completeness(BaseModel):
    """How many of the year's requirements are met: one required value in one year is one requirement."""

    met: int
    applicable: int
    open_plausibility_findings: int


class IndicatorProvenance(BaseModel):
    computed_at: datetime
    draft_head_token: UUID | None = None
    """The instance's draft head when an open submission's figures were computed."""
    instance_revision: int | None = None
    """The published revision a final submission's figures were computed from."""
    template_revision: int | None = None
    settings: dict[str, SettingValue] = Field(default_factory=dict)
    """The parameter values the template prescribed for the computation."""


class SubmissionIndicators(BaseModel):
    status: Literal['computed', 'not_computable']
    reason: str | None = None
    """Why the model could not be computed. A draft can legitimately fail halfway through data entry."""
    year: int
    values: dict[str, IndicatorValue] = Field(default_factory=dict)
    completeness: Completeness | None = None
    declarations: dict[str, SettingValue | None] = Field(default_factory=dict)
    """Effective values of the parameters the template names as the reporter's own declarations."""
    provenance: IndicatorProvenance
