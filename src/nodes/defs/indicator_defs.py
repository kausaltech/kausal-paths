"""
What a model reports about one year of a submission, and how each figure composes.

The declaration lives in the instance spec, so a framework's template names its
own nodes and parameters and no node identifier is written into the code that
stores or sums the figures.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

type IndicatorComposition = Literal['sum', 'weighted_mean', 'none']


class IndicatorSetting(BaseModel):
    """One parameter value the figures are computed under, whatever a visitor's session holds."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    parameter: str
    value: bool | float | str


class IndicatorSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    id: str
    node: str
    """Identifier of the node whose output for the submission's year is stored."""

    dimensions: tuple[str, ...] = ()
    """The dimensions kept; the node's other dimensions are summed away."""

    composition: IndicatorComposition = 'sum'
    """
    How the figure composes over several submissions.

    ``sum`` adds it cell by cell. ``weighted_mean`` divides the summed products of
    value and weight by the summed weights; it is never averaged. ``none`` is a
    figure that says nothing about a territory, such as an intensity.
    """

    weight: str | None = None
    """The indicator that weights a ``weighted_mean``."""

    settings: tuple[IndicatorSetting, ...] = ()
    """Parameter values for this indicator alone, on top of the shared ones."""

    @model_validator(mode='after')
    def validate_composition(self) -> Self:
        if (self.composition == 'weighted_mean') != (self.weight is not None):
            raise ValueError(f'Indicator {self.id}: a weight belongs to a weighted_mean, and a weighted_mean needs one')
        if self.dimensions and self.composition != 'sum':
            raise ValueError(f'Indicator {self.id}: only a summed indicator keeps dimensions')
        return self


class SubmissionIndicatorsSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    settings: tuple[IndicatorSetting, ...] = ()
    """The parameter values every indicator is computed under, e.g. the settings a method prescribes."""

    indicators: tuple[IndicatorSpec, ...] = Field(default=())

    declarations: tuple[str, ...] = ()
    """Parameters whose effective values are recorded with the figures: what the reporter declared."""

    @model_validator(mode='after')
    def validate_references(self) -> Self:
        by_id = {indicator.id: indicator for indicator in self.indicators}
        if len(by_id) != len(self.indicators):
            raise ValueError('Indicator ids must be unique')
        for indicator in self.indicators:
            if indicator.weight is None:
                continue
            weight = by_id.get(indicator.weight)
            if weight is None or weight.composition != 'sum' or weight.settings != indicator.settings:
                raise ValueError(
                    f'Indicator {indicator.id}: its weight must be a summed indicator computed under the same settings'
                )
        return self
