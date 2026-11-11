"""GraphQL types for the figures stored on submissions, and for sums of them."""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import strawberry as sb

from frameworks.composition import ComposedIndicator  # noqa: TC001

if TYPE_CHECKING:
    from frameworks.models import IndicatorValue, SubmissionIndicators


@sb.enum(name='IndicatorStatus')
class IndicatorStatusEnum(Enum):
    COMPUTED = 'computed'
    NOT_COMPUTABLE = 'not_computable'


@sb.enum(name='IndicatorComposition')
class IndicatorCompositionEnum(Enum):
    SUM = 'sum'
    WEIGHTED_MEAN = 'weighted_mean'
    NONE = 'none'


@sb.type(name='IndicatorCategory')
class IndicatorCategoryType:
    dimension: str
    category: str


@sb.type(name='IndicatorCell')
class IndicatorCellType:
    categories: list[IndicatorCategoryType]
    value: float


@sb.type(name='IndicatorValue', description='One declared figure of one year, or the composition of several.')
class IndicatorValueType:
    id: str
    unit: str
    dimensions: list[str]
    composition: IndicatorCompositionEnum
    weight: str | None = sb.field(description='The indicator that weights a weighted mean.')
    total: float | None = sb.field(description='The figure over all cells; null when there is no value for the year.')
    cells: list[IndicatorCellType] = sb.field(description='The figure by the kept dimensions.')

    @classmethod
    def from_value(cls, identifier: str, value: IndicatorValue) -> IndicatorValueType:
        return cls(
            id=identifier,
            unit=value.unit,
            dimensions=value.dimensions,
            composition=IndicatorCompositionEnum(value.composition),
            weight=value.weight,
            total=value.total,
            cells=[
                IndicatorCellType(
                    categories=[IndicatorCategoryType(dimension=d, category=c) for d, c in cell.categories.items()],
                    value=cell.value,
                )
                for cell in value.cells
            ],
        )


@sb.type(name='ComposedIndicator', description='One figure composed over the final submissions of a territory.')
class ComposedIndicatorType:
    value: IndicatorValueType
    contributor_count: int = sb.field(description='How many submissions the figure rests on.')

    @classmethod
    def from_composed(cls, identifier: str, composed: ComposedIndicator) -> ComposedIndicatorType:
        return cls(value=IndicatorValueType.from_value(identifier, composed.value), contributor_count=composed.contributors)


@sb.type(name='IndicatorCompleteness')
class IndicatorCompletenessType:
    met: int = sb.field(description='Requirements of the year that are met.')
    applicable: int = sb.field(description='Requirements that apply in the year: one required value is one requirement.')
    open_plausibility_findings: int

    @sb.field(description='Met requirements as a share of the applicable ones; null when none apply.')
    def share(self) -> float | None:
        return self.met / self.applicable if self.applicable else None


@sb.type(name='IndicatorSettingValue', description='One parameter and its value; exactly one of the value fields is set.')
class IndicatorSettingValueType:
    parameter: str
    bool_value: bool | None
    number_value: float | None
    string_value: str | None

    @classmethod
    def from_value(cls, parameter: str, value: bool | float | str | None) -> IndicatorSettingValueType:
        return cls(
            parameter=parameter,
            bool_value=value if isinstance(value, bool) else None,
            number_value=float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
            string_value=value if isinstance(value, str) else None,
        )


@sb.type(name='SubmissionIndicators')
class SubmissionIndicatorsType:
    """
    The figures a model gives for a submission's year.

    An open submission's are rewritten as the draft changes; a final submission's
    are frozen with it.
    """

    status: IndicatorStatusEnum
    reason: str | None = sb.field(description='Why the model could not be computed.')
    year: int
    values: list[IndicatorValueType]
    completeness: IndicatorCompletenessType | None
    declarations: list[IndicatorSettingValueType] = sb.field(description="The reporter's own declarations, as in effect.")
    settings: list[IndicatorSettingValueType] = sb.field(description='The parameter values the figures were computed under.')
    computed_at: datetime
    is_stale: bool = sb.field(
        description='The draft has changed since the figures were computed. Never true for a final submission.'
    )

    @sb.field(description='One figure by its identifier.')
    def value(self, id: str) -> IndicatorValueType | None:
        return next((value for value in self.values if value.id == id), None)

    @classmethod
    def from_indicators(cls, indicators: SubmissionIndicators, *, is_stale: bool) -> SubmissionIndicatorsType:
        completeness = indicators.completeness
        return cls(
            status=IndicatorStatusEnum(indicators.status),
            reason=indicators.reason,
            year=indicators.year,
            values=[IndicatorValueType.from_value(identifier, value) for identifier, value in indicators.values.items()],
            completeness=(
                IndicatorCompletenessType(
                    met=completeness.met,
                    applicable=completeness.applicable,
                    open_plausibility_findings=completeness.open_plausibility_findings,
                )
                if completeness is not None
                else None
            ),
            declarations=[
                IndicatorSettingValueType.from_value(parameter, value) for parameter, value in indicators.declarations.items()
            ],
            settings=[
                IndicatorSettingValueType.from_value(parameter, value)
                for parameter, value in indicators.provenance.settings.items()
            ],
            computed_at=indicators.provenance.computed_at,
            is_stale=is_stale,
        )
