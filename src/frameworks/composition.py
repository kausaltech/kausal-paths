"""
Compose the stored figures of several submissions into one.

Each figure carries its own rule (`IndicatorValue.composition`), frozen with
its submission, so composing submissions finalised under different template
revisions needs nothing from today's template. A figure whose contributors
disagree on unit, dimensions or rule is not composed: a published sum adds
what was delivered and converts nothing.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from frameworks.models import IndicatorCell, IndicatorValue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from frameworks.models import SubmissionIndicators


@dataclass(frozen=True)
class ComposedIndicator:
    value: IndicatorValue
    contributors: int
    """How many submissions the figure rests on. Fewer than were asked for when some gave no value."""


@dataclass(frozen=True)
class Composition:
    indicators: dict[str, ComposedIndicator]
    incompatible: list[str]
    """Figures left out because their contributors disagree on unit, dimensions or rule."""
    template_revisions: int
    """How many different template revisions the contributors were finalised under."""


def _signature(value: IndicatorValue) -> tuple[str, tuple[str, ...], str, str | None]:
    return value.unit, tuple(value.dimensions), value.composition, value.weight


def _sum(values: Sequence[IndicatorValue]) -> IndicatorValue:
    first = values[0]
    cells: dict[tuple[str, ...], float] = defaultdict(float)
    for value in values:
        for cell in value.cells:
            cells[tuple(cell.categories[dimension] for dimension in first.dimensions)] += cell.value
    return first.model_copy(
        update={
            'total': sum(value.total for value in values if value.total is not None),
            'cells': [
                IndicatorCell(categories=dict(zip(first.dimensions, key, strict=True)), value=total)
                for key, total in sorted(cells.items())
            ],
        }
    )


def compose(members: Sequence[SubmissionIndicators]) -> Composition:
    """Compose every figure the computed members hold, each by its own rule."""
    computed = [member for member in members if member.status == 'computed']
    identifiers = sorted({identifier for member in computed for identifier in member.values})
    indicators: dict[str, ComposedIndicator] = {}
    incompatible: list[str] = []
    for identifier in identifiers:
        holders = [member for member in computed if identifier in member.values]
        signatures = {_signature(member.values[identifier]) for member in holders}
        if len(signatures) != 1:
            incompatible.append(identifier)
            continue
        first = holders[0].values[identifier]
        valued = [member for member in holders if member.values[identifier].total is not None]
        if first.composition == 'none' or not valued:
            continue
        if first.composition == 'sum':
            indicators[identifier] = ComposedIndicator(_sum([member.values[identifier] for member in valued]), len(valued))
            continue
        # A weighted mean: products of value and weight over the summed weights, never an average of the values.
        assert first.weight is not None
        if len({_signature(member.values[first.weight]) for member in valued if first.weight in member.values}) > 1:
            incompatible.append(identifier)
            continue
        weighted = [
            (member.values[identifier].total or 0.0, weight.total)
            for member in valued
            if (weight := member.values.get(first.weight)) is not None and weight.total is not None
        ]
        total_weight = sum(weight for _, weight in weighted)
        if not weighted or total_weight == 0:
            continue
        mean = sum(value * weight for value, weight in weighted) / total_weight
        indicators[identifier] = ComposedIndicator(first.model_copy(update={'total': mean, 'cells': []}), len(weighted))
    revisions = {member.provenance.template_revision for member in computed}
    return Composition(indicators=indicators, incompatible=incompatible, template_revisions=len(revisions))
