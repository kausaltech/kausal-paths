"""
Fallback plausibility ranges derived from a dataset's own history.

Where no curated reference covers a metric, its past values are still evidence
of how it usually moves. For each metric of an editable dataset, the log ratios
between consecutive years of every cell are pooled, and a year-over-year band is
set around their median, several robust spreads wide. A band derived this way
says nothing about what is plausible for a municipality; it says a value is far
outside how this series moves, which is what a slipped factor of 1000, kWh typed
for MWh, or a misplaced decimal point look like.

The bands are computed when the dataset is checked, not stored: a stored band
would freeze the history it was derived from. They are guesses to be tuned, and
the constants below are set by hand, not derived.

- The median and MAD are robust, so a typo, or a real break, already in the
  history hardly moves the band; no value has to be left out to check itself.
- Many historical series are interpolated or scaled from national trends and
  are smoother than values a city enters, so a band is never narrower than
  ``1 / MIN_RATIO_SPAN``-``MIN_RATIO_SPAN``.
- Small cells move erratically in relative terms. A cell whose earlier value is
  below ``SMALL_CELL_SHARE`` of the metric's median value is not judged, and is
  not used to derive the band.
- Only positive values are judged. A value of exactly zero is a statement that
  something stopped, not a slip, and a ratio says nothing across a sign change:
  savings in action datasets and sinks among emission factors are negative by
  nature.

The first tuning pass ran over the local German instances: 92 bands over some
20 000 ratios, of which only 8 came out wider than the minimum, so in practice
the minimum is the band. At a minimum of 2 it flagged 1.7% of ratios, most of
them within a factor of 5; at 5 it flags 0.6%, and among those are ÷1000 and
÷100 steps that look like unit and decimal slips.
"""

import math
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING
from uuid import UUID, uuid5

import polars as pl

from datasets.models import DatasetMetricPlausibilityRange, PlausibilitySource
from frameworks.evidence import QUALITY_OF_SPEC_KEY
from nodes.constants import YEAR_COLUMN

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kausal_common.datasets.models import Dataset, DatasetMetric

Range = DatasetMetricPlausibilityRange

METHOD_VERSION = 1
SOURCE_IDENTIFIER = 'dataset-history'
MAD_MULTIPLIER = 5.0
MIN_RATIO_SPAN = 5.0
MIN_RATIOS = 20
SMALL_CELL_SHARE = 0.01
# MAD of a normal distribution times this is its standard deviation.
_MAD_TO_SIGMA = 1.4826
# Legacy grade columns, before `import_quality_evidence` turns them into evidence and a
# `quality_of` projection. A grade is ordinal, so a ratio between two says nothing.
LEGACY_QUALITY_METRIC = 'quality'
_NAMESPACE = UUID('6f0c7f7e-3b8e-4c55-9d27-0e9a2b8f4d61')

SOURCE_METHOD = f"""\
Derived when the dataset is checked, from the dataset itself: the log ratios
between consecutive years of every cell of a metric, pooled over its cells. The
band is the median ratio {MAD_MULTIPLIER:g} robust standard deviations (1.4826 x MAD)
either way, and never narrower than 1/{MIN_RATIO_SPAN:g}-{MIN_RATIO_SPAN:g}. Cells whose
earlier value is below {SMALL_CELL_SHARE:.0%} of the metric's median value are neither
judged nor used, and only positive values are judged. A metric needs {MIN_RATIOS} ratios for a band. The constants are set
by hand; the band guards against order-of-magnitude errors, and says nothing about
what is plausible for a municipality."""


@dataclass(frozen=True)
class HistoryRange:
    """A derived range, and the earlier value below which it does not judge a cell."""

    rule: DatasetMetricPlausibilityRange
    min_reference: float


@cache
def history_source() -> PlausibilitySource:
    """Return the source every derived range cites; it is never saved."""
    return PlausibilitySource(
        uuid=uuid5(_NAMESPACE, SOURCE_IDENTIFIER),
        identifier=SOURCE_IDENTIFIER,
        name="This dataset's history",
        url='',
        revision=f'method-v{METHOD_VERSION}',
        method=SOURCE_METHOD,
        is_example=False,
    )


def eligible(dataset: Dataset) -> bool:
    """Whether a dataset gets derived ranges: only data someone can correct is worth warning about."""
    schema = dataset.schema
    if schema is None or not schema.is_editable or dataset.is_external_placeholder:
        return False
    return schema.metrics.exists()


def _round(value: float) -> float:
    return float(f'{value:.3g}')


def _log_ratios(frame: pl.DataFrame, column: str, dim_cols: list[str]) -> tuple[pl.Series, float] | None:
    """Log ratios between consecutive years of each cell, and the small-cell floor they were taken above."""
    values = frame.select(YEAR_COLUMN, *dim_cols, pl.col(column).cast(pl.Float64).alias('value')).filter(
        pl.col('value').is_finite() & (pl.col('value') > 0)
    )
    if values.is_empty():
        return None
    median = values['value'].median()
    assert isinstance(median, float)
    floor = SMALL_CELL_SHARE * median
    ordered = values.sort(*dim_cols, YEAR_COLUMN)
    earlier_value, earlier_year = pl.col('value').shift(1), pl.col(YEAR_COLUMN).shift(1)
    if dim_cols:
        earlier_value, earlier_year = earlier_value.over(dim_cols), earlier_year.over(dim_cols)
    pairs = ordered.with_columns(earlier=earlier_value, earlier_year=earlier_year).filter(
        (pl.col(YEAR_COLUMN) - pl.col('earlier_year') == 1) & (pl.col('earlier') >= floor)
    )
    return (pairs['value'] / pairs['earlier']).log(), floor


def derive_metric_range(dataset: Dataset, metric: DatasetMetric, frame: pl.DataFrame, dim_cols: list[str]) -> HistoryRange | None:
    column = metric.name or metric.label or str(metric.uuid)
    if column not in frame.columns:
        return None
    derived = _log_ratios(frame, column, dim_cols)
    if derived is None:
        return None
    logs, floor = derived
    if logs.len() < MIN_RATIOS:
        return None
    median = logs.median()
    mad = (logs - median).abs().median()
    assert isinstance(median, float)
    assert isinstance(mad, float)
    spread = MAD_MULTIPLIER * _MAD_TO_SIGMA * mad
    lower = _round(min(math.exp(median - spread), 1 / MIN_RATIO_SPAN))
    upper = _round(max(math.exp(median + spread), MIN_RATIO_SPAN))
    rule = DatasetMetricPlausibilityRange(
        uuid=uuid5(_NAMESPACE, f'{dataset.uuid}/{metric.uuid}/{METHOD_VERSION}'),
        source=history_source(),
        metric=metric,
        identifier=SOURCE_IDENTIFIER,
        selection={},
        aggregation=Range.Aggregation.CELL,
        denominator=Range.Denominator.NONE,
        reference=Range.Reference.PREVIOUS_YEAR,
        max_gap_years=1,
        lower=lower,
        upper=upper,
        sample_size=logs.len(),
        revision=METHOD_VERSION,
        enabled=True,
    )
    return HistoryRange(rule=rule, min_reference=floor)


def derive_history_ranges(
    dataset: Dataset, frame: pl.DataFrame, dim_cols: list[str], curated: Iterable[DatasetMetricPlausibilityRange]
) -> list[HistoryRange]:
    """
    Derive a range for each metric that no curated cell range covers.

    A curated sum range does not displace a derived one: attribution already brings
    their findings about one mistyped cell together.
    """
    if not eligible(dataset):
        return []
    assert dataset.schema is not None
    covered = {rule.metric_id for rule in curated if rule.aggregation == Range.Aggregation.CELL}
    ranges = []
    for metric in dataset.schema.metrics.all():
        if metric.pk in covered or QUALITY_OF_SPEC_KEY in (metric.spec or {}):
            continue
        if metric.name == LEGACY_QUALITY_METRIC and not metric.unit:
            continue
        derived = derive_metric_range(dataset, metric, frame, dim_cols)
        if derived is not None:
            ranges.append(derived)
    return ranges
