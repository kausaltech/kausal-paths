"""Temporal cleanup preserves the legacy interpolation and extension policy."""

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from common.polars import DataFrameMeta, to_ppdf
from nodes.units import unit_registry

if TYPE_CHECKING:
    from nodes.context import Context

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize('missing_year', [True, False])
@pytest.mark.parametrize('dimensioned', [True, False])
def test_interpolation_and_backward_fill(context: Context, missing_year: bool, dimensioned: bool) -> None:
    frame = pl.DataFrame({
        'Year': [2020, 2021, 2022, 2023, 2024],
        'Value': [None, 2.0, None, 6.0, None],
        'Other': [1.0, None, None, 7.0, None],
        'Forecast': [False, False, None, True, True],
    })
    keys = ['Year']
    if missing_year:
        frame = frame.filter(pl.col('Year') != 2022)
    if dimensioned:
        frame = frame.with_columns(pl.lit('buildings').alias('sector'))
        keys.append('sector')
    units = dict.fromkeys(['Value', 'Other'], unit_registry.kg)
    df = to_ppdf(frame, DataFrameMeta(units=units, primary_keys=keys))
    result = df.paths._add_missing_years(df, context)
    expected = pl.DataFrame({
        'Year': [2020, 2021, 2022, 2023, 2024],
        'Value': [2.0, 2.0, 4.0, 6.0, None],
        'Other': [1.0, 3.0, 5.0, 7.0, None],
        'Forecast': [False, False, True, True, True],
    })
    if dimensioned:
        expected = expected.with_columns(pl.lit('buildings').cast(pl.Categorical).alias('sector'))
    assert_frame_equal(result.sort(keys).select(expected.columns), expected)
    assert result.get_meta().is_equal(DataFrameMeta(units=units, primary_keys=keys))


def test_unsorted_years_keep_legacy_join_order(context: Context) -> None:
    df = to_ppdf(
        pl.DataFrame({'Year': [2020, 2022, 2021], 'Value': [1.0, None, 3.0]}),
        DataFrameMeta(units={'Value': unit_registry.kg}, primary_keys=['Year']),
    )
    result = df.paths._add_missing_years(df, context)
    assert_frame_equal(
        result.sort('Year'),
        pl.DataFrame({'Year': [2020, 2021, 2022], 'Value': [1.0, 3.0, None], 'Forecast': [False] * 3}),
    )


def test_empty_to_zero_keeps_an_empty_dimensioned_frame(context: Context) -> None:
    """An empty frame has nothing to fill; the wide round trip must not cost it its metric column."""
    frame = pl.DataFrame(
        {'Year': [], 'sector': [], 'Value': []},
        schema={'Year': pl.Int64, 'sector': pl.String, 'Value': pl.Float64},
    )
    df = to_ppdf(frame, DataFrameMeta(units={'Value': unit_registry.kg}, primary_keys=['Year', 'sector']))
    result = df.paths._empty_to_zero(df, context)
    assert result.height == 0
    assert result.metric_cols == ['Value']
    assert set(result.primary_keys) == {'Year', 'sector'}
