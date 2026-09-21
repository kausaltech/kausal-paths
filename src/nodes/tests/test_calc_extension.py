import polars as pl
import pytest
from polars.testing import assert_frame_equal

from common.polars import DataFrameMeta, to_ppdf
from nodes.calc import extend_last_historical_value_pl
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize('zero_fill_missing', [True, False])
@pytest.mark.parametrize('categorical', [True, False])
def test_extend_grouped_history(zero_fill_missing: bool, categorical: bool) -> None:
    # Ragged, unsorted groups, multiple metrics, and an existing forecast value.
    frame = pl.DataFrame({
        'Year': [2022, 2020, 2020, 2021],
        'sector': ['a', 'b', 'a', 'b'],
        'unused': [None] * 4,
        'Value': [5.0, None, 2.0, None],
        'Other': [None, 3.0, 1.0, 4.0],
        'Forecast': [True, False, False, False],
    })
    if categorical:
        frame = frame.with_columns(pl.col('sector').cast(pl.Categorical))
    units = dict.fromkeys(['Value', 'Other'], unit_registry.kg)
    df = to_ppdf(frame, meta=DataFrameMeta(units=units, primary_keys=['Year', 'sector', 'unused']))
    result = extend_last_historical_value_pl(df, 2023, zero_fill_missing=zero_fill_missing)
    missing = 0.0 if zero_fill_missing else None
    expected = pl.DataFrame({
        'sector': ['a'] * 4 + ['b'] * 4,
        'Year': [2020, 2021, 2022, 2023] * 2,
        'Value': [2.0, 2.0, 5.0, 5.0, None, None, missing, missing],
        'Other': [1.0, 1.0, 1.0, 1.0, 3.0, 4.0, 4.0, 4.0],
        'Forecast': [False, False, True, True] * 2,
    }).with_columns(pl.col('sector').cast(pl.Categorical))
    assert_frame_equal(result, expected)
    assert result.get_meta() == DataFrameMeta(units=units, primary_keys=['Year', 'sector'])
    assert df.primary_keys == ['Year', 'sector', 'unused']


@pytest.mark.parametrize('end_year', [2019, 2023])
def test_extend_without_dimensions_or_forecast(end_year: int) -> None:
    df = to_ppdf(
        pl.DataFrame({'Year': [2021, 2020], 'Value': [2.0, 1.0]}),
        meta=DataFrameMeta(units={'Value': unit_registry.kg}, primary_keys=['Year']),
    )
    result = extend_last_historical_value_pl(df, end_year)
    years = list(range(2020, max(2021, end_year) + 1))
    assert_frame_equal(
        result,
        pl.DataFrame({
            'Year': years,
            'Value': [1.0] + [2.0] * (len(years) - 1),
            'Forecast': [year > 2021 for year in years],
        }),
    )
    assert result.get_meta() == df.get_meta()


@pytest.mark.parametrize('empty', [True, False])
def test_no_history_is_unchanged(empty: bool) -> None:
    frame = pl.DataFrame({'Year': [2022], 'Value': [2.0], 'Forecast': [True]})
    if empty:
        frame = frame.clear()
    df = to_ppdf(frame, meta=DataFrameMeta(units={'Value': unit_registry.kg}, primary_keys=['Year']))
    assert extend_last_historical_value_pl(df, 2025) is df


@pytest.mark.parametrize('zero_fill_missing', [True, False])
@pytest.mark.parametrize('dimensioned', [True, False])
@pytest.mark.parametrize('end_year', [2019, 2023])
def test_extend_single_observation(zero_fill_missing: bool, dimensioned: bool, end_year: int) -> None:
    frame = pl.DataFrame({
        'Year': [2020],
        'Value': [2.0],
        'Other': pl.Series([None], dtype=pl.Float64),
        'unused': [None],
        'note': ['source metadata'],
    })
    keys = ['Year', 'unused']
    if dimensioned:
        frame = frame.with_columns(pl.lit('buildings').alias('sector'))
        keys.append('sector')
    units = dict.fromkeys(['Value', 'Other'], unit_registry.kg)
    df = to_ppdf(frame, meta=DataFrameMeta(units=units, primary_keys=keys))
    result = extend_last_historical_value_pl(df, end_year, zero_fill_missing=zero_fill_missing)
    years = list(range(2020, max(2020, end_year) + 1))
    expected = pl.DataFrame({
        'Year': years,
        'Value': [2.0] * len(years),
        'Other': pl.Series([None] + [0.0 if zero_fill_missing else None] * (len(years) - 1), dtype=pl.Float64),
        'Forecast': [year > 2020 for year in years],
    })
    if dimensioned:
        expected = expected.with_columns(pl.lit('buildings').cast(pl.Categorical).alias('sector'))
        expected = expected.select('sector', 'Year', 'Value', 'Other', 'Forecast')
    assert_frame_equal(result, expected)
    assert result.get_meta().is_equal(DataFrameMeta(units=units, primary_keys=[key for key in keys if key != 'unused']))
