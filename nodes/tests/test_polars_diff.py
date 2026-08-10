"""Tests for ``PathsDataFrame.diff()``, the year-on-year change of a metric."""

import polars as pl
import pytest

from common import polars as ppl
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def make_df(values: dict[str, list[float | None]], years: list[int]) -> ppl.PathsDataFrame:
    """Build a narrow PathsDataFrame with one `sector` dimension."""
    rows_year: list[int] = []
    rows_sector: list[str] = []
    rows_value: list[float | None] = []
    for sector, vals in values.items():
        rows_year += years
        rows_sector += [sector] * len(years)
        rows_value += vals
    df = pl.DataFrame({
        YEAR_COLUMN: rows_year,
        'sector': rows_sector,
        VALUE_COLUMN: rows_value,
        FORECAST_COLUMN: [False] * len(rows_year),
    })
    meta = ppl.DataFrameMeta(
        units={VALUE_COLUMN: unit_registry.parse_units('kt')},
        primary_keys=[YEAR_COLUMN, 'sector'],
    )
    return ppl.to_ppdf(df, meta=meta)


def values_by_year(df: ppl.PathsDataFrame, sector: str) -> dict[int, float]:
    rows = df.filter(pl.col('sector').eq(sector)).sort(YEAR_COLUMN)
    return dict(zip(rows[YEAR_COLUMN].to_list(), rows[VALUE_COLUMN].to_list(), strict=True))


def test_diff_keeps_first_year_as_zero():
    df = make_df({'a': [1.0, 3.0, 6.0]}, [2020, 2021, 2022])
    out = df.diff(VALUE_COLUMN)

    # The first year has no year to compare against, so its change is zero, and
    # the time scope of the output equals that of the input.
    assert values_by_year(out, 'a') == {2020: 0.0, 2021: 2.0, 2022: 3.0}


def test_diff_divides_unit_by_time():
    df = make_df({'a': [1.0, 3.0]}, [2020, 2021])
    assert df.diff(VALUE_COLUMN).get_unit(VALUE_COLUMN) == unit_registry.parse_units('kt/a')


def test_diff_zeroes_first_year_for_every_dimension_category():
    df = make_df({'a': [1.0, 3.0], 'b': [10.0, 40.0]}, [2020, 2021])
    out = df.diff(VALUE_COLUMN)

    assert values_by_year(out, 'a') == {2020: 0.0, 2021: 2.0}
    assert values_by_year(out, 'b') == {2020: 0.0, 2021: 30.0}


def test_diff_does_not_fabricate_values_for_null_cells():
    """A null input stays null (and is dropped), also in the first year."""
    df = make_df({'a': [1.0, 3.0, 6.0], 'b': [None, 20.0, 45.0]}, [2020, 2021, 2022])
    out = df.diff(VALUE_COLUMN)

    assert values_by_year(out, 'a') == {2020: 0.0, 2021: 2.0, 2022: 3.0}
    # 2020 is null in the input, and 2021 has no year to compare against.
    assert values_by_year(out, 'b') == {2022: 25.0}


@pytest.mark.parametrize('n', [1, 2])
def test_diff_over_several_years(n: int):
    df = make_df({'a': [1.0, 3.0, 6.0, 10.0]}, [2020, 2021, 2022, 2023])
    out = values_by_year(df.diff(VALUE_COLUMN, n=n), 'a')

    assert [out[y] for y in sorted(out)][:n] == [0.0] * n
    assert len(out) == 4
