"""
`compare_outputs`, the cell-by-cell comparison behind `compare_draft_published`.

The command exists to say what a publish would change, so the cases pinned here are the ones
where a wrong answer would mislead: a changed value, a row on one side only, a flipped forecast
flag, and two frames that are equal in content but were built by different contexts.
"""

from typing import TYPE_CHECKING

import polars as pl
import pytest

from common.polars import DataFrameMeta, PathsDataFrame, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.management.commands.compare_draft_published import compare_outputs
from nodes.units import unit_registry

if TYPE_CHECKING:
    from collections.abc import Sequence

# The frames below are literals, but this package's fixtures reach for the DB on setup.
pytestmark = pytest.mark.django_db


type Row = tuple[int, str, float | None, bool]
"""Year, sector, value, forecast flag."""


def _frame(rows: Sequence[Row], unit: str = 'kt/a') -> PathsDataFrame:
    df = pl.DataFrame(
        {
            YEAR_COLUMN: [r[0] for r in rows],
            'sector': pl.Series([r[1] for r in rows], dtype=pl.Categorical),
            VALUE_COLUMN: [r[2] for r in rows],
            FORECAST_COLUMN: [r[3] for r in rows],
        },
        schema_overrides={VALUE_COLUMN: pl.Float64},
    )
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=[YEAR_COLUMN, 'sector'])
    return to_ppdf(df, meta=meta)


BASE: list[Row] = [(2022, 'industry', 10.0, False), (2022, 'households', 5.0, False), (2023, 'industry', 11.0, True)]


def test_equal_frames_in_another_row_order_are_identical():
    cmp = compare_outputs('n', _frame(BASE), _frame(list(reversed(BASE))))
    assert cmp.identical


def test_a_changed_value_is_counted_and_located():
    draft: list[Row] = [(2022, 'industry', 10.0, False), (2022, 'households', 4.0, False), (2023, 'industry', 11.0, True)]
    cmp = compare_outputs('n', _frame(BASE), _frame(draft))
    assert cmp.changed_cells == 1
    assert cmp.largest_year_change == (VALUE_COLUMN, 2022, 15.0, 14.0)
    assert not cmp.problems


def test_a_change_below_the_tolerance_is_not_a_change():
    draft: list[Row] = [(2022, 'industry', 10.0 + 1e-12, False), *BASE[1:]]
    assert compare_outputs('n', _frame(BASE), _frame(draft)).identical


def test_a_value_that_becomes_empty_is_a_change():
    draft: list[Row] = [(2022, 'industry', None, False), *BASE[1:]]
    cmp = compare_outputs('n', _frame(BASE), _frame(draft))
    assert cmp.changed_cells == 1


def test_rows_on_one_side_only_are_reported_per_side():
    draft: list[Row] = [*BASE[:2], (2024, 'industry', 12.0, True)]
    cmp = compare_outputs('n', _frame(BASE), _frame(draft))
    assert '1 rows only in published' in cmp.problems
    assert '1 rows only in draft' in cmp.problems


def test_a_flipped_forecast_flag_is_reported():
    draft: list[Row] = [*BASE[:2], (2023, 'industry', 11.0, False)]
    cmp = compare_outputs('n', _frame(BASE), _frame(draft))
    assert cmp.problems == ['forecast flag differs in 1 rows']
    assert cmp.changed_cells == 0


def test_a_unit_change_stops_the_value_comparison():
    cmp = compare_outputs('n', _frame(BASE), _frame(BASE, unit='Mt/a'))
    assert len(cmp.problems) == 1
    assert cmp.problems[0].startswith('unit of Value')
    assert cmp.changed_cells == 0
