"""Tests for ``PathsDataFrame.drop()``, which keeps the metadata in step with the columns."""

import polars as pl
import pytest

from common import polars as ppl
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def make_df() -> ppl.PathsDataFrame:
    df = pl.DataFrame({YEAR_COLUMN: [2020], 'sector': ['a'], VALUE_COLUMN: [1.0], 'extra': [2.0]})
    meta = ppl.DataFrameMeta(
        units={VALUE_COLUMN: unit_registry.parse_units('kt'), 'extra': unit_registry.parse_units('kt')},
        primary_keys=[YEAR_COLUMN, 'sector'],
    )
    return ppl.to_ppdf(df, meta=meta)


def test_drop_removes_units_and_primary_keys_of_dropped_columns():
    df = make_df().drop(['extra', 'sector'])

    assert df.columns == [YEAR_COLUMN, VALUE_COLUMN]
    assert 'extra' not in df.get_meta().units
    assert df.primary_keys == [YEAR_COLUMN]


def test_drop_of_a_missing_column_names_every_missing_and_available_column():
    df = make_df()

    with pytest.raises(pl.exceptions.ColumnNotFoundError) as excinfo:
        df.drop('nope', ['extra', 'also_nope'])

    message = str(excinfo.value)
    assert 'nope, also_nope' in message
    assert f'available columns: {YEAR_COLUMN}, sector, {VALUE_COLUMN}, extra' in message


def test_non_strict_drop_ignores_missing_columns():
    df = make_df().drop('nope', 'extra', strict=False)

    assert 'extra' not in df.columns
