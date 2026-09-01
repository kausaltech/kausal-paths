"""Tests for the index frame that dimensional metrics are joined against."""

import polars as pl
import pytest

from nodes.constants import YEAR_COLUMN
from nodes.metric import DimensionalMetric, MetricCategory, MetricDimension

pytestmark = pytest.mark.django_db


def _dim() -> MetricDimension:
    return MetricDimension(
        id='sector',
        original_id='sector',
        label='Sector',
        categories=[
            MetricCategory(id='transport', original_id='transport', label='Transport', color=None, order=None),
        ],
    )


def test_year_column_is_typed_when_there_are_years() -> None:
    idx_df = DimensionalMetric.generate_index_df([_dim()], [2020, 2021])
    assert idx_df.schema[YEAR_COLUMN] == pl.Int64
    assert idx_df.height == 2


def test_year_column_is_typed_when_the_node_produced_no_rows() -> None:
    """
    An empty year list must still yield an Int64 year column.

    A node can legitimately compute to nothing -- a weighted average whose weights are all
    missing, for instance. Polars infers Null for a column built from an empty list, and the
    caller then joins this frame against the node's (Int64) output, which used to fail with
    `SchemaError: datatypes of join keys don't match` instead of producing an empty metric.
    """
    idx_df = DimensionalMetric.generate_index_df([_dim()], [])
    assert idx_df.schema[YEAR_COLUMN] == pl.Int64
    assert idx_df.height == 0

    # The join the caller performs must now work rather than raise.
    out = pl.DataFrame(schema={'sector': pl.Utf8, YEAR_COLUMN: pl.Int64, 'Value': pl.Float64})
    joined = idx_df.with_columns(pl.col('sector').cast(pl.Utf8)).join(out, how='left', on=['sector', YEAR_COLUMN], validate='1:1')
    assert joined.height == 0


def test_index_df_with_no_dimensions_still_has_years() -> None:
    idx_df = DimensionalMetric.generate_index_df([], [2020])
    assert idx_df.schema[YEAR_COLUMN] == pl.Int64
    assert idx_df.height == 1
