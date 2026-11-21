"""
Find the rows of an input that a node's result drops, for inputs tagged ``keep_categories``.

A product keeps only the rows its operands share. For a factor that is right, and for an
activity it loses the activity: fuel with no emission factor vanishes from the emissions, with no
error. A binding tagged ``keep_categories`` says that its rows must survive, and `Node` checks it
after computing (`Node._check_kept_categories`). See `docs/plans/dropped-categories.md`.
"""

from typing import TYPE_CHECKING, Final

import polars as pl

from nodes.constants import YEAR_COLUMN

if TYPE_CHECKING:
    from common.polars import PathsDataFrame

KEEP_CATEGORIES_TAG: Final = 'keep_categories'

MAX_REPORTED: Final = 10
"""How many dropped category combinations an error names before it summarises the rest."""


def _keys(df: PathsDataFrame, dims: list[str], valued: pl.Expr) -> pl.DataFrame:
    return pl.DataFrame(df).filter(valued).select([pl.col(YEAR_COLUMN), *[pl.col(dim).cast(pl.Utf8) for dim in dims]]).unique()


def dropped_rows(kept: PathsDataFrame, output: PathsDataFrame) -> pl.DataFrame:
    """
    Return the year and categories of each row of ``kept`` that has no counterpart in ``output``.

    Rows are compared on the year and the dimensions both frames have; a dimension the node sums
    away is not compared. Only rows of ``kept`` with a value other than null and zero count, since a
    zero carries nothing. A year in which ``output`` has no value at all is skipped: a missing year
    is how the route choices of the model fall back (`prefer_by_year`), not a lost row.
    """
    dims = [dim for dim in kept.dim_ids if dim in output.dim_ids]
    if not kept.metric_cols or not output.metric_cols:
        return pl.DataFrame(schema={YEAR_COLUMN: pl.Int64, **dict.fromkeys(dims, pl.Utf8)})
    carries_value = pl.any_horizontal([pl.col(col).is_not_null() & (pl.col(col) != 0) for col in kept.metric_cols])
    rows = _keys(kept, dims, carries_value)
    results = _keys(output, dims, pl.any_horizontal([pl.col(col).is_not_null() for col in output.metric_cols]))
    rows = rows.join(results.select(YEAR_COLUMN).unique(), on=YEAR_COLUMN, how='semi')
    keys = [YEAR_COLUMN, *dims]
    return rows.join(results, on=keys, how='anti', nulls_equal=True).sort(keys)


def describe_dropped(rows: pl.DataFrame) -> str:
    """Describe dropped rows as their category combinations, each with its years."""
    dims = [col for col in rows.columns if col != YEAR_COLUMN]
    if not dims:
        return 'years %s' % ', '.join(str(year) for year in rows[YEAR_COLUMN].to_list())
    grouped = rows.group_by(dims, maintain_order=True).agg(pl.col(YEAR_COLUMN).sort())
    parts = [
        '%s (%s)' % (', '.join(f'{dim}={row[dim]}' for dim in dims), ', '.join(str(year) for year in row[YEAR_COLUMN]))
        for row in grouped.head(MAX_REPORTED).iter_rows(named=True)
    ]
    if grouped.height > MAX_REPORTED:
        parts.append('and %d more' % (grouped.height - MAX_REPORTED))
    return '; '.join(parts)
