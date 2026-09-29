"""
Per-metric qualifier columns: what is known about one metric value at one data point.

A qualifier describes a value without being part of it. It is not a dimension (it is not in
the key) and not a metric (nothing adds it up as a quantity); see
``docs/architecture/metric-dataframe.md``. The metric column ``Value`` is qualified by the
Struct column ``Value__qual``, and the pairing is the name: renaming or dropping a metric takes
its qualifier with it. A projection keeps it only when asked (``PathsDataFrame.qualified``),
because a projection promises exactly the columns it names. Where a helper drops it, the value
simply arrives unqualified -- nobody has said anything about it -- which is the safe direction.

The Struct has one fixed shape, so frames from different sources concatenate and join without
reconciling schemas. A null field means *nobody has said*, which is never the same as a low
grade or a filled value:

- ``quality``: the grade of the value as a score from 0 to 1 (a framework quality level's
  ``score``). Null where the value is ungraded.
- ``graded``: the share of the value's magnitude that carries a grade -- 1 or 0 for a single
  cell, a fraction once cells are added up. It is what keeps an aggregate honest about the part
  nobody graded: ``quality`` is the grade of the graded part, ``quality * graded`` the grade of
  the whole with the ungraded part counted as the lowest grade.
- ``supplied``: whether the value came from its source rather than from an operation that made
  it up -- zero-filling, interpolation, extension. It is how a consumer can still tell which
  years a city actually reported once the frame has been filled so it computes.

Each field states its own rules for the three things generic operations do to values, so the
operations stay mechanical:

- **Adding** values up, over a dimension or across two frames: ``quality`` and ``graded`` are
  means weighted by the magnitude of each value, ``supplied`` is true when any part was. This
  is the Methodenpapier's arithmetic for a Datengüte: each component weighted by its share.
- **Multiplying or dividing**: a factor with no qualifier leaves the other side's qualifier as it
  is. Where both sides have one, ``quality`` and ``graded`` take the lower and ``supplied`` needs
  both.
- **Choosing** one side's value (``coalesce_df``, ``prefer_by_year``): the qualifier follows the
  chosen value.

Filling marks what it filled; see ``carry_over``.
"""

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from common.polars import PathsDataFrame


QUALIFIER_SUFFIX = '__qual'

QUALITY = 'quality'
GRADED = 'graded'
SUPPLIED = 'supplied'

QUALIFIER_DTYPE = pl.Struct({QUALITY: pl.Float64, GRADED: pl.Float64, SUPPLIED: pl.Boolean})


def qualifier_column(metric_col: str) -> str:
    return f'{metric_col}{QUALIFIER_SUFFIX}'


def qualified_metric(col: str) -> str | None:
    """Return the metric column a qualifier column belongs to, or None if ``col`` is not one."""
    if not col.endswith(QUALIFIER_SUFFIX) or col == QUALIFIER_SUFFIX:
        return None
    return col[: -len(QUALIFIER_SUFFIX)]


def make(quality: pl.Expr | None = None, supplied: pl.Expr | None = None) -> pl.Expr:
    """Build a cell-level qualifier: ``graded`` follows from whether ``quality`` is set."""
    q = quality.cast(pl.Float64) if quality is not None else pl.lit(None, dtype=pl.Float64)
    s = supplied.cast(pl.Boolean) if supplied is not None else pl.lit(None, dtype=pl.Boolean)
    return pl.struct(
        q.alias(QUALITY),
        pl.when(q.is_null()).then(pl.lit(0.0)).otherwise(pl.lit(1.0)).alias(GRADED),
        s.alias(SUPPLIED),
    )


FILLED = pl.struct(
    pl.lit(None, dtype=pl.Float64).alias(QUALITY),
    pl.lit(0.0).alias(GRADED),
    pl.lit(value=False).alias(SUPPLIED),
)
"""The qualifier of a value an operation made up: ungraded, and not supplied."""


def _fields(qual: str | None) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """Return (present, quality, graded, supplied) for a qualifier column that may be absent."""
    if qual is None:
        null_f = pl.lit(None, dtype=pl.Float64)
        return pl.lit(value=False), null_f, pl.lit(0.0), pl.lit(None, dtype=pl.Boolean)
    col = pl.col(qual)
    quality = col.struct.field(QUALITY)
    graded = pl.when(quality.is_null()).then(pl.lit(0.0)).otherwise(col.struct.field(GRADED).fill_null(0.0))
    return col.is_not_null(), quality, graded, col.struct.field(SUPPLIED)


def _weight(value: str) -> pl.Expr:
    return pl.col(value).cast(pl.Float64).abs().fill_nan(0.0).fill_null(0.0)


def _nan_to_null(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr.is_nan()).then(pl.lit(None, dtype=pl.Float64)).otherwise(expr)


def reduce_sum(value: str, qual: str) -> pl.Expr:
    """
    Aggregate expression for the qualifier of a sum over a group.

    Where every value in the group is zero there is nothing to weight by, and the fields fall
    back to plain means over the group: the zero-filled cells of an empty template are exactly
    that case, and weighting would turn their grades into 0/0.
    """
    _present, q, g, s = _fields(qual)
    w = _weight(value)
    wg = w * g
    q0 = q.fill_null(0.0)
    quality = pl.when(wg.sum() > 0).then((wg * q0).sum() / wg.sum()).otherwise(_nan_to_null((g * q0).sum() / g.sum()))
    graded = pl.when(w.sum() > 0).then(wg.sum() / w.sum()).otherwise(g.mean())
    supplied = pl.when(s.is_not_null().any()).then(s.any()).otherwise(pl.lit(None, dtype=pl.Boolean))
    return pl.struct(quality.alias(QUALITY), graded.alias(GRADED), supplied.alias(SUPPLIED)).alias(qual)


def combine_sum(out: str, left_value: str, left_qual: str | None, right_value: str, right_qual: str | None) -> pl.Expr:
    """Row-wise qualifier of ``left + right``, both sides already joined into one frame."""
    lp, lq, lg, ls = _fields(left_qual)
    rp, rq, rg, rs = _fields(right_qual)
    # A side that is absent from the row (an outer join's other half) adds nothing and weighs
    # nothing. A side that is present but unqualified still weighs its value, ungraded.
    lw = pl.when(pl.col(left_value).is_null()).then(pl.lit(0.0)).otherwise(_weight(left_value))
    rw = pl.when(pl.col(right_value).is_null()).then(pl.lit(0.0)).otherwise(_weight(right_value))
    lwg, rwg = lw * lg, rw * rg
    lq0, rq0 = lq.fill_null(0.0), rq.fill_null(0.0)
    unweighted_g = lg + rg
    quality = (
        pl
        .when(lwg + rwg > 0)
        .then((lwg * lq0 + rwg * rq0) / (lwg + rwg))
        .when(unweighted_g > 0)
        .then((lg * lq0 + rg * rq0) / unweighted_g)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    sides = lp.cast(pl.Float64) + rp.cast(pl.Float64)
    graded = (
        pl
        .when(lw + rw > 0)
        .then((lwg + rwg) / (lw + rw))
        .when(sides > 0)
        .then((pl.when(lp).then(lg).otherwise(0.0) + pl.when(rp).then(rg).otherwise(0.0)) / sides)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    supplied = pl.when(ls.is_null()).then(rs).when(rs.is_null()).then(ls).otherwise(ls | rs)
    return pl.struct(quality.alias(QUALITY), graded.alias(GRADED), supplied.alias(SUPPLIED)).alias(out)


def combine_product(out: str, left_qual: str | None, right_qual: str | None) -> pl.Expr:
    """Row-wise qualifier of ``left * right`` or ``left / right``."""
    if left_qual is None or right_qual is None:
        only = left_qual or right_qual
        assert only is not None
        return pl.col(only).alias(out)
    lp, lq, lg, ls = _fields(left_qual)
    rp, rq, rg, rs = _fields(right_qual)
    supplied = pl.when(ls.is_null()).then(rs).when(rs.is_null()).then(ls).otherwise(ls & rs)
    combined = pl.struct(
        pl.min_horizontal(lq, rq).alias(QUALITY),
        pl.min_horizontal(lg, rg).alias(GRADED),
        supplied.alias(SUPPLIED),
    )
    return pl.when(lp & rp).then(combined).when(lp).then(pl.col(left_qual)).otherwise(pl.col(right_qual)).alias(out)


def choose(out: str, take_left: pl.Expr, left_qual: str | None, right_qual: str | None) -> pl.Expr:
    """Row-wise qualifier of a value chosen from one side: it follows the chosen value."""
    left = pl.col(left_qual) if left_qual is not None else pl.lit(None, dtype=QUALIFIER_DTYPE)
    right = pl.col(right_qual) if right_qual is not None else pl.lit(None, dtype=QUALIFIER_DTYPE)
    return pl.when(take_left).then(left).otherwise(right).alias(out)


def carry_over(before: PathsDataFrame, after: PathsDataFrame, *, start: bool = False) -> PathsDataFrame:
    """
    Give the result of a fill operation the qualifiers of the frame it filled.

    A fill operation adds rows or replaces nulls, and most of them go through a wide pivot that
    drops every column it does not know. Rather than teaching each of them about qualifiers,
    the qualifiers are carried over afterwards by key: a cell that held a value before keeps its
    qualifier, and a cell that did not -- a new row, or a null that is now a number -- is marked
    ``FILLED``.

    ``start`` makes a qualifier for a frame that had none, recording only what was supplied.
    ``empty_to_zero`` needs it: a zero it wrote is otherwise indistinguishable from a zero
    someone reported, which is the one thing a consumer choosing between sources has to know.
    Other fills only keep a record that already exists, so a frame nobody qualified stays as
    light as it was.
    """
    metrics = [m for m in after.metric_cols if m in before.columns]
    quals = {m: qualifier_column(m) for m in metrics if qualifier_column(m) in before.columns}
    if not quals and start:
        before = before.with_columns([make(supplied=pl.col(m).is_not_null()).alias(qualifier_column(m)) for m in metrics])
        quals = {m: qualifier_column(m) for m in metrics}
    if not quals:
        return after
    keys = [key for key in before.primary_keys if key in after.columns]
    if not keys or set(keys) != set(after.primary_keys):
        return after
    from common import polars as ppl

    meta = after.get_meta()
    carried = pl.DataFrame(after).drop([q for q in quals.values() if q in after.columns])
    source = pl.DataFrame(before).select([
        *[pl.col(key).cast(carried.schema[key]) for key in keys],
        *[
            pl.when(pl.col(m).is_not_null()).then(pl.col(q)).otherwise(pl.lit(None, dtype=QUALIFIER_DTYPE)).alias(q)
            for m, q in quals.items()
        ],
    ])
    joined = carried.join(source, on=keys, how='left', nulls_equal=True)
    joined = joined.with_columns([
        pl.when(pl.col(q).is_null() & pl.col(m).is_not_null()).then(FILLED).otherwise(pl.col(q)).alias(q)
        for m, q in quals.items()
    ])
    return ppl.to_ppdf(joined, meta=meta)


def qualifier_columns(columns: Sequence[str], metric_cols: Sequence[str]) -> dict[str, str]:
    """Map each metric in ``metric_cols`` that has a qualifier in ``columns`` to that qualifier."""
    present = set(columns)
    return {m: qualifier_column(m) for m in metric_cols if qualifier_column(m) in present}
