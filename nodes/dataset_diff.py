from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.units import Unit


@dataclass
class SchemaDiff:
    dvc_row_count: int
    db_row_count: int
    dvc_only_cols: list[str]
    db_only_cols: list[str]
    dtype_diffs: dict[str, tuple[pl.DataType, pl.DataType]] = field(default_factory=dict)
    unit_diffs: dict[str, tuple[Unit | None, Unit | None]] = field(default_factory=dict)
    pk_diff: tuple[list[str], list[str]] | None = None

    @property
    def identical(self) -> bool:
        return (
            not self.dvc_only_cols
            and not self.db_only_cols
            and not self.dtype_diffs
            and not self.unit_diffs
            and self.pk_diff is None
        )


@dataclass
class RowDiff:
    dvc_only: pl.DataFrame
    db_only: pl.DataFrame
    value_diffs: pl.DataFrame
    matched_count: int
    pk_cols: list[str]
    value_cols: list[str]


def normalize_df(df: PathsDataFrame) -> pl.DataFrame:
    """Cast to common types, sort by primary keys, and return a plain polars DataFrame."""
    meta = df.get_meta()
    out: pl.DataFrame = pl.DataFrame(df.to_dict())
    for col in out.columns:
        if out.schema[col] == pl.Categorical:
            out = out.with_columns(pl.col(col).cast(pl.Utf8))
    sort_cols = [c for c in meta.primary_keys if c in out.columns]
    if sort_cols:
        out = out.sort(sort_cols)
    col_order = sort_cols + sorted(c for c in out.columns if c not in sort_cols)
    return out.select(col_order)


def align_dtypes(a: pl.DataFrame, b: pl.DataFrame, cols: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cast mismatched columns to Utf8 so joins and comparisons work."""
    for col in cols:
        if a.schema[col] != b.schema[col]:
            a = a.with_columns(pl.col(col).cast(pl.Utf8))
            b = b.with_columns(pl.col(col).cast(pl.Utf8))
    return a, b


def compute_schema_diff(dvc_df: PathsDataFrame, db_df: PathsDataFrame) -> SchemaDiff:
    dvc_meta = dvc_df.get_meta()
    db_meta = db_df.get_meta()

    dvc_cols = set(dvc_df.columns)
    db_cols = set(db_df.columns)

    dtype_diffs: dict[str, tuple[pl.DataType, pl.DataType]] = {}
    for col in sorted(dvc_cols & db_cols):
        if dvc_df.schema[col] != db_df.schema[col]:
            dtype_diffs[col] = (dvc_df.schema[col], db_df.schema[col])

    unit_diffs: dict[str, tuple[Unit | None, Unit | None]] = {}
    all_metric_cols = sorted(set(dvc_meta.units.keys()) | set(db_meta.units.keys()))
    for col in all_metric_cols:
        dvc_unit = dvc_meta.units.get(col)
        db_unit = db_meta.units.get(col)
        if dvc_unit != db_unit:
            unit_diffs[col] = (dvc_unit, db_unit)

    pk_diff: tuple[list[str], list[str]] | None = None
    if set(dvc_meta.primary_keys) != set(db_meta.primary_keys):
        pk_diff = (dvc_meta.primary_keys, db_meta.primary_keys)

    return SchemaDiff(
        dvc_row_count=len(dvc_df),
        db_row_count=len(db_df),
        dvc_only_cols=sorted(dvc_cols - db_cols),
        db_only_cols=sorted(db_cols - dvc_cols),
        dtype_diffs=dtype_diffs,
        unit_diffs=unit_diffs,
        pk_diff=pk_diff,
    )


def compute_row_diff(dvc_df: PathsDataFrame, db_df: PathsDataFrame) -> RowDiff | None:
    common_cols = sorted(set(dvc_df.columns) & set(db_df.columns))
    if not common_cols:
        return None

    pk_cols = sorted(set(dvc_df.get_meta().primary_keys) & set(db_df.get_meta().primary_keys))
    if not pk_cols:
        return None

    value_cols = [c for c in common_cols if c not in pk_cols]

    dvc_norm_full = normalize_df(dvc_df)
    db_norm_full = normalize_df(db_df)
    internal_cols = sorted(c for c in db_norm_full.columns if c.startswith('_'))

    dvc_norm = dvc_norm_full.select(common_cols)
    db_norm = db_norm_full.select(common_cols + internal_cols)
    dvc_norm, db_norm = align_dtypes(dvc_norm, db_norm, common_cols)

    dvc_only = dvc_norm.join(db_norm, on=pk_cols, how='anti', nulls_equal=True)
    db_only = db_norm.join(dvc_norm, on=pk_cols, how='anti', nulls_equal=True)

    if not value_cols:
        return RowDiff(
            dvc_only=dvc_only,
            db_only=db_only,
            value_diffs=pl.DataFrame(),
            matched_count=0,
            pk_cols=pk_cols,
            value_cols=value_cols,
        )

    joined = dvc_norm.join(db_norm, on=pk_cols, how='inner', suffix='_db', nulls_equal=True)

    diff_exprs = [pl.col(col).ne_missing(pl.col(f'{col}_db')).alias(f'_diff_{col}') for col in value_cols]
    joined = joined.with_columns(diff_exprs)
    any_diff = pl.any_horizontal(*[pl.col(f'_diff_{col}') for col in value_cols])
    differing = joined.filter(any_diff)

    keep_cols = pk_cols + value_cols + [f'{col}_db' for col in value_cols]
    internal_in_differing = [c for c in differing.columns if c.startswith('_') and not c.startswith('_diff_')]
    keep_cols += internal_in_differing
    value_diffs = differing.select([c for c in keep_cols if c in differing.columns])

    return RowDiff(
        dvc_only=dvc_only,
        db_only=db_only,
        value_diffs=value_diffs,
        matched_count=len(joined),
        pk_cols=pk_cols,
        value_cols=value_cols,
    )
