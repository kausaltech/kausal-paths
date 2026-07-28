"""
Apply a port binding's transformations.

Transformations adapt a source to what a port expects: they select, filter,
reshape and relabel. They never compute the node's output — that is the node's
own pipeline (``nodes.pipeline``), which is why these are transformations and
not operations. Note that adapting is not always shape-only: flattening sums
over a dimension and unit coercion rescales values.

One literal pass, in the order stored. The list is the complete recipe — the
stages that used to be hardcoded around the filters are transformations too
(see the legacy markers in ``nodes.defs.transform_def``), so nothing implicit
happens between them.

Deliberately decoupled from ``Dataset``: it takes a frame, the transformations
and a small environment. That is what lets edge bindings run the same list
later, and what will make the ``MetricDataFrame`` migration a change of the
transformation bodies rather than a redesign.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

import polars as pl
from loguru import logger

from common import polars as ppl
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.defs.transform_def import (
    AssignDimensionOp,
    DropNullsOp,
    EnsureUnitOp,
    FilterColumnOp,
    FilterDimensionOp,
    FilterTemporalOp,
    IndexTemporalOp,
    RemapLegacyYearsOp,
    RenameColumnOp,
    RenameItemOp,
    SelectMetricOp,
    SetForecastFromOp,
    TagOperationOp,
)

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.datasets import Dataset
    from nodes.defs.transform_def import PortTransformOp


class PipelineError(Exception):
    """A transform operation could not be applied."""


@dataclass
class PipelineEnv:
    """
    What the transformations need besides the frame.

    ``metric_column`` is what ``select_metric`` resolves to: the binding names
    the metric, the op only says where the selection happens.

    ``dataset`` is optional and only used by the legacy stage markers — for
    tag-operation identity, and to raise ``DatasetError`` instead of a bare
    ``PipelineError`` where callers already handle it.
    """

    context: Context
    dataset: Dataset | None = None
    metric_column: str | None = None

    @property
    def source_id(self) -> str:
        return self.dataset.id if self.dataset is not None else '<unknown>'

    def fail(self, msg: str) -> NoReturn:
        from nodes.exceptions import DatasetError

        if self.dataset is not None:
            raise DatasetError(self.dataset, msg)
        raise PipelineError(msg)


def apply_port_transformations(
    df: ppl.PathsDataFrame,
    transformations: list[PortTransformOp],
    env: PipelineEnv,
) -> ppl.PathsDataFrame:
    """Run the transformations against the frame, in order."""
    for op in transformations:
        df = apply_operation(df, op, env)
    return df


def apply_operation(  # noqa: C901, PLR0911, PLR0912
    df: ppl.PathsDataFrame,
    op: PortTransformOp,
    env: PipelineEnv,
) -> ppl.PathsDataFrame:
    match op:
        case RenameColumnOp():
            return _rename_column(df, op, env)
        case SelectMetricOp():
            return _select_metric(df, env)
        case IndexTemporalOp():
            return _index_temporal(df)
        case RemapLegacyYearsOp():
            return _remap_legacy_years(df, env)
        case SetForecastFromOp():
            return _set_forecast_from(df, op)
        case FilterColumnOp():
            return _guard_not_empty(_filter_column(df, op, env), df, op, env)
        case FilterDimensionOp():
            return _guard_not_empty(_filter_dimension(df, op, env), df, op, env)
        case AssignDimensionOp():
            return _guard_not_empty(_assign_dimension(df, op, env), df, op, env)
        case RenameItemOp():
            return _guard_not_empty(_rename_item(df, op, env), df, op, env)
        case TagOperationOp():
            return _tag_operation(df, op, env)
        case FilterTemporalOp():
            return _filter_temporal(df, op)
        case DropNullsOp():
            return df.drop_nulls()
        case EnsureUnitOp():
            return _ensure_unit(df, op)


def _guard_not_empty(
    df: ppl.PathsDataFrame,
    before: ppl.PathsDataFrame,
    op: PortTransformOp,
    env: PipelineEnv,
) -> ppl.PathsDataFrame:
    """Fail when an operation filtered everything away: that is a configuration error, not a result."""
    if len(df) == 0:
        logger.error('Nothing left after {} on {}; input was:\n{}', op.kind, env.source_id, before)
        env.fail(f'Nothing left after {op.kind}. See the original frame in the log.')
    return df


# --- Legacy stage markers ---------------------------------------------------


def _select_metric(df: ppl.PathsDataFrame, env: PipelineEnv) -> ppl.PathsDataFrame:
    """
    Alias the bound column to ``Value``, drop its nulls and narrow the frame.

    Which column that is comes from the binding, not from the op.
    """
    column = env.metric_column
    if column is None:
        env.fail('select_metric has no column to select: the binding names no metric')
    if column not in df.columns:
        env.fail(f"Column '{column}' not found in dataset '{env.source_id}'. Available columns: {', '.join(df.columns)}")
    df = df.with_columns(pl.col(column).alias(VALUE_COLUMN))
    df = df.filter(pl.col(VALUE_COLUMN).is_not_null())
    cols = [YEAR_COLUMN, VALUE_COLUMN, *df.dim_ids]
    if FORECAST_COLUMN in df.columns:
        cols.append(FORECAST_COLUMN)
    return ppl.to_ppdf(df.lazy().select(cols).collect(), meta=df.get_meta().select(cols))


def _index_temporal(df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
    if YEAR_COLUMN in df.columns and YEAR_COLUMN not in df.primary_keys:
        return df.add_to_index(YEAR_COLUMN)
    return df


def _remap_legacy_years(df: ppl.PathsDataFrame, env: PipelineEnv) -> ppl.PathsDataFrame:
    """
    Turn placeholder year numbers into real years.

    Legacy DVC datasets encode the reference year as 0 or 1 and the target year
    as 100 or 101. Frames that hold no such year are left untouched, which is
    why the check is on the data rather than on configuration.
    """
    if YEAR_COLUMN not in df.columns:
        return df
    ldf = df.lazy()
    if ldf.filter((pl.col(YEAR_COLUMN) < 200).first()).collect().is_empty():
        return df

    instance = env.context.instance
    baseline_year = instance.reference_year
    adjustment = -1  # DVC and DB use year 1 for reference year; offset by -1 so year 1 → baseline_year.

    # Older datasets used Year=0 for reference year and Year=100 for target year.
    # Remap them to Year=1 and Year=101 so the offset formula below handles both.
    ldf = ldf.with_columns(
        pl
        .when(pl.col(YEAR_COLUMN) == 0)
        .then(pl.lit(1))
        .when(pl.col(YEAR_COLUMN) == 100)
        .then(pl.lit(101))
        .otherwise(pl.col(YEAR_COLUMN))
        .alias(YEAR_COLUMN),
    )
    ldf = ldf.with_columns(
        pl
        .when(pl.col(YEAR_COLUMN) < 90)
        .then(pl.col(YEAR_COLUMN) + pl.lit(baseline_year + adjustment))
        .otherwise(pl.col(YEAR_COLUMN))
        .alias(YEAR_COLUMN),
    )
    target_year = instance.target_year
    ldf = ldf.with_columns(
        pl
        .when((pl.col(YEAR_COLUMN) >= 90) & (pl.col(YEAR_COLUMN) < 200))
        .then(pl.col(YEAR_COLUMN) + pl.lit(target_year + adjustment) - pl.lit(100))
        .otherwise(pl.col(YEAR_COLUMN))
        .alias(YEAR_COLUMN),
    )
    ldf = ldf.with_columns(pl.col(YEAR_COLUMN).cast(int).alias(YEAR_COLUMN))

    meta = df.get_meta()
    # FIXME Duplicates may occur when baseline year overlaps with existing data points.
    ldf = ldf.unique(subset=meta.primary_keys, keep='last', maintain_order=True)
    return ppl.to_ppdf(ldf.collect(), meta=meta)


def _tag_operation(df: ppl.PathsDataFrame, op: TagOperationOp, env: PipelineEnv) -> ppl.PathsDataFrame:
    tag = op.tag
    if tag == 'ignore_content':
        logger.warning("Dataset {} has tag 'ignore_content', which is not supported.", env.source_id)
        return df
    if not df.paths.has_operation(tag):
        return df

    # FIXME Don't let DatasetNodes get double preparation of gpc. Remove when you get rid of DatasetNodes
    if tag == 'prepare_gpc_dataset' and _is_bound_to_dataset_node(env):
        return df
    return df.paths.get_operation(tag)(df, env.context)


def _is_bound_to_dataset_node(env: PipelineEnv) -> bool:
    from nodes.gpc import DatasetNode

    dataset = env.dataset
    if dataset is None:
        return False
    return any(
        isinstance(node, DatasetNode) and any(ds is dataset for ds in node.input_dataset_instances)
        for node in env.context.nodes.values()
    )


# --- Dimension and column transformations ----------------------------------------


def _filter_column(df: ppl.PathsDataFrame, op: FilterColumnOp, env: PipelineEnv) -> ppl.PathsDataFrame:
    col = op.column
    val = op.value
    mask = None
    if op.values:
        mask = pl.col(col).is_in(op.values)
    if val:
        mask = pl.col(col) == val
    if op.ref:
        pval = env.context.get_parameter_value(op.ref, required=True)
        if isinstance(pval, float):
            pval = int(pval)
        mask = pl.col(col) == str(pval)
    if mask is not None:
        if op.exclude:
            mask = ~mask
        df = df.filter(mask)

    if op.flatten:
        if VALUE_COLUMN in df.columns:
            df = df.filter(~pl.col(VALUE_COLUMN).is_nan())
        df = df.paths.sum_over_dims(col)
    elif op.drop_col:
        df = df.drop(col)
    return df


def _filter_dimension(df: ppl.PathsDataFrame, op: FilterDimensionOp, env: PipelineEnv) -> ppl.PathsDataFrame:
    dim_id = op.dimension
    if op.groups:
        dim = env.context.dimensions[dim_id]
        grp_s = dim.ids_to_groups(dim.series_to_ids_pl(df[dim_id]))
        df = df.filter(grp_s.is_in(op.groups))
    elif op.categories:
        expr = pl.col(dim_id).is_in(op.categories)
        if op.exclude:
            expr = pl.col(dim_id).is_null() | ~expr
        df = df.filter(expr)
    if op.flatten:
        if VALUE_COLUMN in df.columns:
            df = df.filter(~pl.col(VALUE_COLUMN).is_nan())
        df = df.paths.sum_over_dims(dim_id)
    return df


def _assign_dimension(df: ppl.PathsDataFrame, op: AssignDimensionOp, env: PipelineEnv) -> ppl.PathsDataFrame:
    dim_id = op.dimension
    cat_id = op.category
    context = env.context
    if dim_id in context.dimensions:
        dim = context.dimensions[dim_id]
        if dim_id in df.dim_ids:
            env.fail(f'Cannot assign dimension {dim_id}: the frame already has it')
        if cat_id not in dim.cat_map:
            env.fail(f'Category {cat_id} not found in dimension {dim_id}')
    return df.with_columns(pl.lit(cat_id).alias(dim_id)).add_to_index(dim_id)


def _rename_column(df: ppl.PathsDataFrame, op: RenameColumnOp, env: PipelineEnv) -> ppl.PathsDataFrame:
    col = op.column
    new_name = op.new_name
    if col not in df.columns:
        env.fail(f'Column {col} not found. Available columns are {df.columns}')
    if new_name:
        if new_name in df.columns:
            df = df.drop(new_name)
        df = df.rename({col: new_name})
    return df


def _rename_item(df: ppl.PathsDataFrame, op: RenameItemOp, env: PipelineEnv) -> ppl.PathsDataFrame:
    if not op.new_item:
        env.fail('rename_item must have a new item value.')
    # str.replace_all requires Utf8; cast if the column is not string (e.g. Categorical)
    series = pl.col(op.column)
    if df.schema[op.column] != pl.Utf8:
        series = series.cast(pl.Utf8)
    return df.with_columns(series.str.replace_all(re.escape(op.old_item), op.new_item))


# --- Output shaping ---------------------------------------------------------


def _set_forecast_from(df: ppl.PathsDataFrame, op: SetForecastFromOp) -> ppl.PathsDataFrame:
    """
    Mark values from the given year onwards as forecast.

    A frame that already states its forecast status keeps it: the dataset knows
    better than the binding does.
    """
    if FORECAST_COLUMN in df.columns:
        return df
    meta = df.get_meta()
    df = df.with_columns(
        pl.when(pl.col(YEAR_COLUMN) >= op.year).then(pl.lit(value=True)).otherwise(pl.lit(value=False)).alias(FORECAST_COLUMN),
    )
    return ppl.to_ppdf(df, meta=meta)


def _filter_temporal(df: ppl.PathsDataFrame, op: FilterTemporalOp) -> ppl.PathsDataFrame:
    if op.max_year:
        df = df.filter(pl.col(YEAR_COLUMN) <= op.max_year)
    if op.min_year:
        df = df.filter(pl.col(YEAR_COLUMN) >= op.min_year)
    return df


def _ensure_unit(df: ppl.PathsDataFrame, op: EnsureUnitOp) -> ppl.PathsDataFrame:
    for col in df.columns:
        if col in [FORECAST_COLUMN, YEAR_COLUMN, *df.dim_ids]:
            continue
        if col in df.metric_cols:
            df = df.ensure_unit(col, op.unit)
        else:
            df = df.set_unit(col, op.unit)
    return df
