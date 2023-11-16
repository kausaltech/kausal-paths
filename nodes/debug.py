import polars as pl

import common.polars as ppl
from nodes.constants import BASELINE_VALUE_COLUMN, DEFAULT_METRIC, STACKABLE_QUANTITIES, YEAR_COLUMN

from .node import Node


def _apply_filters(df: ppl.PathsDataFrame, filters: list[str]) -> ppl.PathsDataFrame:
    exprs = []
    years = []
    for f in filters:
        if f[0].isdecimal():
            if '-' in f:
                start, end = f.split('-')
                for year in range(int(start), int(end) + 1):
                    years.append(year)
            else:
                years.append(int(f))
        elif f == 'S':
            df = df.sort(by=[YEAR_COLUMN, *df.dim_ids])
        elif ':' in f:
            dim, cat = f.split(':')
            exprs.append(pl.col(dim).eq(cat))
        else:
            cat = f
            any_cols = [pl.col(dim).eq(cat) for dim in df.dim_ids]
            e = pl.any_horizontal(any_cols)
            exprs.append(e)
    if exprs:
        df = df.filter(pl.all_horizontal(exprs))
    if years:
        df = df.filter(pl.col(YEAR_COLUMN).is_in(years))
    return df


def _get_output_with_baseline(node: Node, filters: list[str] | None):
    df = node.get_output_pl()
    meta = df.get_meta()

    if node.context.active_normalization:
        norm = node.context.active_normalization
        for m in node.output_metrics.values():
            _, df = norm.normalize_output(m, df)
    else:
        norm = None

    if filters:
        df = _apply_filters(df, filters)

    if meta.dim_ids:
        df = df.paths.to_wide()
        if node.quantity in STACKABLE_QUANTITIES:
            df = df.with_columns(pl.sum_horizontal(df.metric_cols).alias('Total'))
        return df

    if node.baseline_values is not None:
        m = node.output_metrics[DEFAULT_METRIC]
        df = (
            df.with_columns(node.baseline_values[m.column_id].alias(BASELINE_VALUE_COLUMN))
            .set_unit(BASELINE_VALUE_COLUMN, node.baseline_values.get_unit(m.column_id))
        )

        if norm:
            bm = m.copy()
            bm.column_id = BASELINE_VALUE_COLUMN
            _, df = norm.normalize_output(m, df)

    return df

def print_node_output(node: Node, only_years: list[int] | None = None, filters: list[str] | None = None):
    df = _get_output_with_baseline(node, filters)
    if only_years:
        df = df.filter(pl.col(YEAR_COLUMN).is_in(only_years))

    if filters is not None:
        pass
    node.print(df)


def plot_node_output(node: Node, filters: list[str] | None = None):
    df = _get_output_with_baseline(node, filters)
    plot_node(node, df)


def plot_node(node: Node, df: ppl.PathsDataFrame):
    try:
        import plotext as plt
    except ImportError:
        return
    meta = df.get_meta()
    if meta.dim_ids:
        return
    df = df.paths.to_wide()
    x = df[YEAR_COLUMN]
    unique_units = set(meta.units.values())
    plt.title(node.name)
    plt.subplots(1, len(unique_units))
    for idx, unit in enumerate(unique_units):
        plt.subplot(1, idx + 1)
        plt.xlabel('Year')
        plt.ylabel(unit)
        for col, unit in meta.units.items():
            if unit != unit:
                continue
            y = df[col]
            plt.plot(x, y, label=col)
    plt.theme('dark')
    plt.show()
