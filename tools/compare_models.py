#!/usr/bin/env python
"""Compare numerical outputs between nzc.yaml and nzc_legacy.yaml using the Python API."""  # noqa: EXE001, RUF100

# ruff: noqa: E402
import os
import sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'paths.settings')

import django

django.setup()

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from nodes.constants import YEAR_COLUMN
from nodes.instance_loader import InstanceLoader

if TYPE_CHECKING:
    from common import polars as ppl

CONFIGS = {
    'new': 'configs/nzc.yaml',
    'legacy': 'configs/nzc_legacy.yaml',
}

NODES = sys.argv[1:] if len(sys.argv) > 1 else ['net_emissions', 'total_cost']
COMPARE_YEARS = [2018, 2030, 2045, 2060]


def load_context(config: str):
    loader = InstanceLoader.from_yaml(Path(config))
    return loader.instance.context


def get_sector_totals(ctx, node_id: str) -> ppl.PathsDataFrame | None:
    """Get the node output summed over all non-year dimensions."""
    try:
        node = ctx.get_node(node_id)
        df = node.get_output_pl()
        metric_cols = list(df.metric_cols)
        agg = df.group_by(YEAR_COLUMN).agg([pl.col(c).sum() for c in metric_cols])
        agg = agg.sort(YEAR_COLUMN)
    except Exception as e:
        print(f'  ERROR computing {node_id}: {e}')
        return None
    else:
        return agg


def get_by_sector(ctx, node_id: str) -> ppl.PathsDataFrame | None:
    """Get the node output by sector dimension."""
    try:
        node = ctx.get_node(node_id)
        df = node.get_output_pl()
    except Exception as e:
        print(f'  ERROR computing {node_id}: {e}')
        return None
    else:
        return df


print('Loading contexts...')
ctxs = {}
for label, config in CONFIGS.items():
    print(f'  Loading {label} ({config})...')
    try:
        ctxs[label] = load_context(config)
    except Exception as e:
        print(f'  ERROR: {e}')

for node_id in NODES:
    print(f'\n{"=" * 70}')
    print(f'NODE: {node_id}')
    print('=' * 70)

    dfs = {}
    for label in CONFIGS:
        if label not in ctxs:
            continue
        ctx = ctxs[label]
        if node_id not in ctx.nodes:
            print(f'  {label}: node not found')
            continue
        print(f'  Computing {label}...')
        df = get_sector_totals(ctx, node_id)
        if df is not None:
            dfs[label] = df

    if len(dfs) < 2:
        print('  Cannot compare: one or both models failed')
        continue

    new_df = dfs['new']
    leg_df = dfs['legacy']

    # Compare totals at specific years
    print(f'\n  {"Year":<8} {"New total":>15} {"Legacy total":>15} {"Diff":>12} {"RelDiff":>10}')
    print(f'  {"-" * 62}')

    joined = new_df.join(leg_df, on=YEAR_COLUMN, how='outer', suffix='_leg')
    joined = joined.sort(YEAR_COLUMN)

    for row in joined.rows(named=True):
        year = row[YEAR_COLUMN]
        if year not in COMPARE_YEARS:
            continue
        new_vals = [v for k, v in row.items() if k != YEAR_COLUMN and not k.endswith('_leg') and v is not None]
        leg_vals = [v for k, v in row.items() if k.endswith('_leg') and v is not None]
        new_total = sum(new_vals) if new_vals else float('nan')
        leg_total = sum(leg_vals) if leg_vals else float('nan')
        diff = new_total - leg_total
        ref = max(abs(new_total), abs(leg_total))
        rel_diff = abs(diff) / ref if ref > 0 else 0
        flag = ' *** MISMATCH' if rel_diff > 0.01 else ''
        print(f'  {year:<8} {new_total:>15.4f} {leg_total:>15.4f} {diff:>12.4f} {rel_diff:>10.2%}{flag}')

    # Compare by first shared dimension at 2030
    print('\n  Sector breakdown at 2030:')
    new_full = get_by_sector(ctxs['new'], node_id)
    leg_full = get_by_sector(ctxs['legacy'], node_id)

    if new_full is not None and leg_full is not None:
        y = 2030
        new_y = new_full.filter(pl.col(YEAR_COLUMN) == y)
        leg_y = leg_full.filter(pl.col(YEAR_COLUMN) == y)

        new_dims = new_full.dim_ids
        leg_dims = leg_full.dim_ids
        print(f'    New dimensions: {new_dims}')
        print(f'    Legacy dimensions: {leg_dims}')

        shared_dims = [d for d in new_dims if d in leg_dims]
        primary_dim = shared_dims[0] if shared_dims else None

        if primary_dim:
            new_agg = new_y.group_by(primary_dim).agg(pl.col(c).sum() for c in new_full.metric_cols).sort(primary_dim)
            leg_agg = leg_y.group_by(primary_dim).agg(pl.col(c).sum() for c in leg_full.metric_cols).sort(primary_dim)

            joined_sectors = new_agg.join(leg_agg, on=primary_dim, how='outer', suffix='_leg').sort(primary_dim)
            print(f'    {"Sector":<40} {"New":>12} {"Legacy":>12} {"Diff":>10}')
            print(f'    {"-" * 76}')
            for row in joined_sectors.rows(named=True):
                dim_val = row[primary_dim]
                new_vals = [v for k, v in row.items() if k != primary_dim and not k.endswith('_leg') and isinstance(v, float)]
                leg_vals = [v for k, v in row.items() if k.endswith('_leg') and isinstance(v, float)]
                nv = sum(new_vals) if new_vals else 0.0
                lv = sum(leg_vals) if leg_vals else 0.0
                diff = nv - lv
                flag = ' ***' if abs(diff) > 1 else ''
                print(f'    {dim_val!s:<40} {nv:>12.4f} {lv:>12.4f} {diff:>10.4f}{flag}')
