"""
IPython bootstrap and plotting helpers for the notebooks in this directory.

The generic half of this module -- `get_context` and `get_nodes`, which have nothing to
do with notebooks -- now lives in `tools/instance_support.py`, because the command-line
tools that use it moved to `tools/` and `notebooks/*` is excluded from ruff and mypy.
They are re-exported here so an existing notebook importing them from this module keeps
working.
"""

from __future__ import annotations

import os
import sys
from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from tools.instance_support import get_context, get_nodes

if TYPE_CHECKING:
    from nodes.node import Node

__all__ = ['get_context', 'get_datasets', 'get_nodes', 'initialize_notebook_env', 'plot_node', 'plotly_theme']


def initialize_notebook_env():
    from IPython import get_ipython

    path = Path(__file__).parent.parent
    if not (path / Path('manage.py')).exists():
        raise Exception('Unable to find project root')
    if str(path) not in sys.path:
        sys.path.append(str(path))

    ip = get_ipython()
    assert ip is not None
    assert ip.extension_manager is not None
    if 'IPython.extensions.autoreload' not in ip.extension_manager.loaded:
        ip.run_line_magic(magic_name='reload_ext', line='autoreload')
        ip.run_line_magic(magic_name='autoreload', line='2')
    ip.run_line_magic(magic_name='matplotlib', line='ipympl')

    os.environ['DJANGO_ALLOW_ASYNC_UNSAFE'] = '1'

    from kausal_common.development.django import init_django

    init_django()


plotly_theme: str = 'ggplot2'


def get_datasets(instance_id: str):
    context = get_context(instance_id)
    context.generate_baseline_values()
    datasets = {key.replace('/', '_').replace('-', '_'): val for key, val in context.dvc_datasets.items()}
    kls = namedtuple('Datasets', list(datasets))  # type: ignore[misc]  # noqa: PYI024
    obj = kls(**datasets)
    return obj


def plot_node(node: Node):
    from plotly import express as px

    from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN

    df = node.get_output_pl()
    for metric in node.output_metrics.values():
        m_col = metric.column_id
        unit = df.get_unit(m_col)
        mdf = df.select([YEAR_COLUMN, FORECAST_COLUMN, *df.dim_ids, m_col])
        mdf = mdf.clear_unit(m_col)
        dim_ids = [(dim_id, len(mdf[dim_id].unique())) for dim_id in df.dim_ids]
        dim_ids = sorted(dim_ids, key=lambda x: x[1], reverse=True)
        color_col = None
        facet_col = None
        if dim_ids:
            color_col, _ = dim_ids.pop(0)
            if dim_ids:
                facet_col, _ = dim_ids.pop(0)
                assert not dim_ids

        labels = {YEAR_COLUMN: 'Year', m_col: '%s (%s)' % (metric.label or m_col, str(unit))}
        fig = px.line(mdf.to_dict(as_series=False), x=YEAR_COLUMN, y=m_col, color=color_col, facet_col=facet_col, labels=labels)
        fc_years = df.filter(pl.col(FORECAST_COLUMN))[YEAR_COLUMN].unique().sort()
        fc_start = fc_years.min()
        fc_end = fc_years.max()
        fig.add_vrect(
            fc_start,
            fc_end,
            fillcolor='grey',
            opacity=0.2,
            annotation_text='Forecast',
            annotation_position='top left',
        )
        return fig
    return None
