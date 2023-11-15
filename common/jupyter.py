from __future__ import annotations

from typing import TYPE_CHECKING
from collections import namedtuple

import polars as pl

from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN

if TYPE_CHECKING:
    from nodes.node import Node
    from nodes.context import Context


_django_initialized = False
plotly_theme: str = 'ggplot2'


def _init_django():
    global _django_initialized
    if _django_initialized:
        return
    import os
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "paths.settings")
    django.setup()
    _django_initialized = True


def _enable_autoreload():
    from IPython import get_ipython
    ip = get_ipython()
    if 'IPython.extensions.autoreload' in ip.extension_manager.loaded:  # noqa
        return
    ip.magic(r'%load_ext autoreload')  # noqa
    ip.magic(r'%autoreload 2')  # noqa


def _get_context(instance_id: str):
    from nodes.instance import InstanceLoader
    from common import polars_ext  # noqa

    _enable_autoreload()
    _init_django()

    # Disable locals when running in notebook
    from rich import traceback
    traceback.install(show_locals=False)

    config_fn = 'configs/%s.yaml' % instance_id
    loader = InstanceLoader.from_yaml(config_fn)
    context = loader.context
    context.cache.clear()
    return context


class NotebookNodes(dict[str, 'Node']):
    context: Context


def get_nodes(instance_id: str):
    context = _get_context(instance_id)
    out = NotebookNodes(context.nodes)
    out.context = context
    return out


def get_datasets(instance_id: str):
    context = _get_context(instance_id)
    context.generate_baseline_values()
    name = '%sDatasets' % (instance_id.capitalize())
    datasets = {key.replace('/', '_').replace('-', '_'): val for key, val in context.dvc_datasets.items()}
    kls = namedtuple(name, list(datasets))  # type: ignore
    obj = kls(**datasets)
    return obj


def plot_node(node: Node):
    from plotly import express as px

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
        if len(dim_ids):
            color_col, _ = dim_ids.pop(0)
            if len(dim_ids):
                facet_col, _ = dim_ids.pop(0)
                assert not len(dim_ids)

        labels = {
            YEAR_COLUMN: 'Year',
            m_col: '%s (%s)' % (metric.label or m_col, str(unit))
        }
        fig = px.line(
            mdf.to_dict(as_series=False),
            x=YEAR_COLUMN,
            y=m_col,
            color=color_col,
            facet_col=facet_col,
            labels=labels
        )
        fc_years = df.filter(pl.col(FORECAST_COLUMN))[YEAR_COLUMN].unique().sort()
        fc_start = fc_years.min()
        fc_end = fc_years.max()
        fig.add_vrect(
            fc_start, fc_end, fillcolor='grey', opacity=0.2, annotation_text='Forecast',
            annotation_position='top left',
        )
        return fig
