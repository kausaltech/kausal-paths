from __future__ import annotations

import os
import sys
from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.models import InstanceConfig
    from nodes.node import Node


def initialize_notebook_env():
    from IPython import get_ipython  # pyright: ignore

    path = Path(__file__).parent.parent
    if not (path / Path('manage.py')).exists():
        raise Exception("Unable to find project root")
    if str(path) not in sys.path:
        sys.path.append(str(path))

    ip = get_ipython()
    assert ip is not None
    assert ip.extension_manager is not None
    if 'IPython.extensions.autoreload' not in ip.extension_manager.loaded:
        ip.magic(r'%reload_ext autoreload')
        ip.magic(r'%autoreload 2')
    ip.magic(r'%matplotlib ipympl')

    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "1"

    from kausal_common.development.django import init_django
    init_django()


plotly_theme: str = 'ggplot2'


def _get_instance_from_db(instance_id: str) -> None | InstanceConfig:
    from nodes.models import InstanceConfig

    ic = InstanceConfig.objects.filter(identifier=instance_id).first()
    if ic is None:
        return None
    return ic


def get_context(instance_id: str):
    from common import polars_ext  # noqa: F401
    from nodes.instance_loader import InstanceLoader

    ic = _get_instance_from_db(instance_id)
    if ic is not None:
        return ic.get_instance().context

    project_root = Path(__file__).parent.parent
    config_fn = (Path(project_root) / 'configs' / ('%s.yaml' % instance_id)).resolve()
    loader = InstanceLoader.from_yaml(config_fn)
    context = loader.context
    context.cache.clear()
    return context


class NotebookNodes(dict[str, 'Node']):
    context: Context


def get_nodes(instance_id: str):
    context = get_context(instance_id)
    out = NotebookNodes(context.nodes)
    out.context = context
    return out


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
    return None
