from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from django.http import Http404, JsonResponse

import polars as pl
import sentry_sdk

from paths.context import realm_context

from nodes.constants import VALUE_COLUMN
from nodes.models import NodeConfig

if TYPE_CHECKING:
    from paths.types import PathsAdminRequest

    from nodes.node import Node


class CategoryOption(TypedDict):
    id: str
    label: str


class DimensionOption(TypedDict):
    id: str
    label: str
    categories: list[CategoryOption]


class NodeGraphOptions(TypedDict):
    dimensions: list[DimensionOption]


def _categories_with_data(node: Node) -> dict[str, set[str]] | None:
    """
    Return, per output dimension, the categories for which the node has a value.

    Computing the node is the only way to know, and it can fail; then None is returned and the
    caller offers every category of each dimension instead, so the editor is not blocked.
    """
    try:
        with node.context.run():
            df = node.get_output_pl()
    except Exception as err:
        sentry_sdk.capture_exception(err)
        return None
    value_col = VALUE_COLUMN if VALUE_COLUMN in df.columns else None
    if value_col is None:
        metric = node.get_default_output_metric(required=False)
        value_col = metric.column_id if metric is not None else None
    if value_col is not None and value_col in df.columns:
        df = df.filter(pl.col(value_col).is_not_null())
    return {dim_id: set(df[dim_id].cast(pl.Utf8).unique().to_list()) for dim_id in node.output_dimensions if dim_id in df.columns}


def node_graph_options(request: PathsAdminRequest) -> JsonResponse:
    """
    Return the dimensions of a node and the categories it has data for.

    Used by the admin form of `ParameterPanelGraphBlock` to offer only what can be chosen for the
    node. Only nodes of the instance being edited are answered.
    """
    node_pk = request.GET.get('node', '')
    if not realm_context.is_set() or not node_pk.isdigit():
        raise Http404
    ic = realm_context.get().realm
    node_config = NodeConfig.objects.filter(instance=ic, pk=int(node_pk)).first()
    if node_config is None:
        raise Http404
    node = ic.get_instance().context.nodes.get(node_config.identifier)
    if node is None:
        raise Http404

    with_data = _categories_with_data(node)
    dimensions: list[DimensionOption] = []
    for dim_id, dim in node.output_dimensions.items():
        present = with_data.get(dim_id) if with_data is not None else None
        categories = [
            CategoryOption(id=cat.id, label=str(cat.label)) for cat in dim.categories if present is None or cat.id in present
        ]
        dimensions.append(DimensionOption(id=dim_id, label=str(dim.label), categories=categories))
    return JsonResponse(NodeGraphOptions(dimensions=dimensions))
