from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nodes.exceptions import NodeError

from .context import Context
from .dimensions import Dimension, DimensionCategory

if TYPE_CHECKING:
    from .node import Node


@dataclass
class EdgeDimension:
    categories: list[DimensionCategory]
    exclude: bool
    flatten: bool

    @classmethod
    def from_config(cls, dc: dict, context: Context, node: Node, node_dims: dict[str, Dimension]) -> tuple[str, EdgeDimension]:
        if 'id' not in dc:
            # If 'id' is not supplied, assume it's the first and only dimension
            if len(node_dims) == 1:
                dim_id, dim = list(node_dims.items())[0]
            else:
                raise NodeError(node, 'dimension id not supplied')
        else:
            dim_id = dc['id']
            if dim_id not in context.dimensions:
                raise NodeError(node, 'dimension %s not found' % dim_id)
            dim = context.dimensions[dim_id]

        flatten = dc.get('flatten')
        exclude = dc.get('exclude')
        cat_ids = dc.get('categories')
        groups = dc.get('groups')
        if groups is not None:
            if cat_ids is None:
                cat_ids = []
            for gid in groups:
                cats = dim.get_cats_for_group(gid)
                cat_ids += [cat.id for cat in cats]

        if cat_ids is None:
            cats = []
            if flatten not in (None, True) or exclude not in (None, True):
                raise Exception("When categories are not supplied, you must not supply 'flatten' or 'exclude'")
            flatten = True
            exclude = True
        else:
            cats = [dim.get(cat_id) for cat_id in cat_ids]
            flatten = bool(flatten)
            exclude = bool(exclude)
        return (dim_id, cls(categories=cats, exclude=exclude, flatten=flatten))


@dataclass
class Edge:
    input_node: Node
    output_node: Node
    tags: list[str] = field(default_factory=list)
    from_dimensions: dict[str, EdgeDimension] = field(default_factory=dict)
    to_dimensions: dict[str, EdgeDimension] | None = None
    metrics: list[str] = field(default_factory=list)

    _input_port_id: str | None = None
    _output_port_id: str | None = None

    def __post_init__(self):
        self.tags = self.tags.copy()
        self.from_dimensions = self.from_dimensions.copy()
        if self.to_dimensions is not None:
            self.to_dimensions = self.to_dimensions.copy()
        self.metrics = self.metrics.copy()

    @classmethod
    def from_config(cls, config: dict | str, node: Node, is_output: bool, context: Context) -> Edge:
        if isinstance(config, str):
            other_id = config
        else:
            s = config.get('id')
            if s is None:
                raise NodeError(node, 'node id not given in edge definition')
            assert isinstance(s, str)
            other_id = s
        assert isinstance(other_id, str)
        other = context.nodes.get(other_id)
        if other is None:
            raise NodeError(node, 'node %s not found' % other_id)

        args: dict[str, Any] = {}
        args['output_node'], args['input_node'] = (other, node) if is_output else (node, other)
        if isinstance(config, dict):
            tags = config.get('tags', [])
            if isinstance(tags, str):
                tags = [tags]
            args['tags'] = tags

            dcs = config.get('from_dimensions', [])
            ndims: dict[str, EdgeDimension] = {}
            for dc in dcs:
                dim_id, ed = EdgeDimension.from_config(dc, context, node, args['input_node'].output_dimensions)
                ndims[dim_id] = ed
            args['from_dimensions'] = ndims

            dcs = config.get('to_dimensions', None)
            if dcs is not None:
                ndims = {}
                for dc in dcs:
                    dim_id, ed = EdgeDimension.from_config(dc, context, node, args['output_node'].input_dimensions)
                    ndims[dim_id] = ed
                args['to_dimensions'] = ndims

            metrics = config.get('metrics', [])
            if metrics:
                args['metrics'] = metrics

        return Edge(**args)
