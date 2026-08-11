import math
from typing import TYPE_CHECKING

import polars as pl
import pytest

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN
from nodes.dimensions import Dimension, DimensionCategory
from nodes.edges import Edge, EdgeDimension
from nodes.exceptions import NodeError
from nodes.tests.factories import NodeFactory
from nodes.units import unit_registry

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.context import Context
    from nodes.node import Node

pytestmark = pytest.mark.django_db


def test_node_get_downstream_nodes(context: Context, node):
    output_node = NodeFactory(context=context)
    node.add_output_node(output_node)
    context.finalize_nodes()
    expected = [output_node]
    assert node.get_downstream_nodes() == expected


def test_node_get_upstream_nodes(context, node):
    input_node = NodeFactory(context=context)
    node.add_input_node(input_node)
    expected = [input_node]
    assert node.get_upstream_nodes() == expected


# --- Edge pipeline parity ---------------------------------------------------
#
# _get_output_for_target() runs the typed transformation pipeline at the edge
# boundary. These pin the semantics the legacy from/to_dimensions interpreter
# had, where the shared executor's dataset behavior differs.


def _sector_frame(values: list[float]) -> PathsDataFrame:
    df = pl.DataFrame({
        YEAR_COLUMN: [2020] * len(values),
        'sector': ['buildings', 'transport'][: len(values)],
        VALUE_COLUMN: values,
    })
    meta = DataFrameMeta(
        units={VALUE_COLUMN: unit_registry.parse_units('kt/a')},
        primary_keys=[YEAR_COLUMN, 'sector'],
    )
    return to_ppdf(df, meta=meta)


def _connect(source: Node, target: Node, **edge_kwargs) -> Edge:
    edge = Edge(input_node=source, output_node=target, **edge_kwargs)
    source.edges.append(edge)
    return edge


def test_edge_flatten_does_not_prune_nan(context: Context, node: Node):
    """A bare from_dimensions flatten sums as-is; NaN must propagate, not be dropped."""
    target = NodeFactory.create(context=context)
    _connect(
        node,
        target,
        from_dimensions={'sector': EdgeDimension(categories=[], exclude=True, flatten=True)},
    )
    df = node._get_output_for_target(_sector_frame([1.0, math.nan]), target)
    assert 'sector' not in df.dim_ids
    assert len(df) == 1
    assert math.isnan(df[VALUE_COLUMN][0])


def test_edge_filter_tolerates_declared_all_null_dimension(context: Context, node: Node):
    """A dimension the node declares but that was pruned from the frame is a no-op filter."""
    target = NodeFactory.create(context=context)
    node.output_dimensions['scope'] = Dimension(
        id='scope', label='Scope', categories=[DimensionCategory(id='scope1', label='Scope 1')]
    )
    _connect(
        node,
        target,
        from_dimensions={
            'scope': EdgeDimension(categories=[DimensionCategory(id='scope1', label='Scope 1')], exclude=False, flatten=True),
        },
    )
    df_in = _sector_frame([1.0, 2.0])
    df = node._get_output_for_target(df_in, target)
    assert df.to_dicts() == df_in.to_dicts()


def test_edge_filter_undeclared_missing_dimension_fails(context: Context, node: Node):
    target = NodeFactory.create(context=context)
    _connect(
        node,
        target,
        from_dimensions={'nope': EdgeDimension(categories=[], exclude=True, flatten=True)},
    )
    with pytest.raises(NodeError):
        node._get_output_for_target(_sector_frame([1.0, 2.0]), target)


def test_edge_assign_dimension_creates_categorical(context: Context, node: Node):
    """to_dimensions category assignment creates an indexed Categorical column."""
    target = NodeFactory.create(context=context)
    _connect(
        node,
        target,
        from_dimensions={'sector': EdgeDimension(categories=[], exclude=True, flatten=True)},
        to_dimensions={'ghg': EdgeDimension(categories=[DimensionCategory(id='co2', label='CO2')], exclude=False, flatten=False)},
    )
    df = node._get_output_for_target(_sector_frame([1.0, 2.0]), target)
    assert df.dim_ids == ['ghg']
    assert df['ghg'].dtype == pl.Categorical
    assert df['ghg'].to_list() == ['co2']


def test_edge_bare_to_dimension_asserts_but_does_not_flatten(context: Context, node: Node):
    """A bare to_dimensions entry (parsed as exclude+flatten) is a shape assertion, not an operation."""
    target = NodeFactory.create(context=context)
    _connect(
        node,
        target,
        to_dimensions={'sector': EdgeDimension(categories=[], exclude=True, flatten=True)},
    )
    df_in = _sector_frame([1.0, 2.0])
    df = node._get_output_for_target(df_in, target)
    assert df.dim_ids == ['sector']
    assert df.to_dicts() == df_in.to_dicts()


def test_edge_output_dimension_assertion_fails_on_mismatch(context: Context, node: Node):
    target = NodeFactory.create(context=context)
    _connect(
        node,
        target,
        to_dimensions={'ghg': EdgeDimension(categories=[], exclude=True, flatten=True)},
    )
    with pytest.raises(NodeError):
        node._get_output_for_target(_sector_frame([1.0, 2.0]), target)
