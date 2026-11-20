"""
Every node class carries the qualifiers of its inputs into its output.

A qualifier (``<metric>__qual``, see `common.qualifiers`) says something about a value, such as
whether it was reported or how good it is. Whatever a node does to the values, the qualifier
must come out the other side, combined by the rule its definition declares. A class that loses
it gives the user a figure whose grade silently vanished one step downstream, and a model whose
classes disagree on this is hard to reason about.

Each case gives a node class one input node and one input dataset, both qualified, and checks
that the output is qualified too. A new class that combines inputs belongs in ``CASES``.
"""

from typing import TYPE_CHECKING

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common import qualifiers
from nodes.constants import VALUE_COLUMN
from nodes.defs.node_defs import FormulaConfig, NodeSpec
from nodes.edges import Edge
from nodes.formula import FormulaNode
from nodes.tests.node_input_harness import add_binding, binding
from nodes.tests.test_add_multiply_semantics import (
    _additive,
    _additive2,
    _attach,
    _connect,
    _FixedOutputNode,
    _generic,
    _make_context,
    _multiplicative,
    _multiplicative2,
    _ppdf,
    _reported,
)
from nodes.units import unit_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from nodes.context import Context
    from nodes.node import Node

pytestmark = pytest.mark.django_db

QUAL = qualifiers.qualifier_column(VALUE_COLUMN)


def _qualified_source(context: Context, identifier: str, value: float, unit: str) -> _FixedOutputNode:
    return _FixedOutputNode(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity='energy',
        fixed_df=_reported(_ppdf([(2020, value)], unit=unit), reported=True),
    )


def _with_dataset(node: Node, unit: str) -> Node:
    _attach(node, 'measured', _reported(_ppdf([(2020, 2.0)], unit=unit), reported=True))
    return node


def _sum_v1(context: Context) -> Node:
    node = _with_dataset(_additive(context, 'out'), 'kWh')
    _connect(_qualified_source(context, 'source', 1.0, 'kWh'), node)
    return node


def _sum_v2(context: Context) -> Node:
    node = _with_dataset(_additive2(context, 'out'), 'kWh')
    _connect(_qualified_source(context, 'source', 1.0, 'kWh'), node)
    return node


def _product_v1(context: Context) -> Node:
    node = _with_dataset(_multiplicative(context, 'out'), 'kW')
    _connect(_qualified_source(context, 'source', 3.0, 'h'), node)
    return node


def _product_v2(context: Context) -> Node:
    node = _with_dataset(_multiplicative2(context, 'out'), 'kW')
    source = _qualified_source(context, 'source', 3.0, 'h')
    edge = Edge(input_node=source, output_node=node, tags=[])
    source.add_edge(edge)
    node.add_edge(edge)
    add_binding(
        node,
        binding(
            'factors',
            position=len(node.runtime_input_bindings),
            source_kind='node',
            source_id=source.id,
            source=source,
            value_loader=lambda: source.get_output_pl(target_node=node),
        ),
    )
    return node


def _generic_sum(context: Context) -> Node:
    node = _with_dataset(_generic(context, 'out', 'add_datasets,add'), 'kWh')
    _connect(_qualified_source(context, 'source', 1.0, 'kWh'), node)
    return node


def _formula(context: Context) -> Node:
    node = FormulaNode(
        id='out',
        context=context,
        name=TranslatedString('out', default_language='en'),
        unit=unit_registry.parse_units('kWh'),
        quantity='energy',
    )
    node._spec = NodeSpec(type_config=FormulaConfig(formula='first + second * 2'))
    for identifier, value in (('first', 1.0), ('second', 2.0)):
        source = _qualified_source(context, identifier, value, 'kWh')
        edge = Edge(input_node=source, output_node=node, tags=[])
        source.add_edge(edge)
        node.add_edge(edge)
    return node


# The v1 classes return pandas and convert back, and the pandas conversion leaves qualifiers out
# by design, so they cannot carry one. They are to be retired rather than fixed; see
# docs/plans/additive-multiplicative-modernization.md.
_V1 = pytest.mark.xfail(strict=True, reason='v1 classes compute in pandas, which carries no qualifiers')

CASES: list[tuple[str, Callable[[Context], Node]] | object] = [
    pytest.param('additive2', _sum_v2, id='AdditiveNode2'),
    pytest.param('multiplicative2', _product_v2, id='MultiplicativeNode2'),
    pytest.param('generic', _generic_sum, id='GenericNode'),
    pytest.param('formula', _formula, id='FormulaNode'),
    pytest.param('additive', _sum_v1, id='AdditiveNode', marks=_V1),
    pytest.param('multiplicative', _product_v1, id='MultiplicativeNode', marks=_V1),
]


@pytest.mark.parametrize(('name', 'build'), CASES)
def test_the_output_carries_the_qualifier_of_the_inputs(name: str, build: Callable[[Context], Node]) -> None:
    node = build(_make_context(f'qualifier-conformance-{name}'))

    out = node.compute()

    assert QUAL in out.columns, f'{type(node).__name__} lost the qualifier of its inputs'
    reported = out[QUAL].struct.field(qualifiers.REPORTED).to_list()
    assert reported == [{qualifiers.ANY: True, qualifiers.ALL: True}] * len(out)
    assert out[VALUE_COLUMN].is_not_null().all()


def test_a_projection_inside_a_node_cannot_drop_the_qualifier() -> None:
    """The cause of the v2 loss: a hand-written projection of Year, Value and Forecast."""
    df = _reported(_ppdf([(2020, 1.0)]), reported=True)

    projected = df.select([*df.primary_keys, pl.col(VALUE_COLUMN), 'Forecast'])

    assert QUAL in projected.columns
