"""
Tests for ``from_year(df, year)`` and one-argument ``sum_dim(df)`` in the formula language.

It exists for quantities that only begin at a date, such as a carbon budget counted from
1 January 2020: the rows before must be absent, not zero, or a later subtraction invents
values for years the quantity does not cover.
"""

from typing import TYPE_CHECKING

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.defs.node_defs import FormulaConfig, NodeSpec
from nodes.edges import Edge
from nodes.exceptions import NodeError
from nodes.formula import FormulaNode
from nodes.node import Node
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry

if TYPE_CHECKING:
    from common import polars as ppl
    from nodes.context import Context

pytestmark = pytest.mark.django_db


class _FixedOutputNode(Node):
    """A leaf node whose output is a fixed, caller-supplied PathsDataFrame. Test-only."""

    allow_unknown_dimensions: bool = False

    def __init__(self, *args, fixed_df: ppl.PathsDataFrame, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_df = fixed_df

    def compute(self) -> ppl.PathsDataFrame:
        return self._fixed_df


def _context(identifier: str) -> Context:
    instance = InstanceFactory.create(id=identifier, name=identifier)
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name=identifier)
    return instance.context


def _formula_over(context: Context, formula: str, rows: list[tuple[int, float]], unit: str = 'kt') -> FormulaNode:
    df = pl.DataFrame(
        {YEAR_COLUMN: [r[0] for r in rows], VALUE_COLUMN: [r[1] for r in rows], FORECAST_COLUMN: [False] * len(rows)},
        schema={YEAR_COLUMN: pl.Int64, VALUE_COLUMN: pl.Float64, FORECAST_COLUMN: pl.Boolean},
    )
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kt/a')}, primary_keys=[YEAR_COLUMN])
    source = _FixedOutputNode(
        id='emissions',
        context=context,
        name=TranslatedString('emissions', default_language='en'),
        unit=unit_registry.parse_units('kt/a'),
        quantity='emissions',
        fixed_df=to_ppdf(df, meta),
    )
    target = FormulaNode(
        id='target',
        context=context,
        name=TranslatedString('target', default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity='mass',
    )
    target._spec = NodeSpec(type_config=FormulaConfig(formula=formula))
    edge = Edge(input_node=source, output_node=target, tags=[])
    source.add_edge(edge)
    target.add_edge(edge)
    return target


def test_from_year_drops_the_years_before():
    target = _formula_over(
        _context('from-year'), 'cumulative(from_year(emissions, 2020))', [(2018, 5.0), (2019, 5.0), (2020, 1.0), (2021, 2.0)]
    )

    out = target.compute()

    assert dict(zip(out[YEAR_COLUMN].to_list(), out[VALUE_COLUMN].to_list(), strict=True)) == {2020: 1.0, 2021: 3.0}


def test_sum_dim_with_one_argument_sums_every_dimension():
    context = _context('sum-dim-all')
    df = pl.DataFrame(
        {
            YEAR_COLUMN: [2020, 2020, 2020, 2021],
            'sector': ['a', 'a', 'b', 'b'],
            'energy_carrier': ['x', 'y', 'x', 'x'],
            VALUE_COLUMN: [1.0, 2.0, 4.0, 8.0],
            FORECAST_COLUMN: [False] * 4,
        },
    )
    meta = DataFrameMeta(
        units={VALUE_COLUMN: unit_registry.parse_units('kt/a')}, primary_keys=[YEAR_COLUMN, 'sector', 'energy_carrier']
    )
    target = _formula_over(context, 'sum_dim(emissions)', [], unit='kt/a')
    source = target.input_nodes[0]
    assert isinstance(source, _FixedOutputNode)
    source._fixed_df = to_ppdf(df, meta)
    source.allow_unknown_dimensions = True  # a test leaf carries dimensions it does not declare

    out = target.compute()

    assert out.dim_ids == []
    assert dict(zip(out[YEAR_COLUMN].to_list(), out[VALUE_COLUMN].to_list(), strict=True)) == {2020: 7.0, 2021: 8.0}


def test_from_year_requires_a_literal_year():
    target = _formula_over(_context('from-year-literal'), 'from_year(emissions, emissions)', [(2020, 1.0)])

    with pytest.raises(NodeError, match='literal year'):
        target.compute()
