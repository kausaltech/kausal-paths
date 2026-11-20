"""
Tests for ``constant_frame(value[, 'unit'])`` and unary minus in the formula language.

``constant_frame`` replaces ``ConstantNode``: one value in every model year, usable inline in a
downstream formula or as the whole formula of a node that exists to document the constant.
"""

from typing import TYPE_CHECKING

import pytest

from kausal_common.i18n.pydantic import TranslatedString

from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.defs.node_defs import FormulaConfig, NodeSpec
from nodes.exceptions import NodeError
from nodes.formula import FormulaNode, analyze_formula_dimensions, analyze_formula_units
from nodes.tests.test_formula_from_year import _context, _formula_over
from nodes.units import unit_registry
from params.param import BoolParameter

if TYPE_CHECKING:
    from common import polars as ppl
    from nodes.context import Context

pytestmark = pytest.mark.django_db


def _constant_node(context: Context, formula: str, unit: str = 'dimensionless') -> FormulaNode:
    node = FormulaNode(
        id='constant',
        context=context,
        name=TranslatedString('constant', default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity='fraction',
    )
    node._spec = NodeSpec(type_config=FormulaConfig(formula=formula))
    return node


def _values(df: ppl.PathsDataFrame) -> dict[int, float]:
    return dict(zip(df[YEAR_COLUMN].to_list(), df[VALUE_COLUMN].to_list(), strict=True))


def test_constant_frame_covers_every_model_year():
    context = _context('constant-years')
    instance = context.instance

    out = _constant_node(context, 'constant_frame(0.5)').compute()

    assert out[YEAR_COLUMN].to_list() == list(range(instance.minimum_historical_year, instance.model_end_year + 1))
    assert set(out[VALUE_COLUMN].to_list()) == {0.5}
    assert not out[FORECAST_COLUMN].any()
    assert out.dim_ids == []


@pytest.mark.parametrize(('value', 'expected'), [(True, 1.0), (False, 0.0)])
def test_constant_frame_turns_a_boolean_parameter_into_one_or_zero(value: bool, expected: float):
    node = _constant_node(_context(f'constant-bool-{value}'), '1 - constant_frame(condition)')
    param = BoolParameter(local_id='condition', is_customizable=False)
    param.set(value)
    node.add_parameter(param)

    assert set(node.compute()[VALUE_COLUMN].to_list()) == {1.0 - expected}


def test_constant_frame_labels_a_bare_number_with_the_unit():
    out = _constant_node(_context('constant-unit'), "constant_frame(1000, 'kt/a')", unit='Mt/a').compute()

    assert out[VALUE_COLUMN].to_list() == pytest.approx([1.0] * out.height)


def test_constant_frame_refuses_a_data_frame():
    target = _formula_over(_context('constant-frame-df'), 'constant_frame(emissions)', [(2020, 1.0)])

    with pytest.raises(NodeError, match='not a data frame'):
        target.compute()


def test_unary_minus_negates_a_frame_and_a_number():
    context = _context('unary-minus')
    rows = [(2020, 3.0), (2021, -2.0)]

    assert _values(_formula_over(context, '-emissions', rows, unit='kt/a').compute()) == {2020: -3.0, 2021: 2.0}
    assert _values(_constant_node(_context('unary-minus-number'), 'constant_frame(-0.25)').compute())[2020] == -0.25
    assert _values(_formula_over(_context('unary-plus'), '+emissions', rows, unit='kt/a').compute()) == {2020: 3.0, 2021: -2.0}


def test_unary_not_is_still_refused():
    with pytest.raises(NodeError, match='Unary operators'):
        _constant_node(_context('unary-not'), 'not constant_frame(1)').compute()


def test_the_analysers_know_constant_frame():
    units = analyze_formula_units("constant_frame(1000, 'kt/a')", {})
    assert units.errors == []
    assert units.unit == unit_registry.parse_units('kt/a')
    assert analyze_formula_units("constant_frame(1, 'no_such_unit')", {}).errors != []
    assert analyze_formula_dimensions('constant_frame(1)', {}).dims == set()
