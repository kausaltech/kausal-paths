"""
Tests for the CHP allocation functions of the formula language, and for ``conforms_to_bisko``.

``chp_heat_share`` is the weighted split, ``exergetic_heat_share`` its exergetic case and
``bisko_heat_share`` BISKO's case of that (`nodes.chp_allocation`). They replace ``BiskoChpNode``
and ``BiskoExergeticAllocationNode`` in the BISKO 2.0 template, so that it needs no
``GenericNode``. ``ChpNode`` keeps its own implementation for the frozen BISKO 1.3 module, and the
tests check that each function agrees with the method of ``ChpNode`` it corresponds to.
"""

from typing import TYPE_CHECKING

import pytest

from kausal_common.i18n.pydantic import TranslatedString

from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.defs.node_defs import FormulaConfig, NodeSpec
from nodes.exceptions import NodeError
from nodes.formula import FormulaNode, analyze_formula_dimensions, analyze_formula_units
from nodes.generic import BiskoChpNode, ChpNode
from nodes.tests.test_chp import (
    END_YEAR,
    _connect,
    _FixedOutputNode,
    _heat_by_year,
    _make_chp_node,
    _make_context,
    _Series,
    _series_df,
)
from nodes.units import unit_registry

if TYPE_CHECKING:
    import polars as pl

    from nodes.context import Context

pytestmark = pytest.mark.django_db

HEAT_SHARE = 'hold_into_forecast(bisko_heat_share(electricity_fraction, t_supply, t_return))'
GATE = (
    "conforms_to_bisko(allocation) * max(zero_fill(emissions + constant_frame(0, 'kt/a')) != 0, "
    "zero_fill(consumption + constant_frame(0, 'GWh/a')) == 0)"
)

# The instance factory gives minimum_historical_year 2010 and maximum_historical_year 2018.
MIN_HISTORICAL_YEAR = 2010
MAX_HISTORICAL_YEAR = 2018

FRACTION = _Series([42.0, 40.0, 38.0], [2016, 2017, 2018], unit='%')
SUPPLY = _Series([110.0, 105.0, 100.0], [2016, 2017, 2018], unit='degC')
RETURN = _Series([60.0, 58.0, 55.0], [2016, 2017, 2018], unit='degC')


def _formula_node(context: Context, formula: str, identifier: str = 'out', unit: str = 'dimensionless') -> FormulaNode:
    node = FormulaNode(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity='fraction',
    )
    node._spec = NodeSpec(type_config=FormulaConfig(formula=formula))
    return node


def _source(context: Context, identifier: str, series: _Series) -> _FixedOutputNode:
    unit = series.unit or 'dimensionless'
    return _FixedOutputNode(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity='fraction',
        fixed_df=_series_df({VALUE_COLUMN: series.values}, years=series.years, units={VALUE_COLUMN: unit}),
    )


def _share_node(context: Context, formula: str, inputs: dict[str, _Series]) -> FormulaNode:
    node = _formula_node(context, formula, identifier='heat_share', unit='%')
    for tag, series in inputs.items():
        _connect(_source(context, tag, series), node, tag)
    return node


def _heat_share_node(
    context: Context, fraction: _Series = FRACTION, supply: _Series = SUPPLY, ret: _Series = RETURN, formula: str = HEAT_SHARE
) -> FormulaNode:
    return _share_node(context, formula, {'electricity_fraction': fraction, 't_supply': supply, 't_return': ret})


def _by_year(df: pl.DataFrame) -> dict[int, float]:
    return dict(zip(df[YEAR_COLUMN].to_list(), df[VALUE_COLUMN].to_list(), strict=True))


def test_the_heat_share_is_the_district_heating_share_of_the_bisko_chp_node():
    context = _make_context('bisko-heat-share-same')
    chp = _make_chp_node(
        context,
        cls=BiskoChpNode,
        identifier='chp',
        inputs={'electricity_fraction': FRACTION, 't_supply': SUPPLY, 't_return': RETURN},
    )
    expected = _heat_by_year(chp.compute())

    out = _heat_share_node(context).compute().ensure_unit(VALUE_COLUMN, 'dimensionless')

    assert _by_year(out) == pytest.approx(expected, rel=1e-12)
    forecast = dict(zip(out[YEAR_COLUMN].to_list(), out[FORECAST_COLUMN].to_list(), strict=True))
    assert not any(forecast[year] for year in (2016, 2017, 2018))
    assert all(forecast[year] for year in range(2019, END_YEAR + 1))


def test_the_weighted_split_is_the_efficiency_method_of_the_chp_node():
    context = _make_context('chp-heat-share-efficiency')
    el_eff = _Series([0.40, 0.42, 0.45], [2016, 2017, 2018], unit='dimensionless')
    heat_eff = _Series([0.90, 0.90, 0.88], [2016, 2017, 2018], unit='dimensionless')
    chp = _make_chp_node(
        context,
        method='efficiency',
        inputs={
            'electricity_fraction': FRACTION,
            'electricity_reference_efficiency': el_eff,
            'heat_reference_efficiency': heat_eff,
        },
    )
    expected = _heat_by_year(chp.compute())

    node = _share_node(
        context,
        'chp_heat_share(electricity_fraction, 1 / electricity_reference_efficiency, 1 / heat_reference_efficiency)',
        {'electricity_fraction': FRACTION, 'electricity_reference_efficiency': el_eff, 'heat_reference_efficiency': heat_eff},
    )
    out = node.compute().ensure_unit(VALUE_COLUMN, 'dimensionless')

    assert _by_year(out) == pytest.approx({year: expected[year] for year in (2016, 2017, 2018)}, rel=1e-12)


def test_equal_weights_split_by_energy_alone():
    context = _make_context('chp-heat-share-energy')
    node = _share_node(
        context,
        'chp_heat_share(electricity_fraction, constant_frame(1), constant_frame(1))',
        {'electricity_fraction': FRACTION},
    )

    out = node.compute().ensure_unit(VALUE_COLUMN, 'dimensionless')

    assert _by_year(out) == pytest.approx({2016: 0.58, 2017: 0.60, 2018: 0.62})


def test_weights_must_be_positive():
    context = _make_context('chp-heat-share-weights')
    node = _share_node(
        context,
        'chp_heat_share(electricity_fraction, constant_frame(1), constant_frame(0))',
        {'electricity_fraction': FRACTION},
    )

    with pytest.raises(NodeError, match='weights must be positive'):
        node.compute()


def test_the_exergetic_split_is_the_work_potential_method_of_the_chp_node():
    context = _make_context('chp-heat-share-exergetic')
    ambient = _Series([20.0, 18.0, 15.0], [2016, 2017, 2018], unit='degC')
    inputs = {'electricity_fraction': FRACTION, 't_supply': SUPPLY, 't_return': RETURN, 't_ambient': ambient}
    expected = _heat_by_year(_make_chp_node(context, cls=ChpNode, method='work_potential', inputs=dict(inputs)).compute())

    node = _share_node(context, 'exergetic_heat_share(electricity_fraction, t_supply, t_return, t_ambient)', inputs)
    out = node.compute().ensure_unit(VALUE_COLUMN, 'dimensionless')

    assert _by_year(out) == pytest.approx({year: expected[year] for year in (2016, 2017, 2018)}, rel=1e-12)


def test_the_bisko_split_is_the_exergetic_split_at_283_kelvin():
    context = _make_context('chp-heat-share-bisko-exergetic')
    bisko = _heat_share_node(context, formula='bisko_heat_share(electricity_fraction, t_supply, t_return)').compute()
    exergetic = _heat_share_node(
        context, formula="exergetic_heat_share(electricity_fraction, t_supply, t_return, constant_frame(283, 'K'))"
    ).compute()

    assert _by_year(bisko) == pytest.approx(_by_year(exergetic), rel=1e-12)


def test_a_year_without_every_input_has_no_share():
    context = _make_context('bisko-heat-share-gap')
    supply = _Series([110.0, None, 100.0], [2016, 2017, 2018], unit='degC')

    out = _heat_share_node(context, supply=supply, formula='bisko_heat_share(electricity_fraction, t_supply, t_return)')

    assert out.compute()[YEAR_COLUMN].to_list() == [2016, 2018]


def test_blank_inputs_give_no_share_rather_than_an_error():
    context = _make_context('bisko-heat-share-blank')
    blank = _Series([None, None], [2017, 2018], unit='degC')

    out = _heat_share_node(context, ret=blank).compute()

    assert out.height == 0


@pytest.mark.parametrize(
    ('fraction', 'supply', 'ret', 'message'),
    [
        (_Series([120.0], [2018], unit='%'), SUPPLY, RETURN, 'between 0 and 1'),
        (FRACTION, _Series([50.0], [2018], unit='degC'), _Series([60.0], [2018], unit='degC'), 'not be below'),
        (FRACTION, SUPPLY, _Series([5.0], [2018], unit='degC'), 'must exceed the ambient'),
    ],
)
def test_impossible_inputs_are_refused(fraction: _Series, supply: _Series, ret: _Series, message: str):
    context = _make_context(f'bisko-heat-share-refuse-{message[:8]}')

    with pytest.raises(NodeError, match=message):
        _heat_share_node(context, fraction=fraction, supply=supply, ret=ret).compute()


def test_a_number_is_not_accepted_as_an_input():
    context = _make_context('bisko-heat-share-number')
    node = _heat_share_node(context, formula='bisko_heat_share(electricity_fraction, 373, t_return)')

    with pytest.raises(NodeError, match='needs 3 series'):
        node.compute()


def test_a_formula_calling_bisko_heat_share_declares_a_conforming_allocation():
    context = _make_context('bisko-heat-share-declares')

    assert _formula_node(context, HEAT_SHARE).allocation_conforms_to_bisko
    assert not _formula_node(context, 'electricity_fraction * 0.5').allocation_conforms_to_bisko


def _gate(context: Context, allocation: FormulaNode, emissions: list[tuple[int, float]], consumption: list[tuple[int, float]]):
    gate = _formula_node(context, GATE, identifier='gate')
    _connect(allocation, gate, 'allocation')
    for tag, rows, unit in (('emissions', emissions, 'kt/a'), ('consumption', consumption, 'GWh/a')):
        source = _FixedOutputNode(
            id=f'dh_{tag}',
            context=context,
            name=TranslatedString(tag, default_language='en'),
            unit=unit_registry.parse_units(unit),
            quantity='emissions' if tag == 'emissions' else 'energy',
            fixed_df=_series_df({VALUE_COLUMN: [r[1] for r in rows]}, years=[r[0] for r in rows], units={VALUE_COLUMN: unit}),
        )
        _connect(source, gate, tag)
    return gate.compute()


def test_the_gate_passes_with_a_conforming_split_and_allocated_emissions():
    context = _make_context('bisko-gate-pass')
    out = _gate(context, _heat_share_node(context), [(2016, 120.0), (2017, 118.0)], [(2016, 400.0), (2017, 390.0)])

    values = _by_year(out)
    # Every year of the model, never null; years without district heating pass as not applicable.
    assert list(values) == list(range(MIN_HISTORICAL_YEAR, END_YEAR + 1))
    assert set(values.values()) == {1.0}
    forecast = dict(zip(out[YEAR_COLUMN].to_list(), out[FORECAST_COLUMN].to_list(), strict=True))
    assert forecast == {year: year > MAX_HISTORICAL_YEAR for year in values}


def test_the_gate_fails_where_district_heating_has_no_allocated_emissions():
    context = _make_context('bisko-gate-nothing')
    out = _gate(context, _heat_share_node(context), [(2016, 120.0), (2017, 0.0)], [(2016, 400.0), (2017, 390.0)])

    values = _by_year(out)
    assert values[2016] == 1.0
    assert values[2017] == 0.0
    assert values[2018] == 1.0


def test_the_gate_fails_in_every_year_when_the_split_is_not_bisko():
    context = _make_context('bisko-gate-method')
    allocation = _heat_share_node(context, formula='electricity_fraction * 0 + 0.2')

    out = _gate(context, allocation, [(2016, 120.0)], [(2016, 400.0)])

    assert set(out[VALUE_COLUMN].to_list()) == {0.0}


def test_conforms_to_bisko_needs_an_allocation_node():
    context = _make_context('bisko-gate-not-allocation')
    gate = _formula_node(context, 'conforms_to_bisko(allocation)', identifier='gate')
    _connect(_source(context, 'plain', FRACTION), gate, 'allocation')

    with pytest.raises(NodeError, match='does not say whether its allocation conforms'):
        gate.compute()


@pytest.mark.parametrize(
    'formula',
    [
        'chp_heat_share(f, ts, tr) * conforms_to_bisko(a)',
        'exergetic_heat_share(f, ts, tr, a) * conforms_to_bisko(a)',
        'bisko_heat_share(f, ts, tr) * conforms_to_bisko(a)',
    ],
)
def test_the_formula_analyses_know_the_functions(formula: str):
    units = analyze_formula_units(
        formula,
        {name: unit_registry.parse_units(unit) for name, unit in (('f', '%'), ('ts', 'degC'), ('tr', 'degC'), ('a', '%'))},
    )
    dims = analyze_formula_dimensions(formula, {'f': set(), 'ts': set(), 'tr': set(), 'a': set()})

    assert units.unit == unit_registry.dimensionless
    assert not units.warnings
    assert dims.dims == set()
    assert not dims.warnings
