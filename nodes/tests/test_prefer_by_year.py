"""
Tests for ``prefer_by_year``: per-year choice between a city's own data and a default series.

The mechanism exists because a city moves from national default statistics to its own data
collection one reporting year at a time. The properties that matter are therefore about the
*granularity* of the choice, not about arithmetic: a covered year comes wholly from the city's
data, an uncovered year wholly from the default, and an empty own-data template selects the
default rather than reading as a series of reported zeros.
"""

from typing import TYPE_CHECKING

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.edges import Edge
from nodes.formula import FormulaNode
from nodes.node import Node
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry
from params.param import StringParameter

if TYPE_CHECKING:
    from common import polars as ppl

pytestmark = pytest.mark.django_db


def _ppdf(rows: list[tuple[int, float | None]], unit: str = 'MWh/a') -> ppl.PathsDataFrame:
    df = pl.DataFrame(
        {
            YEAR_COLUMN: [r[0] for r in rows],
            VALUE_COLUMN: [r[1] for r in rows],
            FORECAST_COLUMN: [False] * len(rows),
        },
        schema={YEAR_COLUMN: pl.Int64, VALUE_COLUMN: pl.Float64, FORECAST_COLUMN: pl.Boolean},
    )
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=[YEAR_COLUMN])
    return to_ppdf(df, meta)


def _dimmed(rows: list[tuple[int, str, float | None]], unit: str = 'MWh/a') -> ppl.PathsDataFrame:
    df = pl.DataFrame(
        {
            YEAR_COLUMN: [r[0] for r in rows],
            'energy_carrier': [r[1] for r in rows],
            VALUE_COLUMN: [r[2] for r in rows],
            FORECAST_COLUMN: [False] * len(rows),
        },
        schema={
            YEAR_COLUMN: pl.Int64,
            'energy_carrier': pl.Utf8,
            VALUE_COLUMN: pl.Float64,
            FORECAST_COLUMN: pl.Boolean,
        },
    )
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=[YEAR_COLUMN, 'energy_carrier'])
    return to_ppdf(df, meta)


def _as_map(df: ppl.PathsDataFrame) -> dict[int, float]:
    return dict(zip(df[YEAR_COLUMN].to_list(), df[VALUE_COLUMN].to_list(), strict=True))


def test_own_years_win_and_default_fills_the_rest():
    own = _ppdf([(2021, 10.0), (2022, 11.0)])
    default = _ppdf([(2019, 1.0), (2020, 2.0), (2021, 3.0), (2022, 4.0)])

    out = own.paths.prefer_by_year(default)

    assert _as_map(out) == {2019: 1.0, 2020: 2.0, 2021: 10.0, 2022: 11.0}


def test_empty_own_data_selects_the_default_everywhere():
    """An own-data template nobody has filled in must not read as a series of reported zeros."""
    own = _ppdf([])
    default = _ppdf([(2020, 2.0), (2021, 3.0)])

    out = own.paths.prefer_by_year(default)

    assert _as_map(out) == {2020: 2.0, 2021: 3.0}


def test_all_null_own_data_selects_the_default_everywhere():
    own = _ppdf([(2020, None), (2021, None)])
    default = _ppdf([(2020, 2.0), (2021, 3.0)])

    out = own.paths.prefer_by_year(default)

    assert _as_map(out) == {2020: 2.0, 2021: 3.0}


def test_a_reported_zero_is_data_and_still_wins():
    """BISKO allows an entered zero. It is a value, so it selects the city's data for that year."""
    own = _ppdf([(2021, 0.0)])
    default = _ppdf([(2020, 2.0), (2021, 3.0)])

    out = own.paths.prefer_by_year(default)

    assert _as_map(out) == {2020: 2.0, 2021: 0.0}


def test_a_covered_year_is_never_mixed_with_the_default():
    """
    The city supplied electricity for 2021 but not district heating.

    2021 is a covered year, so it is served wholly from the city's data and the gap stays a
    gap -- the availability nodes report it. Filling it from the default would produce a year
    that is partly one source and partly the other.
    """
    own = _dimmed([(2021, 'electricity', 10.0)])
    default = _dimmed([
        (2020, 'electricity', 1.0),
        (2020, 'district_heating', 2.0),
        (2021, 'electricity', 3.0),
        (2021, 'district_heating', 4.0),
    ])

    out = own.paths.prefer_by_year(default)
    got = {(r[YEAR_COLUMN], r['energy_carrier']): r[VALUE_COLUMN] for r in out.to_dicts()}

    assert got == {
        (2020, 'electricity'): 1.0,
        (2020, 'district_heating'): 2.0,
        (2021, 'electricity'): 10.0,
    }


def test_units_are_reconciled_to_the_preferred_frame():
    own = _ppdf([(2021, 1.0)], unit='GWh/a')
    default = _ppdf([(2020, 2000.0)], unit='MWh/a')

    out = own.paths.prefer_by_year(default)

    assert out.get_unit(VALUE_COLUMN) == unit_registry.parse_units('GWh/a')
    assert _as_map(out) == {2020: 2.0, 2021: 1.0}


def test_dimension_mismatch_is_refused():
    own = _dimmed([(2021, 'electricity', 10.0)])
    default = _ppdf([(2020, 2.0)])

    with pytest.raises(ValueError, match='Dimensions must match'):
        own.paths.prefer_by_year(default)


# --- Through the formula language -------------------------------------------------------------


class _FixedOutputNode(Node):
    """A leaf node whose output is a fixed, caller-supplied PathsDataFrame. Test-only."""

    def __init__(self, *args, fixed_df: ppl.PathsDataFrame, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_df = fixed_df

    def compute(self) -> ppl.PathsDataFrame:
        return self._fixed_df


def _make_context(identifier: str):
    instance = InstanceFactory.create(id=identifier, name=identifier)
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name=identifier)
    return instance.context


def _fixed_node(context, identifier: str, rows: list[tuple[int, float | None]]) -> _FixedOutputNode:
    return _FixedOutputNode(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units('MWh/a'),
        quantity='energy',
        fixed_df=_ppdf(rows),
    )


def _connect(input_node: Node, output_node: Node) -> None:
    edge = Edge(input_node=input_node, output_node=output_node, tags=[])
    input_node.add_edge(edge)
    output_node.add_edge(edge)


def test_formula_node_prefers_own_years_over_the_default():
    """The whole point, exercised the way a model config uses it."""
    context = _make_context('prefer-by-year-formula')
    target = FormulaNode(
        id='road_transport_energy',
        context=context,
        name=TranslatedString('target', default_language='en'),
        unit=unit_registry.parse_units('MWh/a'),
        quantity='energy',
    )
    target.parameters['formula'] = StringParameter(local_id='formula', value='prefer_by_year(own_data, default_data)')
    own_data = _fixed_node(context, 'own_data', [(2022, 500.0), (2023, 550.0)])
    default_data = _fixed_node(context, 'default_data', [(2020, 100.0), (2021, 110.0), (2022, 120.0), (2023, 130.0)])
    _connect(own_data, target)
    _connect(default_data, target)

    assert _as_map(target.compute()) == {2020: 100.0, 2021: 110.0, 2022: 500.0, 2023: 550.0}


# --- The coverage argument ---------------------------------------------------------------------


def _flags(rows: list[tuple[int, float | None]]) -> ppl.PathsDataFrame:
    return _ppdf(rows, unit='dimensionless')


def test_coverage_overrides_zero_filled_values():
    """
    The trap the coverage argument exists for.

    An own-data node reading an empty template needs ``empty_to_zero`` to produce a dimensioned
    frame at all. Those zeros are real values in the frame, so without coverage they would claim
    every year and suppress the default entirely. Coverage comes from a DataAvailabilityNode,
    which inspects the dataset before zero-filling, and says the truth: nothing was supplied.
    """
    own_zero_filled = _ppdf([(2020, 0.0), (2021, 0.0), (2022, 0.0)])
    default = _ppdf([(2020, 1.0), (2021, 2.0), (2022, 3.0)])
    nothing_supplied = _flags([(2020, 0.0), (2021, 0.0), (2022, 0.0)])

    without = own_zero_filled.paths.prefer_by_year(default)
    assert _as_map(without) == {2020: 0.0, 2021: 0.0, 2022: 0.0}  # the balance destroyed

    with_coverage = own_zero_filled.paths.prefer_by_year(default, nothing_supplied)
    assert _as_map(with_coverage) == {2020: 1.0, 2021: 2.0, 2022: 3.0}  # the default preserved


def test_coverage_selects_only_the_supplied_years():
    """The city started collecting its own data in 2022; 2020-21 stay on the default."""
    own = _ppdf([(2020, 0.0), (2021, 0.0), (2022, 500.0)])
    default = _ppdf([(2020, 1.0), (2021, 2.0), (2022, 3.0)])
    supplied_from_2022 = _flags([(2020, 0.0), (2021, 0.0), (2022, 1.0)])

    out = own.paths.prefer_by_year(default, supplied_from_2022)

    assert _as_map(out) == {2020: 1.0, 2021: 2.0, 2022: 500.0}


def test_coverage_of_zero_still_yields_the_citys_reported_zero():
    """
    A city that reports zero consumption for a year it did supply gets its zero, not the default.

    BISKO allows an entered zero, so coverage -- not the value -- decides the source. This is why
    fabricated zeros cannot be tolerated in the value frame: the function cannot tell them apart.
    """
    own = _ppdf([(2022, 0.0)])
    default = _ppdf([(2021, 2.0), (2022, 3.0)])
    supplied_2022 = _flags([(2022, 1.0)])

    out = own.paths.prefer_by_year(default, supplied_2022)

    assert _as_map(out) == {2021: 2.0, 2022: 0.0}


def test_null_coverage_flags_do_not_count_as_coverage():
    own = _ppdf([(2021, 9.0), (2022, 9.0)])
    default = _ppdf([(2021, 2.0), (2022, 3.0)])
    coverage = _flags([(2021, None), (2022, 1.0)])

    out = own.paths.prefer_by_year(default, coverage)

    assert _as_map(out) == {2021: 2.0, 2022: 9.0}


def test_formula_node_three_argument_form_matches_the_documented_example():
    """Exercises the tag names and formula written in docs/imputing-data-into-nodes.md."""
    context = _make_context('prefer-by-year-coverage')
    target = FormulaNode(
        id='vehicle_kilometers',
        context=context,
        name=TranslatedString('target', default_language='en'),
        unit=unit_registry.parse_units('MWh/a'),
        quantity='energy',
    )
    target.parameters['formula'] = StringParameter(local_id='formula', value='prefer_by_year(own, default, coverage)')
    own = _fixed_node(context, 'own', [(2020, 0.0), (2021, 0.0), (2022, 500.0)])
    default = _fixed_node(context, 'default', [(2020, 100.0), (2021, 110.0), (2022, 120.0)])
    coverage = _FixedOutputNode(
        id='coverage',
        context=context,
        name=TranslatedString('coverage', default_language='en'),
        unit=unit_registry.parse_units('dimensionless'),
        quantity='fraction',
        fixed_df=_flags([(2020, 0.0), (2021, 0.0), (2022, 1.0)]),
    )
    for n in (own, default, coverage):
        _connect(n, target)

    assert _as_map(target.compute()) == {2020: 100.0, 2021: 110.0, 2022: 500.0}
