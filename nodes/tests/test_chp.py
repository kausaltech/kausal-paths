from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.datasets import Dataset
from nodes.dimensions import Dimension, DimensionCategory
from nodes.edges import Edge
from nodes.exceptions import NodeError
from nodes.generic import BISKO_T_RETURN, BiskoChpNode, BiskoExergeticAllocationNode, ChpNode
from nodes.node import Node
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry
from params.param import NumberParameter, StringParameter

if TYPE_CHECKING:
    from typing import Any

    from common.polars import PathsDataFrame
    from nodes.context import Context

pytestmark = pytest.mark.django_db

# The instance factory gives reference_year 1990, minimum_historical_year 2010,
# maximum_historical_year 2018 and target_year (= model_end_year) 2030.
START_YEAR = 1990
END_YEAR = 2030


@dataclass
class _FixedDataset(Dataset):
    """A dataset returning a caller-supplied PathsDataFrame. Test-only."""

    fixed_df: PathsDataFrame | None = None

    def load_internal(self) -> PathsDataFrame:
        assert self.fixed_df is not None
        return self.fixed_df

    def hash_data(self) -> dict[str, Any]:
        return {'id': self.id}


class _FixedOutputNode(Node):
    """A leaf node whose output is a fixed, caller-supplied PathsDataFrame. Test-only."""

    def __init__(self, *args, fixed_df: PathsDataFrame, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_df = fixed_df

    def compute(self) -> PathsDataFrame:
        return self._fixed_df


def _make_context(identifier: str) -> Context:
    instance = InstanceFactory.create(id=identifier, name=identifier)
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name=identifier)
    return instance.context


def _series_df(columns: dict[str, list[float]], years: list[int], units: dict[str, str]) -> PathsDataFrame:
    df = pl.DataFrame({YEAR_COLUMN: years, **columns}).with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
    meta = DataFrameMeta(
        units={col: unit_registry.parse_units(unit) for col, unit in units.items()},
        primary_keys=[YEAR_COLUMN],
    )
    return to_ppdf(df, meta)


def _make_chp_node(
    context: Context,
    cls: type[ChpNode] = ChpNode,
    identifier: str = 'chp',
    params: dict[str, float | str] | None = None,
) -> ChpNode:
    node = cls(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units('dimensionless'),
        quantity='fraction',
    )
    for key, value in (params or {}).items():
        if isinstance(value, str):
            node.parameters[key] = StringParameter(local_id=key, value=value)
        else:
            node.parameters[key] = NumberParameter(local_id=key, value=value)
    return node


def _attach_dataset(node: ChpNode, df: PathsDataFrame) -> None:
    node.input_dataset_instances.append(_FixedDataset(id='chp_parameters', context=node.context, fixed_df=df))


def _attach_series_node(node: ChpNode, identifier: str, df: PathsDataFrame, tag: str) -> None:
    source = _FixedOutputNode(
        id=identifier,
        context=node.context,
        name=TranslatedString(identifier, default_language='en'),
        unit=df.get_unit(VALUE_COLUMN),
        quantity='fraction',
        fixed_df=df,
    )
    edge = Edge(input_node=source, output_node=node, tags=[tag])
    source.add_edge(edge)
    node.add_edge(edge)


def _heat_by_year(df: PathsDataFrame) -> dict[int, float]:
    sub = df.filter(pl.col('energy_carrier') == 'district_heating')
    return {row[YEAR_COLUMN]: row[VALUE_COLUMN] for row in sub.to_dicts()}


def _expected_heat_fraction(f_el: float, t_supply: float, t_return: float = BISKO_T_RETURN) -> float:
    z_heat = 1.0 - t_return / t_supply
    a_el = f_el / (f_el + z_heat * (1.0 - f_el))
    return 1.0 - a_el


def test_constant_parameters_give_a_flat_split_over_the_whole_model_span():
    context = _make_context('chp-constant')
    node = _make_chp_node(context, params={'method': 'bisko', 'electricity_fraction': 0.3, 't_supply': 373.0})

    df = node.compute()
    heat = _heat_by_year(df)

    assert min(heat) == START_YEAR
    assert max(heat) == END_YEAR
    expected = _expected_heat_fraction(0.3, 373.0)
    assert all(value == pytest.approx(expected) for value in heat.values())

    # The two fractions sum to 1 in every year.
    totals = df.group_by(YEAR_COLUMN).agg(pl.col(VALUE_COLUMN).sum())
    assert all(row[VALUE_COLUMN] == pytest.approx(1.0) for row in totals.to_dicts())

    # Years past the last historical one are marked as forecast.
    flags = {row[YEAR_COLUMN]: row[FORECAST_COLUMN] for row in df.to_dicts()}
    assert flags[2018] is False
    assert flags[2019] is True


def test_annual_series_from_dataset_makes_the_split_vary_by_year():
    context = _make_context('chp-dataset')
    node = _make_chp_node(context, params={'method': 'bisko'})
    _attach_dataset(
        node,
        _series_df(
            {'electricity_fraction': [0.30, 0.35, 0.40], 't_supply': [373.0, 368.0, 363.0]},
            years=[2016, 2017, 2018],
            units={'electricity_fraction': 'dimensionless', 't_supply': 'K'},
        ),
    )

    heat = _heat_by_year(node.compute())

    assert heat[2016] == pytest.approx(_expected_heat_fraction(0.30, 373.0))
    assert heat[2017] == pytest.approx(_expected_heat_fraction(0.35, 368.0))
    assert heat[2018] == pytest.approx(_expected_heat_fraction(0.40, 363.0))
    # A varying operating point is the whole point: the fractions must actually differ.
    assert heat[2016] != pytest.approx(heat[2018])


def test_series_is_held_constant_outside_the_years_it_covers():
    context = _make_context('chp-extension')
    node = _make_chp_node(context, params={'method': 'bisko', 't_supply': 373.0})
    _attach_dataset(
        node,
        _series_df(
            {'electricity_fraction': [0.30, 0.40]},
            years=[2016, 2018],
            units={'electricity_fraction': 'dimensionless'},
        ),
    )

    df = node.compute()
    heat = _heat_by_year(df)

    assert min(heat) == START_YEAR
    assert max(heat) == END_YEAR
    assert heat[START_YEAR] == pytest.approx(_expected_heat_fraction(0.30, 373.0))  # held back
    assert heat[2017] == pytest.approx(_expected_heat_fraction(0.35, 373.0))  # interpolated over the gap
    assert heat[END_YEAR] == pytest.approx(_expected_heat_fraction(0.40, 373.0))  # held forward

    flags = {row[YEAR_COLUMN]: row[FORECAST_COLUMN] for row in df.to_dicts()}
    assert flags[2018] is False  # the last observed year is history
    assert flags[2019] is True  # everything held forward from it is forecast


def test_a_series_may_come_from_a_tagged_input_node():
    context = _make_context('chp-node-input')
    node = _make_chp_node(context, params={'method': 'bisko', 't_supply': 373.0})
    fraction_df = _series_df({VALUE_COLUMN: [0.30, 0.40]}, years=[2017, 2018], units={VALUE_COLUMN: 'dimensionless'})
    _attach_series_node(node, 'chp_electricity_share', fraction_df, tag='electricity_fraction')

    heat = _heat_by_year(node.compute())

    assert heat[2017] == pytest.approx(_expected_heat_fraction(0.30, 373.0))
    assert heat[2018] == pytest.approx(_expected_heat_fraction(0.40, 373.0))


def test_a_series_given_in_per_cent_is_converted():
    context = _make_context('chp-units')
    node = _make_chp_node(context, params={'method': 'bisko', 't_supply': 373.0})
    _attach_dataset(
        node,
        _series_df({'electricity_fraction': [30.0]}, years=[2018], units={'electricity_fraction': '%'}),
    )

    heat = _heat_by_year(node.compute())
    assert heat[2018] == pytest.approx(_expected_heat_fraction(0.30, 373.0))


def test_supplying_an_input_twice_is_an_error():
    context = _make_context('chp-conflict')
    node = _make_chp_node(context, params={'method': 'bisko', 't_supply': 373.0})
    _attach_dataset(
        node,
        _series_df({'electricity_fraction': [0.3]}, years=[2018], units={'electricity_fraction': 'dimensionless'}),
    )
    _attach_series_node(
        node,
        'chp_electricity_share',
        _series_df({VALUE_COLUMN: [0.4]}, years=[2018], units={VALUE_COLUMN: 'dimensionless'}),
        tag='electricity_fraction',
    )

    with pytest.raises(NodeError, match='keep only one of them'):
        node.compute()


def test_a_missing_input_names_all_three_ways_of_supplying_it():
    context = _make_context('chp-missing')
    node = _make_chp_node(context, params={'method': 'bisko', 'electricity_fraction': 0.3})

    with pytest.raises(NodeError, match="'t_supply'"):
        node.compute()


def test_a_fraction_outside_zero_to_one_is_an_error():
    context = _make_context('chp-range')
    node = _make_chp_node(context, params={'method': 'bisko', 't_supply': 373.0})
    _attach_dataset(
        node,
        _series_df(
            {'electricity_fraction': [0.3, 1.4]},
            years=[2017, 2018],
            units={'electricity_fraction': 'dimensionless'},
        ),
    )

    with pytest.raises(NodeError, match='between 0 and 1'):
        node.compute()


def test_a_supply_temperature_below_the_return_temperature_is_an_error():
    context = _make_context('chp-temperature')
    node = _make_chp_node(context, params={'method': 'bisko', 'electricity_fraction': 0.3, 't_supply': 273.0})

    with pytest.raises(NodeError, match='must exceed the return temperature'):
        node.compute()


def test_an_unknown_method_is_an_error():
    context = _make_context('chp-method')
    node = _make_chp_node(context, params={'method': 'guesswork', 'electricity_fraction': 0.3})

    with pytest.raises(NodeError, match='must be one of'):
        node.compute()


def test_energy_content_method_splits_by_energy_alone():
    context = _make_context('chp-energy-content')
    node = _make_chp_node(context, params={'method': 'energy_content', 'electricity_fraction': 0.3})

    heat = _heat_by_year(node.compute())
    assert heat[2018] == pytest.approx(0.7)  # z_i = 1, so the split is the energy split


def test_efficiency_method_uses_the_reference_efficiencies():
    context = _make_context('chp-efficiency')
    node = _make_chp_node(
        context,
        params={
            'method': 'efficiency',
            'electricity_fraction': 0.3,
            'electricity_reference_efficiency': 0.4,
            'heat_reference_efficiency': 0.9,
        },
    )

    heat = _heat_by_year(node.compute())
    a_el = (0.3 / 0.4) / (0.3 / 0.4 + 0.7 / 0.9)
    assert heat[2018] == pytest.approx(1.0 - a_el)


def test_work_potential_method_takes_the_return_temperature_from_the_config():
    context = _make_context('chp-work-potential')
    node = _make_chp_node(
        context,
        params={'method': 'work_potential', 'electricity_fraction': 0.3, 't_supply': 373.0, 't_return': 313.0},
    )

    heat = _heat_by_year(node.compute())
    assert heat[2018] == pytest.approx(_expected_heat_fraction(0.3, 373.0, t_return=313.0))


def test_bisko_node_computes_the_same_split_as_the_generic_node_set_to_bisko():
    generic = _make_chp_node(
        _make_context('chp-generic'),
        params={'method': 'bisko', 'electricity_fraction': 0.3, 't_supply': 373.0},
    )
    bisko = _make_chp_node(
        _make_context('chp-bisko'),
        cls=BiskoChpNode,
        params={'electricity_fraction': 0.3, 't_supply': 373.0},
    )

    assert _heat_by_year(bisko.compute()) == pytest.approx(_heat_by_year(generic.compute()))


def test_bisko_node_does_not_accept_the_parameters_the_standard_fixes():
    """The config loader rejects any parameter not in allowed_parameters, so this is a load-time gate."""
    allowed = {p.local_id for p in BiskoChpNode.allowed_parameters}

    assert 'method' not in allowed
    assert 't_return' not in allowed
    assert 'electricity_reference_efficiency' not in allowed
    assert 'heat_reference_efficiency' not in allowed
    assert {'electricity_fraction', 't_supply'} <= allowed


def test_bisko_node_refuses_a_return_temperature_series():
    context = _make_context('chp-bisko-t-return')
    node = _make_chp_node(context, cls=BiskoChpNode, params={'electricity_fraction': 0.3, 't_supply': 373.0})
    _attach_dataset(node, _series_df({'t_return': [300.0]}, years=[2018], units={'t_return': 'K'}))

    with pytest.raises(NodeError, match='silently ignored'):
        node.compute()


def test_a_dataset_with_no_usable_column_is_an_error():
    context = _make_context('chp-bad-dataset')
    node = _make_chp_node(context, params={'method': 'bisko', 't_supply': 373.0})
    _attach_dataset(node, _series_df({'something_else': [1.0]}, years=[2018], units={'something_else': 'dimensionless'}))

    with pytest.raises(NodeError, match='no usable metric column'):
        node.compute()


# --------------------------------------------------------------------------------------
# BiskoExergeticAllocationNode -- the criterion 6 conformity gate
# --------------------------------------------------------------------------------------


def _make_gate(context: Context, identifier: str = 'gate') -> BiskoExergeticAllocationNode:
    return BiskoExergeticAllocationNode(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units('dimensionless'),
        quantity='probability',
    )


def _connect(input_node: Node, output_node: Node, tag: str) -> None:
    edge = Edge(input_node=input_node, output_node=output_node, tags=[tag])
    input_node.add_edge(edge)
    output_node.add_edge(edge)


def _attach_emissions(gate: BiskoExergeticAllocationNode, rows: list[tuple[int, float]]) -> None:
    df = _series_df({VALUE_COLUMN: [r[1] for r in rows]}, years=[r[0] for r in rows], units={VALUE_COLUMN: 'kt/a'})
    source = _FixedOutputNode(
        id='dh_emissions',
        context=gate.context,
        name=TranslatedString('dh_emissions', default_language='en'),
        unit=unit_registry.parse_units('kt/a'),
        quantity='emissions',
        fixed_df=df,
    )
    _connect(source, gate, 'emissions')


def _attach_consumption(gate: BiskoExergeticAllocationNode, rows: list[tuple[int, float]]) -> None:
    df = _series_df({VALUE_COLUMN: [r[1] for r in rows]}, years=[r[0] for r in rows], units={VALUE_COLUMN: 'GWh/a'})
    source = _FixedOutputNode(
        id='dh_consumption',
        context=gate.context,
        name=TranslatedString('dh_consumption', default_language='en'),
        unit=unit_registry.parse_units('GWh/a'),
        quantity='energy',
        fixed_df=df,
    )
    _connect(source, gate, 'consumption')


def _gate_by_year(node: BiskoExergeticAllocationNode) -> dict[int, float]:
    return {row[YEAR_COLUMN]: row[VALUE_COLUMN] for row in node.compute().to_dicts()}


def test_gate_passes_when_the_method_conforms_and_something_was_allocated():
    context = _make_context('gate-pass')
    gate = _make_gate(context)
    allocation = _make_chp_node(context, cls=BiskoChpNode, params={'electricity_fraction': 0.3, 't_supply': 373.0})
    _connect(allocation, gate, 'allocation')
    _attach_emissions(gate, [(2016, 120.0), (2017, 118.0), (2018, 115.0)])

    result = _gate_by_year(gate)

    assert result[2016] == 1.0
    assert result[2018] == 1.0
    # Full model span, and never null -- a null here fails baseline validation downstream.
    assert min(result) == START_YEAR
    assert max(result) == END_YEAR
    assert all(v is not None for v in result.values())


def test_gate_fails_in_years_with_nothing_to_allocate():
    """A conforming method applied to an empty balance evidences nothing."""
    context = _make_context('gate-empty-years')
    gate = _make_gate(context)
    allocation = _make_chp_node(context, cls=BiskoChpNode, params={'electricity_fraction': 0.3, 't_supply': 373.0})
    _connect(allocation, gate, 'allocation')
    _attach_emissions(gate, [(2016, 120.0), (2017, 0.0), (2018, 115.0)])

    result = _gate_by_year(gate)

    assert result[2016] == 1.0
    assert result[2017] == 0.0  # zero emissions -> not evidenced
    assert result[2018] == 1.0
    assert result[1990] == 0.0  # before the series -> no emissions at all


def test_gate_fails_when_the_method_does_not_conform():
    context = _make_context('gate-wrong-method')
    gate = _make_gate(context)
    allocation = _make_chp_node(context, params={'method': 'energy_content', 'electricity_fraction': 0.3})
    _connect(allocation, gate, 'allocation')
    _attach_emissions(gate, [(2016, 120.0), (2018, 115.0)])

    result = _gate_by_year(gate)

    assert result[2016] == 0.0
    assert result[2018] == 0.0


def test_gate_accepts_work_potential_at_the_bisko_return_temperature():
    """work_potential with t_return = 283 K is numerically the bisko method, so it conforms."""
    context = _make_context('gate-work-potential-283')
    gate = _make_gate(context)
    allocation = _make_chp_node(
        context,
        params={
            'method': 'work_potential',
            'electricity_fraction': 0.3,
            't_supply': 373.0,
            't_return': BISKO_T_RETURN,
        },
    )
    _connect(allocation, gate, 'allocation')
    _attach_emissions(gate, [(2018, 115.0)])

    assert _gate_by_year(gate)[2018] == 1.0


def test_gate_rejects_work_potential_at_another_return_temperature():
    context = _make_context('gate-work-potential-313')
    gate = _make_gate(context)
    allocation = _make_chp_node(
        context,
        params={'method': 'work_potential', 'electricity_fraction': 0.3, 't_supply': 373.0, 't_return': 313.0},
    )
    _connect(allocation, gate, 'allocation')
    _attach_emissions(gate, [(2018, 115.0)])

    assert _gate_by_year(gate)[2018] == 0.0


def test_gate_rejects_a_return_temperature_that_moves_between_years():
    """BISKO fixes the return temperature, so a series cannot conform even if it passes through 283."""
    context = _make_context('gate-work-potential-series')
    allocation = _make_chp_node(context, params={'method': 'work_potential', 'electricity_fraction': 0.3, 't_supply': 373.0})
    _attach_dataset(allocation, _series_df({'t_return': [BISKO_T_RETURN]}, years=[2018], units={'t_return': 'K'}))

    assert allocation.allocation_conforms_to_bisko is False


def test_bisko_chp_node_declares_a_fixed_conforming_allocation():
    context = _make_context('declaration')
    node = _make_chp_node(context, cls=BiskoChpNode, params={'electricity_fraction': 0.3, 't_supply': 373.0})

    assert node.allocation_method == 'bisko'
    assert node.allocation_method_is_fixed is True
    assert node.allocation_conforms_to_bisko is True


def test_generic_chp_node_declares_a_settable_method():
    context = _make_context('declaration-generic')
    node = _make_chp_node(context, params={'method': 'bisko', 'electricity_fraction': 0.3, 't_supply': 373.0})

    assert node.allocation_method == 'bisko'
    assert node.allocation_method_is_fixed is False  # conforming, but by a setting
    assert node.allocation_conforms_to_bisko is True


def test_efficiency_method_does_not_conform_to_bisko():
    context = _make_context('declaration-efficiency')
    node = _make_chp_node(
        context,
        params={
            'method': 'efficiency',
            'electricity_fraction': 0.3,
            'electricity_reference_efficiency': 0.4,
            'heat_reference_efficiency': 0.9,
        },
    )

    assert node.allocation_conforms_to_bisko is False


def test_gate_requires_exactly_one_allocation_input():
    context = _make_context('gate-no-allocation')
    gate = _make_gate(context)
    _attach_emissions(gate, [(2018, 115.0)])

    with pytest.raises(NodeError, match="exactly one input node tagged 'allocation'"):
        gate.compute()


def test_gate_rejects_an_allocation_node_that_declares_nothing():
    context = _make_context('gate-not-a-chp-node')
    gate = _make_gate(context)
    other = _FixedOutputNode(
        id='not_a_chp_node',
        context=context,
        name=TranslatedString('not_a_chp_node', default_language='en'),
        unit=unit_registry.parse_units('dimensionless'),
        quantity='fraction',
        fixed_df=_series_df({VALUE_COLUMN: [1.0]}, years=[2018], units={VALUE_COLUMN: 'dimensionless'}),
    )
    _connect(other, gate, 'allocation')
    _attach_emissions(gate, [(2018, 115.0)])

    with pytest.raises(NodeError, match='does not declare whether its allocation conforms'):
        gate.compute()


def test_gate_requires_exactly_one_emissions_input():
    context = _make_context('gate-no-emissions')
    gate = _make_gate(context)
    allocation = _make_chp_node(context, cls=BiskoChpNode, params={'electricity_fraction': 0.3, 't_supply': 373.0})
    _connect(allocation, gate, 'allocation')

    with pytest.raises(NodeError, match="exactly one input node tagged 'emissions'"):
        gate.compute()


def test_gate_rejects_emissions_that_still_carry_a_dimension():
    """Forgetting the flatten on the emissions edge is the realistic way to get this wrong."""
    context = _make_context('gate-dimensional-emissions')
    gate = _make_gate(context)
    allocation = _make_chp_node(context, cls=BiskoChpNode, params={'electricity_fraction': 0.3, 't_supply': 373.0})
    _connect(allocation, gate, 'allocation')

    context.dimensions['energy_carrier'] = Dimension(
        id='energy_carrier',
        label=TranslatedString('Energy carrier', default_language='en'),
        categories=[
            DimensionCategory(id='district_heating', label=TranslatedString('DH', default_language='en')),
            DimensionCategory(id='electricity', label=TranslatedString('El', default_language='en')),
        ],
    )
    df = pl.DataFrame({
        YEAR_COLUMN: [2018, 2018],
        'energy_carrier': ['district_heating', 'electricity'],
        VALUE_COLUMN: [115.0, 40.0],
    }).with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
    meta = DataFrameMeta(
        units={VALUE_COLUMN: unit_registry.parse_units('kt/a')},
        primary_keys=[YEAR_COLUMN, 'energy_carrier'],
    )
    source = _FixedOutputNode(
        id='dh_emissions_by_carrier',
        context=context,
        name=TranslatedString('dh_emissions_by_carrier', default_language='en'),
        unit=unit_registry.parse_units('kt/a'),
        quantity='emissions',
        output_dimension_ids=['energy_carrier'],
        fixed_df=to_ppdf(df, meta),
    )
    _connect(source, gate, 'emissions')

    with pytest.raises(NodeError, match='must be a single series'):
        gate.compute()


def test_a_year_without_district_heating_is_not_applicable_rather_than_failing():
    """
    BISKO nowhere requires a city to have a heat network.

    Without the consumption input the gate cannot tell "no network" from "network, nothing
    allocated" and calls both a failure. That made a city with no district heating permanently
    non-conform, and it is why an all-zero balance -- which BISKO allows explicitly, "der Eintrag
    kann auch Null sein" -- failed criterion 6. The second Prüflauf recorded it as a defect.
    """
    context = _make_context('gate-no-network')
    gate = _make_gate(context)
    allocation = _make_chp_node(context, cls=BiskoChpNode, params={'electricity_fraction': 0.3, 't_supply': 373.0})
    _connect(allocation, gate, 'allocation')
    _attach_emissions(gate, [(2016, 0.0), (2017, 0.0), (2018, 0.0)])
    _attach_consumption(gate, [(2016, 0.0), (2017, 0.0), (2018, 0.0)])

    by_year = _gate_by_year(gate)
    assert by_year[2016] == 1.0
    assert by_year[2017] == 1.0
    assert by_year[2018] == 1.0


def test_district_heating_with_no_allocated_emissions_still_fails():
    """The other half of the same distinction: a network that allocated nothing is a failure."""
    context = _make_context('gate-network-no-emissions')
    gate = _make_gate(context)
    allocation = _make_chp_node(context, cls=BiskoChpNode, params={'electricity_fraction': 0.3, 't_supply': 373.0})
    _connect(allocation, gate, 'allocation')
    _attach_emissions(gate, [(2017, 0.0), (2018, 0.0)])
    _attach_consumption(gate, [(2017, 240.0), (2018, 260.0)])

    by_year = _gate_by_year(gate)
    assert by_year[2017] == 0.0
    assert by_year[2018] == 0.0


def test_the_consumption_input_must_be_a_single_series():
    context = _make_context('gate-consumption-dims')
    gate = _make_gate(context)
    allocation = _make_chp_node(context, cls=BiskoChpNode, params={'electricity_fraction': 0.3, 't_supply': 373.0})
    _connect(allocation, gate, 'allocation')
    _attach_emissions(gate, [(2018, 115.0)])

    context.dimensions['energy_carrier'] = Dimension(
        id='energy_carrier',
        label=TranslatedString('Energy carrier', default_language='en'),
        categories=[
            DimensionCategory(id='district_heating', label=TranslatedString('DH', default_language='en')),
            DimensionCategory(id='electricity', label=TranslatedString('El', default_language='en')),
        ],
    )
    df = pl.DataFrame({
        YEAR_COLUMN: [2018, 2018],
        'energy_carrier': ['district_heating', 'electricity'],
        VALUE_COLUMN: [240.0, 90.0],
    }).with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
    meta = DataFrameMeta(
        units={VALUE_COLUMN: unit_registry.parse_units('GWh/a')},
        primary_keys=[YEAR_COLUMN, 'energy_carrier'],
    )
    source = _FixedOutputNode(
        id='dh_consumption_by_carrier',
        context=context,
        name=TranslatedString('dh_consumption_by_carrier', default_language='en'),
        unit=unit_registry.parse_units('GWh/a'),
        quantity='energy',
        output_dimension_ids=['energy_carrier'],
        fixed_df=to_ppdf(df, meta),
    )
    _connect(source, gate, 'consumption')

    with pytest.raises(NodeError, match='must be a single series'):
        gate.compute()
