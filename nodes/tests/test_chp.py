from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.datasets import Dataset
from nodes.edges import Edge
from nodes.exceptions import NodeError
from nodes.generic import BISKO_T_RETURN, BiskoChpNode, ChpNode
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
