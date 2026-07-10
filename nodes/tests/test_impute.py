from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common import polars as ppl
from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.dimensions import Dimension, DimensionCategory
from nodes.edges import Edge
from nodes.exceptions import NodeError
from nodes.formula import FormulaNode
from nodes.generic import GenericNode
from nodes.node import Node
from nodes.simple import AdditiveNode, MultiplicativeNode
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry
from params.param import StringParameter

if TYPE_CHECKING:
    import pandas as pd

    from common.polars import PathsDataFrame
    from nodes.context import Context

pytestmark = pytest.mark.django_db


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


def _ppdf(rows: list[tuple[int, float, bool]], unit: str = 'kWh', primary_keys: list[str] | None = None) -> PathsDataFrame:
    primary_keys = primary_keys or [YEAR_COLUMN]
    df = pl.DataFrame({
        YEAR_COLUMN: [r[0] for r in rows],
        VALUE_COLUMN: [r[1] for r in rows],
        FORECAST_COLUMN: [r[2] for r in rows],
    })
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=primary_keys)
    return to_ppdf(df, meta)


def _make_node(context: Context, cls: type[Node], identifier: str, unit: str = 'kWh', quantity: str = 'energy', **kwargs) -> Node:
    return cls(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity=quantity,
        **kwargs,
    )


def _make_fixed_node(
    context: Context, identifier: str, rows: list[tuple[int, float, bool]], unit: str = 'kWh', quantity: str = 'energy'
) -> _FixedOutputNode:
    node = _make_node(context, _FixedOutputNode, identifier, unit=unit, quantity=quantity, fixed_df=_ppdf(rows, unit=unit))
    assert isinstance(node, _FixedOutputNode)
    return node


def _connect(input_node: Node, output_node: Node, tags: list[str] | None = None) -> None:
    edge = Edge(input_node=input_node, output_node=output_node, tags=tags or [])
    input_node.add_edge(edge)
    output_node.add_edge(edge)


def _values_by_year(df: pd.DataFrame | PathsDataFrame) -> dict[int, float]:
    assert isinstance(df, ppl.PathsDataFrame)
    return {row[YEAR_COLUMN]: row[VALUE_COLUMN] for row in df.to_dicts()}


def test_additive_node_impute_overlays_and_falls_back():
    context = _make_context('additive-impute')
    target = _make_node(context, AdditiveNode, 'target')
    base_source = _make_fixed_node(context, 'base_source', [(2020, 1.0, False), (2021, 1.0, False)])
    impute_source = _make_fixed_node(context, 'impute_source', [(2020, 99.0, False)])

    _connect(base_source, target)
    _connect(impute_source, target, tags=['impute'])

    result = _values_by_year(target.compute())
    assert result[2020] == 99.0  # impute wins where it has a value
    assert result[2021] == 1.0  # target's own computed value survives elsewhere


def test_multiplicative_node_impute_overlays_and_falls_back():
    context = _make_context('multiplicative-impute')
    target = _make_node(context, MultiplicativeNode, 'target', unit='kWh')
    power = _make_fixed_node(context, 'power', [(2020, 2.0, False), (2021, 2.0, False)], unit='kW')
    time = _make_fixed_node(context, 'time', [(2020, 3.0, False), (2021, 3.0, False)], unit='h')
    impute_source = _make_fixed_node(context, 'impute_source', [(2020, 99.0, False)], unit='kWh')

    _connect(power, target)
    _connect(time, target)
    _connect(impute_source, target, tags=['impute'])

    result = _values_by_year(target.compute())
    assert result[2020] == 99.0  # impute wins
    assert result[2021] == 6.0  # target's own computed (2 * 3) survives elsewhere


def test_formula_node_impute_overlays_and_falls_back():
    context = _make_context('formula-impute')
    target = _make_node(context, FormulaNode, 'target')
    target.parameters['formula'] = StringParameter(local_id='formula', value='node_a')
    node_a = _make_fixed_node(context, 'node_a', [(2020, 1.0, False), (2021, 1.0, False)])
    impute_source = _make_fixed_node(context, 'impute_source', [(2020, 99.0, False)])

    _connect(node_a, target)
    _connect(impute_source, target, tags=['impute'])

    result = _values_by_year(target.compute())
    assert result[2020] == 99.0  # impute wins
    assert result[2021] == 1.0  # target's own computed value survives elsewhere


def test_generic_node_impute_overlays_and_falls_back():
    context = _make_context('generic-impute')
    target = _make_node(context, GenericNode, 'target')
    target.parameters['operations'] = StringParameter(local_id='operations', value='add,impute')
    base_source = _make_fixed_node(context, 'base_source', [(2020, 1.0, False), (2021, 1.0, False)])
    impute_source = _make_fixed_node(context, 'impute_source', [(2020, 99.0, False)])

    _connect(base_source, target)
    _connect(impute_source, target, tags=['impute'])

    result = _values_by_year(target.compute())
    assert result[2020] == 99.0  # impute wins
    assert result[2021] == 1.0  # target's own computed value survives elsewhere


def test_impute_dimension_mismatch_raises():
    context = _make_context('impute-dim-mismatch')
    vehicle_type = Dimension(id='vehicle_type', label='Vehicle type', categories=[DimensionCategory(id='a', label='A')])
    context.dimensions['vehicle_type'] = vehicle_type

    target = _make_node(context, AdditiveNode, 'target')
    base_source = _make_fixed_node(context, 'base_source', [(2020, 1.0, False)])

    # impute_source legitimately has a dimension the target doesn't, so both nodes'
    # own outputs are internally valid, but they can't be imputed onto one another.
    impute_source_df = _ppdf([(2020, 99.0, False)])
    impute_source_df = impute_source_df.with_columns(pl.lit('a').alias('vehicle_type').cast(pl.Categorical))
    impute_source_df = to_ppdf(
        impute_source_df,
        DataFrameMeta(units=impute_source_df.get_meta().units, primary_keys=[YEAR_COLUMN, 'vehicle_type']),
    )
    impute_source = _make_node(
        context, _FixedOutputNode, 'impute_source', fixed_df=impute_source_df, output_dimension_ids=['vehicle_type']
    )

    _connect(base_source, target)
    _connect(impute_source, target, tags=['impute'])

    with pytest.raises(NodeError, match='Dimensions must match for imputing'):
        target.compute()
