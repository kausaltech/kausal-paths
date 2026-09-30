from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common import polars as ppl, qualifiers
from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.edges import Edge
from nodes.exceptions import NodeError
from nodes.formula import FormulaNode
from nodes.node import Node
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, ScenarioFactory
from nodes.units import unit_registry
from params.param import NumberParameter, StringParameter

if TYPE_CHECKING:
    import pandas as pd

    from common.polars import PathsDataFrame
    from nodes.context import Context

pytestmark = pytest.mark.django_db


class _ParamMultiplierNode(Node):
    """Leaf node whose output is a fixed base value times a local parameter. Test-only."""

    def compute(self) -> PathsDataFrame:
        multiplier = float(self.get_parameter('multiplier').value)
        df = pl.DataFrame({
            YEAR_COLUMN: [2020],
            VALUE_COLUMN: [1.0],
            FORECAST_COLUMN: [False],
        })
        meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kWh')}, primary_keys=[YEAR_COLUMN])
        pdf = to_ppdf(df, meta)
        return pdf.with_columns((pl.col(VALUE_COLUMN) * multiplier).alias(VALUE_COLUMN))


class _QualifiedNode(_ParamMultiplierNode):
    def compute(self) -> PathsDataFrame:
        return (
            super().compute().with_columns(qualifiers.make(quality=pl.lit(0.5), reported=pl.lit(value=True)).alias('Value__qual'))
        )


def _make_context(identifier: str) -> Context:
    instance = InstanceFactory.create(id=identifier, name=identifier)
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name=identifier)
    return instance.context


def _make_node(context: Context, cls: type[Node], identifier: str, unit: str = 'kWh', quantity: str = 'energy') -> Node:
    return cls(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity=quantity,
    )


def _connect(input_node: Node, output_node: Node) -> None:
    edge = Edge(input_node=input_node, output_node=output_node)
    input_node.add_edge(edge)
    output_node.add_edge(edge)


def _first_value(df: pd.DataFrame | PathsDataFrame) -> float:
    assert isinstance(df, ppl.PathsDataFrame)
    return df.to_dicts()[0][VALUE_COLUMN]


def test_output_with_scenario_uses_named_scenario_not_active_one():
    context = _make_context('output-with-scenario')
    # `_make_context` already creates and activates a 'default' scenario.
    assert context.active_scenario.id == 'default'

    source = _make_node(context, _ParamMultiplierNode, 'source')
    multiplier = NumberParameter(local_id='multiplier', value=1.0)
    source.add_parameter(multiplier)
    context.add_node(source)

    other_scenario = ScenarioFactory.create(id='other')
    other_scenario.add_parameter(multiplier, 5.0)
    context.add_scenario(other_scenario)

    target = _make_node(context, FormulaNode, 'target')
    target.parameters['formula'] = StringParameter(local_id='formula', value="output_with_scenario(source, 'other')")
    _connect(source, target)

    result = target.compute()

    # The formula picked up the 'other' scenario's parameter value...
    assert _first_value(result) == 5.0
    # ...and the active scenario's own value was restored afterwards.
    assert multiplier.value == 1.0
    assert context.active_scenario.id == 'default'


def test_output_with_scenario_rejects_unknown_node_reference():
    context = _make_context('output-with-scenario-bad-node')

    target = _make_node(context, FormulaNode, 'target')
    target.parameters['formula'] = StringParameter(local_id='formula', value="output_with_scenario(missing, 'default')")

    with pytest.raises(NodeError, match='must be a reference to an input node'):
        target.compute()


@pytest.mark.parametrize(('function', 'expected'), [('quality', 0.5), ('graded', 1.0), ('reported', 1.0)])
def test_formula_reads_nested_assessment_and_reporting(function: str, expected: float) -> None:
    context = _make_context(f'qualifier-{function}')
    source = _make_node(context, _QualifiedNode, 'source')
    source.add_parameter(NumberParameter(local_id='multiplier', value=1.0))
    context.add_node(source)
    target = _make_node(context, FormulaNode, 'target', unit='dimensionless', quantity='fraction')
    target.parameters['formula'] = StringParameter(local_id='formula', value=f'{function}(source)')
    _connect(source, target)
    assert _first_value(target.compute()) == expected


def test_remove_subset_formula_and_analysis() -> None:
    from nodes.formula import analyze_formula_dimensions, analyze_formula_units

    context = _make_context('remove-subset')
    target = _make_node(context, FormulaNode, 'target')
    for name, value in [('total', 100.0), ('subset', 20.0)]:
        source = _make_node(context, _QualifiedNode, name)
        source.add_parameter(NumberParameter(local_id='multiplier', value=value))
        context.add_node(source)
        _connect(source, target)
    target.parameters['formula'] = StringParameter(local_id='formula', value='remove_subset(total, subset)')
    result = target.compute()
    assert _first_value(result) == 80.0
    assert result['Value__qual'][0]['quality'] == {'score': 0.5, 'coverage': 1.0}
    dims = analyze_formula_dimensions('remove_subset(total, subset)', {'total': {'sector'}, 'subset': {'sector'}})
    assert dims.dims == {'sector'}
    assert not dims.errors
    assert not dims.warnings
    units = analyze_formula_units(
        'remove_subset(total, subset)', {'total': unit_registry.parse_units('kWh'), 'subset': unit_registry.parse_units('kWh')}
    )
    assert units.unit == unit_registry.parse_units('kWh')
    assert not units.errors
    assert not units.warnings


def test_dataset_cleaning_preserves_endpoint_grades_and_interpolates_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _make_context('qualified-cleaning')
    node = _make_node(context, _ParamMultiplierNode, 'source')
    frame = to_ppdf(
        pl.DataFrame({'Year': [2020, 2022], 'energy_carrier': ['gas', 'gas'], 'Value': [10.0, 20.0]}),
        meta=DataFrameMeta(units={'Value': unit_registry.parse_units('kWh')}, primary_keys=['Year', 'energy_carrier']),
    ).with_columns(qualifiers.make(quality=pl.lit(1.0), reported=pl.lit(value=True)).alias('Value__qual'))
    monkeypatch.setattr(node, 'get_input_dataset_pl', lambda **_kwargs: frame)
    result = node.get_cleaned_dataset()
    assert result is not None
    original = result.filter(pl.col('Year') == 2020)['Value__qual'][0]
    interpolated = result.filter(pl.col('Year') == 2021)['Value__qual'][0]
    assert original == {'reported': True, 'quality': {'score': 1.0, 'coverage': 1.0}}
    assert interpolated == {'reported': False, 'quality': {'score': 1.0, 'coverage': 1.0}}
