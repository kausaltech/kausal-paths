"""Tests for the ``keep_categories`` binding tag: a node fails if its result drops values of such an input."""

from typing import TYPE_CHECKING
from uuid import uuid4

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.defs.binding_def import DatasetBindingDef, NodePortRef
from nodes.defs.node_defs import FormulaConfig, NodeSpec
from nodes.dimensions import Dimension, DimensionCategory
from nodes.edges import Edge
from nodes.exceptions import NodeError
from nodes.formula import FormulaNode
from nodes.kept_categories import KEEP_CATEGORIES_TAG, describe_dropped, dropped_rows
from nodes.runtime_input import RuntimeInputBinding
from nodes.tests.test_chp import _FixedOutputNode, _make_context
from nodes.units import unit_registry

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.context import Context
    from nodes.node import Node

pytestmark = pytest.mark.django_db

CARRIERS = ['coal', 'gas', 'waste']


def _frame(rows: list[tuple[int, str, float | None]], unit: str) -> PathsDataFrame:
    df = pl.DataFrame(
        {
            YEAR_COLUMN: [year for year, _carrier, _value in rows],
            'energy_carrier': [carrier for _year, carrier, _value in rows],
            VALUE_COLUMN: [value for _year, _carrier, value in rows],
        },
        schema={YEAR_COLUMN: pl.Int64, 'energy_carrier': pl.Utf8, VALUE_COLUMN: pl.Float64},
    ).with_columns(pl.col('energy_carrier').cast(pl.Categorical), pl.lit(value=False).alias(FORECAST_COLUMN))
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=[YEAR_COLUMN, 'energy_carrier'])
    return to_ppdf(df, meta)


def _by_year(rows: list[tuple[int, float]], unit: str) -> PathsDataFrame:
    df = pl.DataFrame({YEAR_COLUMN: [r[0] for r in rows], VALUE_COLUMN: [r[1] for r in rows]}).with_columns(
        pl.lit(value=False).alias(FORECAST_COLUMN)
    )
    return to_ppdf(df, DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=[YEAR_COLUMN]))


ACTIVITY = _frame([(2018, 'coal', 5.0), (2018, 'gas', 10.0), (2018, 'waste', 0.0), (2019, 'coal', 6.0)], 'MWh/a')


def test_a_row_without_a_counterpart_is_reported():
    output = _frame([(2018, 'gas', 2.0), (2019, 'coal', 1.0)], 'kt/a')

    rows = dropped_rows(ACTIVITY, output)

    assert rows.to_dicts() == [{YEAR_COLUMN: 2018, 'energy_carrier': 'coal'}]
    assert describe_dropped(rows) == 'energy_carrier=coal (2018)'


def test_a_zero_row_carries_nothing_and_is_not_reported():
    output = _frame([(2018, 'coal', 1.0), (2018, 'gas', 2.0), (2019, 'coal', 1.0)], 'kt/a')

    assert dropped_rows(ACTIVITY, output).is_empty()


def test_a_year_the_result_has_no_value_in_is_not_reported():
    """A missing year is the fallback of a route choice, not a lost row."""
    output = _frame([(2018, 'coal', 1.0), (2018, 'gas', 2.0)], 'kt/a')

    assert dropped_rows(ACTIVITY, output).is_empty()


def test_a_dimension_the_node_sums_away_is_not_compared():
    assert dropped_rows(ACTIVITY, _by_year([(2018, 3.0), (2019, 1.0)], 'kt/a')).is_empty()


def test_a_null_result_counts_as_dropped():
    output = _frame([(2018, 'coal', None), (2018, 'gas', 2.0), (2019, 'coal', 1.0)], 'kt/a')

    assert dropped_rows(ACTIVITY, output).to_dicts() == [{YEAR_COLUMN: 2018, 'energy_carrier': 'coal'}]


def _declare_carriers(context: Context) -> None:
    context.dimensions['energy_carrier'] = Dimension(
        id='energy_carrier',
        label=TranslatedString('Energy carrier', default_language='en'),
        categories=[DimensionCategory(id=c, label=TranslatedString(c, default_language='en')) for c in CARRIERS],
    )


def _source(context: Context, identifier: str, df: PathsDataFrame, unit: str) -> _FixedOutputNode:
    return _FixedOutputNode(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity='energy' if identifier == 'activity' else 'emission_factor',
        output_dimension_ids=['energy_carrier'],
        fixed_df=df,
    )


def _connect(source: Node, target: Node, tags: list[str]) -> None:
    edge = Edge(input_node=source, output_node=target, tags=tags)
    source.add_edge(edge)
    target.add_edge(edge)


def _product(context: Context, activity_tags: list[str]) -> FormulaNode:
    """Return activity times factor, with a factor table that has no coal."""
    _declare_carriers(context)
    node = FormulaNode(
        id='emissions',
        context=context,
        name=TranslatedString('emissions', default_language='en'),
        unit=unit_registry.parse_units('t/a'),
        quantity='emissions',
        input_dimension_ids=['energy_carrier'],
        output_dimension_ids=['energy_carrier'],
    )
    node._spec = NodeSpec(type_config=FormulaConfig(formula='activity * factor'))
    _connect(_source(context, 'activity', ACTIVITY, 'MWh/a'), node, activity_tags)
    factors = _frame([(2018, 'gas', 200.0), (2018, 'waste', 300.0), (2019, 'gas', 200.0)], 'g/kWh')
    _connect(_source(context, 'factor', factors, 'g/kWh'), node, ['factor'])
    return node


def test_a_product_that_drops_a_tagged_activity_fails_naming_it():
    context = _make_context('kept-categories-tagged')
    node = _product(context, ['activity', KEEP_CATEGORIES_TAG])

    with pytest.raises(NodeError, match=r"drops values of input 'activity'.*energy_carrier=coal \(2018\)"):
        node.get_output_pl()


def test_without_the_tag_the_product_drops_the_activity_as_before():
    context = _make_context('kept-categories-untagged')
    node = _product(context, ['activity'])

    out = node.get_output_pl()

    assert set(out['energy_carrier'].cast(pl.Utf8).to_list()) == {'gas', 'waste'}


def test_the_tag_on_a_graph_binding_is_checked():
    """Instances loaded from a graph read their inputs through bindings, which carry the tags."""
    context = _make_context('kept-categories-binding')
    _declare_carriers(context)
    node = _source(context, 'emissions', _frame([(2018, 'gas', 2.0), (2019, 'coal', 1.0)], 't/a'), 't/a')
    definition = DatasetBindingDef(
        id=uuid4(),
        port_ref=NodePortRef(node_id='emissions', port_id=uuid4()),
        tags=['activity', KEEP_CATEGORIES_TAG],
    )
    node.bind_runtime_inputs((
        RuntimeInputBinding(
            id=definition.id,
            port_role='activity',
            position=0,
            source_kind='dataset',
            source=None,
            value_loader=lambda: ACTIVITY,
            source_id='kommune/activity',
            definition=definition,
        ),
    ))

    with pytest.raises(NodeError, match=r"input 'kommune/activity'.*energy_carrier=coal \(2018\)"):
        node.get_output_pl()
