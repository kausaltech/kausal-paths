from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.dimensions import Dimension, DimensionCategory
from nodes.edges import Edge
from nodes.exceptions import NodeError
from nodes.generic import GenericNode
from nodes.node import Node
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.tests.node_input_harness import add_binding, binding
from nodes.units import unit_registry
from params.param import NumberParameter, StringParameter

if TYPE_CHECKING:
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


def _ppdf(
    rows: list[tuple[int, float, bool]],
    unit: str = 'kWh',
    category: str | None = None,
) -> PathsDataFrame:
    data: dict[str, list] = {
        YEAR_COLUMN: [r[0] for r in rows],
        VALUE_COLUMN: [r[1] for r in rows],
        FORECAST_COLUMN: [r[2] for r in rows],
    }
    primary_keys = [YEAR_COLUMN]
    df = pl.DataFrame(data)
    if category is not None:
        df = df.with_columns(pl.lit(category).cast(pl.Categorical).alias('sector'))
        primary_keys = [YEAR_COLUMN, 'sector']
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=primary_keys)
    return to_ppdf(df, meta)


def _make_target(context: Context, operations: str, trend_years: float | None = None, **kwargs) -> GenericNode:
    node = GenericNode(
        id='target',
        context=context,
        name=TranslatedString('target', default_language='en'),
        unit=unit_registry.parse_units('kWh'),
        quantity='energy',
        **kwargs,
    )
    node.parameters['operations'] = StringParameter(local_id='operations', value=operations)
    if trend_years is not None:
        node.parameters['trend_years'] = NumberParameter(local_id='trend_years', value=trend_years)
    return node


def _connect(input_node: Node, output_node: Node) -> None:
    edge = Edge(input_node=input_node, output_node=output_node, tags=[])
    input_node.add_edge(edge)
    output_node.add_edge(edge)
    add_binding(
        output_node,
        binding(
            'additive',
            position=len(output_node.runtime_input_bindings),
            source_kind='node',
            source_id=input_node.id,
            source=input_node,
            value_loader=lambda: input_node.get_output_pl(target_node=output_node),
        ),
    )


def _rows(df: PathsDataFrame) -> dict[int, tuple[float, bool]]:
    return {row[YEAR_COLUMN]: (row[VALUE_COLUMN], row[FORECAST_COLUMN]) for row in df.to_dicts()}


def test_trendline_fits_only_the_requested_window():
    context = _make_context('trendline-window')
    # A flat stretch, then a rise of 2/a over the last three years. With trend_years=3
    # only the rise is fitted, so the flat years must not drag the slope down.
    source = _FixedOutputNode(
        id='source',
        context=context,
        name=TranslatedString('source', default_language='en'),
        unit=unit_registry.parse_units('kWh'),
        quantity='energy',
        fixed_df=_ppdf([
            (2012, 10.0, False),
            (2013, 10.0, False),
            (2014, 10.0, False),
            (2015, 10.0, False),
            (2016, 10.0, False),
            (2017, 12.0, False),
            (2018, 14.0, False),
        ]),
    )
    target = _make_target(context, 'add,trendline', trend_years=3)
    _connect(source, target)

    rows = _rows(target.compute())
    assert rows[2016] == (10.0, False)  # history untouched
    assert rows[2018] == (14.0, False)
    assert rows[2019][0] == pytest.approx(16.0)
    assert rows[2019][1] is True
    assert rows[2030][0] == pytest.approx(38.0)  # 14 + 2 * 12
    assert max(rows) == context.model_end_year


def test_trendline_uses_all_history_without_the_parameter():
    context = _make_context('trendline-all-history')
    source = _FixedOutputNode(
        id='source',
        context=context,
        name=TranslatedString('source', default_language='en'),
        unit=unit_registry.parse_units('kWh'),
        quantity='energy',
        fixed_df=_ppdf([(2014, 10.0, False), (2015, 11.0, False), (2016, 12.0, False), (2017, 13.0, False)]),
    )
    target = _make_target(context, 'add,trendline')
    _connect(source, target)

    rows = _rows(target.compute())
    assert rows[2018][0] == pytest.approx(14.0)
    assert rows[2025][0] == pytest.approx(21.0)


def test_trendline_replaces_existing_forecast():
    context = _make_context('trendline-replaces-forecast')
    source = _FixedOutputNode(
        id='source',
        context=context,
        name=TranslatedString('source', default_language='en'),
        unit=unit_registry.parse_units('kWh'),
        quantity='energy',
        fixed_df=_ppdf([
            (2016, 10.0, False),
            (2017, 11.0, False),
            (2018, 12.0, False),
            (2019, 100.0, True),  # a forecast from upstream, discarded by the trendline
        ]),
    )
    target = _make_target(context, 'add,trendline')
    _connect(source, target)

    rows = _rows(target.compute())
    assert rows[2019][0] == pytest.approx(13.0)


def test_trendline_fits_each_category_separately():
    context = _make_context('trendline-dims')
    context.dimensions['sector'] = Dimension(
        id='sector',
        label='Sector',
        categories=[DimensionCategory(id='a', label='A'), DimensionCategory(id='b', label='B')],
    )
    rising = _ppdf([(2016, 10.0, False), (2017, 11.0, False), (2018, 12.0, False)], category='a')
    falling = _ppdf([(2016, 10.0, False), (2017, 8.0, False), (2018, 6.0, False)], category='b')
    df = rising.paths.concat_vertical(falling)
    source = _FixedOutputNode(
        id='source',
        context=context,
        name=TranslatedString('source', default_language='en'),
        unit=unit_registry.parse_units('kWh'),
        quantity='energy',
        fixed_df=df,
        output_dimension_ids=['sector'],
    )
    target = _make_target(context, 'add,trendline', output_dimension_ids=['sector'])
    _connect(source, target)

    out = target.compute()
    by_cat = {(row['sector'], row[YEAR_COLUMN]): row[VALUE_COLUMN] for row in out.to_dicts()}
    assert by_cat[('a', 2020)] == pytest.approx(14.0)
    assert by_cat[('b', 2020)] == pytest.approx(2.0)


def test_trendline_refuses_a_single_historical_year():
    context = _make_context('trendline-single-year')
    source = _FixedOutputNode(
        id='source',
        context=context,
        name=TranslatedString('source', default_language='en'),
        unit=unit_registry.parse_units('kWh'),
        quantity='energy',
        fixed_df=_ppdf([(2018, 10.0, False)]),
    )
    target = _make_target(context, 'add,trendline')
    _connect(source, target)

    with pytest.raises(NodeError, match='fewer than two historical years'):
        target.compute()


def test_trendline_fits_every_metric_column():
    """Two metrics round-trip through wide format, each with its own line."""
    context = _make_context('trendline-metrics')
    target = _make_target(context, 'trendline')
    df = pl.DataFrame({
        YEAR_COLUMN: [2016, 2017, 2018],
        VALUE_COLUMN: [10.0, 11.0, 12.0],
        'Cost': [100.0, 90.0, 80.0],
        FORECAST_COLUMN: [False, False, False],
    })
    df = to_ppdf(
        df,
        DataFrameMeta(
            units={VALUE_COLUMN: unit_registry.parse_units('kWh'), 'Cost': unit_registry.parse_units('EUR')},
            primary_keys=[YEAR_COLUMN],
        ),
    )

    out = target._operation_trendline(df)
    assert out is not None
    by_year = {row[YEAR_COLUMN]: row for row in out.to_dicts()}
    assert by_year[2020][VALUE_COLUMN] == pytest.approx(14.0)
    assert by_year[2020]['Cost'] == pytest.approx(60.0)
    assert out.get_unit('Cost') == unit_registry.parse_units('EUR')


def test_trendline_does_not_fabricate_rows_for_a_sparse_category():
    """A category that stops before the window gets no fitted forecast, not a null one."""
    context = _make_context('trendline-sparse')
    context.dimensions['sector'] = Dimension(
        id='sector',
        label='Sector',
        categories=[DimensionCategory(id='a', label='A'), DimensionCategory(id='b', label='B')],
    )
    target = _make_target(context, 'trendline', trend_years=2, output_dimension_ids=['sector'])
    live = _ppdf([(2016, 10.0, False), (2017, 11.0, False), (2018, 12.0, False)], category='a')
    retired = _ppdf([(2016, 5.0, False)], category='b')
    df = live.paths.concat_vertical(retired)

    out = target._operation_trendline(df)
    assert out is not None
    rows = {(row['sector'], row[YEAR_COLUMN]): row[VALUE_COLUMN] for row in out.to_dicts()}
    assert rows[('a', 2020)] == pytest.approx(14.0)
    assert rows[('b', 2016)] == 5.0  # history kept as it is
    assert ('b', 2020) not in rows  # no line to extrapolate, so no forecast row
