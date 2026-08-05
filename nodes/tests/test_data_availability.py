from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.datasets import Dataset
from nodes.exceptions import NodeError
from nodes.simple import DataAvailabilityNode
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.context import Context

pytestmark = pytest.mark.django_db


class _FixedRawDataset(Dataset):
    """A dataset that hands out a caller-supplied frame, going through `post_process()`. Test-only."""

    raw_df: PathsDataFrame | None = None

    def load_internal(self) -> PathsDataFrame:
        assert self.raw_df is not None
        return self.post_process(self.raw_df)

    def hash_data(self) -> dict[str, Any]:
        return {'id': self.id}


def _make_context(identifier: str) -> Context:
    instance = InstanceFactory.create(id=identifier, name=identifier)
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name=identifier)
    return instance.context


def _make_node(context: Context, datasets: list[Dataset]) -> DataAvailabilityNode:
    return DataAvailabilityNode(
        id='availability',
        context=context,
        name=TranslatedString('availability', default_language='en'),
        unit=unit_registry.parse_units('dimensionless'),
        quantity='fraction',
        input_datasets=datasets,
    )


def _make_dataset(context: Context, df: PathsDataFrame, *, interpolate: bool = False) -> _FixedRawDataset:
    ds = _FixedRawDataset(id='raw', context=context, interpolate=interpolate)
    ds.raw_df = df
    return ds


def _values_by_year(df: PathsDataFrame) -> dict[int, float]:
    return {row[YEAR_COLUMN]: row[VALUE_COLUMN] for row in df.to_dicts()}


def _sparse_df(unit: str = 'kWh') -> PathsDataFrame:
    """2011 and 2014 have values, 2012 is null and 2013 is missing altogether."""
    df = pl.DataFrame({
        YEAR_COLUMN: [2011, 2012, 2014],
        VALUE_COLUMN: [5.0, None, 7.0],
        FORECAST_COLUMN: [False, False, False],
    })
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=[YEAR_COLUMN])
    return to_ppdf(df, meta)


def test_missing_and_null_cells_are_zero_and_present_ones_one():
    context = _make_context('availability-basic')
    node = _make_node(context, [_make_dataset(context, _sparse_df())])

    values = _values_by_year(node.compute())
    assert values[2011] == 1.0  # a value exists
    assert values[2012] == 0.0  # explicit null
    assert values[2013] == 0.0  # year missing from the data
    assert values[2014] == 1.0
    assert values[2010] == 0.0  # before the data starts
    assert values[context.model_end_year] == 0.0  # after it ends


def test_zero_counts_as_a_value():
    context = _make_context('availability-zero')
    df = pl.DataFrame({YEAR_COLUMN: [2011], VALUE_COLUMN: [0.0], FORECAST_COLUMN: [False]})
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kWh')}, primary_keys=[YEAR_COLUMN])
    node = _make_node(context, [_make_dataset(context, to_ppdf(df, meta))])

    assert _values_by_year(node.compute())[2011] == 1.0


def test_output_spans_the_model_range_and_is_dimensionless():
    context = _make_context('availability-range')
    node = _make_node(context, [_make_dataset(context, _sparse_df())])

    df = node.compute()
    years = df[YEAR_COLUMN].to_list()
    assert years[0] == context.instance.minimum_historical_year
    assert years[-1] == context.model_end_year
    assert len(years) == len(set(years))
    assert df.get_unit(VALUE_COLUMN) == unit_registry.parse_units('dimensionless')
    node.validate_output(df)


def test_forecast_column_follows_the_maximum_historical_year():
    context = _make_context('availability-forecast')
    max_hist_year = context.instance.maximum_historical_year
    assert max_hist_year is not None
    node = _make_node(context, [_make_dataset(context, _sparse_df())])

    forecast_by_year = {row[YEAR_COLUMN]: row[FORECAST_COLUMN] for row in node.compute().to_dicts()}
    assert forecast_by_year[max_hist_year] is False
    assert forecast_by_year[max_hist_year + 1] is True


def test_interpolation_configured_on_the_binding_is_ignored():
    context = _make_context('availability-interpolate')
    ds = _make_dataset(context, _sparse_df(), interpolate=True)
    node = _make_node(context, [ds])

    values = _values_by_year(node.compute())
    assert values[2012] == 0.0  # would be interpolated into a value without the node's opt-out
    assert values[2013] == 0.0
    assert ds.interpolate is False


def test_dimensions_get_a_full_grid_of_the_categories_in_the_data():
    context = _make_context('availability-dims')
    df = pl.DataFrame({
        YEAR_COLUMN: [2011, 2012],
        'sector': ['a', 'b'],
        VALUE_COLUMN: [5.0, 7.0],
        FORECAST_COLUMN: [False, False],
    })
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kWh')}, primary_keys=[YEAR_COLUMN, 'sector'])
    node = _make_node(context, [_make_dataset(context, to_ppdf(df, meta))])

    out = node.compute()
    by_cell = {(row[YEAR_COLUMN], row['sector']): row[VALUE_COLUMN] for row in out.to_dicts()}
    assert by_cell[(2011, 'a')] == 1.0
    assert by_cell[(2011, 'b')] == 0.0  # category exists in the data, but not for this year
    assert by_cell[(2012, 'b')] == 1.0
    assert set(out['sector'].unique()) == {'a', 'b'}
    n_years = context.model_end_year - context.instance.minimum_historical_year + 1
    assert len(out) == 2 * n_years


def test_ragged_category_combinations_are_not_invented():
    """Combinations the data never uses must stay out; otherwise they look missing forever."""
    context = _make_context('availability-ragged')
    df = pl.DataFrame({
        YEAR_COLUMN: [2011, 2011],
        'carrier': ['coal', 'solar'],
        'measured_as': ['fuel_input', 'heat_output'],
        VALUE_COLUMN: [5.0, 7.0],
        FORECAST_COLUMN: [False, False],
    })
    meta = DataFrameMeta(
        units={VALUE_COLUMN: unit_registry.parse_units('kWh')},
        primary_keys=[YEAR_COLUMN, 'carrier', 'measured_as'],
    )
    node = _make_node(context, [_make_dataset(context, to_ppdf(df, meta))])

    out = node.compute()
    combos = {(row['carrier'], row['measured_as']) for row in out.to_dicts()}
    assert combos == {('coal', 'fuel_input'), ('solar', 'heat_output')}  # not the 2 x 2 cross product
    n_years = context.model_end_year - context.instance.minimum_historical_year + 1
    assert len(out) == 2 * n_years
    # Every combination the data uses is covered in the one year the data has.
    year_2011 = out.filter(pl.col(YEAR_COLUMN) == 2011)
    assert year_2011[VALUE_COLUMN].sum() == 2.0


def test_each_metric_column_gets_its_own_flag_column():
    context = _make_context('availability-metrics')
    from nodes.node import NodeMetric

    df = pl.DataFrame({
        YEAR_COLUMN: [2011, 2012],
        'energy': [5.0, None],
        'emissions': [None, 3.0],
        FORECAST_COLUMN: [False, False],
    })
    meta = DataFrameMeta(
        units={'energy': unit_registry.parse_units('kWh'), 'emissions': unit_registry.parse_units('t')},
        primary_keys=[YEAR_COLUMN],
    )
    node = DataAvailabilityNode(
        id='availability',
        context=context,
        name=TranslatedString('availability', default_language='en'),
        unit=None,
        quantity=None,
        output_metrics={
            'energy': NodeMetric(unit='dimensionless', quantity='fraction', id='energy', column_id='energy'),
            'emissions': NodeMetric(unit='dimensionless', quantity='fraction', id='emissions', column_id='emissions'),
        },
        input_datasets=[_make_dataset(context, to_ppdf(df, meta))],
    )

    out = node.compute()
    by_year = {row[YEAR_COLUMN]: row for row in out.to_dicts()}
    assert by_year[2011]['energy'] == 1.0
    assert by_year[2011]['emissions'] == 0.0
    assert by_year[2012]['energy'] == 0.0
    assert by_year[2012]['emissions'] == 1.0


def test_metric_columns_and_dimensions_together():
    context = _make_context('availability-metrics-dims')
    from nodes.node import NodeMetric

    df = pl.DataFrame({
        YEAR_COLUMN: [2011, 2011, 2012],
        'carrier': ['coal', 'solar', 'coal'],
        'energy': [5.0, 2.0, None],
        'emissions': [None, 1.0, 3.0],
        FORECAST_COLUMN: [False, False, False],
    })
    meta = DataFrameMeta(
        units={'energy': unit_registry.parse_units('kWh'), 'emissions': unit_registry.parse_units('t')},
        primary_keys=[YEAR_COLUMN, 'carrier'],
    )
    node = DataAvailabilityNode(
        id='availability',
        context=context,
        name=TranslatedString('availability', default_language='en'),
        unit=None,
        quantity=None,
        output_metrics={
            'energy': NodeMetric(unit='dimensionless', quantity='fraction', id='energy', column_id='energy'),
            'emissions': NodeMetric(unit='dimensionless', quantity='fraction', id='emissions', column_id='emissions'),
        },
        input_datasets=[_make_dataset(context, to_ppdf(df, meta))],
    )

    out = node.compute()
    by_cell = {(row[YEAR_COLUMN], row['carrier']): row for row in out.to_dicts()}
    assert by_cell[(2011, 'coal')]['energy'] == 1.0
    assert by_cell[(2011, 'coal')]['emissions'] == 0.0
    assert by_cell[(2011, 'solar')]['emissions'] == 1.0
    assert by_cell[(2012, 'coal')]['energy'] == 0.0  # null in a row that exists
    assert by_cell[(2012, 'solar')]['energy'] == 0.0  # row missing altogether
    assert by_cell[(2013, 'coal')]['emissions'] == 0.0  # year missing altogether


def test_input_nodes_are_rejected():
    context = _make_context('availability-nodes')
    from nodes.edges import Edge

    source = _make_node(context, [_make_dataset(context, _sparse_df())])
    node = _make_node(context, [_make_dataset(context, _sparse_df())])
    node.id = 'availability2'
    edge = Edge(input_node=source, output_node=node)
    source.add_edge(edge)
    node.add_edge(edge)

    with pytest.raises(NodeError, match='only inspects its input dataset'):
        node.compute()
