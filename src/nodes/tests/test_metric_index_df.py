"""Tests for the index frame that dimensional metrics are joined against."""

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common import polars as ppl
from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN
from nodes.metric import DimensionalMetric, MetricCategory, MetricDimension
from nodes.node import Node, NodeMetric
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def _dim() -> MetricDimension:
    return MetricDimension(
        id='sector',
        original_id='sector',
        label='Sector',
        categories=[
            MetricCategory(id='transport', original_id='transport', label='Transport', color=None, order=None),
        ],
    )


def test_year_column_is_typed_when_there_are_years() -> None:
    idx_df = DimensionalMetric.generate_index_df([_dim()], [2020, 2021])
    assert idx_df.schema[YEAR_COLUMN] == pl.Int64
    assert idx_df.height == 2


def test_year_column_is_typed_when_the_node_produced_no_rows() -> None:
    """
    An empty year list must still yield an Int64 year column.

    A node can legitimately compute to nothing -- a weighted average whose weights are all
    missing, for instance. Polars infers Null for a column built from an empty list, and the
    caller then joins this frame against the node's (Int64) output, which used to fail with
    `SchemaError: datatypes of join keys don't match` instead of producing an empty metric.
    """
    idx_df = DimensionalMetric.generate_index_df([_dim()], [])
    assert idx_df.schema[YEAR_COLUMN] == pl.Int64
    assert idx_df.height == 0

    # The join the caller performs must now work rather than raise.
    out = pl.DataFrame(schema={'sector': pl.Utf8, YEAR_COLUMN: pl.Int64, 'Value': pl.Float64})
    joined = idx_df.with_columns(pl.col('sector').cast(pl.Utf8)).join(out, how='left', on=['sector', YEAR_COLUMN], validate='1:1')
    assert joined.height == 0


def test_index_df_with_no_dimensions_still_has_years() -> None:
    idx_df = DimensionalMetric.generate_index_df([], [2020])
    assert idx_df.schema[YEAR_COLUMN] == pl.Int64
    assert idx_df.height == 1


class _TwoMetricNode(Node):
    def compute(self) -> ppl.PathsDataFrame:
        raw = pl.DataFrame({
            YEAR_COLUMN: [2020, 2021],
            'energy': [1.0, 2.0],
            'cost': [10.0, 20.0],
            FORECAST_COLUMN: [False, True],
        })
        return ppl.to_ppdf(
            raw,
            ppl.DataFrameMeta(
                units={'energy': unit_registry.parse_units('kWh/a'), 'cost': unit_registry.parse_units('EUR/a')},
                primary_keys=[YEAR_COLUMN],
            ),
        )


def test_metric_of_a_node_with_several_output_metrics() -> None:
    """The metric is built from the chosen metric's column, which the output does not call Value."""
    instance = InstanceFactory.create(id='two_metric_node', name='Two metric node')
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name='Two metric node')
    context = instance.context
    node = _TwoMetricNode(
        id='two_metrics',
        context=context,
        name=TranslatedString('Two metrics', default_language='en'),
        output_metrics={
            'energy': NodeMetric(id='energy', column_id='energy', quantity='energy', unit=unit_registry.parse_units('kWh/a')),
            'cost': NodeMetric(id='cost', column_id='cost', quantity='currency', unit=unit_registry.parse_units('EUR/a')),
        },
    )
    context.add_node(node)
    with context.run():
        energy = DimensionalMetric.from_node(node, metric=node.output_metrics['energy'])
        cost = DimensionalMetric.from_node(node, metric=node.output_metrics['cost'])
    assert energy is not None
    assert cost is not None
    assert energy.values == [1.0, 2.0]
    assert cost.values == [10.0, 20.0]
    assert cost.unit == unit_registry.parse_units('EUR/a')
    assert energy.forecast_from == 2021
