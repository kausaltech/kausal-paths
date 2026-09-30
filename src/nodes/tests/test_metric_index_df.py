"""Tests for the index frame that dimensional metrics are joined against."""

from typing import TYPE_CHECKING

from pydantic import ValidationError

import polars as pl
import pytest

from common import polars as ppl, qualifiers
from nodes.constants import YEAR_COLUMN
from nodes.metric import BooleanQualifier, CoveredScoreQualifier, DimensionalMetric, MetricCategory, MetricDimension
from nodes.metric_gen import _indexed_output_data, metric_from_dataframe_standalone
from nodes.units import unit_registry

if TYPE_CHECKING:
    from django.test import Client

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


def test_qualifiers_follow_values_through_dense_index_and_round_trip() -> None:
    frame = ppl.to_ppdf(
        pl.DataFrame({
            'Year': [2021, 2020],
            'sector': ['transport', 'industry'],
            'Energy': [0.0, 12.0],
        }).with_columns(qualifiers.make(quality=pl.lit(0.5), reported=pl.lit(value=True)).alias('Energy__qual')),
        meta=ppl.DataFrameMeta(units={'Energy': unit_registry.parse_units('MWh/a')}, primary_keys=['Year', 'sector']),
    )
    metric = metric_from_dataframe_standalone(frame, 'Energy', metric_id='test', metric_name='Test')
    assert metric.qualifiers is not None
    assert len(metric.values) == 4
    assert len(metric.qualifiers) == 2
    for q in metric.qualifiers:
        assert q.id == f'test:{q.identifier}'
        assert len(q.values if isinstance(q, BooleanQualifier) else q.scores) == 4
    other = DimensionalMetric.model_validate({**metric.model_dump(), 'id': 'other'})
    assert {q.id for q in metric.qualifiers}.isdisjoint(q.id for q in other.qualifiers)
    restored = metric.to_df()
    rows = {(r['sector'], r['Year']): r for r in restored.to_dicts()}
    assert rows[('transport', 2021)]['Value'] == 0
    assert rows[('transport', 2021)]['Value__qual'] == {
        'quality': {'score': 0.5, 'coverage': 1.0},
        'reported': True,
    }
    # A serializer-created zero is not a reported zero or a grade D.
    assert rows[('industry', 2021)]['Value'] == 0
    assert rows[('industry', 2021)]['Value__qual'] == {
        'reported': None,
        'quality': {'score': None, 'coverage': None},
    }


def test_dropped_values_also_drop_their_qualifier_slots() -> None:
    frame = ppl.to_ppdf(
        pl.DataFrame({'Year': [2020, 2021], 'Value': [None, 1.0], 'Forecast': [False, False]}).with_columns(
            qualifiers.make(quality=pl.lit(0.0), reported=pl.lit(value=True)).alias('Value__qual')
        ),
        meta=ppl.DataFrameMeta(units={'Value': unit_registry.parse_units('MWh/a')}, primary_keys=['Year']),
    )
    data = _indexed_output_data([], frame, dropped_not_filled=True)
    assert data.values == [1.0]
    assert data.qualifiers is not None
    assessment = next(q for q in data.qualifiers if isinstance(q, CoveredScoreQualifier))
    assert assessment.scores == [0.0]
    assert assessment.coverage == [1.0]
    reporting = next(q for q in data.qualifiers if isinstance(q, BooleanQualifier))
    assert reporting.values == [True]


@pytest.mark.parametrize('invalid', ['length', 'coverage_length', 'duplicate'])
def test_qualifier_contract_is_validated(invalid: str) -> None:
    frame = ppl.to_ppdf(
        pl.DataFrame({'Year': [2020], 'Value': [1.0]}),
        meta=ppl.DataFrameMeta(units={'Value': unit_registry.parse_units('MWh/a')}, primary_keys=['Year']),
    )
    metric = metric_from_dataframe_standalone(frame, 'Value', metric_id='test', metric_name='Test')
    payload = metric.model_dump()
    q = CoveredScoreQualifier(identifier='quality', scores=[1.0], coverage=[1.0])
    if invalid == 'length':
        q.scores = []
    elif invalid == 'coverage_length':
        q.coverage = []
    payload['qualifiers'] = [q, q] if invalid == 'duplicate' else [q]
    with pytest.raises(ValidationError):
        DimensionalMetric.model_validate(payload)


def test_graphql_qualifier_columns_have_metric_scoped_ids(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    from paths.tests.graphql import PathsTestClient

    from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeFactory

    instance = InstanceFactory.create()
    config = InstanceConfigFactory.create(identifier=instance.id, instance=instance)
    node = NodeFactory.create(context=instance.context)
    frame = ppl.to_ppdf(
        pl.DataFrame({'Year': [2020, 2021], 'Value': [10.0, 20.0]}).with_columns(
            qualifiers.make(quality=pl.col('Year').replace_strict({2020: 1.0, 2021: 0.0}), reported=pl.lit(value=True)).alias(
                'Value__qual'
            )
        ),
        meta=ppl.DataFrameMeta(units={'Value': unit_registry.parse_units('MWh/a')}, primary_keys=['Year']),
    )
    metric = metric_from_dataframe_standalone(frame, 'Value', metric_id=node.id, metric_name='Test')
    monkeypatch.setattr(node, 'outcome', metric, raising=False)
    gql_client = PathsTestClient(client)
    gql_client.set_instance(config)
    data = gql_client.query_data(
        """
        query($id: ID!) {
            node(id: $id) {
                outcome {
                    id values
                    qualifiers {
                        __typename
                        ... on BooleanQualifierType { id identifier values }
                        ... on CoveredScoreQualifierType { id identifier scores coverage }
                    }
                }
            }
        }
    """,
        variables={'id': node.id},
    )
    output = data['node']['outcome']
    columns = {q['identifier']: q for q in output['qualifiers']}
    assert columns['quality'] == {
        '__typename': 'CoveredScoreQualifierType',
        'id': f'{node.id}:quality',
        'identifier': 'quality',
        'scores': [1.0, 0.0],
        'coverage': [1.0, 1.0],
    }
    assert columns['reported']['id'] == f'{node.id}:reported'
    assert columns['reported']['values'] == [True, True]
