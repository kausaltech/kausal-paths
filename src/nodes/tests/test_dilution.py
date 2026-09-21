from typing import TYPE_CHECKING
from unittest.mock import Mock

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from common import polars as ppl
from nodes.costs import DilutionNode
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


if TYPE_CHECKING:
    from collections.abc import Sequence


def frame(years: list[int], values: Sequence[float | int | None], unit: str, *, dimension: bool = False) -> ppl.PathsDataFrame:
    df = pl.DataFrame({'Year': years, 'Value': values, 'Forecast': [year > 2020 for year in years]})
    keys = ['Year']
    if dimension:
        df = df.with_columns(pl.lit('cars').alias('mode'))
        keys.append('mode')
    return ppl.to_ppdf(df, meta=ppl.DataFrameMeta(units={'Value': unit_registry.parse_units(unit)}, primary_keys=keys))


def node(existing: ppl.PathsDataFrame, incoming: ppl.PathsDataFrame, rates: ppl.PathsDataFrame, end_year: int) -> DilutionNode:
    result = Mock(spec=DilutionNode)
    inputs = {'existing': existing, 'incoming': incoming, 'removing': rates, 'inserting': rates}
    result.get_input_node.side_effect = lambda *, tag: Mock(get_output_pl=Mock(return_value=inputs[tag]))
    result.get_end_year.return_value = end_year
    return result


@pytest.mark.parametrize('dimension', [False, True])
@pytest.mark.parametrize('integer', [False, True])
def test_recurrence_preserves_history_broadcasting_and_step_coercion(dimension: bool, integer: bool) -> None:
    historical = frame([2019, 2020], [12, 10] if integer else [12.0, 10.0], 'kg', dimension=dimension)
    incoming = frame([2021, 2022, 2023], [4.0, 2.0, 6.0], 'kg')
    rates = frame([2021, 2022, 2023], [0.25, 0.5, 1.0], '1/a')
    actual = DilutionNode.compute(node(historical, incoming, rates, 2023))
    values = [12, 10, 8, 5, 6] if integer else [12.0, 10.0, 8.5, 5.25, 6.0]
    expected = frame([2019, 2020, 2021, 2022, 2023], values, 'kg', dimension=dimension)
    assert_frame_equal(actual.select(expected.columns), expected, check_exact=True)
    assert actual.get_meta().is_equal(expected.get_meta())


def test_missing_input_year_propagates_nulls() -> None:
    historical = frame([2020], [10.0], 'kg')
    incoming = frame([2021, 2023], [4.0, 6.0], 'kg')
    rates = frame([2021, 2022, 2023], [0.25, 0.5, 1.0], '1/a')
    actual = DilutionNode.compute(node(historical, incoming, rates, 2023))
    assert actual['Value'].to_list() == [10.0, 8.5, None, None]


def test_no_extension_returns_original() -> None:
    historical = frame([2019, 2020], [12.0, 10.0], 'kg')
    assert DilutionNode.compute(node(historical, historical, frame([2020], [0.5], '1/a'), 2020)) is historical


def test_duplicate_indices_still_raise() -> None:
    historical = frame([2020, 2020], [10.0, 12.0], 'kg')
    incoming = frame([2021], [4.0], 'kg')
    rates = frame([2021], [0.25], '1/a')
    with pytest.raises(Exception, match='duplicated index rows'):
        DilutionNode.compute(node(historical, incoming, rates, 2021))
