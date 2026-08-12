from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import polars as pl
import pytest

from common import polars as ppl
from nodes.datasets import DBDataset, DVCDataset

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset as DBDatasetModel

    from nodes.context import Context


pytestmark = pytest.mark.django_db


def test_db_dataset_reuses_deserialized_dataframe_within_context(monkeypatch: pytest.MonkeyPatch) -> None:
    context = cast('Context', SimpleNamespace(db_dataset_dfs={}))
    db_dataset = cast('DBDatasetModel', SimpleNamespace(pk=1))
    raw_df = ppl.to_ppdf(pl.DataFrame({'Year': [2020], 'Value': [1.0]}))
    calls = 0

    def deserialize_df(
        cls: type[DBDataset],
        ds_in: DBDatasetModel,
        *,
        include_data_point_primary_keys: bool = False,
    ) -> ppl.PathsDataFrame:
        nonlocal calls
        assert ds_in is db_dataset
        assert include_data_point_primary_keys is False
        calls += 1
        return raw_df

    monkeypatch.setattr(DBDataset, 'deserialize_df', classmethod(deserialize_df))
    monkeypatch.setattr(DBDataset, '_filter_and_process_df', lambda _self, df: df)

    first = DBDataset(id='dataset', context=context, db_dataset_obj=db_dataset)
    second = DBDataset(id='dataset', context=context, db_dataset_obj=db_dataset)

    first_df = first.load_internal()
    assert first.load_internal() is first_df
    second_df = second.load_internal()

    assert calls == 1
    assert context.db_dataset_dfs[1] is raw_df
    assert first_df is not raw_df
    assert second_df is not raw_df
    assert second_df is not first_df


def _frame_with(columns: dict[str, list], primary_keys: list[str], units: dict[str, str]) -> ppl.PathsDataFrame:
    from nodes.units import unit_registry

    meta = ppl.DataFrameMeta(
        units={col: unit_registry.parse_units(unit) for col, unit in units.items()},
        primary_keys=primary_keys,
    )
    return ppl.PathsDataFrame._from_pydf(pl.DataFrame(columns)._df, meta=meta)


def test_dvc_dataset_drops_provenance_columns_so_dvc_matches_db() -> None:
    """
    `Source` and `Comment` must not survive into a DVC-loaded frame.

    They ride to DVC as ordinary columns because the parquet has nowhere else to put
    them, and `load_dvc_dataset` reads them back into DataSource / DataPointComment
    records -- so the database path never surfaces them. Leaving them on the DVC path
    gave the same dataset two different column sets depending on which source the
    instance was configured for.
    """
    df = _frame_with(
        {
            'Year': [2020, 2021],
            'sector': ['a', 'b'],
            'source': ['S1', 'S2'],
            'comment': ['note', 'note'],
            'Value': [1.0, 2.0],
        },
        primary_keys=['Year', 'sector'],
        units={'Value': 'MWh/a'},
    )

    out = DVCDataset._drop_reserved_columns(df)

    assert out.columns == ['Year', 'sector', 'Value']
    assert out.primary_keys == ['Year', 'sector']
    assert out.metric_cols == ['Value']
    assert out.height == 2


def test_dvc_dataset_keeps_a_dimension_that_happens_to_be_called_source() -> None:
    """
    A reserved name that is genuinely an index column is data, not provenance.

    `upload_new_dataset` keeps the reserved names out of `index_columns`, so anything
    still in `primary_keys` under one of them was put there deliberately.
    """
    df = _frame_with(
        {'Year': [2020], 'source': ['grid'], 'comment': ['ignore me'], 'Value': [1.0]},
        primary_keys=['Year', 'source'],
        units={'Value': 'MWh/a'},
    )

    out = DVCDataset._drop_reserved_columns(df)

    assert out.columns == ['Year', 'source', 'Value']


def test_dvc_dataset_keeps_a_metric_that_happens_to_be_called_description() -> None:
    """A reserved name carrying a unit is a metric, and dropping it would lose data."""
    df = _frame_with(
        {'Year': [2020], 'sector': ['a'], 'description': [3.0], 'Value': [1.0]},
        primary_keys=['Year', 'sector'],
        units={'Value': 'MWh/a', 'description': 'MWh/a'},
    )

    out = DVCDataset._drop_reserved_columns(df)

    assert 'description' in out.columns
