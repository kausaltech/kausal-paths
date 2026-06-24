from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import polars as pl
import pytest

from common import polars as ppl
from nodes.datasets import DBDataset

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
