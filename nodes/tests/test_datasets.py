from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import polars as pl
import pytest

from common import polars as ppl
from nodes.datasets import DBDataset, DVCDataset
from nodes.defs.transform_def import AssignDimensionOp, FilterColumnOp, InterpolateOp, RemapLegacyYearsOp

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset as DBDatasetModel

    from nodes.context import Context


pytestmark = pytest.mark.django_db


def _hash_context(
    *,
    parameter_values: dict[str, str] | None = None,
    reference_year: int = 2020,
    target_year: int = 2030,
) -> Context:
    values = parameter_values or {}

    def get_parameter(parameter_id: str, *, required: bool = True) -> SimpleNamespace:
        assert required
        return SimpleNamespace(calculate_hash=lambda: f'{parameter_id}:{values[parameter_id]}')

    instance = SimpleNamespace(
        reference_year=reference_year,
        target_year=target_year,
        model_end_year=target_year,
        minimum_historical_year=2010,
        maximum_historical_year=2022,
    )
    return cast(
        'Context',
        SimpleNamespace(
            dataset_repo_spec=SimpleNamespace(
                url='https://example.test/datasets.git',
                commit='0123456789abcdef',
            ),
            dimensions={},
            get_parameter=get_parameter,
            instance=instance,
        ),
    )


def test_dvc_dataset_hash_includes_resolved_filter_parameter_values() -> None:
    transformation = FilterColumnOp(column='municipality', ref='municipality_name')
    mainz = DVCDataset(
        id='bisko/other_transport_energy',
        context=_hash_context(parameter_values={'municipality_name': 'Mainz'}),
        transformations=[transformation],
    )
    duesseldorf = DVCDataset(
        id='bisko/other_transport_energy',
        context=_hash_context(parameter_values={'municipality_name': 'Düsseldorf'}),
        transformations=[transformation],
    )

    assert mainz.get_cache_key() != duesseldorf.get_cache_key()
    assert mainz.hash_data()['pipeline']['transformations'][0]['parameter'] == 'municipality_name:Mainz'


def test_dvc_dataset_hash_includes_years_resolved_by_legacy_remapping() -> None:
    mainz = DVCDataset(
        id='shared/dataset',
        context=_hash_context(reference_year=2019, target_year=2035),
        transformations=[RemapLegacyYearsOp()],
    )
    duesseldorf = DVCDataset(
        id='shared/dataset',
        context=_hash_context(reference_year=2020, target_year=2045),
        transformations=[RemapLegacyYearsOp()],
    )

    assert mainz.get_cache_key() != duesseldorf.get_cache_key()


def test_dataset_hash_allows_assignment_of_an_instance_external_dimension() -> None:
    dataset = DVCDataset(
        id='shared/dataset',
        context=_hash_context(),
        transformations=[AssignDimensionOp(dimension='external', category='value')],
    )

    assert dataset.hash_data()['pipeline']['transformations'][0]['dimension'] is None


def test_operation_cache_version_invalidates_the_dataset_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _hash_context()
    original = DVCDataset(
        id='shared/dataset',
        context=context,
        transformations=[FilterColumnOp(column='sector', value='first')],
    )
    original_key = original.get_cache_key()

    monkeypatch.setattr(FilterColumnOp, 'cache_version', FilterColumnOp.cache_version + 1)
    versioned = DVCDataset(
        id='shared/dataset',
        context=context,
        transformations=[FilterColumnOp(column='sector', value='first')],
    )

    assert versioned.get_cache_key() != original_key


def test_source_overlay_runs_before_temporal_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    """Framework overlays can consume UUID join keys before interpolation removes them."""
    dataset = DVCDataset(
        id='framework/dataset',
        context=_hash_context(),
        transformations=[InterpolateOp()],
    )
    raw = _frame_with(
        {
            'Year': [2020, 2022],
            'uuid': ['measure', 'measure'],
            'Value': [1.0, 3.0],
        },
        primary_keys=['Year', 'uuid'],
        units={'Value': 'MWh/a'},
    )
    observed_join_keys: list[str] = []

    def overlay(_self: DVCDataset, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        observed_join_keys.extend(df['uuid'].to_list())
        return df.drop('uuid')

    monkeypatch.setattr(DVCDataset, 'before_temporal_fill', overlay)

    result = dataset._filter_and_process_df.__wrapped__(dataset, raw)

    assert observed_join_keys == ['measure', 'measure']
    assert result['Year'].to_list() == [2020, 2021, 2022]
    assert result['Value'].to_list() == [1.0, 2.0, 3.0]


def test_db_dataset_hash_includes_the_shared_binding_pipeline() -> None:
    db_dataset = cast(
        'DBDatasetModel',
        SimpleNamespace(pk=1, last_modified_at=datetime(2026, 1, 1, tzinfo=UTC)),
    )
    first = DBDataset(
        id='dataset',
        context=_hash_context(),
        db_dataset_obj=db_dataset,
        transformations=[FilterColumnOp(column='sector', value='first')],
    )
    second = DBDataset(
        id='dataset',
        context=_hash_context(),
        db_dataset_obj=db_dataset,
        transformations=[FilterColumnOp(column='sector', value='second')],
    )

    assert first.get_cache_key() != second.get_cache_key()


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
