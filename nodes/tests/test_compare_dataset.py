from __future__ import annotations

import polars as pl
import pytest

from common.polars import DataFrameMeta, PathsDataFrame, to_ppdf
from nodes.dataset_diff import align_dtypes, compute_row_diff, compute_schema_diff, normalize_df
from nodes.units import unit_registry


@pytest.fixture(autouse=True)
def instance_config():
    return None


def _make_ppdf(data: dict, primary_keys: list[str], units: dict[str, str] | None = None) -> PathsDataFrame:
    df = pl.DataFrame(data)
    parsed_units = {}
    if units:
        parsed_units = {col: unit_registry.parse_units(u) for col, u in units.items()}
    meta = DataFrameMeta(units=parsed_units, primary_keys=primary_keys)
    return to_ppdf(df, meta)


# --- compute_schema_diff tests ---


def test_schema_diff_identical():
    df = _make_ppdf(
        {'Year': [2020, 2021], 'value': [1.0, 2.0]},
        primary_keys=['Year'],
        units={'value': 'kg/year'},
    )
    sd = compute_schema_diff(df, df)
    assert sd.identical is True


def test_schema_diff_different_row_count():
    dvc = _make_ppdf(
        {'Year': [2020, 2021], 'value': [1.0, 2.0]},
        primary_keys=['Year'],
        units={'value': 'kg/year'},
    )
    db = _make_ppdf(
        {'Year': [2020], 'value': [1.0]},
        primary_keys=['Year'],
        units={'value': 'kg/year'},
    )
    sd = compute_schema_diff(dvc, db)
    assert sd.identical is True
    assert sd.dvc_row_count == 2
    assert sd.db_row_count == 1


def test_schema_diff_different_primary_keys():
    dvc = _make_ppdf(
        {'Year': [2020], 'sector': ['a'], 'value': [1.0]},
        primary_keys=['Year', 'sector'],
        units={'value': 'kg'},
    )
    db = _make_ppdf(
        {'Year': [2020], 'sector': ['a'], 'value': [1.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    sd = compute_schema_diff(dvc, db)
    assert sd.identical is False
    assert sd.pk_diff is not None


def test_schema_diff_extra_columns():
    dvc = _make_ppdf(
        {'Year': [2020], 'value': [1.0], 'extra_dvc': [0.5]},
        primary_keys=['Year'],
        units={'value': 'kg', 'extra_dvc': 'kg'},
    )
    db = _make_ppdf(
        {'Year': [2020], 'value': [1.0], 'extra_db': [0.5]},
        primary_keys=['Year'],
        units={'value': 'kg', 'extra_db': 'kg'},
    )
    sd = compute_schema_diff(dvc, db)
    assert sd.identical is False
    assert 'extra_dvc' in sd.dvc_only_cols
    assert 'extra_db' in sd.db_only_cols


def test_schema_diff_different_dtypes():
    dvc = _make_ppdf(
        {'Year': [2020], 'value': [1.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    db_raw = pl.DataFrame({'Year': [2020], 'value': [1]})
    db = to_ppdf(db_raw, DataFrameMeta(
        units={'value': unit_registry.parse_units('kg')},
        primary_keys=['Year'],
    ))
    sd = compute_schema_diff(dvc, db)
    assert sd.identical is False
    assert 'value' in sd.dtype_diffs


def test_schema_diff_different_units():
    dvc = _make_ppdf(
        {'Year': [2020], 'value': [1.0]},
        primary_keys=['Year'],
        units={'value': 'kg/year'},
    )
    db = _make_ppdf(
        {'Year': [2020], 'value': [1.0]},
        primary_keys=['Year'],
        units={'value': 'metric_ton/year'},
    )
    sd = compute_schema_diff(dvc, db)
    assert sd.identical is False
    assert 'value' in sd.unit_diffs


# --- normalize_df tests ---


def test_normalize_df_sorts_by_primary_keys():
    df = _make_ppdf(
        {'Year': [2021, 2020, 2022], 'value': [2.0, 1.0, 3.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    result = normalize_df(df)
    assert result['Year'].to_list() == [2020, 2021, 2022]
    assert result['value'].to_list() == [1.0, 2.0, 3.0]


def test_normalize_df_casts_categorical_to_string():
    raw = pl.DataFrame({
        'Year': [2020, 2020],
        'sector': pl.Series(['a', 'b']).cast(pl.Categorical),
        'value': [1.0, 2.0],
    })
    df = to_ppdf(raw, DataFrameMeta(
        units={'value': unit_registry.parse_units('kg')},
        primary_keys=['Year', 'sector'],
    ))
    result = normalize_df(df)
    assert result.schema['sector'] == pl.Utf8


def test_normalize_df_returns_plain_dataframe():
    df = _make_ppdf(
        {'Year': [2020], 'value': [1.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    result = normalize_df(df)
    assert type(result) is pl.DataFrame


# --- align_dtypes tests ---


def test_align_dtypes_casts_mismatched():
    dvc = pl.DataFrame({'Year': [2020], 'value': [1.0]})
    db = pl.DataFrame({'Year': [2020], 'value': [1]})
    dvc_out, db_out = align_dtypes(dvc, db, ['Year', 'value'])
    assert dvc_out.schema['value'] == pl.Utf8
    assert db_out.schema['value'] == pl.Utf8


def test_align_dtypes_leaves_matching_types():
    dvc = pl.DataFrame({'Year': [2020], 'value': [1.0]})
    db = pl.DataFrame({'Year': [2020], 'value': [2.0]})
    dvc_out, db_out = align_dtypes(dvc, db, ['Year', 'value'])
    assert dvc_out.schema['value'] == pl.Float64
    assert db_out.schema['value'] == pl.Float64


# --- compute_row_diff tests ---


def test_row_diff_identical():
    df = _make_ppdf(
        {'Year': [2020, 2021], 'value': [1.0, 2.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    rd = compute_row_diff(df, df)
    assert rd is not None
    assert len(rd.dvc_only) == 0
    assert len(rd.db_only) == 0
    assert len(rd.value_diffs) == 0


def test_row_diff_detects_exclusive_rows():
    dvc = _make_ppdf(
        {'Year': [2020, 2021, 2022], 'value': [1.0, 2.0, 3.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    db = _make_ppdf(
        {'Year': [2020, 2021, 2023], 'value': [1.0, 2.0, 4.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    rd = compute_row_diff(dvc, db)
    assert rd is not None
    assert len(rd.dvc_only) == 1
    assert len(rd.db_only) == 1


def test_row_diff_detects_value_differences():
    dvc = _make_ppdf(
        {'Year': [2020, 2021], 'value': [1.0, 2.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    db = _make_ppdf(
        {'Year': [2020, 2021], 'value': [1.0, 9.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    rd = compute_row_diff(dvc, db)
    assert rd is not None
    assert len(rd.value_diffs) == 1


def test_row_diff_handles_null_primary_keys():
    dvc = _make_ppdf(
        {'Year': [2020, 2020], 'sector': [None, 'a'], 'value': [1.0, 2.0]},
        primary_keys=['Year', 'sector'],
        units={'value': 'kg'},
    )
    db = _make_ppdf(
        {'Year': [2020, 2020], 'sector': [None, 'a'], 'value': [1.0, 2.0]},
        primary_keys=['Year', 'sector'],
        units={'value': 'kg'},
    )
    rd = compute_row_diff(dvc, db)
    assert rd is not None
    assert len(rd.dvc_only) == 0
    assert len(rd.db_only) == 0
    assert len(rd.value_diffs) == 0


def test_row_diff_null_pk_with_differing_values():
    dvc = _make_ppdf(
        {'Year': [2020], 'sector': [None], 'value': [1.0]},
        primary_keys=['Year', 'sector'],
        units={'value': 'kg'},
    )
    db = _make_ppdf(
        {'Year': [2020], 'sector': [None], 'value': [5.0]},
        primary_keys=['Year', 'sector'],
        units={'value': 'kg'},
    )
    rd = compute_row_diff(dvc, db)
    assert rd is not None
    assert len(rd.dvc_only) == 0
    assert len(rd.db_only) == 0
    assert len(rd.value_diffs) == 1


def test_row_diff_null_values_in_metric_cols():
    dvc = _make_ppdf(
        {'Year': [2020, 2021], 'value': [None, 2.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    db = _make_ppdf(
        {'Year': [2020, 2021], 'value': [1.0, 2.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    rd = compute_row_diff(dvc, db)
    assert rd is not None
    assert len(rd.value_diffs) == 1


def test_row_diff_both_null_values_match():
    dvc = _make_ppdf(
        {'Year': [2020, 2021], 'value': [None, 2.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    db = _make_ppdf(
        {'Year': [2020, 2021], 'value': [None, 2.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    rd = compute_row_diff(dvc, db)
    assert rd is not None
    assert len(rd.value_diffs) == 0


def test_row_diff_multiple_value_columns():
    dvc = _make_ppdf(
        {'Year': [2020], 'emissions': [100.0], 'mileage': [50.0]},
        primary_keys=['Year'],
        units={'emissions': 'kg', 'mileage': 'km'},
    )
    db = _make_ppdf(
        {'Year': [2020], 'emissions': [100.0], 'mileage': [99.0]},
        primary_keys=['Year'],
        units={'emissions': 'kg', 'mileage': 'km'},
    )
    rd = compute_row_diff(dvc, db)
    assert rd is not None
    assert len(rd.value_diffs) == 1


def test_row_diff_different_column_order():
    dvc = _make_ppdf(
        {'Year': [2020, 2021], 'sector': ['a', 'b'], 'value': [1.0, 2.0]},
        primary_keys=['Year', 'sector'],
        units={'value': 'kg'},
    )
    db_raw = pl.DataFrame({'value': [2.0, 1.0], 'sector': ['b', 'a'], 'Year': [2021, 2020]})
    db = to_ppdf(db_raw, DataFrameMeta(
        units={'value': unit_registry.parse_units('kg')},
        primary_keys=['Year', 'sector'],
    ))
    rd = compute_row_diff(dvc, db)
    assert rd is not None
    assert len(rd.dvc_only) == 0
    assert len(rd.db_only) == 0
    assert len(rd.value_diffs) == 0


def test_row_diff_different_row_order():
    dvc = _make_ppdf(
        {'Year': [2021, 2020], 'value': [2.0, 1.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    db = _make_ppdf(
        {'Year': [2020, 2021], 'value': [1.0, 2.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    rd = compute_row_diff(dvc, db)
    assert rd is not None
    assert len(rd.value_diffs) == 0


def test_row_diff_mismatched_dtypes_aligned_and_compared():
    dvc = _make_ppdf(
        {'Year': [2020], 'value': [1.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    db_raw = pl.DataFrame({'Year': [2020], 'value': [1]})
    db = to_ppdf(db_raw, DataFrameMeta(
        units={'value': unit_registry.parse_units('kg')},
        primary_keys=['Year'],
    ))
    rd = compute_row_diff(dvc, db)
    assert rd is not None
    assert len(rd.dvc_only) == 0
    assert len(rd.db_only) == 0
    assert len(rd.value_diffs) == 1


def test_row_diff_no_common_columns():
    dvc = _make_ppdf(
        {'Year': [2020], 'val_a': [1.0]},
        primary_keys=['Year'],
        units={'val_a': 'kg'},
    )
    db = _make_ppdf(
        {'date': [2020], 'val_b': [1.0]},
        primary_keys=['date'],
        units={'val_b': 'kg'},
    )
    assert compute_row_diff(dvc, db) is None


def test_row_diff_no_common_primary_keys():
    dvc = _make_ppdf(
        {'Year': [2020], 'value': [1.0]},
        primary_keys=['Year'],
        units={'value': 'kg'},
    )
    db = _make_ppdf(
        {'Year': [2020], 'value': [1.0]},
        primary_keys=['value'],
        units={},
    )
    assert compute_row_diff(dvc, db) is None
