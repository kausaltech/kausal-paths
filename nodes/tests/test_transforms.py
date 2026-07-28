"""
Tests for the port transform pipeline executor.

The executor replaced a hardcoded loading sequence in ``DatasetWithFilters``.
These pin the stages whose *position* in that sequence was load-bearing, since
that is what a literal pass over the operations has to reproduce.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import polars as pl
import pytest

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.defs.transform_def import (
    DropNullsOp,
    EnsureUnitOp,
    FilterDimensionOp,
    FilterTemporalOp,
    IndexTemporalOp,
    RemapLegacyYearsOp,
    RenameColumnOp,
    SelectMetricOp,
    SetForecastFromOp,
)
from nodes.transforms import PipelineEnv, apply_port_pipeline
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.context import Context
    from nodes.defs.transform_def import PortTransformOp


def _env(reference_year: int = 2020, target_year: int = 2030) -> PipelineEnv:
    """Build a minimal environment; only the year-remapping op reads the instance."""
    context = cast(
        'Context',
        SimpleNamespace(instance=SimpleNamespace(reference_year=reference_year, target_year=target_year)),
    )
    return PipelineEnv(context=context)


def _wide_ppdf() -> PathsDataFrame:
    """Build a frame shaped like a wide DVC dataset: several metric columns, no Forecast."""
    df = pl.DataFrame({
        YEAR_COLUMN: [2020, 2021, 2022],
        'Cars': [1.0, 2.0, 3.0],
        'Trucks': [10.0, None, 30.0],
    })
    meta = DataFrameMeta(
        units={'Cars': unit_registry.parse_units('kt/a'), 'Trucks': unit_registry.parse_units('kt/a')},
        primary_keys=[YEAR_COLUMN],
    )
    return to_ppdf(df, meta)


def _run(df: PathsDataFrame, ops: list[PortTransformOp], env: PipelineEnv | None = None) -> PathsDataFrame:
    return apply_port_pipeline(df, ops, env or _env())


def test_select_metric_aliases_the_bound_column_and_narrows_the_frame():
    env = _env()
    env.metric_column = 'Cars'

    result = _run(_wide_ppdf(), [SelectMetricOp()], env)

    assert result.columns == [YEAR_COLUMN, VALUE_COLUMN]
    assert result[VALUE_COLUMN].to_list() == [1.0, 2.0, 3.0]


def test_select_metric_drops_rows_where_the_selected_metric_is_null():
    """The selection carries a not-null filter; dropping it would change row counts."""
    env = _env()
    env.metric_column = 'Trucks'

    result = _run(_wide_ppdf(), [SelectMetricOp()], env)

    assert result[YEAR_COLUMN].to_list() == [2020, 2022]


def test_select_metric_without_a_bound_column_fails_loudly():
    with pytest.raises(Exception, match='no column to select'):
        _run(_wide_ppdf(), [SelectMetricOp()], _env())


def test_remap_legacy_years_maps_placeholder_years_onto_real_ones():
    """Legacy DVC datasets encode the reference year as 0 or 1, the target year as 100 or 101."""
    df = to_ppdf(
        pl.DataFrame({YEAR_COLUMN: [0, 1, 100, 101], VALUE_COLUMN: [1.0, 2.0, 3.0, 4.0]}),
        DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kt/a')}, primary_keys=[YEAR_COLUMN]),
    )

    result = _run(df, [IndexTemporalOp(), RemapLegacyYearsOp()], _env(reference_year=2020, target_year=2030))

    # 0 and 1 both mean the reference year, 100 and 101 both the target year, so
    # each pair collapses and the later row wins.
    assert result[YEAR_COLUMN].to_list() == [2020, 2030]
    assert result[VALUE_COLUMN].to_list() == [2.0, 4.0]


def test_remap_legacy_years_leaves_real_years_alone():
    df = to_ppdf(
        pl.DataFrame({YEAR_COLUMN: [2020, 2021], VALUE_COLUMN: [1.0, 2.0]}),
        DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kt/a')}, primary_keys=[YEAR_COLUMN]),
    )

    result = _run(df, [IndexTemporalOp(), RemapLegacyYearsOp()])

    assert result[YEAR_COLUMN].to_list() == [2020, 2021]


def test_set_forecast_from_marks_years_from_the_given_one():
    df = to_ppdf(
        pl.DataFrame({YEAR_COLUMN: [2020, 2021, 2022], VALUE_COLUMN: [1.0, 2.0, 3.0]}),
        DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kt/a')}, primary_keys=[YEAR_COLUMN]),
    )

    result = _run(df, [SetForecastFromOp(year=2021)])

    assert result[FORECAST_COLUMN].to_list() == [False, True, True]


def test_set_forecast_from_does_not_override_a_frame_that_states_its_own():
    """The dataset knows its forecast status better than the binding does."""
    df = to_ppdf(
        pl.DataFrame({
            YEAR_COLUMN: [2020, 2021],
            VALUE_COLUMN: [1.0, 2.0],
            FORECAST_COLUMN: [True, False],
        }),
        DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kt/a')}, primary_keys=[YEAR_COLUMN]),
    )

    result = _run(df, [SetForecastFromOp(year=2021)])

    assert result[FORECAST_COLUMN].to_list() == [True, False]


def test_filter_temporal_and_drop_nulls_shape_the_output():
    df = to_ppdf(
        pl.DataFrame({YEAR_COLUMN: [2018, 2020, 2021, 2025], VALUE_COLUMN: [1.0, None, 3.0, 4.0]}),
        DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kt/a')}, primary_keys=[YEAR_COLUMN]),
    )

    result = _run(df, [FilterTemporalOp(min_year=2020, max_year=2021), DropNullsOp()])

    assert result[YEAR_COLUMN].to_list() == [2021]


def test_ensure_unit_converts_rather_than_relabels():
    df = to_ppdf(
        pl.DataFrame({YEAR_COLUMN: [2020], VALUE_COLUMN: [1.0]}),
        DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kt/a')}, primary_keys=[YEAR_COLUMN]),
    )

    result = _run(df, [EnsureUnitOp(unit=unit_registry.parse_units('t/a'))])

    assert result.get_unit(VALUE_COLUMN) == unit_registry.parse_units('t/a')
    assert result[VALUE_COLUMN].to_list() == [1000.0]


def test_rename_column_runs_before_selection_so_it_can_expose_the_column():
    """
    Order matters, and this is the case that proves it.

    Real configs rename a dataset's own column headings (``Vuosi`` to ``Year``)
    before the metric selection can address them, so a pipeline that selected
    first would break.
    """
    df = to_ppdf(
        pl.DataFrame({'Vuosi': [2020, 2021], 'Päästöt': [1.0, 2.0]}),
        DataFrameMeta(units={'Päästöt': unit_registry.parse_units('kt/a')}, primary_keys=[]),
    )
    env = _env()
    env.metric_column = 'Päästöt'

    result = _run(df, [RenameColumnOp(column='Vuosi', new_name=YEAR_COLUMN), SelectMetricOp()], env)

    assert result.columns == [YEAR_COLUMN, VALUE_COLUMN]
    assert result[YEAR_COLUMN].to_list() == [2020, 2021]


def test_filter_dimension_flattens_by_summing_over_the_dimension():
    df = to_ppdf(
        pl.DataFrame({
            YEAR_COLUMN: [2020, 2020, 2021, 2021],
            'sector': ['a', 'b', 'a', 'b'],
            VALUE_COLUMN: [1.0, 2.0, 3.0, 4.0],
        }),
        DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kt/a')}, primary_keys=[YEAR_COLUMN, 'sector']),
    )

    result = _run(df, [FilterDimensionOp(dimension='sector', flatten=True)])

    assert 'sector' not in result.columns
    assert sorted(result[VALUE_COLUMN].to_list()) == [3.0, 7.0]


def test_filtering_everything_away_is_an_error():
    """An empty result means the configuration is wrong, and silence would hide it."""
    df = to_ppdf(
        pl.DataFrame({YEAR_COLUMN: [2020], 'sector': ['a'], VALUE_COLUMN: [1.0]}),
        DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units('kt/a')}, primary_keys=[YEAR_COLUMN, 'sector']),
    )

    with pytest.raises(Exception, match='Nothing left after filter_dimension'):
        _run(df, [FilterDimensionOp(dimension='sector', categories=['nonexistent'])])


def test_dataset_level_forecast_default_enters_the_pipeline():
    """
    A default inherited from the dataset must behave like one declared on the binding.

    `_promote_dataset_forecast_defaults` moves an unambiguous forecast year onto
    the dataset and clears the binding overrides, so those bindings rely on the
    fallback. Since synthesis is an operation now, setting the old field alone
    would silently drop the Forecast column.
    """
    from nodes.defs.node_defs import InputDatasetDef
    from nodes.defs.transform_def import with_forecast_from

    ops = InputDatasetDef(id='some/dataset', column='C').to_transform_pipeline().operations

    with_default = with_forecast_from(ops, 2030)

    assert [op.kind for op in with_default] == [
        'select_metric',
        'index_temporal',
        'remap_legacy_years',
        'set_forecast_from',
    ]


def test_a_binding_forecast_year_is_not_overridden_by_the_dataset_default():
    from nodes.defs.node_defs import InputDatasetDef
    from nodes.defs.transform_def import forecast_from_operations, with_forecast_from

    ops = InputDatasetDef(id='some/dataset', forecast_from=2025).to_transform_pipeline().operations

    assert forecast_from_operations(with_forecast_from(ops, 2030)) == 2025


def test_explanations_still_describe_a_pipeline_shaped_dataset_config():
    """
    The explanation text must not depend on which config source the instance uses.

    Database-backed instances carry an ordered pipeline where YAML carries flat
    fields, and the explanations were written against the latter.
    """
    from nodes.explanations import _flat_keys_from_operations

    config = {
        'id': 'some/dataset',
        'operations': [
            {'kind': 'rename_column', 'column': 'Vuosi', 'new_name': 'Year'},
            {'kind': 'set_forecast_from', 'year': 2025},
            {'kind': 'filter_column', 'column': 'action', 'value': 'x'},
            {'kind': 'filter_dimension', 'dimension': 'sector', 'categories': ['a'], 'flatten': True},
            {'kind': 'assign_dimension', 'dimension': 'sector', 'category': 'b'},
            {'kind': 'rename_item', 'column': 'sector', 'old_item': 'old', 'new_item': 'new'},
            {'kind': 'drop_nulls'},
        ],
    }

    flat = _flat_keys_from_operations(config)

    assert flat['forecast_from'] == 2025
    assert flat['dropna'] is True
    assert flat['filters'] == [
        {'rename_col': 'Vuosi', 'value': 'Year'},
        {'column': 'action', 'value': 'x'},
        {'dimension': 'sector', 'categories': ['a'], 'flatten': True},
        {'dimension': 'sector', 'assign_category': 'b'},
        {'rename_item': 'sector|old', 'value': 'new'},
    ]


def test_flat_key_translation_leaves_yaml_shaped_configs_alone():
    from nodes.explanations import _flat_keys_from_operations

    config = {'id': 'some/dataset', 'forecast_from': 2025, 'filters': [{'column': 'action', 'value': 'x'}]}

    assert _flat_keys_from_operations(config) is config
