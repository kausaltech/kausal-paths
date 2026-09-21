from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from common import polars as ppl
from datasets.models import PreparedDataset
from datasets.prepared import PreparedDatasetStore, PreparedRecipe, deserialize_frame, serialize_frame
from datasets.tests.test_manifests import SPEC, manifest_for
from nodes.context import Context
from nodes.datasets import DVCDataset
from nodes.defs.transform_def import FilterColumnOp, RemapLegacyYearsOp, SelectMetricOp, TagOperationOp
from nodes.units import unit_registry

if TYPE_CHECKING:
    from pytest_django import DjangoAssertNumQueries

pytestmark = pytest.mark.django_db


def frame() -> ppl.PathsDataFrame:
    result = ppl.to_ppdf(
        pl.DataFrame({'Year': [2020, 2021], 'sector': ['a', 'b'], 'Value': [1.0, 2.0]}),
        ppl.DataFrameMeta(units={'Value': unit_registry.parse_units('kg')}, primary_keys=['Year', 'sector']),
    )
    result._explanation = ['source preparation']
    return result


def context() -> Context:
    ctx = Context.__new__(Context)
    ctx.skip_cache = False
    ctx.sample_size = 0
    ctx.dataset_repo_spec = SPEC
    ctx.dvc_source_manifest = manifest_for('activity')
    ctx.nodes = {}
    ctx.perf_context = MagicMock()
    return ctx


def test_round_trip_preserves_frame_and_metadata() -> None:
    original = frame().with_columns(pl.col('sector').cast(pl.Categorical))
    payload, meta = serialize_frame(original)
    actual = deserialize_frame(payload, meta)
    assert_frame_equal(actual, original)
    assert actual.get_meta().is_equal(original.get_meta())
    assert actual._explanation == original._explanation


def test_reads_are_batched_and_frames_are_isolated(django_assert_num_queries: DjangoAssertNumQueries) -> None:
    recipes = [PreparedRecipe({'test': index}) for index in range(3)]
    writer = PreparedDatasetStore()
    for recipe in recipes:
        writer.put(recipe, frame())
    reader = PreparedDatasetStore()
    with django_assert_num_queries(1):
        reader.prefetch(recipes)
        for recipe in recipes:
            loaded = reader.get(recipe)
            assert loaded is not None
            loaded._explanation.append('consumer-specific change')
            again = reader.get(recipe)
            assert again is not None
            assert again._explanation == ['source preparation']


def test_corrupt_payload_is_rebuilt() -> None:
    recipe = PreparedRecipe({'test': 'corrupt'})
    PreparedDataset.objects.create(key=recipe.key, recipe=recipe.content, payload=b'broken', frame_metadata={})
    store = PreparedDatasetStore()
    assert store.get(recipe) is None
    store.put(recipe, frame())
    loaded = PreparedDatasetStore().get(recipe)
    assert loaded is not None
    assert_frame_equal(loaded, frame())


def test_source_hash_metadata_and_recipe_invalidate_but_revision_does_not() -> None:
    ctx = context()
    dataset = DVCDataset(id='activity', context=ctx, column='Value', transformations=[SelectMetricOp()])
    original = dataset.prepared_recipe()
    assert original is not None
    assert ctx.dvc_source_manifest is not None
    source = ctx.dvc_source_manifest.datasets['activity']
    ctx.dvc_source_manifest.datasets['activity'] = source.model_copy(update={'revision': 'c' * 40})
    assert dataset.prepared_recipe() == original
    for changed in [{'content_hash': 'd' * 32}, {'units': {'Value': 'g'}}, {'index_columns': ['sector']}]:
        ctx.dvc_source_manifest.datasets['activity'] = source.model_copy(update=cast('dict[str, Any]', changed))
        assert dataset.prepared_recipe() != original
    ctx.dvc_source_manifest.datasets['activity'] = source
    dataset.tags.append('empty_to_zero')
    assert dataset.prepared_recipe() != original
    dataset.tags.clear()
    dataset.column = 'Other'
    assert dataset.prepared_recipe() != original


def test_parameter_and_year_dependencies_invalidate() -> None:
    ctx = context()
    ctx.get_parameter = MagicMock(return_value=SimpleNamespace(calculate_hash=lambda: 'a'))  # type: ignore[method-assign]
    dataset = DVCDataset(id='activity', context=ctx, transformations=[FilterColumnOp(column='sector', ref='sector')])
    original = dataset.prepared_recipe()
    ctx.get_parameter.return_value.calculate_hash = lambda: 'b'
    assert dataset.prepared_recipe() != original
    ctx.instance = MagicMock(reference_year=2020, target_year=2030)
    dataset.transformations = [RemapLegacyYearsOp()]
    original = dataset.prepared_recipe()
    ctx.instance.target_year = 2040
    assert dataset.prepared_recipe() != original


def test_opaque_operations_stop_the_cached_prefix() -> None:
    dataset = DVCDataset(
        id='activity',
        context=context(),
        transformations=[FilterColumnOp(column='sector', value='a'), TagOperationOp(tag='custom'), SelectMetricOp()],
    )
    assert dataset.prepared_prefix_length() == 1


def test_cross_context_reuse_still_runs_instance_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    first = DVCDataset(id='activity', context=context(), transformations=[FilterColumnOp(column='sector', value='a')])
    second = DVCDataset(id='activity', context=context(), transformations=first.transformations)
    monkeypatch.setattr(DVCDataset, 'cache_get', lambda _self: None)
    monkeypatch.setattr(DVCDataset, 'cache_set', lambda _self, _df: None)
    source = MagicMock()
    first.context.load_dvc_dataset = MagicMock(return_value=source)  # type: ignore[method-assign]
    second.context.load_dvc_dataset = MagicMock(side_effect=AssertionError('source read on cache hit'))  # type: ignore[method-assign]
    with (
        patch.object(DVCDataset, '_convert_dvc_dataset', return_value=frame()),
        patch.object(first, 'before_temporal_fill', side_effect=lambda df: df.with_columns(pl.col('Value') + 10)),
        patch.object(second, 'before_temporal_fill', side_effect=lambda df: df.with_columns(pl.col('Value') + 20)),
    ):
        assert first.load_internal()['Value'].to_list() == [11.0]
        assert second.load_internal()['Value'].to_list() == [21.0]
    first.context.load_dvc_dataset.assert_called_once()
    assert PreparedDataset.objects.count() == 1


def test_db_backed_framework_binding_has_no_dvc_recipe() -> None:
    from frameworks.datasets import FrameworkMeasureDVCDataset2

    dataset = FrameworkMeasureDVCDataset2(id='activity', context=context(), db_dataset_obj=MagicMock())
    assert dataset.dvc_source_id() is None
    assert dataset.prepared_recipe() is None


def test_implementation_version_invalidates(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = DVCDataset(id='activity', context=context(), transformations=[SelectMetricOp()], column='Value')
    recipe = dataset.prepared_recipe()
    monkeypatch.setattr(SelectMetricOp, 'cache_version', SelectMetricOp.cache_version + 1)
    assert dataset.prepared_recipe() != recipe
