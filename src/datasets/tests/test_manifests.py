from contextlib import nullcontext
from datetime import UTC, datetime
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.core.management.base import CommandError

import polars as pl
import pytest
from dvc_pandas import DatasetLoader, DatasetManifest, RepositoryManifest

from datasets.management.commands.prepare_dvc_manifests import Command
from datasets.manifests import persisted_manifest
from datasets.models import DVCSourceManifest
from nodes.context import Context
from nodes.defs.instance_defs import DatasetRepoSpec
from nodes.tests.factories import InstanceConfigFactory

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from pytest_django import DjangoAssertNumQueries

pytestmark = pytest.mark.django_db
REVISION = 'a' * 40
SPEC = DatasetRepoSpec(url='https://example.invalid/data.git', commit=REVISION, dvc_remote='storage')


def manifest_for(*identifiers: str) -> RepositoryManifest:
    return RepositoryManifest(
        repository_url=SPEC.url,
        revision=REVISION,
        datasets={
            identifier: DatasetManifest(
                repository_url=SPEC.url,
                revision=REVISION,
                identifier=identifier,
                content_hash='b' * 32,
                object_url='s3://example/files/md5/bb/' + 'b' * 30,
                modified_at=datetime(2026, 1, 1, tzinfo=UTC),
                units={'Value': 'kg'},
                index_columns=['Year'],
            )
            for identifier in identifiers
        },
    )


def save_manifest(manifest: RepositoryManifest) -> DVCSourceManifest:
    return DVCSourceManifest.objects.create(
        repository_url=SPEC.url,
        revision=REVISION,
        remote_name='storage',
        format_version=1,
        dataset_identifier=next(iter(manifest.datasets)),
        content=next(iter(manifest.datasets.values())).model_dump(mode='json'),
    )


def test_persisted_hit_never_initializes_repository(django_assert_num_queries: DjangoAssertNumQueries) -> None:
    expected = manifest_for('activity')
    save_manifest(expected)
    repository = MagicMock(side_effect=AssertionError('Git must not be initialized'))
    with django_assert_num_queries(1):
        assert persisted_manifest(SPEC, {'activity'}, repository) == expected


@pytest.mark.parametrize('content', [{}, {'version': 99}])
def test_missing_or_invalid_metadata_is_repaired(content: dict[str, Any]) -> None:
    row = save_manifest(manifest_for('activity'))
    row.content = content
    row.save()
    repository = MagicMock()
    repository.return_value.get_manifest.return_value = manifest_for('activity')
    assert persisted_manifest(SPEC, {'activity'}, repository) == manifest_for('activity')
    row.refresh_from_db()
    assert DatasetManifest.model_validate(row.content) == manifest_for('activity').datasets['activity']


def test_concurrent_additions_are_merged() -> None:
    repository = MagicMock()

    def resolve(identifiers: list[str]) -> RepositoryManifest:
        assert identifiers == ['activity']
        # Another worker finishes its network lookup before this worker persists.
        save_manifest(manifest_for('other'))
        return manifest_for('activity')

    repository.return_value.get_manifest.side_effect = resolve
    result = persisted_manifest(SPEC, {'activity'}, repository)
    assert result is not None
    assert set(result.datasets) == {'activity'}
    repository.reset_mock()
    assert persisted_manifest(SPEC, {'activity', 'other'}, repository) == manifest_for('activity', 'other')
    repository.assert_not_called()


def test_new_subset_only_resolves_missing_entries() -> None:
    save_manifest(manifest_for('activity'))
    repository = MagicMock()
    repository.return_value.get_manifest.return_value = manifest_for('other')
    result = persisted_manifest(SPEC, {'activity', 'other'}, repository)
    repository.return_value.get_manifest.assert_called_once_with(['other'])
    assert result is not None
    assert set(result.datasets) == {'activity', 'other'}


@pytest.mark.parametrize('commit', [None, 'main', 'abc123'])
def test_moving_or_abbreviated_revision_uses_git(commit: str | None, django_assert_num_queries: DjangoAssertNumQueries) -> None:
    repository = MagicMock()
    with django_assert_num_queries(0):
        assert persisted_manifest(DatasetRepoSpec(url=SPEC.url, commit=commit), {'activity'}, repository) is None
    repository.assert_not_called()


def test_context_prefetch_and_load_without_git(tmp_path: Path) -> None:
    import hashlib

    parquet = tmp_path / 'source.parquet'
    frame = pl.DataFrame({'Year': [2020], 'Value': [1.0]})
    frame.write_parquet(parquet)
    source = (
        manifest_for('activity')
        .datasets['activity']
        .model_copy(
            update={
                'object_url': parquet.as_uri(),
                'content_hash': hashlib.md5(parquet.read_bytes(), usedforsecurity=False).hexdigest(),
            }
        )
    )
    save_manifest(RepositoryManifest(repository_url=SPEC.url, revision=REVISION, datasets={'activity': source}))
    context = Context.__new__(Context)
    context.dataset_repo_spec = SPEC
    context.dvc_datasets = {}
    context.dvc_manifest_loader = DatasetLoader(cache_root=tmp_path / 'cache')
    context.start_perf_span = MagicMock(return_value=nullcontext((None, None)))  # type: ignore[method-assign]
    with (
        patch.object(Context, 'get_all_dvc_dataset_ids', return_value={'activity'}),
        patch('dvc_pandas.Repository', side_effect=AssertionError('Git forbidden')),
    ):
        context.warm_dvc_cache()
        parquet.unlink()
        loaded = context.load_dvc_dataset('activity')
    assert loaded.df is not None
    assert loaded.df.equals(frame)
    assert loaded.manifest == source
    assert 'dataset_repo' not in context.__dict__


def test_preparation_selection_and_failure_continuation() -> None:
    first = InstanceConfigFactory.create(name='first', identifier='first', in_customer_use=True)
    second = InstanceConfigFactory.create(name='second', identifier='second', in_customer_use=True)
    InstanceConfigFactory.create(name='unused')
    InstanceConfigFactory.create(name='inactive', in_customer_use=True, is_active=False)
    with (
        patch.object(Command, 'prepare_instance', side_effect=[ValueError('network error'), True]) as prepare,
        pytest.raises(CommandError, match='Failed instances: first'),
    ):
        call_command('prepare_dvc_manifests', '--in-customer-use', stdout=StringIO())
    assert [call.args[0].pk for call in prepare.call_args_list] == [first.pk, second.pk]


@pytest.mark.parametrize('fail', [False, True])
def test_preparation_cleans_runtime(fail: bool) -> None:
    config = MagicMock()
    instance = config.enter_instance_context.return_value.__enter__.return_value
    if fail:
        from unittest.mock import PropertyMock

        with patch.object(type(instance.context), 'dvc_source_manifest', new_callable=PropertyMock, create=True) as prop:
            prop.side_effect = ValueError('metadata unavailable')
            with pytest.raises(ValueError, match='metadata unavailable'):
                Command().prepare_instance(config)
    else:
        assert Command().prepare_instance(config)
    instance.clean.assert_called_once()


def test_read_only_lookup_does_not_resolve_missing_rows() -> None:
    save_manifest(manifest_for('activity'))
    result = persisted_manifest(SPEC, {'activity', 'absent'}, None, resolve_missing=False)
    assert result == manifest_for('activity')
    assert DVCSourceManifest.objects.count() == 1


def test_remote_selection_is_part_of_identity() -> None:
    save_manifest(manifest_for('activity'))
    other_spec = SPEC.model_copy(update={'dvc_remote': 'other'})
    repository = MagicMock()
    repository.return_value.get_manifest.return_value = manifest_for('activity')
    persisted_manifest(other_spec, {'activity'}, repository)
    repository.assert_called_once()
    assert DVCSourceManifest.objects.count() == 2
