from datetime import timedelta
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.utils import timezone

import pytest

from datasets.management.commands.prune_dataset_caches import Command
from datasets.models import DVCSourceManifest, PreparedDataset
from datasets.tests.test_manifests import SPEC, manifest_for, save_manifest

pytestmark = pytest.mark.django_db


def test_cleanup_retains_references_and_recent_entries() -> None:
    referenced = save_manifest(manifest_for('current'))
    stale = save_manifest(manifest_for('stale'))
    recent = save_manifest(manifest_for('recent'))
    DVCSourceManifest.objects.filter(pk__in=[referenced.pk, stale.pk]).update(updated_at=timezone.now() - timedelta(days=10))
    PreparedDataset.objects.create(key='stale', recipe={}, payload=b'', frame_metadata={})
    PreparedDataset.objects.filter(key='stale').update(created_at=timezone.now() - timedelta(days=10))
    sources = {(SPEC.url, SPEC.commit, 'storage', 'current', 1)}
    with patch.object(Command, 'collect_references', return_value=(sources, set())):
        call_command('prune_dataset_caches', stdout=StringIO())
        assert DVCSourceManifest.objects.count() == 3
        call_command('prune_dataset_caches', '--delete', stdout=StringIO())
    assert set(DVCSourceManifest.objects.values_list('pk', flat=True)) == {referenced.pk, recent.pk}
    assert not PreparedDataset.objects.exists()


def test_failed_reference_scan_never_deletes() -> None:
    row = save_manifest(manifest_for('keep'))
    DVCSourceManifest.objects.filter(pk=row.pk).update(updated_at=timezone.now() - timedelta(days=10))
    with (
        patch.object(Command, 'collect_references', side_effect=ValueError('invalid model')),
        pytest.raises(CommandError, match='no cache entries were deleted'),
    ):
        call_command('prune_dataset_caches', '--delete')
    assert DVCSourceManifest.objects.filter(pk=row.pk).exists()
